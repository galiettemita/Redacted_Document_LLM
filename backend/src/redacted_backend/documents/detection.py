from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
import pytesseract
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from redacted_backend.documents.extract import extract_visible_words
from redacted_backend.documents.masks import discover_candidate_masks
from redacted_backend.documents.preparation import iter_page_images
from redacted_backend.domain.passages import PassageRect
from redacted_backend.domain.visible_examples import (
    MaskCandidate,
    MaskDiscovery,
    MaskKind,
    PageWord,
    PageWords,
)


class DetectedPassage(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    page_number: int = Field(ge=1)
    rects: list[PassageRect] = Field(min_length=1)
    kinds: list[MaskKind] = Field(min_length=1)
    source_mask_ids: list[str] = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)


class DetectionResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    passages: list[DetectedPassage]
    pages_scanned: int = Field(ge=1)
    coverage_confident: bool
    uncertain_pages: list[int]
    warnings: list[str]


class PassageDetectionError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


PageImage = np.ndarray | Image.Image
Extractor = Callable[[Path], list[PageWords]]
Renderer = Callable[[Path], Iterable[PageImage]]
Discoverer = Callable[..., MaskDiscovery]


def detect_passages(
    pdf_path: Path,
    *,
    _extractor: Extractor = extract_visible_words,
    _renderer: Renderer = iter_page_images,
    _discoverer: Discoverer = discover_candidate_masks,
) -> DetectionResult:
    try:
        pages = _extractor(pdf_path)
        images = iter(_renderer(pdf_path))
        discoveries: list[MaskDiscovery] = []
        for page in pages:
            try:
                image = next(images)
            except StopIteration as error:
                raise PassageDetectionError(
                    "page_render_mismatch",
                    "The extracted and rendered page counts do not match",
                ) from error
            discoveries.append(
                _discoverer(image, page.words, page_number=page.page_number)
            )
        try:
            next(images)
        except StopIteration:
            pass
        else:
            raise PassageDetectionError(
                "page_render_mismatch",
                "The extracted and rendered page counts do not match",
            )
    except PassageDetectionError:
        raise
    except pytesseract.TesseractNotFoundError as error:
        raise PassageDetectionError(
            "ocr_unavailable",
            "OCR is unavailable for this document",
        ) from error
    except Exception as error:
        raise PassageDetectionError(
            "pdf_processing_failed",
            "The PDF could not be scanned safely",
        ) from error
    return group_detected_passages(pages, discoveries)


def group_detected_passages(
    pages: list[PageWords],
    discoveries: list[MaskDiscovery],
) -> DetectionResult:
    words_by_page = {page.page_number: page.words for page in pages}
    discoveries_by_page = {item.page_number: item for item in discoveries}
    uncertain_pages = sorted(
        page.page_number
        for page in pages
        if not discoveries_by_page.get(
            page.page_number,
            MaskDiscovery(
                page_number=page.page_number,
                masks=[],
                coverage_confident=False,
                warnings=["mask_discovery_missing"],
            ),
        ).coverage_confident
    )
    warnings = [
        f"page {page_number}: {warning}"
        for page_number in sorted(discoveries_by_page)
        for warning in discoveries_by_page[page_number].warnings
    ]
    missing_pages = sorted(set(words_by_page) - set(discoveries_by_page))
    warnings.extend(
        f"page {page_number}: mask_discovery_missing" for page_number in missing_pages
    )

    grouped_by_page: dict[int, list[DetectedPassage]] = defaultdict(list)
    for page_number in sorted(discoveries_by_page):
        masks = [
            candidate
            for candidate in discoveries_by_page[page_number].masks
            if candidate.confirmed and candidate.kind != MaskKind.MANUAL
        ]
        grouped_by_page[page_number] = _group_page_masks(
            masks,
            words_by_page.get(page_number, []),
        )

    ordered: list[DetectedPassage] = []
    for page_number in sorted(grouped_by_page):
        page_passages = grouped_by_page[page_number]
        column_starts = _column_starts(page_passages)
        ordered.extend(
            sorted(
                page_passages,
                key=lambda passage: (
                    _column_for(passage, column_starts),
                    min(rect.y for rect in passage.rects),
                    min(rect.x for rect in passage.rects),
                ),
            )
        )

    return DetectionResult(
        passages=ordered,
        pages_scanned=len(pages),
        coverage_confident=not uncertain_pages,
        uncertain_pages=uncertain_pages,
        warnings=warnings,
    )


def _group_page_masks(
    masks: list[MaskCandidate],
    words: list[PageWord],
) -> list[DetectedPassage]:
    ordered = sorted(masks, key=lambda item: (item.rect.y, item.rect.x, item.id))
    remaining = set(range(len(ordered)))
    groups: list[list[MaskCandidate]] = []
    while remaining:
        seed = min(remaining)
        remaining.remove(seed)
        component = [seed]
        pending = [seed]
        while pending:
            current = pending.pop()
            for candidate in sorted(remaining):
                if not _belongs_to_same_passage(
                    ordered[current].rect,
                    ordered[candidate].rect,
                    words,
                ):
                    continue
                remaining.remove(candidate)
                component.append(candidate)
                pending.append(candidate)
        groups.append([ordered[index] for index in component])

    return [
        DetectedPassage(
            page_number=group[0].page_number,
            rects=[item.rect for item in sorted(group, key=_mask_order_key)],
            kinds=[item.kind for item in sorted(group, key=_mask_order_key)],
            source_mask_ids=[item.id for item in sorted(group, key=_mask_order_key)],
            confidence=min(item.confidence for item in group),
        )
        for group in groups
    ]


def _belongs_to_same_passage(
    first: PassageRect,
    second: PassageRect,
    words: list[PageWord],
) -> bool:
    if _readable_word_between(first, second, words):
        return False
    horizontal_gap = max(
        0.0,
        max(first.x, second.x) - min(first.x + first.width, second.x + second.width),
    )
    vertical_gap = max(
        0.0,
        max(first.y, second.y) - min(first.y + first.height, second.y + second.height),
    )
    vertical_overlap = _overlap(
        first.y,
        first.y + first.height,
        second.y,
        second.y + second.height,
    )
    horizontal_overlap = _overlap(
        first.x,
        first.x + first.width,
        second.x,
        second.x + second.width,
    )
    same_line_touching = (
        vertical_overlap >= 0.5 * min(first.height, second.height)
        and horizontal_gap <= 0.01
    )
    same_column_wrapped = vertical_gap <= 1.5 * max(first.height, second.height) and (
        horizontal_overlap >= 0.25 * min(first.width, second.width)
        or abs(first.x - second.x) <= 0.12
    )
    return same_line_touching or same_column_wrapped


def _readable_word_between(
    first: PassageRect,
    second: PassageRect,
    words: list[PageWord],
) -> bool:
    left, right = sorted((first, second), key=lambda rect: rect.x)
    top, bottom = sorted((first, second), key=lambda rect: rect.y)
    for word in words:
        center_x = word.x + word.width / 2
        center_y = word.y + word.height / 2
        horizontally_between = (
            left.x + left.width < center_x < right.x
            and min(first.y, second.y) - word.height
            <= center_y
            <= max(first.y + first.height, second.y + second.height) + word.height
        )
        vertically_between = (
            top.y + top.height < center_y < bottom.y
            and min(first.x, second.x) - word.width
            <= center_x
            <= max(first.x + first.width, second.x + second.width) + word.width
        )
        if horizontally_between or vertically_between:
            return True
    return False


def _column_starts(passages: list[DetectedPassage]) -> list[float]:
    starts: list[float] = []
    for passage in sorted(
        passages, key=lambda item: min(rect.x for rect in item.rects)
    ):
        start = min(rect.x for rect in passage.rects)
        if not starts or start - starts[-1] > 0.25:
            starts.append(start)
    return starts or [0.0]


def _column_for(passage: DetectedPassage, starts: list[float]) -> int:
    start = min(rect.x for rect in passage.rects)
    return min(range(len(starts)), key=lambda index: abs(start - starts[index]))


def _overlap(
    first_start: float, first_end: float, second_start: float, second_end: float
) -> float:
    return max(0.0, min(first_end, second_end) - max(first_start, second_start))


def _mask_order_key(mask: MaskCandidate) -> tuple[float, float, str]:
    return mask.rect.y, mask.rect.x, mask.id
