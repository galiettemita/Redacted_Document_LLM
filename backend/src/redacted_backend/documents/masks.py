from __future__ import annotations

import statistics
from itertools import pairwise

import cv2
import numpy as np
from PIL import Image

from redacted_backend.domain.passages import PassageRect
from redacted_backend.domain.visible_examples import (
    MaskCandidate,
    MaskDiscovery,
    PageWord,
)


def discover_candidate_masks(
    page_image: np.ndarray | Image.Image,
    words: list[PageWord],
    *,
    page_number: int,
) -> MaskDiscovery:
    image = np.asarray(page_image)
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.astype(np.uint8)
    height, width = gray.shape[:2]
    _, black_binary = cv2.threshold(gray, 55, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(
        black_binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    candidates: list[MaskCandidate] = []
    for x, y, candidate_width, candidate_height in sorted(
        (cv2.boundingRect(contour) for contour in contours),
        key=lambda rect: (rect[1], rect[0]),
    ):
        normalized_width = candidate_width / width
        normalized_height = candidate_height / height
        if (
            normalized_width < 0.02
            or normalized_height < 0.008
            or normalized_width > 0.95
            or normalized_height > 0.5
        ):
            continue
        candidates.append(
            MaskCandidate(
                id=f"auto-black-{page_number}-{len(candidates) + 1}",
                page_number=page_number,
                rect=PassageRect(
                    x=x / width,
                    y=y / height,
                    width=normalized_width,
                    height=normalized_height,
                ),
                kind="black",
                confidence=1.0,
                confirmed=True,
            )
        )

    candidates.extend(
        _covered_embedded_word_candidates(
            gray=gray,
            words=words,
            page_number=page_number,
            starting_index=len(candidates),
        )
    )
    candidates.extend(_white_gap_candidates(words, page_number, len(candidates)))
    coverage_confident = len(words) >= 3 or (len(words) >= 2 and bool(candidates))
    warnings = [] if coverage_confident else ["insufficient_visible_content"]
    return MaskDiscovery(
        page_number=page_number,
        masks=candidates,
        coverage_confident=coverage_confident,
        warnings=warnings,
    )


def _covered_embedded_word_candidates(
    *,
    gray: np.ndarray,
    words: list[PageWord],
    page_number: int,
    starting_index: int,
) -> list[MaskCandidate]:
    height, width = gray.shape[:2]
    ratios: list[tuple[PageWord, float]] = []
    for word in words:
        if word.source.value != "embedded":
            continue
        left = max(0, min(width - 1, round(word.x * width)))
        top = max(0, min(height - 1, round(word.y * height)))
        right = max(left + 1, min(width, round((word.x + word.width) * width)))
        bottom = max(top + 1, min(height, round((word.y + word.height) * height)))
        crop = gray[top:bottom, left:right]
        ratios.append((word, float(np.mean(crop < 180))))
    if sum(ratio >= 0.02 for _, ratio in ratios) < 2:
        return []
    candidates: list[MaskCandidate] = []
    for word, ratio in ratios:
        if ratio >= 0.005:
            continue
        candidates.append(
            MaskCandidate(
                id=(
                    f"auto-white-covered-{page_number}-"
                    f"{starting_index + len(candidates) + 1}"
                ),
                page_number=page_number,
                rect=word.rect,
                kind="white",
                confidence=0.95,
                confirmed=True,
            )
        )
    return candidates


def _white_gap_candidates(
    words: list[PageWord],
    page_number: int,
    starting_index: int,
) -> list[MaskCandidate]:
    character_widths = [
        word.width / len(word.text)
        for word in words
        if word.text and len(word.text) > 0
    ]
    if len(character_widths) < 2:
        return []
    gap_threshold = max(0.08, statistics.median(character_widths) * 12)
    by_line: dict[tuple[int, int], list[PageWord]] = {}
    for word in words:
        by_line.setdefault((word.block_id, word.line_id), []).append(word)

    candidates: list[MaskCandidate] = []
    seen: set[tuple[float, float, float, float]] = set()
    for line_words in by_line.values():
        _append_white_gaps(
            candidates,
            seen,
            line_words,
            page_number=page_number,
            starting_index=starting_index,
            gap_threshold=gap_threshold,
            cross_block=False,
        )
    for line_words in _visual_lines(words):
        _append_white_gaps(
            candidates,
            seen,
            line_words,
            page_number=page_number,
            starting_index=starting_index,
            gap_threshold=gap_threshold,
            cross_block=True,
        )
    return candidates


def _append_white_gaps(
    candidates: list[MaskCandidate],
    seen: set[tuple[float, float, float, float]],
    line_words: list[PageWord],
    *,
    page_number: int,
    starting_index: int,
    gap_threshold: float,
    cross_block: bool,
) -> None:
    ordered = sorted(line_words, key=lambda word: word.x)
    for left, right in pairwise(ordered):
        if cross_block and (left.block_id, left.line_id) == (
            right.block_id,
            right.line_id,
        ):
            continue
        gap_start = left.x + left.width
        gap_width = right.x - gap_start
        if gap_width < gap_threshold or (cross_block and gap_width > 0.20):
            continue
        y = min(left.y, right.y)
        bottom = max(left.y + left.height, right.y + right.height)
        key = tuple(round(value, 6) for value in (gap_start, y, gap_width, bottom - y))
        if key in seen:
            continue
        seen.add(key)
        candidates.append(
            MaskCandidate(
                id=(f"auto-white-{page_number}-{starting_index + len(candidates) + 1}"),
                page_number=page_number,
                rect=PassageRect(
                    x=gap_start,
                    y=y,
                    width=gap_width,
                    height=bottom - y,
                ),
                kind="white",
                confidence=0.8,
                confirmed=True,
            )
        )


def _visual_lines(words: list[PageWord]) -> list[list[PageWord]]:
    lines: list[list[PageWord]] = []
    for word in sorted(words, key=lambda item: (item.y, item.x)):
        center = word.y + word.height / 2
        matching = next(
            (
                line
                for line in lines
                if abs(
                    center
                    - statistics.median(item.y + item.height / 2 for item in line)
                )
                <= 0.6
                * max(word.height, statistics.median(item.height for item in line))
            ),
            None,
        )
        if matching is None:
            lines.append([word])
        else:
            matching.append(word)
    return lines
