from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

import numpy as np
import pymupdf
import pytesseract
from PIL import Image

from redacted_backend.documents.extract import extract_visible_words
from redacted_backend.documents.dates import (
    VisibleDateDetectionError,
    detect_visible_document_date,
)
from redacted_backend.documents.masks import discover_candidate_masks
from redacted_backend.documents.visible_example import (
    VisibleExampleError,
    build_visible_example,
)
from redacted_backend.domain.documents import DocumentRecord, VisibleDateEvidence
from redacted_backend.domain.passages import PassageRecord
from redacted_backend.domain.visible_examples import (
    MaskDiscovery,
    PageWords,
    VisibleDocument,
    VisibleExample,
)

PageImage = np.ndarray | Image.Image
Extractor = Callable[[Path], list[PageWords]]
Renderer = Callable[[Path], list[PageImage]]
Discoverer = Callable[..., MaskDiscovery]


class VisibleContextPreparationError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class PreparedVisibleContext:
    example: VisibleExample
    confirmed_mask_count: int
    visible_word_count: int
    date_evidence: VisibleDateEvidence


def prepare_visible_context(
    pdf_path: Path,
    document: DocumentRecord,
    passages: list[PassageRecord],
    target_id: UUID,
    *,
    _extractor: Extractor = extract_visible_words,
    _renderer: Renderer | None = None,
    _discoverer: Discoverer = discover_candidate_masks,
) -> PreparedVisibleContext:
    renderer = _renderer or render_page_images
    try:
        pages = _extractor(pdf_path)
        page_images = renderer(pdf_path)
        if len(pages) != len(page_images):
            raise VisibleContextPreparationError(
                "page_render_mismatch",
                "The extracted and rendered page counts do not match",
            )
        discoveries = [
            _discoverer(
                page_image,
                page.words,
                page_number=page.page_number,
            )
            for page, page_image in zip(pages, page_images, strict=True)
        ]
    except VisibleContextPreparationError:
        raise
    except pytesseract.TesseractNotFoundError as error:
        raise VisibleContextPreparationError(
            "ocr_unavailable",
            "OCR is unavailable for this document",
        ) from error
    except Exception as error:
        raise VisibleContextPreparationError(
            "pdf_processing_failed",
            "The PDF could not be processed safely",
        ) from error

    try:
        date_evidence = detect_visible_document_date(pages, discoveries)
    except VisibleDateDetectionError as error:
        raise VisibleContextPreparationError(error.code, str(error)) from error

    visible_document = VisibleDocument(
        id=document.id,
        visible_date=date_evidence.value,
        pages=pages,
        mask_discoveries=discoveries,
    )
    try:
        example = build_visible_example(visible_document, passages, target_id)
    except VisibleExampleError as error:
        raise VisibleContextPreparationError(error.code, str(error)) from error

    return PreparedVisibleContext(
        example=example,
        confirmed_mask_count=sum(
            candidate.confirmed
            for discovery in discoveries
            for candidate in discovery.masks
        ),
        visible_word_count=len(example.visible_words),
        date_evidence=date_evidence,
    )


def render_page_images(pdf_path: Path) -> list[Image.Image]:
    document = pymupdf.open(pdf_path)
    try:
        images: list[Image.Image] = []
        for page in document:
            pixmap = page.get_pixmap(matrix=pymupdf.Matrix(2, 2), alpha=False)
            images.append(
                Image.frombytes(
                    "RGB",
                    (pixmap.width, pixmap.height),
                    pixmap.samples,
                )
            )
        return images
    finally:
        document.close()
