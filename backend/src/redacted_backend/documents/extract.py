from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pymupdf
import pytesseract
from PIL import Image
from pytesseract import Output

from redacted_backend.domain.visible_examples import PageWord, PageWords


@dataclass(frozen=True)
class OcrWord:
    text: str
    left: int
    top: int
    width: int
    height: int
    confidence: float
    block_id: int = 0
    line_id: int = 0


OcrEngine = Callable[[Image.Image], list[OcrWord]]


def extract_visible_words(
    pdf_path: Path,
    *,
    _ocr_engine: OcrEngine | None = None,
) -> list[PageWords]:
    ocr_engine = _ocr_engine or _tesseract_words
    document = pymupdf.open(pdf_path)
    try:
        pages: list[PageWords] = []
        for page_index, pdf_page in enumerate(document):
            embedded = _embedded_words(pdf_page, page_index + 1)
            if len(embedded) >= 3:
                pages.append(
                    PageWords(
                        page_number=page_index + 1,
                        width_points=pdf_page.rect.width,
                        height_points=pdf_page.rect.height,
                        extraction_source="embedded",
                        words=embedded,
                    )
                )
                continue

            pixmap = pdf_page.get_pixmap(matrix=pymupdf.Matrix(2, 2), alpha=False)
            image = Image.frombytes(
                "RGB",
                (pixmap.width, pixmap.height),
                pixmap.samples,
            )
            ocr_words = _normalize_ocr_words(
                ocr_engine(image),
                page_number=page_index + 1,
                image_width=image.width,
                image_height=image.height,
            )
            pages.append(
                PageWords(
                    page_number=page_index + 1,
                    width_points=pdf_page.rect.width,
                    height_points=pdf_page.rect.height,
                    extraction_source="ocr",
                    words=ocr_words,
                )
            )
        return pages
    finally:
        document.close()


def _embedded_words(page: pymupdf.Page, page_number: int) -> list[PageWord]:
    words: list[PageWord] = []
    for raw in page.get_text("words", sort=False):
        x0, y0, x1, y1, text, block_id, line_id, _word_id = raw[:8]
        cleaned = str(text).strip()
        if not cleaned or x1 <= x0 or y1 <= y0:
            continue
        words.append(
            PageWord(
                text=cleaned,
                page_number=page_number,
                x=max(0.0, x0 / page.rect.width),
                y=max(0.0, y0 / page.rect.height),
                width=min(1.0, (x1 - x0) / page.rect.width),
                height=min(1.0, (y1 - y0) / page.rect.height),
                block_id=int(block_id),
                line_id=int(line_id),
                source="embedded",
                confidence=1.0,
            )
        )
    return words


def _normalize_ocr_words(
    words: list[OcrWord],
    *,
    page_number: int,
    image_width: int,
    image_height: int,
) -> list[PageWord]:
    normalized: list[PageWord] = []
    for item in words:
        cleaned = item.text.strip()
        if not cleaned or item.width <= 0 or item.height <= 0:
            continue
        normalized.append(
            PageWord(
                text=cleaned,
                page_number=page_number,
                x=item.left / image_width,
                y=item.top / image_height,
                width=item.width / image_width,
                height=item.height / image_height,
                block_id=item.block_id,
                line_id=item.line_id,
                source="ocr",
                confidence=max(0.0, min(1.0, item.confidence)),
            )
        )
    return normalized


def _tesseract_words(image: Image.Image) -> list[OcrWord]:
    data = pytesseract.image_to_data(
        image,
        output_type=Output.DICT,
        config="--oem 1 --psm 3",
    )
    words: list[OcrWord] = []
    for index, text in enumerate(data["text"]):
        cleaned = text.strip()
        if not cleaned:
            continue
        confidence = float(data["conf"][index])
        if confidence < 0:
            continue
        words.append(
            OcrWord(
                text=cleaned,
                left=int(data["left"][index]),
                top=int(data["top"][index]),
                width=int(data["width"][index]),
                height=int(data["height"][index]),
                confidence=confidence / 100,
                block_id=int(data["block_num"][index]),
                line_id=int(data["line_num"][index]),
            )
        )
    return words
