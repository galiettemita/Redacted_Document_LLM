from __future__ import annotations

import shutil
from pathlib import Path

import pymupdf
import pytest

from redacted_backend.documents.extract import OcrWord, extract_visible_words


def test_uses_injected_ocr_when_a_page_has_no_usable_text_layer(tmp_path: Path) -> None:
    pdf_path = tmp_path / "image-only.pdf"
    document = pymupdf.open()
    document.new_page()
    document.save(pdf_path)
    document.close()

    def fake_ocr(_image):
        return [
            OcrWord(
                text="VISIBLE",
                left=100,
                top=100,
                width=120,
                height=30,
                confidence=0.99,
            )
        ]

    pages = extract_visible_words(pdf_path, _ocr_engine=fake_ocr)

    assert pages[0].extraction_source == "ocr"
    assert [word.text for word in pages[0].words] == ["VISIBLE"]


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="Tesseract not installed")
def test_real_tesseract_reads_the_public_scanned_fixture() -> None:
    fixture = (
        Path(__file__).parents[1] / "fixtures" / "pdfs" / "scanned-visible-layout.pdf"
    )

    pages = extract_visible_words(fixture)

    visible_text = " ".join(word.text for word in pages[0].words).upper()
    assert pages[0].extraction_source == "ocr"
    assert "VISIBLE" in visible_text
    assert "BULLETIN" in visible_text
