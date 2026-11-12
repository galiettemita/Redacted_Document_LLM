from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from uuid import UUID

import numpy as np
import pytesseract
import pytest

from redacted_backend.documents.preparation import (
    VisibleContextPreparationError,
    prepare_visible_context,
)
from redacted_backend.domain.documents import DocumentRecord
from redacted_backend.domain.passages import PassageRect
from tests.documents.factories import discovery, mask, page, passage, word

DOCUMENT_ID = UUID("00000000-0000-0000-0000-000000000001")
TARGET_ID = UUID("00000000-0000-0000-0000-000000000010")


def document_record(*, visible_date: date | None = date(1945, 5, 8)) -> DocumentRecord:
    now = datetime.now(UTC)
    return DocumentRecord(
        id=DOCUMENT_ID,
        filename="visible-bulletin.pdf",
        sha256="a" * 64,
        media_type="application/pdf",
        page_count=1,
        status="uploaded",
        storage_key=f"sha256/aa/aa/{'a' * 64}.pdf",
        visible_date=visible_date,
        date_provenance=("manual_visible_document" if visible_date else None),
        created_at=now,
        expires_at=now + timedelta(hours=24),
    )


def target_passage():
    return passage(
        str(TARGET_ID),
        rects=[PassageRect(x=0.35, y=0.2, width=0.18, height=0.025)],
    )


def visible_page():
    return page(
        word("alpha", x=0.05, y=0.2, block_id=1, line_id=1),
        word("bravo", x=0.12, y=0.2, block_id=1, line_id=1),
        word("delta", x=0.19, y=0.2, block_id=1, line_id=1),
        word("HIDDEN", x=0.55, y=0.2, block_id=1, line_id=1),
    )


def test_preparation_feeds_each_rendered_page_to_mask_discovery() -> None:
    rendered = np.full((100, 100), 255, dtype=np.uint8)
    seen: list[tuple[object, list[str], int]] = []

    def discover(image, words, *, page_number):
        seen.append((image, [item.text for item in words], page_number))
        return discovery(
            page_number,
            mask(
                "auto-black-1-1",
                x=0.35,
                y=0.2,
                width=0.18,
                height=0.025,
            ),
        )

    result = prepare_visible_context(
        Path("stored.pdf"),
        document_record(),
        [target_passage()],
        TARGET_ID,
        _extractor=lambda _path: [visible_page()],
        _renderer=lambda _path: [rendered],
        _discoverer=discover,
    )

    assert seen == [(rendered, ["alpha", "bravo", "delta", "HIDDEN"], 1)]
    assert result.confirmed_mask_count == 1
    assert result.visible_word_count == 4


def test_preparation_never_serializes_text_covered_by_a_confirmed_mask() -> None:
    result = prepare_visible_context(
        Path("stored.pdf"),
        document_record(),
        [target_passage()],
        TARGET_ID,
        _extractor=lambda _path: [visible_page()],
        _renderer=lambda _path: [np.zeros((10, 10), dtype=np.uint8)],
        _discoverer=lambda _image, _words, *, page_number: discovery(
            page_number,
            mask(
                "covered-hidden",
                x=0.54,
                y=0.19,
                width=0.09,
                height=0.05,
            ),
        ),
    )

    serialized = result.example.model_dump_json()
    assert "HIDDEN" not in serialized
    assert "answer" not in serialized.lower()


@pytest.mark.parametrize(
    ("failure", "expected_code"),
    [
        (
            pytesseract.TesseractNotFoundError(),
            "ocr_unavailable",
        ),
        (
            RuntimeError("damaged file"),
            "pdf_processing_failed",
        ),
    ],
)
def test_preparation_converts_processing_failures_to_stable_safe_codes(
    failure: Exception,
    expected_code: str,
) -> None:
    def fail(_path):
        raise failure

    with pytest.raises(VisibleContextPreparationError) as error:
        prepare_visible_context(
            Path("stored.pdf"),
            document_record(),
            [target_passage()],
            TARGET_ID,
            _extractor=fail,
            _renderer=lambda _path: [],
        )

    assert error.value.code == expected_code
    assert "damaged file" not in str(error.value)


@pytest.mark.parametrize(
    ("discoveries", "expected_code"),
    [
        ([discovery(1, coverage_confident=False)], "mask_coverage_uncertain"),
        ([discovery(1)], "insufficient_visible_text_for_length"),
    ],
)
def test_preparation_preserves_safe_visible_example_failure_codes(
    discoveries,
    expected_code: str,
) -> None:
    pages = [visible_page()] if expected_code == "mask_coverage_uncertain" else [page()]

    with pytest.raises(VisibleContextPreparationError) as error:
        prepare_visible_context(
            Path("stored.pdf"),
            document_record(),
            [target_passage()],
            TARGET_ID,
            _extractor=lambda _path: pages,
            _renderer=lambda _path: [np.zeros((10, 10), dtype=np.uint8)],
            _discoverer=lambda _image, _words, *, page_number: discoveries[0],
        )

    assert error.value.code == expected_code
