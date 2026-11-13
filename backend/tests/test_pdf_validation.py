from pathlib import Path

import pytest

from redacted_backend.documents.validation import (
    PdfLimits,
    PdfValidationError,
    validate_pdf,
)
from tests.pdf_fixtures import (
    make_compression_heavy_pdf,
    make_encrypted_pdf,
    make_pdf,
    slow_pdf_inspector,
)


def write_pdf(path: Path, content: bytes) -> Path:
    path.write_bytes(content)
    return path


def assert_validation_code(path: Path, limits: PdfLimits, code: str) -> None:
    with pytest.raises(PdfValidationError) as error:
        validate_pdf(path, limits)
    assert error.value.code == code


def test_accepts_a_valid_pdf_at_the_configured_size_boundary(tmp_path: Path) -> None:
    pdf_bytes = make_pdf()
    path = write_pdf(tmp_path / "valid.pdf", pdf_bytes)

    metadata = validate_pdf(path, PdfLimits(max_file_bytes=len(pdf_bytes)))

    assert metadata.page_count == 1
    assert metadata.size_bytes == len(pdf_bytes)
    assert PdfLimits().max_file_bytes == 100 * 1024 * 1024


def test_rejects_one_byte_over_the_configured_size_boundary(tmp_path: Path) -> None:
    pdf_bytes = make_pdf()
    path = write_pdf(tmp_path / "too-large.pdf", pdf_bytes)

    assert_validation_code(
        path,
        PdfLimits(max_file_bytes=len(pdf_bytes) - 1),
        "file_too_large",
    )


def test_rejects_an_incorrect_pdf_signature(tmp_path: Path) -> None:
    path = write_pdf(tmp_path / "not.pdf", b"not a PDF")

    assert_validation_code(path, PdfLimits(), "invalid_signature")


def test_rejects_a_malformed_pdf_after_signature_validation(tmp_path: Path) -> None:
    path = write_pdf(tmp_path / "malformed.pdf", b"%PDF-this-is-not-valid")

    assert_validation_code(path, PdfLimits(), "malformed_pdf")


def test_rejects_an_encrypted_pdf(tmp_path: Path) -> None:
    path = write_pdf(tmp_path / "encrypted.pdf", make_encrypted_pdf())

    assert_validation_code(path, PdfLimits(), "encrypted_pdf")


def test_rejects_a_pdf_whose_streams_exceed_the_expansion_limit(tmp_path: Path) -> None:
    path = write_pdf(
        tmp_path / "compressed.pdf",
        make_compression_heavy_pdf(expanded_bytes=8_192),
    )

    assert_validation_code(
        path,
        PdfLimits(max_decompressed_bytes=1_024),
        "decompression_limit",
    )


def test_rejects_a_tiny_pdf_with_an_extreme_page_render_size(tmp_path: Path) -> None:
    path = write_pdf(
        tmp_path / "extreme-page.pdf",
        make_pdf(page_width=10_000, page_height=10_000),
    )

    assert_validation_code(path, PdfLimits(), "page_render_limit")


def test_terminates_pdf_inspection_after_the_parser_timeout(tmp_path: Path) -> None:
    path = write_pdf(tmp_path / "slow.pdf", make_pdf())

    with pytest.raises(PdfValidationError) as error:
        validate_pdf(
            path,
            PdfLimits(parser_timeout_seconds=0.05),
            _inspector=slow_pdf_inspector,
        )

    assert error.value.code == "parser_timeout"


@pytest.mark.parametrize(
    ("page_count", "expected_code"),
    [(500, None), (501, "too_many_pages")],
)
def test_enforces_the_500_page_boundary(
    tmp_path: Path,
    page_count: int,
    expected_code: str | None,
) -> None:
    path = write_pdf(tmp_path / f"{page_count}.pdf", make_pdf(page_count=page_count))

    if expected_code is None:
        assert validate_pdf(path, PdfLimits()).page_count == 500
    else:
        assert_validation_code(path, PdfLimits(), expected_code)
