import pytest
from pydantic import ValidationError

from redacted_backend.config import Settings


@pytest.mark.parametrize(
    ("field", "unsafe_value"),
    [
        ("document_retention_hours", 25),
        ("pdf_max_file_bytes", 100 * 1024 * 1024 + 1),
        ("pdf_max_pages", 501),
        ("pdf_max_decompressed_bytes", 512 * 1024 * 1024 + 1),
        ("pdf_max_rendered_page_pixels", 25_000_001),
        ("pdf_parser_timeout_seconds", 10.01),
    ],
)
def test_runtime_configuration_cannot_raise_pdf_or_retention_safety_ceilings(
    field: str,
    unsafe_value: float,
) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **{field: unsafe_value})
