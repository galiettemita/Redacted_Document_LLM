from __future__ import annotations

import math
import multiprocessing
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pymupdf
from pydantic import BaseModel, ConfigDict, Field


class PdfLimits(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    max_file_bytes: int = Field(default=100 * 1024 * 1024, ge=1)
    max_pages: int = Field(default=500, ge=1)
    max_decompressed_bytes: int = Field(default=512 * 1024 * 1024, ge=1)
    max_rendered_page_pixels: int = Field(default=25_000_000, ge=1)
    parser_timeout_seconds: float = Field(default=10.0, gt=0)


class PdfMetadata(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    page_count: int = Field(ge=1)
    size_bytes: int = Field(ge=1)
    decompressed_bytes: int = Field(ge=0)


class PdfValidationError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


class _InspectionError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


Inspector = Callable[[Path, int, int], dict[str, Any]]


def validate_pdf(
    path: Path,
    limits: PdfLimits,
    *,
    _inspector: Inspector | None = None,
) -> PdfMetadata:
    size_bytes = path.stat().st_size
    if size_bytes > limits.max_file_bytes:
        raise PdfValidationError(
            "file_too_large",
            f"PDF exceeds the {limits.max_file_bytes}-byte upload limit",
        )
    if size_bytes < 5:
        raise PdfValidationError("invalid_signature", "File is not a PDF")
    with path.open("rb") as source:
        if source.read(5) != b"%PDF-":
            raise PdfValidationError("invalid_signature", "File is not a PDF")

    result = _inspect_with_timeout(
        path,
        limits.max_decompressed_bytes,
        limits.max_rendered_page_pixels,
        limits.parser_timeout_seconds,
        _inspector or _inspect_pdf,
    )
    page_count = int(result["page_count"])
    if page_count < 1:
        raise PdfValidationError("malformed_pdf", "PDF has no pages")
    if page_count > limits.max_pages:
        raise PdfValidationError(
            "too_many_pages",
            f"PDF exceeds the {limits.max_pages}-page limit",
        )
    return PdfMetadata(
        page_count=page_count,
        size_bytes=size_bytes,
        decompressed_bytes=int(result["decompressed_bytes"]),
    )


def _inspect_with_timeout(
    path: Path,
    max_decompressed_bytes: int,
    max_rendered_page_pixels: int,
    timeout_seconds: float,
    inspector: Inspector,
) -> dict[str, Any]:
    context = multiprocessing.get_context("spawn")
    parent_connection, child_connection = context.Pipe(duplex=False)
    process = context.Process(
        target=_inspection_process,
        args=(
            child_connection,
            path,
            max_decompressed_bytes,
            max_rendered_page_pixels,
            inspector,
        ),
        daemon=True,
    )
    process.start()
    child_connection.close()
    process.join(timeout_seconds)
    if process.is_alive():
        process.terminate()
        process.join(1)
        if process.is_alive():
            process.kill()
            process.join(1)
        parent_connection.close()
        raise PdfValidationError("parser_timeout", "PDF parser timed out")
    if not parent_connection.poll():
        parent_connection.close()
        raise PdfValidationError("malformed_pdf", "PDF parser failed")
    status, payload = parent_connection.recv()
    parent_connection.close()
    if status == "error":
        raise PdfValidationError(payload["code"], payload["message"])
    return payload


def _inspection_process(
    connection,
    path: Path,
    max_decompressed_bytes: int,
    max_rendered_page_pixels: int,
    inspector: Inspector,
) -> None:
    try:
        connection.send(
            (
                "ok",
                inspector(
                    path,
                    max_decompressed_bytes,
                    max_rendered_page_pixels,
                ),
            )
        )
    except _InspectionError as error:
        connection.send(("error", {"code": error.code, "message": error.message}))
    # This child process is the isolation boundary for an untrusted parser. Any
    # unexpected parser/library failure must become a generic validation error.
    except Exception:  # noqa: BLE001
        connection.send(
            (
                "error",
                {"code": "malformed_pdf", "message": "PDF parser rejected the file"},
            )
        )
    finally:
        connection.close()


def _inspect_pdf(
    path: Path,
    max_decompressed_bytes: int,
    max_rendered_page_pixels: int,
) -> dict[str, int]:
    try:
        document = pymupdf.open(path)
    except Exception as error:
        raise _InspectionError(
            "malformed_pdf", "PDF parser rejected the file"
        ) from error
    try:
        if document.needs_pass:
            raise _InspectionError("encrypted_pdf", "Encrypted PDFs are not accepted")
        for page in document:
            rendered_width = math.ceil(page.rect.width * 2)
            rendered_height = math.ceil(page.rect.height * 2)
            if rendered_width * rendered_height > max_rendered_page_pixels:
                raise _InspectionError(
                    "page_render_limit",
                    "A PDF page exceeds the safe rendering size",
                )
        decompressed_bytes = 0
        for xref in range(1, document.xref_length()):
            if not document.xref_is_stream(xref):
                continue
            stream = document.xref_stream(xref)
            decompressed_bytes += len(stream)
            if decompressed_bytes > max_decompressed_bytes:
                raise _InspectionError(
                    "decompression_limit",
                    "PDF expanded streams exceed the safety limit",
                )
        return {
            "page_count": document.page_count,
            "decompressed_bytes": decompressed_bytes,
        }
    except _InspectionError:
        raise
    except Exception as error:
        raise _InspectionError(
            "malformed_pdf", "PDF parser rejected the file"
        ) from error
    finally:
        document.close()
