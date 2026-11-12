from __future__ import annotations

import time
from pathlib import Path

import pymupdf


def make_pdf(*, page_count: int = 1, text: str = "Visible bulletin text") -> bytes:
    document = pymupdf.open()
    for page_number in range(page_count):
        page = document.new_page()
        page.insert_text((72, 72), f"{text} page {page_number + 1}")
    try:
        return document.tobytes()
    finally:
        document.close()


def make_encrypted_pdf() -> bytes:
    document = pymupdf.open()
    document.new_page().insert_text((72, 72), "Visible encrypted bulletin")
    try:
        return document.tobytes(
            encryption=pymupdf.PDF_ENCRYPT_AES_256,
            owner_pw="owner-secret",
            user_pw="user-secret",
        )
    finally:
        document.close()


def make_compression_heavy_pdf(*, expanded_bytes: int = 8_192) -> bytes:
    document = pymupdf.open()
    document.new_page()
    stream_xref = document.get_new_xref()
    document.update_object(stream_xref, "<<>>")
    document.update_stream(stream_xref, b"A" * expanded_bytes, compress=True)
    try:
        return document.tobytes(garbage=0)
    finally:
        document.close()


def slow_pdf_inspector(_path: Path, _max_decompressed_bytes: int):
    time.sleep(1)
    return {"page_count": 1, "decompressed_bytes": 0}
