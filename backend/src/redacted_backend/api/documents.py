from __future__ import annotations

import hashlib
import re
import unicodedata
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Annotated, BinaryIO
from uuid import UUID

from fastapi import APIRouter, File, HTTPException, Response, UploadFile, status

from redacted_backend.config import Settings
from redacted_backend.db.base import Database
from redacted_backend.db.repositories.documents import DocumentRepository
from redacted_backend.db.repositories.passages import PassageRepository
from redacted_backend.documents.validation import (
    PdfLimits,
    PdfValidationError,
    validate_pdf,
)
from redacted_backend.domain.documents import DocumentMetadataUpdate, DocumentRecord
from redacted_backend.storage.local import LocalDocumentStore


def create_documents_router(
    settings: Settings,
    database: Database | None,
    document_store: LocalDocumentStore | None,
) -> APIRouter:
    router = APIRouter()

    @router.post(
        "/documents",
        response_model=DocumentRecord,
        status_code=status.HTTP_201_CREATED,
    )
    def upload_document(file: Annotated[UploadFile, File()]) -> DocumentRecord:
        ready_database, ready_store = _require_services(database, document_store)
        if file.content_type not in {"application/pdf", "application/x-pdf"}:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail={
                    "code": "unsupported_media_type",
                    "message": "Select a PDF file",
                },
            )
        sha256 = _hash_stream(file.file, settings.pdf_max_file_bytes)
        file.file.seek(0)
        stored = ready_store.put(file.file, sha256)
        try:
            metadata = validate_pdf(
                stored.local_path,
                _pdf_limits(settings),
            )
        except PdfValidationError as error:
            with ready_database.session_factory() as session:
                if (
                    DocumentRepository(session).live_reference_count(stored.storage_key)
                    == 0
                ):
                    ready_store.delete(stored.storage_key)
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail={"code": error.code, "message": error.message},
            ) from error

        try:
            with ready_database.session_factory() as session, session.begin():
                record = DocumentRepository(session).create(
                    filename=sanitize_filename(file.filename),
                    sha256=stored.sha256,
                    media_type="application/pdf",
                    page_count=metadata.page_count,
                    storage_key=stored.storage_key,
                    expires_at=datetime.now(UTC)
                    + timedelta(hours=settings.document_retention_hours),
                )
            return record
        except Exception:
            with ready_database.session_factory() as session:
                if (
                    DocumentRepository(session).live_reference_count(stored.storage_key)
                    == 0
                ):
                    ready_store.delete(stored.storage_key)
            raise

    @router.patch(
        "/documents/{document_id}/metadata",
        response_model=DocumentRecord,
    )
    def update_document_metadata(
        document_id: UUID,
        update: DocumentMetadataUpdate,
    ) -> DocumentRecord:
        ready_database, _ = _require_services(database, document_store)
        with ready_database.session_factory() as session, session.begin():
            documents = DocumentRepository(session)
            existing = documents.get(document_id)
            record = documents.set_visible_date(
                document_id,
                visible_date=update.visible_date,
                provenance=update.date_provenance,
            )
            if existing is not None and (
                existing.visible_date != update.visible_date
                or existing.date_provenance != update.date_provenance
            ):
                PassageRepository(session).invalidate_document_date(document_id)
        if record is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
        return record

    @router.delete(
        "/documents/{document_id}",
        status_code=status.HTTP_204_NO_CONTENT,
    )
    def delete_document(document_id: UUID) -> Response:
        ready_database, ready_store = _require_services(database, document_store)
        with ready_database.session_factory() as session, session.begin():
            repository = DocumentRepository(session)
            record = repository.mark_deleted(document_id)
            if record is None:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
            PassageRepository(session).delete_for_document(document_id)
            live_references = repository.live_reference_count(record.storage_key)
        if live_references == 0:
            ready_store.delete(record.storage_key)
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    return router


def _require_services(
    database: Database | None,
    document_store: LocalDocumentStore | None,
) -> tuple[Database, LocalDocumentStore]:
    if database is None or document_store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "backend_not_configured",
                "message": "Document storage is not configured",
            },
        )
    return database, document_store


def _hash_stream(stream: BinaryIO, max_file_bytes: int) -> str:
    digest = hashlib.sha256()
    size_bytes = 0
    while chunk := stream.read(1024 * 1024):
        size_bytes += len(chunk)
        if size_bytes > max_file_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail={
                    "code": "file_too_large",
                    "message": f"PDF exceeds the {max_file_bytes}-byte upload limit",
                },
            )
        digest.update(chunk)
    return digest.hexdigest()


def sanitize_filename(filename: str | None) -> str:
    candidate = (filename or "document.pdf").replace("\\", "/").rsplit("/", 1)[-1]
    candidate = unicodedata.normalize("NFKC", candidate)
    candidate = "".join(
        character if character.isalnum() or character in " ._()-" else "_"
        for character in candidate
        if not unicodedata.category(character).startswith("C")
    )
    candidate = re.sub(r"_+", "_", candidate).strip(" .")
    if not candidate:
        candidate = "document.pdf"
    if len(candidate) > 255:
        suffix = Path(candidate).suffix[:16]
        candidate = f"{candidate[: 255 - len(suffix)]}{suffix}"
    return candidate


def _pdf_limits(settings: Settings) -> PdfLimits:
    return PdfLimits(
        max_file_bytes=settings.pdf_max_file_bytes,
        max_pages=settings.pdf_max_pages,
        max_decompressed_bytes=settings.pdf_max_decompressed_bytes,
        parser_timeout_seconds=settings.pdf_parser_timeout_seconds,
    )
