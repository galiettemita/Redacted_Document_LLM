from __future__ import annotations

from datetime import date
from typing import Literal
from uuid import UUID

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.exc import IntegrityError

from redacted_backend.db.base import Database
from redacted_backend.db.repositories.documents import DocumentRepository
from redacted_backend.db.repositories.passages import (
    PassageConflictError,
    PassageRepository,
)
from redacted_backend.db.repositories.visible_examples import (
    VisibleExampleRepository,
    artifact_example,
)
from redacted_backend.documents.preparation import (
    VisibleContextPreparationError,
    prepare_visible_context,
)
from redacted_backend.domain.documents import DocumentStatus
from redacted_backend.domain.passages import PassageStatus
from redacted_backend.storage.local import LocalDocumentStore


class VisibleContextSummary(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    passage_id: UUID
    passage_revision: int = Field(ge=1)
    status: Literal["context_ready"] = "context_ready"
    document_date: date
    length_cue: Literal[
        "a few words",
        "part of a sentence",
        "about a sentence",
        "a paragraph",
    ]
    estimated_words: int = Field(ge=1)
    visible_word_count: int = Field(ge=0)
    confirmed_mask_count: int = Field(ge=0)
    serialization_version: Literal["visible-transcript-v1"]


def create_context_router(
    database: Database | None,
    document_store: LocalDocumentStore | None,
) -> APIRouter:
    router = APIRouter()

    @router.post(
        "/passages/{passage_id}:prepare-context",
        response_model=VisibleContextSummary,
    )
    def prepare_context(passage_id: UUID) -> VisibleContextSummary:
        ready_database, ready_store = _require_services(database, document_store)
        with ready_database.session_factory() as session:
            passages = PassageRepository(session)
            passage = passages.get(passage_id)
            if passage is None or passage.status == PassageStatus.DELETED:
                raise _not_found()
            document = DocumentRepository(session).get(passage.document_id)
            if document is None or document.status == DocumentStatus.DELETED:
                raise _not_found()
            existing = VisibleExampleRepository(session).get_current(
                passage.id,
                passage.revision,
            )
            if existing is not None:
                return _summary(existing)
            if document.visible_date is None:
                raise _preparation_error(
                    VisibleContextPreparationError(
                        "missing_visible_date",
                        "Save a date copied from the visible document first",
                    )
                )
            all_passages = passages.list_live_for_document(document.id)
            expected_revision = passage.revision
            pdf_path = ready_store.resolve(document.storage_key)

        try:
            prepared = prepare_visible_context(
                pdf_path,
                document,
                all_passages,
                passage.id,
            )
        except VisibleContextPreparationError as error:
            raise _preparation_error(error) from error

        try:
            with ready_database.session_factory() as session, session.begin():
                passages = PassageRepository(session)
                current = passages.get(passage.id)
                if (
                    current is None
                    or current.status == PassageStatus.DELETED
                    or current.revision != expected_revision
                ):
                    raise PassageConflictError(
                        "passage revision changed during preparation"
                    )
                current_document = DocumentRepository(session).get(current.document_id)
                if (
                    current_document is None
                    or current_document.status == DocumentStatus.DELETED
                    or current_document.visible_date != prepared.example.document_date
                ):
                    raise PassageConflictError("document changed during preparation")
                artifacts = VisibleExampleRepository(session)
                existing = artifacts.get_current(current.id, current.revision)
                if existing is not None:
                    return _summary(existing)
                artifact = artifacts.save(
                    prepared,
                    passage_revision=current.revision,
                )
                passages.mark_context_ready(
                    current.id,
                    expected_revision=current.revision,
                )
                return _summary(artifact)
        except PassageConflictError as error:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "code": "passage_revision_changed",
                    "message": str(error),
                },
            ) from error
        except IntegrityError:
            with ready_database.session_factory() as session:
                current = PassageRepository(session).get(passage.id)
                if current is not None:
                    existing = VisibleExampleRepository(session).get_current(
                        current.id,
                        current.revision,
                    )
                    if existing is not None:
                        return _summary(existing)
            raise

    return router


def _summary(artifact) -> VisibleContextSummary:
    example = artifact_example(artifact)
    return VisibleContextSummary(
        passage_id=example.passage_id,
        passage_revision=artifact.passage_revision,
        document_date=example.document_date,
        length_cue=example.length_cue.bucket,
        estimated_words=example.length_cue.estimated_words,
        visible_word_count=artifact.visible_word_count,
        confirmed_mask_count=artifact.confirmed_mask_count,
        serialization_version=example.serialization_version,
    )


def _require_services(
    database: Database | None,
    document_store: LocalDocumentStore | None,
) -> tuple[Database, LocalDocumentStore]:
    if database is None or document_store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "backend_not_configured"},
        )
    return database, document_store


def _not_found() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail={"code": "passage_not_found"},
    )


def _preparation_error(error: VisibleContextPreparationError) -> HTTPException:
    response_status = (
        status.HTTP_503_SERVICE_UNAVAILABLE
        if error.code == "ocr_unavailable"
        else status.HTTP_422_UNPROCESSABLE_CONTENT
    )
    return HTTPException(
        status_code=response_status,
        detail={"code": error.code, "message": str(error)},
    )
