from __future__ import annotations

from datetime import date
from typing import Literal
from uuid import UUID

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.exc import IntegrityError

from redacted_backend.db.base import Database
from redacted_backend.db.repositories.documents import (
    DocumentRepository,
    VisibleDateConflictError,
)
from redacted_backend.db.repositories.passages import (
    PassageConflictError,
    PassageRepository,
)
from redacted_backend.db.repositories.visible_examples import (
    ContextQueryArtifactConflictError,
    VisibleExampleRepository,
    artifact_example,
)
from redacted_backend.documents.preparation import (
    VisibleContextPreparationError,
    prepare_visible_context,
    prepare_visible_contexts,
)
from redacted_backend.domain.documents import DocumentStatus
from redacted_backend.domain.passages import PassageRecord, PassageStatus
from redacted_backend.domain.visible_examples import VisibleExample
from redacted_backend.retrieval.context_query import (
    ContextQueryError,
    prepare_context_query,
)
from redacted_backend.retrieval.types import PreparedContextQuery
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
    context_query_status: Literal["ready"] = "ready"
    context_query_term_count: int = Field(ge=1)
    context_query_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    context_query_version: Literal["context-query-v1"]
    context_stopwords_version: str = Field(min_length=1)


class DocumentVisibleContextsSummary(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    document_id: UUID
    status: Literal["contexts_ready"] = "contexts_ready"
    passage_count: int = Field(ge=0)
    contexts: list[VisibleContextSummary]


class DocumentMaskSetChangedError(ValueError):
    pass


def create_context_router(
    database: Database | None,
    document_store: LocalDocumentStore | None,
) -> APIRouter:
    router = APIRouter()

    @router.post(
        "/documents/{document_id}:prepare-contexts",
        response_model=DocumentVisibleContextsSummary,
    )
    def prepare_document_contexts(
        document_id: UUID,
    ) -> DocumentVisibleContextsSummary:
        ready_database, ready_store = _require_services(database, document_store)
        try:
            with ready_database.session_factory() as session, session.begin():
                document = DocumentRepository(session).get(document_id)
                if document is None or document.status == DocumentStatus.DELETED:
                    raise _not_found()
                passages = PassageRepository(session).list_live_for_document(
                    document.id
                )
                if not passages:
                    return DocumentVisibleContextsSummary(
                        document_id=document.id,
                        passage_count=0,
                        contexts=[],
                    )
                artifacts = VisibleExampleRepository(session)
                existing_summaries: dict[UUID, VisibleContextSummary] = {}
                for passage in passages:
                    existing = artifacts.get_current(passage.id, passage.revision)
                    if existing is None:
                        continue
                    query = _prepare_query(artifact_example(existing))
                    artifacts.ensure_context_query(existing, query)
                    existing_summaries[passage.id] = _summary(existing)
                missing = [
                    passage
                    for passage in passages
                    if passage.id not in existing_summaries
                ]
                if not missing:
                    return DocumentVisibleContextsSummary(
                        document_id=document.id,
                        passage_count=len(passages),
                        contexts=[
                            existing_summaries[passage.id] for passage in passages
                        ],
                    )
                expected_revisions = {
                    passage.id: passage.revision for passage in passages
                }
                expected_mask_set_revision = document.mask_set_revision
                pdf_path = ready_store.resolve(document.storage_key)
        except ContextQueryArtifactConflictError as error:
            raise _query_conflict(error) from error

        try:
            prepared_contexts = prepare_visible_contexts(
                pdf_path,
                document,
                passages,
                [passage.id for passage in missing],
            )
        except VisibleContextPreparationError as error:
            raise _preparation_error(error) from error
        prepared_by_id = {
            prepared.example.passage_id: prepared for prepared in prepared_contexts
        }
        queries = {
            passage_id: _prepare_query(prepared.example)
            for passage_id, prepared in prepared_by_id.items()
        }

        try:
            with ready_database.session_factory() as session, session.begin():
                documents = DocumentRepository(session)
                current_document = documents.get_for_update(document.id)
                if (
                    current_document is None
                    or current_document.status == DocumentStatus.DELETED
                    or current_document.sha256 != document.sha256
                    or current_document.storage_key != document.storage_key
                ):
                    raise PassageConflictError("document changed during preparation")
                if current_document.mask_set_revision != expected_mask_set_revision:
                    raise DocumentMaskSetChangedError(
                        "document mask set changed during preparation"
                    )
                persisted_document = documents.set_detected_visible_date(
                    current_document.id,
                    prepared_contexts[0].date_evidence,
                )
                if persisted_document is None:
                    raise PassageConflictError("document changed during preparation")

                passage_repository = PassageRepository(session)
                current_passages = passage_repository.list_live_for_document(
                    document.id
                )
                if _passage_set_signature(current_passages) != _passage_set_signature(
                    passages
                ):
                    raise DocumentMaskSetChangedError(
                        "document passage set changed during preparation"
                    )
                artifacts = VisibleExampleRepository(session)
                summaries: list[VisibleContextSummary] = []
                for original in current_passages:
                    current = passage_repository.get(original.id)
                    if (
                        current is None
                        or current.status == PassageStatus.DELETED
                        or current.revision != expected_revisions[original.id]
                    ):
                        raise PassageConflictError(
                            "passage revision changed during preparation"
                        )
                    artifact = artifacts.get_current(current.id, current.revision)
                    if artifact is None:
                        prepared = prepared_by_id.get(current.id)
                        if prepared is None:
                            raise PassageConflictError(
                                "prepared passage context is missing"
                            )
                        artifact = artifacts.save(
                            prepared,
                            queries[current.id],
                            passage_revision=current.revision,
                        )
                    else:
                        query = queries.get(current.id) or _prepare_query(
                            artifact_example(artifact)
                        )
                        artifacts.ensure_context_query(artifact, query)
                    passage_repository.mark_context_ready(
                        current.id,
                        expected_revision=current.revision,
                    )
                    summaries.append(_summary(artifact))
                return DocumentVisibleContextsSummary(
                    document_id=document.id,
                    passage_count=len(summaries),
                    contexts=summaries,
                )
        except DocumentMaskSetChangedError as error:
            raise _mask_set_conflict(error) from error
        except VisibleDateConflictError as error:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={"code": "visible_date_conflict", "message": str(error)},
            ) from error
        except PassageConflictError as error:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={"code": "passage_revision_changed", "message": str(error)},
            ) from error
        except ContextQueryArtifactConflictError as error:
            raise _query_conflict(error) from error
        except IntegrityError:
            for recovery_attempt in range(len(passages) + 1):
                try:
                    with ready_database.session_factory() as session, session.begin():
                        documents = DocumentRepository(session)
                        current_document = documents.get_for_update(document.id)
                        if (
                            current_document is None
                            or current_document.status == DocumentStatus.DELETED
                            or current_document.sha256 != document.sha256
                            or current_document.storage_key != document.storage_key
                        ):
                            raise PassageConflictError(
                                "document changed during preparation"
                            )
                        if (
                            current_document.mask_set_revision
                            != expected_mask_set_revision
                        ):
                            raise DocumentMaskSetChangedError(
                                "document mask set changed during preparation"
                            )
                        passage_repository = PassageRepository(session)
                        current_passages = passage_repository.list_live_for_document(
                            document.id
                        )
                        if _passage_set_signature(
                            current_passages
                        ) != _passage_set_signature(passages):
                            raise DocumentMaskSetChangedError(
                                "document passage set changed during preparation"
                            )
                        artifacts = VisibleExampleRepository(session)
                        summaries: list[VisibleContextSummary] = []
                        for current in current_passages:
                            artifact = artifacts.get_current(
                                current.id,
                                current.revision,
                            )
                            if artifact is None:
                                prepared = prepared_by_id.get(current.id)
                                if prepared is None:
                                    raise PassageConflictError(
                                        "prepared passage context is missing"
                                    )
                                artifact = artifacts.save(
                                    prepared,
                                    queries[current.id],
                                    passage_revision=current.revision,
                                )
                            else:
                                query = _prepare_query(artifact_example(artifact))
                                artifacts.ensure_context_query(artifact, query)
                            passage_repository.mark_context_ready(
                                current.id,
                                expected_revision=current.revision,
                            )
                            summaries.append(_summary(artifact))
                        return DocumentVisibleContextsSummary(
                            document_id=document.id,
                            passage_count=len(summaries),
                            contexts=summaries,
                        )
                except IntegrityError:
                    if recovery_attempt == len(passages):
                        raise
                    continue
                except DocumentMaskSetChangedError as conflict:
                    raise _mask_set_conflict(conflict) from conflict
                except PassageConflictError as conflict:
                    raise HTTPException(
                        status_code=status.HTTP_409_CONFLICT,
                        detail={
                            "code": "passage_revision_changed",
                            "message": str(conflict),
                        },
                    ) from conflict
                except ContextQueryArtifactConflictError as conflict:
                    raise _query_conflict(conflict) from conflict
            raise RuntimeError("document context recovery attempts were exhausted")

    @router.post(
        "/passages/{passage_id}:prepare-context",
        response_model=VisibleContextSummary,
    )
    def prepare_context(passage_id: UUID) -> VisibleContextSummary:
        ready_database, ready_store = _require_services(database, document_store)
        try:
            with ready_database.session_factory() as session, session.begin():
                passages = PassageRepository(session)
                passage = passages.get(passage_id)
                if passage is None or passage.status == PassageStatus.DELETED:
                    raise _not_found()
                document = DocumentRepository(session).get(passage.document_id)
                if document is None or document.status == DocumentStatus.DELETED:
                    raise _not_found()
                artifacts = VisibleExampleRepository(session)
                existing = artifacts.get_current(
                    passage.id,
                    passage.revision,
                )
                if existing is not None:
                    query = _prepare_query(artifact_example(existing))
                    artifacts.ensure_context_query(existing, query)
                    return _summary(existing)
                all_passages = passages.list_live_for_document(document.id)
                expected_revision = passage.revision
                expected_mask_set_revision = document.mask_set_revision
                pdf_path = ready_store.resolve(document.storage_key)
        except ContextQueryArtifactConflictError as error:
            raise _query_conflict(error) from error

        try:
            prepared = prepare_visible_context(
                pdf_path,
                document,
                all_passages,
                passage.id,
            )
        except VisibleContextPreparationError as error:
            raise _preparation_error(error) from error
        query = _prepare_query(prepared.example)

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
                documents = DocumentRepository(session)
                current_document = documents.get_for_update(current.document_id)
                if (
                    current_document is None
                    or current_document.status == DocumentStatus.DELETED
                    or current_document.sha256 != document.sha256
                    or current_document.storage_key != document.storage_key
                ):
                    raise PassageConflictError("document changed during preparation")
                if current_document.mask_set_revision != expected_mask_set_revision:
                    raise DocumentMaskSetChangedError(
                        "document mask set changed during preparation"
                    )
                current_passages = passages.list_live_for_document(current.document_id)
                if _passage_set_signature(current_passages) != _passage_set_signature(
                    all_passages
                ):
                    raise DocumentMaskSetChangedError(
                        "document passage set changed during preparation"
                    )
                persisted_document = documents.set_detected_visible_date(
                    current_document.id,
                    prepared.date_evidence,
                )
                if persisted_document is None:
                    raise PassageConflictError("document changed during preparation")
                artifacts = VisibleExampleRepository(session)
                existing = artifacts.get_current(current.id, current.revision)
                if existing is not None:
                    artifacts.ensure_context_query(existing, query)
                    return _summary(existing)
                artifact = artifacts.save(
                    prepared,
                    query,
                    passage_revision=current.revision,
                )
                passages.mark_context_ready(
                    current.id,
                    expected_revision=current.revision,
                )
                return _summary(artifact)
        except DocumentMaskSetChangedError as error:
            raise _mask_set_conflict(error) from error
        except VisibleDateConflictError as error:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "code": "visible_date_conflict",
                    "message": str(error),
                },
            ) from error
        except PassageConflictError as error:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "code": "passage_revision_changed",
                    "message": str(error),
                },
            ) from error
        except ContextQueryArtifactConflictError as error:
            raise _query_conflict(error) from error
        except IntegrityError:
            with ready_database.session_factory() as session, session.begin():
                documents = DocumentRepository(session)
                current_document = documents.get_for_update(document.id)
                if (
                    current_document is None
                    or current_document.mask_set_revision != expected_mask_set_revision
                ):
                    raise _mask_set_conflict(
                        DocumentMaskSetChangedError(
                            "document mask set changed during preparation"
                        )
                    )
                passage_repository = PassageRepository(session)
                current_passages = passage_repository.list_live_for_document(
                    document.id
                )
                if _passage_set_signature(current_passages) != _passage_set_signature(
                    all_passages
                ):
                    raise _mask_set_conflict(
                        DocumentMaskSetChangedError(
                            "document passage set changed during preparation"
                        )
                    )
                current = passage_repository.get(passage.id)
                if current is not None:
                    artifacts = VisibleExampleRepository(session)
                    existing = artifacts.get_current(
                        current.id,
                        current.revision,
                    )
                    if existing is not None:
                        artifacts.ensure_context_query(existing, query)
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
        context_query_term_count=artifact.context_query_term_count,
        context_query_sha256=artifact.context_query_sha256,
        context_query_version=artifact.context_query_version,
        context_stopwords_version=artifact.context_stopwords_version,
    )


def _prepare_query(example: VisibleExample) -> PreparedContextQuery:
    try:
        return prepare_context_query(example)
    except ContextQueryError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={"code": "context_query_empty", "message": str(error)},
        ) from error


def _query_conflict(error: ContextQueryArtifactConflictError) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_409_CONFLICT,
        detail={"code": "context_query_conflict", "message": str(error)},
    )


def _mask_set_conflict(error: DocumentMaskSetChangedError) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_409_CONFLICT,
        detail={"code": "document_mask_set_changed", "message": str(error)},
    )


def _passage_set_signature(
    passages: list[PassageRecord],
) -> tuple[tuple[UUID, int], ...]:
    return tuple((passage.id, passage.revision) for passage in passages)


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
