from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Iterator
from uuid import UUID

from fastapi import APIRouter, Header, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from redacted_backend.api.readiness import (
    resolve_best_available_corpus_readiness,
    resolve_corpus_readiness,
    resolve_prediction_basis_readiness,
    resolve_staged_bm25_corpus_readiness,
    resolve_worker_readiness,
    select_prediction_basis,
)
from redacted_backend.config import Settings
from redacted_backend.db.base import Database
from redacted_backend.db.models import Document
from redacted_backend.db.repositories.analysis import (
    AnalysisConflictError,
    AnalysisJobRepository,
)
from redacted_backend.db.repositories.documents import DocumentRepository
from redacted_backend.db.repositories.passages import PassageRepository
from redacted_backend.db.repositories.visible_examples import VisibleExampleRepository
from redacted_backend.domain.analysis import (
    AnalysisBlocker,
    AnalysisJobRecord,
    DocumentAnalysisScheduleResult,
    DocumentAnalysisState,
    PredictionBasis,
)
from redacted_backend.domain.readiness import ComponentReadiness
from redacted_backend.jobs.queue import AnalysisDispatcher
from sqlalchemy import select
from sqlalchemy.orm import Session


class ScheduleAnalysisRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    expected_revision: int = Field(ge=1)


def create_analysis_router(
    settings: Settings,
    database: Database | None,
    dispatcher: AnalysisDispatcher | None = None,
) -> APIRouter:
    router = APIRouter()

    @router.post(
        "/documents/{document_id}/analysis",
        response_model=DocumentAnalysisScheduleResult,
        status_code=status.HTTP_202_ACCEPTED,
    )
    def schedule_document_analysis(
        document_id: UUID,
        idempotency_key: str = Header(
            alias="Idempotency-Key", min_length=1, max_length=512
        ),
    ) -> DocumentAnalysisScheduleResult:
        ready_database = _require_database(database)
        _validate_idempotency_key(idempotency_key)
        try:
            with ready_database.session_factory() as session, session.begin():
                _begin_analysis_write_lock(ready_database, session)
                document = DocumentRepository(session).get_live(document_id)
                if document is None:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail={"code": "document_not_found"},
                    )
                _lock_analysis_document(session, document_id)
                passages = PassageRepository(session).list_live_for_document(document_id)
                if not passages:
                    raise HTTPException(
                        status_code=status.HTTP_409_CONFLICT,
                        detail={
                            "code": "visible_context_not_ready",
                            "message": "Prepare visible-only context before analysis",
                        },
                    )
                decisions: dict[
                    UUID,
                    tuple[PredictionBasis, list[AnalysisBlocker]],
                ] = {}
                for passage in passages:
                    artifact = VisibleExampleRepository(session).get_current(
                        passage.id,
                        passage.revision,
                    )
                    if artifact is None:
                        raise HTTPException(
                            status_code=status.HTTP_409_CONFLICT,
                            detail={
                                "code": "visible_context_not_ready",
                                "message": "Prepare visible-only context before analysis",
                            },
                        )
                    _require_complete_query_metadata(artifact)
                    decisions[passage.id] = _analysis_decision(
                        settings,
                        context_query_ready=all(
                            value is not None for value in _query_fields(artifact)
                        ),
                    )

                repository = AnalysisJobRepository(session)
                recovered = repository.recoverable_for_document(document_id)
                decision_bases = {
                    basis
                    for basis, blockers in decisions.values()
                    if not blockers
                }
                candidate_basis = (
                    next(iter(decision_bases))
                    if len(decision_bases) == 1
                    else None
                )
                full_run_basis = (
                    candidate_basis
                    if candidate_basis is not None
                    and all(not blockers for _basis, blockers in decisions.values())
                    and candidate_basis in {
                        PredictionBasis.BEST_AVAILABLE_FOIARCHIVE_V1,
                        PredictionBasis.PAPER_EXACT_V1,
                    }
                    else None
                )
                if full_run_basis is not None:
                    recovered = [
                        job
                        for job in recovered
                        if job.prediction_basis == full_run_basis.value
                    ]
                accepted = {
                    job.passage_id: job
                    for job in recovered
                }
                deferred: list[UUID] = []
                readiness_blockers: list[AnalysisBlocker] = []
                for passage in passages:
                    if passage.id in accepted:
                        continue
                    prediction_basis, decision_blockers = decisions[passage.id]
                    blockers = list(decision_blockers)
                    passage_key = _document_passage_key(
                        idempotency_key,
                        passage.id,
                    )
                    if (
                        prediction_basis
                        == PredictionBasis.BEST_AVAILABLE_FOIARCHIVE_BM25_V1
                        and not blockers
                    ):
                        limit_blocker = repository.staged_prediction_blocker(
                            document_id=document_id,
                            passage_id=passage.id,
                            idempotency_key=passage_key,
                            limit=settings.bm25_stage_max_predictions_per_document,
                        )
                        if limit_blocker is not None:
                            blockers.append(limit_blocker)
                    if blockers:
                        deferred.append(passage.id)
                        readiness_blockers.extend(
                            blocker
                            for blocker in blockers
                            if blocker.component != "staged_prediction_limit"
                        )
                        continue
                    job = repository.schedule(
                        passage_id=passage.id,
                        document_id=document_id,
                        passage_revision=passage.revision,
                        idempotency_key=passage_key,
                        blockers=[],
                        prediction_basis=prediction_basis,
                    )
                    accepted[passage.id] = job

                accepted_jobs = [
                    accepted[passage.id]
                    for passage in passages
                    if passage.id in accepted
                ]
                blockers = (
                    []
                    if accepted_jobs
                    else _unique_blockers(readiness_blockers)
                )
                result = DocumentAnalysisScheduleResult(
                    document_id=document_id,
                    accepted_jobs=accepted_jobs,
                    deferred_passage_ids=deferred,
                    blockers=blockers,
                )
            for job in result.accepted_jobs:
                if job.status != "queued":
                    continue
                _dispatch_job(dispatcher, job)
            return result
        except AnalysisConflictError as error:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={"code": "idempotency_conflict", "message": str(error)},
            ) from error

    @router.get(
        "/documents/{document_id}/analysis",
        response_model=DocumentAnalysisState,
    )
    def get_document_analysis(document_id: UUID) -> DocumentAnalysisState:
        ready_database = _require_database(database)
        with ready_database.session_factory() as session:
            document = DocumentRepository(session).get_live(document_id)
            if document is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail={"code": "document_not_found"},
                )
            jobs = AnalysisJobRepository(session).recoverable_for_document(
                document_id
            )
        return DocumentAnalysisState(document_id=document_id, jobs=jobs)

    @router.post(
        "/passages/{passage_id}/analysis",
        response_model=AnalysisJobRecord,
        status_code=status.HTTP_202_ACCEPTED,
    )
    def schedule_analysis(
        passage_id: UUID,
        body: ScheduleAnalysisRequest,
        idempotency_key: str = Header(
            alias="Idempotency-Key", min_length=1, max_length=512
        ),
    ) -> AnalysisJobRecord:
        ready_database = _require_database(database)
        _validate_idempotency_key(idempotency_key)
        try:
            with ready_database.session_factory() as session, session.begin():
                _begin_analysis_write_lock(ready_database, session)
                passage = PassageRepository(session).get_live(passage_id)
                if passage is None:
                    raise _not_found()
                if passage.revision != body.expected_revision:
                    raise HTTPException(
                        status_code=status.HTTP_409_CONFLICT,
                        detail={
                            "code": "passage_revision_changed",
                            "message": "Passage revision changed before analysis scheduling",
                        },
                    )
                _lock_analysis_document(session, passage.document_id)
                artifact = VisibleExampleRepository(session).get_current(
                    passage.id,
                    passage.revision,
                )
                if artifact is None:
                    raise HTTPException(
                        status_code=status.HTTP_409_CONFLICT,
                        detail={
                            "code": "visible_context_not_ready",
                            "message": "Prepare visible-only context before analysis",
                        },
                    )
                query_fields = (
                    _query_fields(artifact)
                )
                _require_complete_query_metadata(artifact)
                prediction_basis, blockers = _analysis_decision(
                    settings,
                    context_query_ready=all(
                        value is not None for value in query_fields
                    ),
                )
                repository = AnalysisJobRepository(session)
                replayed = (
                    repository.get_by_idempotency(passage.id, idempotency_key)
                    is not None
                )
                if (
                    prediction_basis
                    == PredictionBasis.BEST_AVAILABLE_FOIARCHIVE_BM25_V1
                    and not blockers
                ):
                    limit_blocker = repository.selected_staged_prediction_blocker(
                        document_id=passage.document_id,
                        passage_id=passage.id,
                        idempotency_key=idempotency_key,
                        limit=settings.bm25_stage_max_predictions_per_document,
                    )
                    if limit_blocker is not None:
                        blockers.append(limit_blocker)
                job = repository.schedule(
                    passage_id=passage.id,
                    document_id=passage.document_id,
                    passage_revision=passage.revision,
                    idempotency_key=idempotency_key,
                    blockers=blockers,
                    prediction_basis=prediction_basis,
                )
            if job.status == "queued" and not replayed:
                if dispatcher is None:
                    raise HTTPException(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        detail={
                            "code": "worker_dispatch_unavailable",
                            "message": "Analysis worker dispatch is unavailable",
                        },
                    )
                try:
                    dispatcher.dispatch(job.id)
                except Exception as error:
                    raise HTTPException(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        detail={
                            "code": "worker_dispatch_failed",
                            "message": "Analysis worker dispatch failed; retry with the same idempotency key",
                        },
                    ) from error
            return job
        except AnalysisConflictError as error:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={"code": "idempotency_conflict", "message": str(error)},
            ) from error

    @router.get("/analysis/{job_id}", response_model=AnalysisJobRecord)
    def get_analysis(job_id: UUID) -> AnalysisJobRecord:
        ready_database = _require_database(database)
        with ready_database.session_factory() as session:
            record = AnalysisJobRepository(session).get(job_id)
            document = (
                DocumentRepository(session).get_live(record.document_id)
                if record is not None
                else None
            )
        if record is None or document is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"code": "analysis_job_not_found"},
            )
        return record

    @router.get("/documents/{document_id}/events")
    def document_events(
        document_id: UUID,
        last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
    ) -> StreamingResponse:
        ready_database = _require_database(database)
        with ready_database.session_factory() as session:
            if DocumentRepository(session).get_live(document_id) is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail={"code": "document_not_found"},
                )
        after = _last_event_id(last_event_id)
        return StreamingResponse(
            _event_stream(ready_database, document_id, after),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    return router


def _analysis_decision(
    settings: Settings,
    *,
    context_query_ready: bool = True,
) -> tuple[PredictionBasis, list[AnalysisBlocker]]:
    database = ComponentReadiness(ready=True, detail="ready")
    worker = resolve_worker_readiness(settings)
    corpus = resolve_corpus_readiness(settings)
    staged_bm25_corpus = resolve_staged_bm25_corpus_readiness(settings)
    best_available_corpus = resolve_best_available_corpus_readiness(settings)
    bases = resolve_prediction_basis_readiness(
        settings,
        database=database,
        worker=worker,
        corpus=corpus,
        staged_bm25_corpus=staged_bm25_corpus,
        best_available_corpus=best_available_corpus,
    )
    if not context_query_ready:
        for basis in (
            PredictionBasis.BEST_AVAILABLE_FOIARCHIVE_BM25_V1,
            PredictionBasis.BEST_AVAILABLE_FOIARCHIVE_V1,
            PredictionBasis.PAPER_EXACT_V1,
        ):
            bases[basis] = bases[basis].model_copy(
                update={
                    "operational": False,
                    "ready": False,
                    "detail": "visible context query is unavailable",
                }
            )
    selected = select_prediction_basis(bases)
    if selected is not None:
        return selected, []

    basis = PredictionBasis.MODEL_ONLY_V1
    blockers: list[AnalysisBlocker] = []
    if settings.openai_api_key is None:
        blockers.append(
            AnalysisBlocker(
                component="model",
                detail="model access is not configured",
            )
        )
    elif not settings.allow_paid_model_calls:
        blockers.append(
            AnalysisBlocker(
                component="model",
                detail="paid model calls are not approved",
            )
        )
    elif not settings.allow_model_only_calls:
        blockers.append(
            AnalysisBlocker(
                component="model_only",
                detail="model-only calls are not approved",
            )
        )
    if not worker.ready:
        blockers.append(AnalysisBlocker(component="worker", detail=worker.detail))
    return basis, blockers


def _event_stream(
    database: Database,
    document_id: UUID,
    after: int,
) -> Iterator[str]:
    cursor = after
    last_output = time.monotonic()
    while True:
        with database.session_factory() as session:
            if DocumentRepository(session).get_live(document_id) is None:
                return
            repository = AnalysisJobRepository(session)
            events = repository.events_after(document_id, cursor)
            terminal = repository.document_is_terminal(document_id)
        for event in events:
            cursor = event.id
            data = json.dumps(event.payload, sort_keys=True, separators=(",", ":"))
            yield f"id: {event.id}\nevent: {event.event_type}\ndata: {data}\n\n"
            last_output = time.monotonic()
        if terminal:
            return
        if time.monotonic() - last_output >= 15:
            yield ": heartbeat\n\n"
            last_output = time.monotonic()
        time.sleep(0.75)


def _last_event_id(value: str | None) -> int:
    if value is None or value == "":
        return 0
    try:
        parsed = int(value)
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "invalid_last_event_id"},
        ) from error
    if parsed < 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "invalid_last_event_id"},
        )
    return parsed


def _require_database(database: Database | None) -> Database:
    if database is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "backend_not_configured"},
        )
    return database


def _begin_analysis_write_lock(database: Database, session: Session) -> None:
    if database.engine.dialect.name == "sqlite":
        session.connection().exec_driver_sql("BEGIN IMMEDIATE")


def _lock_analysis_document(session: Session, document_id: UUID) -> None:
    session.scalar(
        select(Document.id)
        .where(Document.id == str(document_id))
        .with_for_update()
    )


def _not_found() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail={"code": "passage_not_found"},
    )


def _query_fields(artifact) -> tuple[object | None, ...]:
    return (
        artifact.context_query,
        artifact.context_query_sha256,
        artifact.context_query_term_count,
        artifact.context_query_version,
        artifact.context_stopwords_version,
    )


def _require_complete_query_metadata(artifact) -> None:
    query_fields = _query_fields(artifact)
    if any(value is None for value in query_fields) and not all(
        value is None for value in query_fields
    ):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "code": "visible_context_not_ready",
                "message": "Prepared context query metadata is incomplete",
            },
        )


def _validate_idempotency_key(idempotency_key: str) -> None:
    if not idempotency_key.isascii():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={"code": "invalid_idempotency_key"},
        )


def _document_passage_key(idempotency_key: str, passage_id: UUID) -> str:
    digest = hashlib.sha256(
        f"{idempotency_key}:{passage_id}".encode("ascii")
    ).hexdigest()
    return f"document-analysis-v1:{digest}"


def _unique_blockers(blockers: list[AnalysisBlocker]) -> list[AnalysisBlocker]:
    unique: dict[tuple[str, str], AnalysisBlocker] = {}
    for blocker in blockers:
        unique[(blocker.component, blocker.detail)] = blocker
    return list(unique.values())


def _dispatch_job(
    dispatcher: AnalysisDispatcher | None,
    job: AnalysisJobRecord,
) -> None:
    if dispatcher is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "worker_dispatch_unavailable",
                "message": "Analysis worker dispatch is unavailable",
            },
        )
    try:
        dispatcher.dispatch(job.id)
    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "worker_dispatch_failed",
                "message": "Analysis worker dispatch failed; retry with the same idempotency key",
            },
        ) from error
