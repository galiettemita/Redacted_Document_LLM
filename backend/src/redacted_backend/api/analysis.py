from __future__ import annotations

import json
import time
from collections.abc import Iterator
from uuid import UUID

from fastapi import APIRouter, Header, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from redacted_backend.api.readiness import resolve_corpus_readiness
from redacted_backend.config import Settings
from redacted_backend.db.base import Database
from redacted_backend.db.repositories.analysis import (
    AnalysisConflictError,
    AnalysisJobRepository,
)
from redacted_backend.db.repositories.passages import PassageRepository
from redacted_backend.db.repositories.visible_examples import VisibleExampleRepository
from redacted_backend.domain.analysis import AnalysisBlocker, AnalysisJobRecord
from redacted_backend.domain.passages import PassageStatus


class ScheduleAnalysisRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    expected_revision: int = Field(ge=1)


def create_analysis_router(settings: Settings, database: Database | None) -> APIRouter:
    router = APIRouter()

    @router.post(
        "/passages/{passage_id}/analysis",
        response_model=AnalysisJobRecord,
        status_code=status.HTTP_202_ACCEPTED,
    )
    def schedule_analysis(
        passage_id: UUID,
        body: ScheduleAnalysisRequest,
        idempotency_key: str = Header(alias="Idempotency-Key", min_length=1, max_length=512),
    ) -> AnalysisJobRecord:
        ready_database = _require_database(database)
        if not idempotency_key.isascii():
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail={"code": "invalid_idempotency_key"},
            )
        try:
            with ready_database.session_factory() as session, session.begin():
                passage = PassageRepository(session).get(passage_id)
                if passage is None or passage.status == PassageStatus.DELETED:
                    raise _not_found()
                if passage.revision != body.expected_revision:
                    raise HTTPException(
                        status_code=status.HTTP_409_CONFLICT,
                        detail={
                            "code": "passage_revision_changed",
                            "message": "Passage revision changed before analysis scheduling",
                        },
                    )
                artifact = VisibleExampleRepository(session).get_current(
                    passage.id,
                    passage.revision,
                )
                if artifact is None or artifact.context_query is None:
                    raise HTTPException(
                        status_code=status.HTTP_409_CONFLICT,
                        detail={
                            "code": "visible_context_not_ready",
                            "message": "Prepare visible-only context before analysis",
                        },
                    )
                return AnalysisJobRepository(session).schedule(
                    passage_id=passage.id,
                    document_id=passage.document_id,
                    passage_revision=passage.revision,
                    idempotency_key=idempotency_key,
                    blockers=_analysis_blockers(settings),
                )
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
        if record is None:
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


def _analysis_blockers(settings: Settings) -> list[AnalysisBlocker]:
    blockers: list[AnalysisBlocker] = []
    corpus = resolve_corpus_readiness(settings)
    if not corpus.ready:
        blockers.append(AnalysisBlocker(component="corpus", detail=corpus.detail))
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
    blockers.append(
        AnalysisBlocker(
            component="worker",
            detail="analysis worker execution is not connected",
        )
    )
    return blockers


def _event_stream(
    database: Database,
    document_id: UUID,
    after: int,
) -> Iterator[str]:
    cursor = after
    last_output = time.monotonic()
    while True:
        with database.session_factory() as session:
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


def _not_found() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail={"code": "passage_not_found"},
    )
