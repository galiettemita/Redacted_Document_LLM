from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from uuid import UUID, uuid4

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from redacted_backend.db.models import AnalysisEvent, AnalysisJob
from redacted_backend.domain.analysis import (
    AnalysisBlocker,
    AnalysisEventRecord,
    AnalysisJobRecord,
    AnalysisStage,
    AnalysisStatus,
)


class AnalysisConflictError(ValueError):
    pass


class AnalysisJobRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def schedule(
        self,
        *,
        passage_id: UUID,
        document_id: UUID,
        passage_revision: int,
        idempotency_key: str,
        blockers: list[AnalysisBlocker],
    ) -> AnalysisJobRecord:
        request_hash = _request_hash(passage_id, passage_revision)
        existing = self._session.scalar(
            select(AnalysisJob).where(
                AnalysisJob.passage_id == str(passage_id),
                AnalysisJob.idempotency_key == idempotency_key,
            )
        )
        if existing is not None:
            if existing.request_hash != request_hash:
                raise AnalysisConflictError(
                    "idempotency key was already used for a different passage revision"
                )
            return _job_record(existing)

        status = AnalysisStatus.BLOCKED if blockers else AnalysisStatus.QUEUED
        stage = AnalysisStage.READINESS if blockers else AnalysisStage.QUERY
        now = datetime.now(UTC)
        model = AnalysisJob(
            id=str(uuid4()),
            passage_id=str(passage_id),
            document_id=str(document_id),
            passage_revision=passage_revision,
            idempotency_key=idempotency_key,
            request_hash=request_hash,
            status=status.value,
            stage=stage.value,
            blockers=[blocker.model_dump(mode="json") for blocker in blockers],
            error_code=None,
            error_detail=None,
            query_payload=None,
            retrieval_trace=None,
            prediction_payload=None,
            created_at=now,
            updated_at=now,
        )
        self._session.add(model)
        self._session.flush()
        record = _job_record(model)
        event_payload = record.model_dump(mode="json")
        event_payload["job_id"] = event_payload.pop("id")
        self._append_event(model, "passage.status", event_payload)
        return record

    def get(self, job_id: UUID | str) -> AnalysisJobRecord | None:
        model = self._session.get(AnalysisJob, str(job_id))
        return _job_record(model) if model is not None else None

    def events_after(
        self,
        document_id: UUID | str,
        event_id: int,
    ) -> list[AnalysisEventRecord]:
        models = self._session.scalars(
            select(AnalysisEvent)
            .where(
                AnalysisEvent.document_id == str(document_id),
                AnalysisEvent.id > event_id,
            )
            .order_by(AnalysisEvent.id)
        ).all()
        return [_event_record(model) for model in models]

    def document_is_terminal(self, document_id: UUID | str) -> bool:
        active = self._session.scalar(
            select(func.count())
            .select_from(AnalysisJob)
            .where(
                AnalysisJob.document_id == str(document_id),
                AnalysisJob.status.in_(
                    (
                        AnalysisStatus.QUEUED.value,
                        AnalysisStatus.RUNNING.value,
                    )
                ),
            )
        )
        return not bool(active)

    def cancel_active_for_passage(
        self,
        passage_id: UUID | str,
        *,
        reason: str,
    ) -> None:
        models = self._session.scalars(
            select(AnalysisJob).where(
                AnalysisJob.passage_id == str(passage_id),
                AnalysisJob.status.in_(
                    (
                        AnalysisStatus.QUEUED.value,
                        AnalysisStatus.RUNNING.value,
                    )
                ),
            )
        ).all()
        now = datetime.now(UTC)
        for model in models:
            model.status = AnalysisStatus.CANCELLED.value
            model.error_code = reason
            model.error_detail = "Analysis cancelled because its passage changed"
            model.updated_at = now
            self._append_event(
                model,
                "passage.status",
                {
                    "job_id": model.id,
                    "passage_id": model.passage_id,
                    "document_id": model.document_id,
                    "passage_revision": model.passage_revision,
                    "status": model.status,
                    "stage": model.stage,
                    "blockers": model.blockers,
                    "error_code": model.error_code,
                    "error_detail": model.error_detail,
                    "created_at": _as_utc(model.created_at).isoformat(),
                    "updated_at": now.isoformat(),
                },
            )
        self._session.flush()

    def _append_event(
        self,
        job: AnalysisJob,
        event_type: str,
        payload: dict[str, object],
    ) -> None:
        sequence = (
            self._session.scalar(
                select(func.max(AnalysisEvent.sequence)).where(
                    AnalysisEvent.job_id == job.id
                )
            )
            or 0
        ) + 1
        self._session.add(
            AnalysisEvent(
                job_id=job.id,
                document_id=job.document_id,
                passage_id=job.passage_id,
                sequence=sequence,
                event_type=event_type,
                payload=payload,
            )
        )
        self._session.flush()


def _request_hash(passage_id: UUID, revision: int) -> str:
    canonical = json.dumps(
        {"passage_id": str(passage_id), "passage_revision": revision},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _job_record(model: AnalysisJob) -> AnalysisJobRecord:
    return AnalysisJobRecord(
        id=model.id,
        passage_id=model.passage_id,
        document_id=model.document_id,
        passage_revision=model.passage_revision,
        status=model.status,
        stage=model.stage,
        blockers=model.blockers,
        error_code=model.error_code,
        error_detail=model.error_detail,
        created_at=_as_utc(model.created_at),
        updated_at=_as_utc(model.updated_at),
    )


def _event_record(model: AnalysisEvent) -> AnalysisEventRecord:
    return AnalysisEventRecord(
        id=model.id,
        job_id=model.job_id,
        document_id=model.document_id,
        passage_id=model.passage_id,
        sequence=model.sequence,
        event_type=model.event_type,
        payload=model.payload,
        created_at=_as_utc(model.created_at),
    )


def _as_utc(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
