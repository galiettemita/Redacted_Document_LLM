from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from uuid import UUID, uuid4

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from redacted_backend.db.models import (
    AnalysisEvent,
    AnalysisJob,
    Passage,
    VisibleExampleArtifact,
)
from redacted_backend.domain.analysis import (
    AnalysisBlocker,
    AnalysisEventRecord,
    AnalysisJobRecord,
    AnalysisStage,
    AnalysisStatus,
    PredictionBasis,
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
        prediction_basis: PredictionBasis,
    ) -> AnalysisJobRecord:
        request_hash = _request_hash(passage_id, passage_revision)
        existing = self._session.scalar(
            select(AnalysisJob).where(
                AnalysisJob.passage_id == str(passage_id),
                AnalysisJob.idempotency_key == idempotency_key,
            )
        )
        if existing is not None:
            return _replayed_job(existing, request_hash)

        status = AnalysisStatus.BLOCKED if blockers else AnalysisStatus.QUEUED
        stage = (
            AnalysisStage.READINESS
            if blockers
            else AnalysisStage.PREDICTION
            if prediction_basis == PredictionBasis.MODEL_ONLY_V1
            else AnalysisStage.QUERY
        )
        now = datetime.now(UTC)
        model = AnalysisJob(
            id=str(uuid4()),
            passage_id=str(passage_id),
            document_id=str(document_id),
            passage_revision=passage_revision,
            prediction_basis=prediction_basis.value,
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
        savepoint = self._session.begin_nested()
        try:
            self._session.add(model)
            self._session.flush()
            savepoint.commit()
        except IntegrityError:
            savepoint.rollback()
            self._session.expire_all()
            winner = self._session.scalar(
                select(AnalysisJob).where(
                    AnalysisJob.passage_id == str(passage_id),
                    AnalysisJob.idempotency_key == idempotency_key,
                )
            )
            if winner is None:
                raise
            return _replayed_job(winner, request_hash)
        record = _job_record(model)
        event_payload = record.model_dump(mode="json")
        event_payload["job_id"] = event_payload.pop("id")
        self._append_event(model, "passage.status", event_payload)
        return record

    def get(self, job_id: UUID | str) -> AnalysisJobRecord | None:
        model = self._session.get(AnalysisJob, str(job_id))
        return _job_record(model) if model is not None else None

    def get_by_idempotency(
        self,
        passage_id: UUID | str,
        idempotency_key: str,
    ) -> AnalysisJobRecord | None:
        model = self._session.scalar(
            select(AnalysisJob).where(
                AnalysisJob.passage_id == str(passage_id),
                AnalysisJob.idempotency_key == idempotency_key,
            )
        )
        return _job_record(model) if model is not None else None

    def recoverable_for_document(
        self,
        document_id: UUID | str,
    ) -> list[AnalysisJobRecord]:
        rows = self._session.execute(
            select(AnalysisJob, Passage)
            .join(Passage, AnalysisJob.passage_id == Passage.id)
            .where(
                AnalysisJob.document_id == str(document_id),
                AnalysisJob.passage_revision == Passage.revision,
                AnalysisJob.status.in_(
                    (
                        AnalysisStatus.QUEUED.value,
                        AnalysisStatus.RUNNING.value,
                        AnalysisStatus.COMPLETE.value,
                    )
                ),
                Passage.status != "deleted",
            )
        ).all()
        latest_by_passage: dict[str, AnalysisJob] = {}
        passages: dict[str, Passage] = {}
        for job, passage in rows:
            passages[passage.id] = passage
            current = latest_by_passage.get(passage.id)
            if current is None or _as_utc(job.updated_at) > _as_utc(
                current.updated_at
            ):
                latest_by_passage[passage.id] = job
        return [
            _job_record(latest_by_passage[passage_id])
            for passage_id in sorted(
                latest_by_passage,
                key=lambda value: _passage_order_key(passages[value]),
            )
        ]

    def recoverable_for_document_basis(
        self,
        document_id: UUID | str,
        prediction_basis: PredictionBasis,
    ) -> list[AnalysisJobRecord]:
        rows = self._session.execute(
            select(AnalysisJob, Passage)
            .join(Passage, AnalysisJob.passage_id == Passage.id)
            .where(
                AnalysisJob.document_id == str(document_id),
                AnalysisJob.passage_revision == Passage.revision,
                AnalysisJob.prediction_basis == prediction_basis.value,
                AnalysisJob.status.in_(
                    (
                        AnalysisStatus.QUEUED.value,
                        AnalysisStatus.RUNNING.value,
                        AnalysisStatus.COMPLETE.value,
                    )
                ),
                Passage.status != "deleted",
            )
        ).all()
        latest_by_passage: dict[str, AnalysisJob] = {}
        passages: dict[str, Passage] = {}
        for job, passage in rows:
            passages[passage.id] = passage
            current = latest_by_passage.get(passage.id)
            if current is None or _as_utc(job.updated_at) > _as_utc(
                current.updated_at
            ):
                latest_by_passage[passage.id] = job
        return [
            _job_record(latest_by_passage[passage_id])
            for passage_id in sorted(
                latest_by_passage,
                key=lambda value: _passage_order_key(passages[value]),
            )
        ]

    def has_active_for_document_outside_basis(
        self,
        document_id: UUID | str,
        prediction_basis: PredictionBasis,
    ) -> bool:
        count = self._session.scalar(
            select(func.count())
            .select_from(AnalysisJob)
            .join(Passage, AnalysisJob.passage_id == Passage.id)
            .where(
                AnalysisJob.document_id == str(document_id),
                AnalysisJob.passage_revision == Passage.revision,
                AnalysisJob.prediction_basis != prediction_basis.value,
                AnalysisJob.status.in_(
                    (
                        AnalysisStatus.QUEUED.value,
                        AnalysisStatus.RUNNING.value,
                    )
                ),
                Passage.status != "deleted",
            )
        )
        return int(count or 0) > 0

    def recoverable_for_passage_basis(
        self,
        *,
        passage_id: UUID | str,
        passage_revision: int,
        prediction_basis: PredictionBasis,
    ) -> AnalysisJobRecord | None:
        model = self._session.scalar(
            select(AnalysisJob)
            .where(
                AnalysisJob.passage_id == str(passage_id),
                AnalysisJob.passage_revision == passage_revision,
                AnalysisJob.prediction_basis == prediction_basis.value,
                AnalysisJob.status.in_(
                    (
                        AnalysisStatus.QUEUED.value,
                        AnalysisStatus.RUNNING.value,
                        AnalysisStatus.COMPLETE.value,
                    )
                ),
            )
            .order_by(AnalysisJob.updated_at.desc(), AnalysisJob.id.desc())
            .limit(1)
        )
        return _job_record(model) if model is not None else None

    def staged_prediction_blocker(
        self,
        *,
        document_id: UUID | str,
        passage_id: UUID | str,
        idempotency_key: str,
        limit: int,
    ) -> AnalysisBlocker | None:
        existing = self._session.scalar(
            select(AnalysisJob.id).where(
                AnalysisJob.passage_id == str(passage_id),
                AnalysisJob.idempotency_key == idempotency_key,
            )
        )
        if existing is not None:
            return None
        prepared = self._session.scalars(
            select(Passage)
            .join(
                VisibleExampleArtifact,
                VisibleExampleArtifact.passage_id == Passage.id,
            )
            .where(
                Passage.document_id == str(document_id),
                Passage.status != "deleted",
                VisibleExampleArtifact.passage_revision == Passage.revision,
                VisibleExampleArtifact.invalidated_at.is_(None),
            )
        ).all()
        eligible_ids = [
            model.id
            for model in sorted(prepared, key=_passage_order_key)[:limit]
        ]
        if str(passage_id) not in eligible_ids:
            return _staged_limit_blocker()
        consuming = self._session.scalar(
            select(func.count())
            .select_from(AnalysisJob)
            .where(
                AnalysisJob.document_id == str(document_id),
                AnalysisJob.prediction_basis
                == PredictionBasis.BEST_AVAILABLE_FOIARCHIVE_BM25_V1.value,
                AnalysisJob.status.in_(
                    (
                        AnalysisStatus.QUEUED.value,
                        AnalysisStatus.RUNNING.value,
                        AnalysisStatus.COMPLETE.value,
                    )
                ),
            )
        )
        if int(consuming or 0) >= limit:
            return _staged_limit_blocker()
        return None

    def selected_staged_prediction_blocker(
        self,
        *,
        document_id: UUID | str,
        passage_id: UUID | str,
        idempotency_key: str,
        limit: int,
    ) -> AnalysisBlocker | None:
        existing = self._session.scalar(
            select(AnalysisJob.id).where(
                AnalysisJob.passage_id == str(passage_id),
                AnalysisJob.idempotency_key == idempotency_key,
            )
        )
        if existing is not None:
            return None
        active = self._session.scalar(
            select(func.count())
            .select_from(AnalysisJob)
            .where(
                AnalysisJob.document_id == str(document_id),
                AnalysisJob.prediction_basis
                == PredictionBasis.BEST_AVAILABLE_FOIARCHIVE_BM25_V1.value,
                AnalysisJob.status.in_(
                    (
                        AnalysisStatus.QUEUED.value,
                        AnalysisStatus.RUNNING.value,
                    )
                ),
            )
        )
        if int(active or 0) >= limit:
            return _staged_limit_blocker()
        return None

    def start(self, job_id: UUID | str) -> AnalysisJobRecord | None:
        model = self._session.get(AnalysisJob, str(job_id), with_for_update=True)
        if model is None:
            return None
        if model.status == AnalysisStatus.QUEUED.value:
            model.status = AnalysisStatus.RUNNING.value
            model.updated_at = datetime.now(UTC)
            self._append_status_event(model)
        self._session.flush()
        return _job_record(model)

    def save_query(
        self,
        job_id: UUID | str,
        payload: dict[str, object],
    ) -> AnalysisJobRecord:
        model = self._required_for_update(job_id)
        self._save_immutable_payload(model, "query_payload", payload)
        if model.stage == AnalysisStage.QUERY.value:
            model.stage = AnalysisStage.RETRIEVAL.value
            model.status = AnalysisStatus.RUNNING.value
            model.updated_at = datetime.now(UTC)
            self._append_status_event(model)
        self._session.flush()
        return _job_record(model)

    def save_retrieval_trace(
        self,
        job_id: UUID | str,
        payload: dict[str, object],
    ) -> AnalysisJobRecord:
        model = self._required_for_update(job_id)
        self._save_immutable_payload(model, "retrieval_trace", payload)
        if model.stage == AnalysisStage.RETRIEVAL.value:
            model.stage = AnalysisStage.PREDICTION.value
            model.status = AnalysisStatus.RUNNING.value
            model.updated_at = datetime.now(UTC)
            self._append_status_event(model)
        self._session.flush()
        return _job_record(model)

    def complete(
        self,
        job_id: UUID | str,
        payload: dict[str, object],
        *,
        page_number: int,
        model_name: str,
    ) -> AnalysisJobRecord:
        model = self._required_for_update(job_id)
        if payload.get("prediction_basis") != model.prediction_basis:
            raise AnalysisConflictError(
                "Prediction payload basis does not match the scheduled job"
            )
        self._save_immutable_payload(model, "prediction_payload", payload)
        if model.status == AnalysisStatus.COMPLETE.value:
            return _job_record(model)
        now = datetime.now(UTC)
        self._append_event(
            model,
            "passage.prediction",
            {
                "job_id": model.id,
                "passage_id": model.passage_id,
                "document_id": model.document_id,
                "passage_revision": model.passage_revision,
                "prediction_basis": model.prediction_basis,
                "page_number": page_number,
                "model": model_name,
                "prediction": payload["prediction"],
                "provider_response_id": payload["provider_response_id"],
                "provider_request_id": payload.get("provider_request_id"),
                "prompt_sha256": payload["prompt_sha256"],
                "request_sha256": payload["request_sha256"],
                "retrieval_trace_sha256": payload["retrieval_trace_sha256"],
                "usage": payload["usage"],
                "estimated_cost_usd": payload["estimated_cost_usd"],
                "pricing_version": payload["pricing_version"],
            },
        )
        model.status = AnalysisStatus.COMPLETE.value
        model.stage = AnalysisStage.COMPLETE.value
        model.updated_at = now
        self._append_status_event(model)
        self._session.flush()
        return _job_record(model)

    def cancel(
        self,
        job_id: UUID | str,
        *,
        code: str,
        detail: str,
    ) -> AnalysisJobRecord | None:
        model = self._session.get(AnalysisJob, str(job_id), with_for_update=True)
        if model is None:
            return None
        if model.status in {
            AnalysisStatus.COMPLETE.value,
            AnalysisStatus.FAILED.value,
            AnalysisStatus.CANCELLED.value,
        }:
            return _job_record(model)
        model.status = AnalysisStatus.CANCELLED.value
        model.error_code = code
        model.error_detail = detail
        model.updated_at = datetime.now(UTC)
        self._append_status_event(model)
        self._session.flush()
        return _job_record(model)

    def fail(
        self,
        job_id: UUID | str,
        *,
        code: str,
        detail: str,
    ) -> AnalysisJobRecord | None:
        model = self._session.get(AnalysisJob, str(job_id), with_for_update=True)
        if model is None:
            return None
        if model.status in {
            AnalysisStatus.COMPLETE.value,
            AnalysisStatus.CANCELLED.value,
        }:
            return _job_record(model)
        model.status = AnalysisStatus.FAILED.value
        model.error_code = code
        model.error_detail = detail
        model.updated_at = datetime.now(UTC)
        self._append_event(
            model,
            "passage.error",
            {
                "job_id": model.id,
                "passage_id": model.passage_id,
                "document_id": model.document_id,
                "passage_revision": model.passage_revision,
                "code": code,
                "detail": detail,
            },
        )
        self._append_status_event(model)
        self._session.flush()
        return _job_record(model)

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

    def document_has_active_jobs(self, document_id: UUID | str) -> bool:
        active = self._session.scalar(
            select(AnalysisJob.id).where(
                AnalysisJob.document_id == str(document_id),
                AnalysisJob.status.in_(
                    (
                        AnalysisStatus.QUEUED.value,
                        AnalysisStatus.RUNNING.value,
                    )
                ),
            ).limit(1)
        )
        return active is not None

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
                    "prediction_basis": model.prediction_basis,
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

    def _required_for_update(self, job_id: UUID | str) -> AnalysisJob:
        model = self._session.get(AnalysisJob, str(job_id), with_for_update=True)
        if model is None:
            raise LookupError("analysis job not found")
        if model.status in {
            AnalysisStatus.BLOCKED.value,
            AnalysisStatus.FAILED.value,
            AnalysisStatus.CANCELLED.value,
        }:
            raise AnalysisConflictError(
                f"analysis job cannot advance from {model.status}"
            )
        return model

    def _save_immutable_payload(
        self,
        model: AnalysisJob,
        attribute: str,
        payload: dict[str, object],
    ) -> None:
        existing = getattr(model, attribute)
        if existing is not None and existing != payload:
            raise AnalysisConflictError(f"{attribute} is immutable once persisted")
        if existing is None:
            setattr(model, attribute, payload)

    def _append_status_event(self, model: AnalysisJob) -> None:
        self._append_event(model, "passage.status", _status_payload(model))


def _request_hash(passage_id: UUID, revision: int) -> str:
    canonical = json.dumps(
        {"passage_id": str(passage_id), "passage_revision": revision},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _staged_limit_blocker() -> AnalysisBlocker:
    return AnalysisBlocker(
        component="staged_prediction_limit",
        detail="This passage is not enabled for staged prediction yet",
    )


def _passage_order_key(model: Passage) -> tuple[int, float, float, str]:
    first_rect = min(model.rects, key=lambda rect: (rect["y"], rect["x"]))
    return model.page_number, first_rect["y"], first_rect["x"], model.id


def _replayed_job(model: AnalysisJob, request_hash: str) -> AnalysisJobRecord:
    if model.request_hash != request_hash:
        raise AnalysisConflictError(
            "idempotency key was already used for a different passage revision"
        )
    return _job_record(model)


def _job_record(model: AnalysisJob) -> AnalysisJobRecord:
    return AnalysisJobRecord(
        id=model.id,
        passage_id=model.passage_id,
        document_id=model.document_id,
        passage_revision=model.passage_revision,
        prediction_basis=model.prediction_basis,
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


def _status_payload(model: AnalysisJob) -> dict[str, object]:
    return {
        "job_id": model.id,
        "passage_id": model.passage_id,
        "document_id": model.document_id,
        "passage_revision": model.passage_revision,
        "prediction_basis": model.prediction_basis,
        "status": model.status,
        "stage": model.stage,
        "blockers": model.blockers,
        "error_code": model.error_code,
        "error_detail": model.error_detail,
        "created_at": _as_utc(model.created_at).isoformat(),
        "updated_at": _as_utc(model.updated_at).isoformat(),
    }


def _as_utc(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
