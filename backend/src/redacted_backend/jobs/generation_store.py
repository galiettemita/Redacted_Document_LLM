from __future__ import annotations

from redacted_backend.db.base import Database
from redacted_backend.db.models import AnalysisJob
from redacted_backend.models.openai_gateway import (
    ModelGatewayError,
    StoredGeneration,
)


class AnalysisGenerationStore:
    """Persist provider results on the owning job before orchestration advances."""

    def __init__(self, database: Database) -> None:
        self.database = database

    def get(self, scope: str, key: str) -> StoredGeneration | None:
        job_id, kind = _identity(scope, key)
        field = _payload_field(kind)
        with self.database.session_factory() as session:
            job = session.get(AnalysisJob, job_id)
            if job is None:
                return None
            _validate_passage_scope(job, scope)
            payload = getattr(job, field)
        if payload is None:
            return None
        return StoredGeneration(
            kind=kind,
            request_sha256=str(payload["request_sha256"]),
            payload=payload,
        )

    def save(self, scope: str, key: str, value: StoredGeneration) -> None:
        job_id, kind = _identity(scope, key)
        if value.kind != kind:
            raise ModelGatewayError(
                "idempotency_conflict",
                "Generation kind does not match its durable job key",
            )
        field = _payload_field(kind)
        with self.database.session_factory() as session, session.begin():
            job = session.get(AnalysisJob, job_id, with_for_update=True)
            if job is None:
                raise ModelGatewayError(
                    "analysis_job_missing",
                    "Generation job no longer exists",
                )
            _validate_passage_scope(job, scope)
            existing = getattr(job, field)
            if existing is not None and existing != value.payload:
                raise ModelGatewayError(
                    "idempotency_conflict",
                    "Generation result is immutable once persisted",
                )
            if existing is None:
                setattr(job, field, value.payload)
                session.flush()


def _identity(scope: str, key: str) -> tuple[str, str]:
    try:
        job_id, kind = key.rsplit(":", 1)
        scope_kind, _passage_id = scope.split(":", 1)
    except ValueError as error:
        raise ModelGatewayError(
            "invalid_idempotency_key",
            "Generation key is not bound to an analysis job",
        ) from error
    if kind not in {"query", "prediction"} or scope_kind != kind or not job_id:
        raise ModelGatewayError(
            "invalid_idempotency_key",
            "Generation key is not bound to the requested stage",
        )
    return job_id, kind


def _payload_field(kind: str) -> str:
    return "query_payload" if kind == "query" else "prediction_payload"


def _validate_passage_scope(job: AnalysisJob, scope: str) -> None:
    _kind, passage_id = scope.split(":", 1)
    if passage_id != job.passage_id:
        raise ModelGatewayError(
            "idempotency_conflict",
            "Generation scope belongs to a different passage",
        )
