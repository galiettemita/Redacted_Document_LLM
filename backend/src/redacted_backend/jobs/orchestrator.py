from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Protocol
from uuid import UUID

from sqlalchemy import text

from redacted_backend.db.base import Database
from redacted_backend.db.models import AnalysisJob, Document, Passage
from redacted_backend.db.repositories.analysis import AnalysisJobRepository
from redacted_backend.db.repositories.visible_examples import (
    VisibleExampleRepository,
    artifact_example,
)
from redacted_backend.domain.analysis import AnalysisStatus
from redacted_backend.domain.documents import DocumentStatus
from redacted_backend.domain.passages import PassageStatus
from redacted_backend.domain.predictions import PredictionRecord, QueryResult
from redacted_backend.domain.retrieval import RetrievalTrace
from redacted_backend.domain.visible_examples import VisibleExample
from redacted_backend.scientific_config import ScientificConfig


class ModelGateway(Protocol):
    def generate_query(
        self,
        example: VisibleExample,
        idempotency_key: str,
    ) -> QueryResult: ...

    def generate_prediction(
        self,
        example: VisibleExample,
        retrieval: RetrievalTrace,
        idempotency_key: str,
    ) -> PredictionRecord: ...


class RetrievalService(Protocol):
    def retrieve_realistic(
        self,
        example: VisibleExample,
        own_query: str,
    ) -> RetrievalTrace: ...


class AnalysisOrchestrator:
    """Resume one revision-bound job from its last durable stage boundary."""

    def __init__(
        self,
        *,
        database: Database,
        model: ModelGateway,
        retrieval: RetrievalService,
        scientific: ScientificConfig | None = None,
    ) -> None:
        self.database = database
        self.model = model
        self.retrieval = retrieval
        self.scientific = scientific or ScientificConfig()

    def run(self, job_id: UUID | str) -> None:
        job_key = str(job_id)
        with _execution_lock(self.database, job_key) as acquired:
            if not acquired:
                return
            self._run_locked(job_key)

    def _run_locked(self, job_key: str) -> None:
        with self.database.session_factory() as session, session.begin():
            record = AnalysisJobRepository(session).start(job_key)
        if record is None or record.status in {
            AnalysisStatus.BLOCKED,
            AnalysisStatus.COMPLETE,
            AnalysisStatus.FAILED,
            AnalysisStatus.CANCELLED,
        }:
            return

        current = self._current(job_key)
        if current is None:
            return
        example, page_number = current

        query = self._stored_query(job_key)
        if query is None:
            query = self.model.generate_query(example, f"{job_key}:query")
            if not self._still_current(job_key):
                return
            with self.database.session_factory() as session, session.begin():
                AnalysisJobRepository(session).save_query(
                    job_key,
                    query.model_dump(mode="json"),
                )

        if not self._still_current(job_key):
            return
        trace = self._stored_trace(job_key)
        if trace is None:
            trace = self.retrieval.retrieve_realistic(example, query.query)
            if not self._still_current(job_key):
                return
            with self.database.session_factory() as session, session.begin():
                AnalysisJobRepository(session).save_retrieval_trace(
                    job_key,
                    trace.model_dump(mode="json"),
                )

        if not self._still_current(job_key):
            return
        prediction = self._stored_prediction(job_key)
        if prediction is None:
            prediction = self.model.generate_prediction(
                example,
                trace,
                f"{job_key}:prediction",
            )
        if not self._still_current(job_key):
            return
        with self.database.session_factory() as session, session.begin():
            AnalysisJobRepository(session).complete(
                job_key,
                prediction.model_dump(mode="json"),
                page_number=page_number,
                model_name=self.scientific.model,
            )

    def _current(self, job_id: str) -> tuple[VisibleExample, int] | None:
        with self.database.session_factory() as session:
            job = session.get(AnalysisJob, job_id)
            if job is None:
                return None
            if job.status in {
                AnalysisStatus.BLOCKED.value,
                AnalysisStatus.COMPLETE.value,
                AnalysisStatus.FAILED.value,
                AnalysisStatus.CANCELLED.value,
            }:
                return None
            passage = session.get(Passage, job.passage_id)
            document = session.get(Document, job.document_id)
            if passage is None or passage.status == PassageStatus.DELETED.value:
                reason = (
                    "passage_deleted",
                    "Passage was deleted before analysis completed",
                )
            elif document is None or document.status == DocumentStatus.DELETED.value:
                reason = (
                    "document_deleted",
                    "Document was deleted before analysis completed",
                )
            elif passage.revision != job.passage_revision:
                reason = (
                    "passage_revision_changed",
                    "Passage revision changed before analysis completed",
                )
            else:
                artifact = VisibleExampleRepository(session).get_current(
                    passage.id,
                    passage.revision,
                )
                if artifact is not None:
                    return artifact_example(artifact), passage.page_number
                reason = (
                    "visible_context_invalidated",
                    "Visible-only context was invalidated before analysis completed",
                )
        self._cancel(job_id, *reason)
        return None

    def _still_current(self, job_id: str) -> bool:
        return self._current(job_id) is not None

    def _cancel(self, job_id: str, code: str, detail: str) -> None:
        with self.database.session_factory() as session, session.begin():
            AnalysisJobRepository(session).cancel(
                job_id,
                code=code,
                detail=detail,
            )

    def _stored_query(self, job_id: str) -> QueryResult | None:
        payload = self._payload(job_id, "query_payload")
        return QueryResult.model_validate(payload) if payload is not None else None

    def _stored_trace(self, job_id: str) -> RetrievalTrace | None:
        payload = self._payload(job_id, "retrieval_trace")
        return RetrievalTrace.model_validate(payload) if payload is not None else None

    def _stored_prediction(self, job_id: str) -> PredictionRecord | None:
        payload = self._payload(job_id, "prediction_payload")
        return PredictionRecord.model_validate(payload) if payload is not None else None

    def _payload(self, job_id: str, field: str) -> dict[str, object] | None:
        with self.database.session_factory() as session:
            job = session.get(AnalysisJob, job_id)
            if job is None:
                return None
            return getattr(job, field)


@contextmanager
def _execution_lock(database: Database, job_id: str) -> Iterator[bool]:
    """Use a crash-safe PostgreSQL session lock to suppress duplicate workers."""

    if database.engine.dialect.name != "postgresql":
        yield True
        return
    with database.engine.connect() as connection:
        acquired = bool(
            connection.execute(
                text("SELECT pg_try_advisory_lock(hashtext(:job_id))"),
                {"job_id": job_id},
            ).scalar_one()
        )
        try:
            yield acquired
        finally:
            if acquired:
                connection.execute(
                    text("SELECT pg_advisory_unlock(hashtext(:job_id))"),
                    {"job_id": job_id},
                )
