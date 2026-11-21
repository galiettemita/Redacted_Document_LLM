from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from sqlalchemy import select

from redacted_backend.db.models import AnalysisEvent, AnalysisJob, Passage
from redacted_backend.db.repositories.analysis import AnalysisJobRepository
from redacted_backend.domain.analysis import PredictionBasis
from redacted_backend.domain.predictions import (
    PredictionRecord,
    QueryResult,
    TokenUsage,
)
from redacted_backend.jobs.orchestrator import AnalysisOrchestrator
from redacted_backend.models.openai_gateway import ModelGatewayError
from tests.api.test_context import (
    create_test_app_and_passage,
    install_fake_preparation,
    request,
)
from tests.models.test_prompts import _retrieval


class RecordingGateway:
    def __init__(self, *, fail_prediction_once: bool = False) -> None:
        self.query_calls: list[str] = []
        self.prediction_calls: list[str] = []
        self.prediction_bases: list[PredictionBasis] = []
        self.model_only_calls: list[str] = []
        self.fail_prediction_once = fail_prediction_once

    def generate_query(
        self,
        example,
        idempotency_key: str,
        *,
        before_generation=None,
    ) -> QueryResult:
        if before_generation is not None and not before_generation():
            raise ModelGatewayError("analysis_cancelled", "job is no longer current")
        self.query_calls.append(idempotency_key)
        return QueryResult(
            passage_id=example.passage_id,
            query="specific actors paris 1960 meeting",
            prompt_sha256="1" * 64,
            request_sha256="2" * 64,
            provider_response_id="resp-query",
            provider_request_id="req-query",
            input_tokens=100,
            usage=_usage(100, 20),
            estimated_cost_usd=0.002,
            pricing_version="test-pricing",
        )

    def generate_prediction(
        self,
        example,
        retrieval,
        idempotency_key: str,
        *,
        prediction_basis: PredictionBasis,
        before_generation=None,
    ) -> PredictionRecord:
        if before_generation is not None and not before_generation():
            raise ModelGatewayError("analysis_cancelled", "job is no longer current")
        self.prediction_calls.append(idempotency_key)
        self.prediction_bases.append(prediction_basis)
        if self.fail_prediction_once:
            self.fail_prediction_once = False
            raise TimeoutError("temporary provider timeout")
        trace_hash = hashlib.sha256(
            retrieval.model_dump_json().encode("utf-8")
        ).hexdigest()
        return PredictionRecord(
            passage_id=example.passage_id,
            prediction_basis=prediction_basis,
            prediction="The missing passage identifies the officials and their planned meeting.",
            prompt_sha256="3" * 64,
            request_sha256="4" * 64,
            retrieval_trace_sha256=trace_hash,
            provider_response_id="resp-prediction",
            provider_request_id="req-prediction",
            input_tokens=2_000,
            usage=_usage(2_000, 120),
            estimated_cost_usd=0.026,
            pricing_version="test-pricing",
        )

    def generate_model_only_prediction(
        self,
        example,
        idempotency_key: str,
        *,
        before_generation=None,
    ) -> PredictionRecord:
        if before_generation is not None and not before_generation():
            raise ModelGatewayError("analysis_cancelled", "job is no longer current")
        self.model_only_calls.append(idempotency_key)
        return PredictionRecord(
            passage_id=example.passage_id,
            prediction_basis=PredictionBasis.MODEL_ONLY_V1,
            prediction="The missing passage names the officials.",
            prompt_sha256="5" * 64,
            request_sha256="6" * 64,
            retrieval_trace_sha256=None,
            provider_response_id="resp-model-only",
            provider_request_id="req-model-only",
            input_tokens=800,
            usage=_usage(800, 80),
            estimated_cost_usd=0.012,
            pricing_version="test-pricing",
        )


class RecordingRetrieval:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def retrieve_realistic(self, example, own_query: str):
        self.calls.append(own_query)
        return _retrieval().model_copy(
            update={
                "context_query_sha256": hashlib.sha256(
                    b"visible before visible after"
                ).hexdigest(),
                "own_query_sha256": hashlib.sha256(
                    own_query.encode("utf-8")
                ).hexdigest(),
            }
        )


def test_model_only_orchestrator_skips_query_and_retrieval(
    tmp_path: Path,
    monkeypatch,
) -> None:
    app, document, passage = create_test_app_and_passage(tmp_path)
    install_fake_preparation(monkeypatch, calls=[])
    _prepare(app, passage.id)
    job_id = _schedule_ready(
        app,
        document.id,
        passage.id,
        passage.revision,
        basis=PredictionBasis.MODEL_ONLY_V1,
    )
    gateway = RecordingGateway()

    AnalysisOrchestrator(
        database=app.state.database,
        model=gateway,
        retrieval=None,
    ).run(job_id)

    with app.state.database.session_factory() as session:
        job = session.get(AnalysisJob, job_id)
        events = session.scalars(
            select(AnalysisEvent)
            .where(AnalysisEvent.job_id == job_id)
            .order_by(AnalysisEvent.sequence)
        ).all()
    assert job is not None
    assert job.status == "complete"
    assert job.query_payload is None
    assert job.retrieval_trace is None
    assert job.prediction_payload["prediction_basis"] == "model_only_v1"
    assert job.prediction_payload["retrieval_trace_sha256"] is None
    assert gateway.query_calls == []
    assert gateway.prediction_calls == []
    assert gateway.model_only_calls == [f"{job_id}:prediction"]
    assert [event.event_type for event in events] == [
        "passage.status",
        "passage.status",
        "passage.prediction",
        "passage.status",
    ]
    assert events[-2].payload["prediction_basis"] == "model_only_v1"


def test_model_only_orchestrator_resumes_a_stored_prediction_without_generation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    app, document, passage = create_test_app_and_passage(tmp_path)
    install_fake_preparation(monkeypatch, calls=[])
    _prepare(app, passage.id)
    job_id = _schedule_ready(
        app,
        document.id,
        passage.id,
        passage.revision,
        basis=PredictionBasis.MODEL_ONLY_V1,
    )
    stored = PredictionRecord(
        passage_id=passage.id,
        prediction_basis=PredictionBasis.MODEL_ONLY_V1,
        prediction="Durably stored prediction",
        prompt_sha256="7" * 64,
        request_sha256="8" * 64,
        retrieval_trace_sha256=None,
        provider_response_id="resp-stored",
        provider_request_id="req-stored",
        input_tokens=700,
        usage=_usage(700, 70),
        estimated_cost_usd=0.01,
        pricing_version="test-pricing",
    )
    with app.state.database.session_factory() as session, session.begin():
        job = session.get(AnalysisJob, job_id)
        assert job is not None
        job.prediction_payload = stored.model_dump(mode="json")
    gateway = RecordingGateway()

    AnalysisOrchestrator(
        database=app.state.database,
        model=gateway,
        retrieval=None,
    ).run(job_id)

    assert gateway.model_only_calls == []
    with app.state.database.session_factory() as session:
        job = session.get(AnalysisJob, job_id)
    assert job is not None
    assert job.status == "complete"
    assert job.prediction_payload["prediction"] == "Durably stored prediction"


def test_deleted_passage_cancels_model_only_before_generation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    app, document, passage = create_test_app_and_passage(tmp_path)
    install_fake_preparation(monkeypatch, calls=[])
    _prepare(app, passage.id)
    job_id = _schedule_ready(
        app,
        document.id,
        passage.id,
        passage.revision,
        basis=PredictionBasis.MODEL_ONLY_V1,
    )
    with app.state.database.session_factory() as session, session.begin():
        model = session.get(Passage, str(passage.id))
        assert model is not None
        model.status = "deleted"
    gateway = RecordingGateway()

    AnalysisOrchestrator(
        database=app.state.database,
        model=gateway,
        retrieval=None,
    ).run(job_id)

    with app.state.database.session_factory() as session:
        job = session.get(AnalysisJob, job_id)
    assert job is not None
    assert job.status == "cancelled"
    assert gateway.model_only_calls == []


def test_model_only_rechecks_revision_immediately_before_generation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    app, document, passage = create_test_app_and_passage(tmp_path)
    install_fake_preparation(monkeypatch, calls=[])
    _prepare(app, passage.id)
    job_id = _schedule_ready(
        app,
        document.id,
        passage.id,
        passage.revision,
        basis=PredictionBasis.MODEL_ONLY_V1,
    )

    class InvalidatingGateway(RecordingGateway):
        def generate_model_only_prediction(
            self,
            example,
            idempotency_key: str,
            *,
            before_generation=None,
        ) -> PredictionRecord:
            with app.state.database.session_factory() as session, session.begin():
                model = session.get(Passage, str(passage.id))
                assert model is not None
                model.revision += 1
            return super().generate_model_only_prediction(
                example,
                idempotency_key,
                before_generation=before_generation,
            )

    gateway = InvalidatingGateway()

    AnalysisOrchestrator(
        database=app.state.database,
        model=gateway,
        retrieval=None,
    ).run(job_id)

    with app.state.database.session_factory() as session:
        job = session.get(AnalysisJob, job_id)
    assert job is not None
    assert job.status == "cancelled"
    assert job.error_code == "passage_revision_changed"
    assert gateway.model_only_calls == []


def test_orchestrator_persists_each_stage_and_emits_one_real_prediction(
    tmp_path: Path,
    monkeypatch,
) -> None:
    app, document, passage = create_test_app_and_passage(tmp_path)
    install_fake_preparation(monkeypatch, calls=[])
    _prepare(app, passage.id)
    job_id = _schedule_ready(app, document.id, passage.id, passage.revision)
    gateway = RecordingGateway()
    retrieval = RecordingRetrieval()

    AnalysisOrchestrator(
        database=app.state.database,
        model=gateway,
        retrieval=retrieval,
    ).run(job_id)

    with app.state.database.session_factory() as session:
        job = session.get(AnalysisJob, job_id)
        events = session.scalars(
            select(AnalysisEvent)
            .where(AnalysisEvent.job_id == job_id)
            .order_by(AnalysisEvent.sequence)
        ).all()
    assert job is not None
    assert job.status == "complete"
    assert job.stage == "complete"
    assert job.query_payload["query"] == "specific actors paris 1960 meeting"
    assert set(job.retrieval_trace["lists"]) == {
        "context_bm25",
        "context_dense",
        "own_bm25",
        "own_dense",
    }
    assert len(job.retrieval_trace["documents"]) > 0
    assert job.prediction_payload["prediction"].startswith("The missing passage")
    assert [event.event_type for event in events] == [
        "passage.status",
        "passage.status",
        "passage.status",
        "passage.status",
        "passage.prediction",
        "passage.status",
    ]
    prediction_event = events[-2].payload
    assert prediction_event["passage_id"] == str(passage.id)
    assert prediction_event["page_number"] == 1
    assert prediction_event["model"] == "gpt-6-astra"
    assert prediction_event["prediction"].startswith("The missing passage")
    assert gateway.query_calls == [f"{job_id}:query"]
    assert gateway.prediction_calls == [f"{job_id}:prediction"]
    assert gateway.prediction_bases == [PredictionBasis.PAPER_EXACT_V1]
    assert retrieval.calls == ["specific actors paris 1960 meeting"]


def test_best_available_orchestrator_preserves_basis_through_prediction_event(
    tmp_path: Path,
    monkeypatch,
) -> None:
    app, document, passage = create_test_app_and_passage(tmp_path)
    install_fake_preparation(monkeypatch, calls=[])
    _prepare(app, passage.id)
    job_id = _schedule_ready(
        app,
        document.id,
        passage.id,
        passage.revision,
        basis=PredictionBasis.BEST_AVAILABLE_FOIARCHIVE_V1,
    )
    gateway = RecordingGateway()
    retrieval = RecordingRetrieval()

    AnalysisOrchestrator(
        database=app.state.database,
        model=gateway,
        retrieval=retrieval,
        prediction_basis=PredictionBasis.BEST_AVAILABLE_FOIARCHIVE_V1,
    ).run(job_id)

    with app.state.database.session_factory() as session:
        job = session.get(AnalysisJob, job_id)
        events = session.scalars(
            select(AnalysisEvent)
            .where(AnalysisEvent.job_id == job_id)
            .order_by(AnalysisEvent.sequence)
        ).all()
    assert job is not None
    assert job.prediction_basis == "best_available_foiarchive_v1"
    assert job.prediction_payload["prediction_basis"] == (
        "best_available_foiarchive_v1"
    )
    assert events[-2].payload["prediction_basis"] == (
        "best_available_foiarchive_v1"
    )
    assert gateway.prediction_bases == [
        PredictionBasis.BEST_AVAILABLE_FOIARCHIVE_V1
    ]
    assert retrieval.calls == ["specific actors paris 1960 meeting"]


def test_basis_bound_runtime_refuses_a_job_from_another_basis(
    tmp_path: Path,
    monkeypatch,
) -> None:
    app, document, passage = create_test_app_and_passage(tmp_path)
    install_fake_preparation(monkeypatch, calls=[])
    _prepare(app, passage.id)
    job_id = _schedule_ready(
        app,
        document.id,
        passage.id,
        passage.revision,
        basis=PredictionBasis.BEST_AVAILABLE_FOIARCHIVE_V1,
    )
    gateway = RecordingGateway()
    retrieval = RecordingRetrieval()

    with pytest.raises(RuntimeError, match="prediction basis"):
        AnalysisOrchestrator(
            database=app.state.database,
            model=gateway,
            retrieval=retrieval,
            prediction_basis=PredictionBasis.PAPER_EXACT_V1,
        ).run(job_id)

    assert gateway.query_calls == []
    assert retrieval.calls == []
    assert gateway.prediction_calls == []


@pytest.mark.parametrize(
    "basis",
    [
        PredictionBasis.PAPER_EXACT_V1,
        PredictionBasis.BEST_AVAILABLE_FOIARCHIVE_V1,
    ],
)
def test_retry_after_prediction_timeout_reuses_query_trace_and_basis(
    tmp_path: Path,
    monkeypatch,
    basis: PredictionBasis,
) -> None:
    app, document, passage = create_test_app_and_passage(tmp_path)
    install_fake_preparation(monkeypatch, calls=[])
    _prepare(app, passage.id)
    job_id = _schedule_ready(
        app,
        document.id,
        passage.id,
        passage.revision,
        basis=basis,
    )
    gateway = RecordingGateway(fail_prediction_once=True)
    retrieval = RecordingRetrieval()
    orchestrator = AnalysisOrchestrator(
        database=app.state.database,
        model=gateway,
        retrieval=retrieval,
        prediction_basis=basis,
    )

    try:
        orchestrator.run(job_id)
    except TimeoutError:
        pass
    else:
        raise AssertionError("temporary provider error should be retryable")
    orchestrator.run(job_id)

    assert gateway.query_calls == [f"{job_id}:query"]
    assert retrieval.calls == ["specific actors paris 1960 meeting"]
    assert gateway.prediction_calls == [
        f"{job_id}:prediction",
        f"{job_id}:prediction",
    ]
    assert gateway.prediction_bases == [basis, basis]
    with app.state.database.session_factory() as session:
        job = session.get(AnalysisJob, job_id)
    assert job is not None and job.status == "complete"
    assert job.prediction_payload["prediction_basis"] == basis.value


def test_revision_change_cancels_resumed_work_before_another_external_call(
    tmp_path: Path,
    monkeypatch,
) -> None:
    app, document, passage = create_test_app_and_passage(tmp_path)
    install_fake_preparation(monkeypatch, calls=[])
    _prepare(app, passage.id)
    job_id = _schedule_ready(app, document.id, passage.id, passage.revision)
    gateway = RecordingGateway()
    retrieval = RecordingRetrieval()
    with app.state.database.session_factory() as session, session.begin():
        job = session.get(AnalysisJob, job_id)
        assert job is not None
        job.status = "running"
        job.stage = "retrieval"
        job.query_payload = gateway.generate_query(
            type("Example", (), {"passage_id": passage.id})(),
            f"{job_id}:query",
        ).model_dump(mode="json")
        model = session.get(Passage, str(passage.id))
        assert model is not None
        model.revision += 1

    AnalysisOrchestrator(
        database=app.state.database,
        model=gateway,
        retrieval=retrieval,
    ).run(job_id)

    with app.state.database.session_factory() as session:
        job = session.get(AnalysisJob, job_id)
    assert job is not None
    assert job.status == "cancelled"
    assert job.error_code == "passage_revision_changed"
    assert retrieval.calls == []
    assert gateway.prediction_calls == []


def test_deleted_passage_cancels_queued_work_without_external_calls(
    tmp_path: Path,
    monkeypatch,
) -> None:
    app, document, passage = create_test_app_and_passage(tmp_path)
    install_fake_preparation(monkeypatch, calls=[])
    _prepare(app, passage.id)
    job_id = _schedule_ready(app, document.id, passage.id, passage.revision)
    gateway = RecordingGateway()
    retrieval = RecordingRetrieval()
    with app.state.database.session_factory() as session, session.begin():
        model = session.get(Passage, str(passage.id))
        assert model is not None
        model.status = "deleted"

    AnalysisOrchestrator(
        database=app.state.database,
        model=gateway,
        retrieval=retrieval,
    ).run(job_id)

    with app.state.database.session_factory() as session:
        job = session.get(AnalysisJob, job_id)
    assert job is not None
    assert job.status == "cancelled"
    assert job.error_code == "passage_deleted"
    assert gateway.query_calls == []
    assert retrieval.calls == []


def _prepare(app, passage_id) -> None:
    import asyncio

    response = asyncio.run(
        request(app, "POST", f"/v1/passages/{passage_id}:prepare-context")
    )
    assert response.status_code == 200


def _schedule_ready(
    app,
    document_id,
    passage_id,
    revision: int,
    *,
    basis: PredictionBasis = PredictionBasis.PAPER_EXACT_V1,
) -> str:
    with app.state.database.session_factory() as session, session.begin():
        job = AnalysisJobRepository(session).schedule(
            passage_id=passage_id,
            document_id=document_id,
            passage_revision=revision,
            idempotency_key="orchestrator-test",
            blockers=[],
            prediction_basis=basis,
        )
    return str(job.id)


def _usage(input_tokens: int, output_tokens: int) -> TokenUsage:
    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        cached_input_tokens=0,
        reasoning_tokens=0,
    )
