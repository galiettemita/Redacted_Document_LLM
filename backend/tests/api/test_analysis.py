from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Barrier
from uuid import UUID

from sqlalchemy import event, func, select
from sqlalchemy.orm import Session

import redacted_backend.api.analysis as analysis_api
import redacted_backend.domain.analysis as analysis_domain
from redacted_backend.db.models import AnalysisEvent, AnalysisJob, Document
from redacted_backend.db.repositories.analysis import AnalysisJobRepository
from redacted_backend.domain.readiness import ComponentReadiness
from tests.api.test_context import (
    create_test_app_and_passage,
    install_fake_preparation,
    request,
)


def test_analysis_request_persists_a_truthful_blocked_job_and_is_idempotent(
    tmp_path: Path,
    monkeypatch,
) -> None:
    app, document, passage = create_test_app_and_passage(tmp_path)
    install_fake_preparation(monkeypatch, calls=[])
    assert (
        asyncio.run(
            request(app, "POST", f"/v1/passages/{passage.id}:prepare-context")
        ).status_code
        == 200
    )
    headers = {"Idempotency-Key": "analysis-key-1"}
    body = {"expected_revision": passage.revision}

    first = asyncio.run(
        request(
            app,
            "POST",
            f"/v1/passages/{passage.id}/analysis",
            headers=headers,
            json=body,
        )
    )
    second = asyncio.run(
        request(
            app,
            "POST",
            f"/v1/passages/{passage.id}/analysis",
            headers=headers,
            json=body,
        )
    )

    assert first.status_code == 202
    assert second.status_code == 202
    assert second.json() == first.json()
    job = first.json()
    assert job["status"] == "blocked"
    assert job["stage"] == "readiness"
    assert job["passage_id"] == str(passage.id)
    assert job["document_id"] == str(document.id)
    assert job["passage_revision"] == passage.revision
    assert job["prediction_basis"] == "model_only_v1"
    assert {blocker["component"] for blocker in job["blockers"]} == {
        "model",
        "worker",
    }
    assert "prediction" not in job
    with app.state.database.session_factory() as session:
        assert session.scalar(select(func.count()).select_from(AnalysisJob)) == 1
        assert session.scalar(select(func.count()).select_from(AnalysisEvent)) == 1
        event = session.scalar(select(AnalysisEvent))
        assert event is not None
        assert event.payload["prediction_basis"] == "model_only_v1"


def test_analysis_repository_keeps_the_original_basis_on_same_key_replay(
    tmp_path: Path,
) -> None:
    prediction_basis = getattr(analysis_domain, "PredictionBasis", None)
    assert prediction_basis is not None
    app, document, passage = create_test_app_and_passage(tmp_path)

    with app.state.database.session_factory() as session, session.begin():
        repository = AnalysisJobRepository(session)
        first = repository.schedule(
            passage_id=passage.id,
            document_id=document.id,
            passage_revision=passage.revision,
            idempotency_key="stable-basis",
            blockers=[],
            prediction_basis=prediction_basis.MODEL_ONLY_V1,
        )
        replay = repository.schedule(
            passage_id=passage.id,
            document_id=document.id,
            passage_revision=passage.revision,
            idempotency_key="stable-basis",
            blockers=[],
            prediction_basis=prediction_basis.PAPER_EXACT_V1,
        )

    assert first.id == replay.id
    assert first.prediction_basis == prediction_basis.MODEL_ONLY_V1
    assert replay.prediction_basis == prediction_basis.MODEL_ONLY_V1
    assert first.stage == analysis_domain.AnalysisStage.PREDICTION

    with app.state.database.session_factory() as session, session.begin():
        exact = AnalysisJobRepository(session).schedule(
            passage_id=passage.id,
            document_id=document.id,
            passage_revision=passage.revision,
            idempotency_key="exact-basis",
            blockers=[],
            prediction_basis=prediction_basis.PAPER_EXACT_V1,
        )
    assert exact.stage == analysis_domain.AnalysisStage.QUERY


def test_concurrent_same_key_scheduling_returns_the_winning_job_and_basis(
    tmp_path: Path,
) -> None:
    app, document, passage = create_test_app_and_passage(tmp_path)
    barrier = Barrier(2)

    def synchronize_job_inserts(session, _flush_context, _instances) -> None:
        if any(isinstance(model, AnalysisJob) for model in session.new):
            barrier.wait(timeout=5)

    def schedule(basis: analysis_domain.PredictionBasis):
        with app.state.database.session_factory() as session, session.begin():
            return AnalysisJobRepository(session).schedule(
                passage_id=passage.id,
                document_id=document.id,
                passage_revision=passage.revision,
                idempotency_key="concurrent-stable-basis",
                blockers=[],
                prediction_basis=basis,
            )

    event.listen(Session, "before_flush", synchronize_job_inserts)
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(schedule, analysis_domain.PredictionBasis.MODEL_ONLY_V1),
                executor.submit(schedule, analysis_domain.PredictionBasis.PAPER_EXACT_V1),
            ]
            records = [future.result(timeout=10) for future in futures]
    finally:
        event.remove(Session, "before_flush", synchronize_job_inserts)

    assert records[0].id == records[1].id
    assert records[0].prediction_basis == records[1].prediction_basis
    assert records[0].stage == records[1].stage
    with app.state.database.session_factory() as session:
        assert session.scalar(select(func.count()).select_from(AnalysisJob)) == 1
        assert session.scalar(select(func.count()).select_from(AnalysisEvent)) == 1


def test_analysis_request_requires_the_exact_prepared_passage_revision(
    tmp_path: Path,
    monkeypatch,
) -> None:
    app, _document, passage = create_test_app_and_passage(tmp_path)
    install_fake_preparation(monkeypatch, calls=[])
    assert (
        asyncio.run(
            request(app, "POST", f"/v1/passages/{passage.id}:prepare-context")
        ).status_code
        == 200
    )

    response = asyncio.run(
        request(
            app,
            "POST",
            f"/v1/passages/{passage.id}/analysis",
            headers={"Idempotency-Key": "wrong-revision"},
            json={"expected_revision": passage.revision + 1},
        )
    )

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "passage_revision_changed"


def test_analysis_request_rejects_a_document_at_its_expiry_deadline(
    tmp_path: Path,
    monkeypatch,
) -> None:
    app, document, passage = create_test_app_and_passage(tmp_path)
    install_fake_preparation(monkeypatch, calls=[])
    assert (
        asyncio.run(
            request(app, "POST", f"/v1/passages/{passage.id}:prepare-context")
        ).status_code
        == 200
    )
    with app.state.database.session_factory() as session, session.begin():
        model = session.get(Document, str(document.id))
        model.expires_at = datetime.now(UTC) - timedelta(seconds=1)

    response = asyncio.run(
        request(
            app,
            "POST",
            f"/v1/passages/{passage.id}/analysis",
            headers={"Idempotency-Key": "expired-document"},
            json={"expected_revision": passage.revision},
        )
    )

    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "passage_not_found"


def test_document_events_are_ordered_sse_and_resume_after_last_event_id(
    tmp_path: Path,
    monkeypatch,
) -> None:
    app, document, passage = create_test_app_and_passage(tmp_path)
    install_fake_preparation(monkeypatch, calls=[])
    assert (
        asyncio.run(
            request(app, "POST", f"/v1/passages/{passage.id}:prepare-context")
        ).status_code
        == 200
    )
    created = asyncio.run(
        request(
            app,
            "POST",
            f"/v1/passages/{passage.id}/analysis",
            headers={"Idempotency-Key": "events-key"},
            json={"expected_revision": passage.revision},
        )
    ).json()

    first = asyncio.run(request(app, "GET", f"/v1/documents/{document.id}/events"))

    assert first.status_code == 200
    assert first.headers["content-type"].startswith("text/event-stream")
    assert "event: passage.status" in first.text
    assert '"job_id":"' + created["id"] + '"' in first.text
    event_id = first.text.split("id: ", 1)[1].split("\n", 1)[0]

    resumed = asyncio.run(
        request(
            app,
            "GET",
            f"/v1/documents/{document.id}/events",
            headers={"Last-Event-ID": event_id},
        )
    )

    assert resumed.status_code == 200
    assert resumed.text == ""


def test_ready_analysis_is_queued_and_dispatched_to_the_durable_worker(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dispatched: list[UUID] = []

    class RecordingDispatcher:
        def dispatch(self, job_id: UUID) -> None:
            dispatched.append(job_id)

    monkeypatch.setattr(
        analysis_api,
        "resolve_worker_readiness",
        lambda _settings: ComponentReadiness(
            ready=True,
            detail="heartbeat current",
        ),
    )
    monkeypatch.setattr(
        analysis_api,
        "resolve_corpus_readiness",
        lambda _settings: ComponentReadiness(
            ready=True,
            detail="corpus and retrieval indexes operational",
        ),
    )
    app, _document, passage = create_test_app_and_passage(
        tmp_path,
        analysis_dispatcher=RecordingDispatcher(),
        settings_overrides={
            "redis_url": "redis://worker.test/0",
            "openai_api_key": "secret",
            "allow_paid_model_calls": True,
            "allow_model_only_calls": True,
            "allow_paper_exact_calls": False,
        },
    )
    install_fake_preparation(monkeypatch, calls=[])
    assert (
        asyncio.run(
            request(app, "POST", f"/v1/passages/{passage.id}:prepare-context")
        ).status_code
        == 200
    )

    response = asyncio.run(
        request(
            app,
            "POST",
            f"/v1/passages/{passage.id}/analysis",
            headers={"Idempotency-Key": "dispatch-key"},
            json={"expected_revision": passage.revision},
        )
    )

    assert response.status_code == 202
    assert response.json()["status"] == "queued"
    assert response.json()["stage"] == "prediction"
    assert response.json()["prediction_basis"] == "model_only_v1"
    assert dispatched == [UUID(response.json()["id"])]


def test_model_only_schedules_when_visible_context_has_no_retrieval_terms(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dispatched: list[UUID] = []

    class RecordingDispatcher:
        def dispatch(self, job_id: UUID) -> None:
            dispatched.append(job_id)

    monkeypatch.setattr(
        analysis_api,
        "resolve_worker_readiness",
        lambda _settings: ComponentReadiness(
            ready=True,
            detail="heartbeat current",
        ),
    )
    app, _document, passage = create_test_app_and_passage(
        tmp_path,
        analysis_dispatcher=RecordingDispatcher(),
        settings_overrides={
            "redis_url": "redis://worker.test/0",
            "openai_api_key": "secret",
            "allow_paid_model_calls": True,
            "allow_model_only_calls": True,
        },
    )
    install_fake_preparation(
        monkeypatch,
        calls=[],
        context_window="the [[TARGET]] of",
    )

    prepared = asyncio.run(
        request(app, "POST", f"/v1/passages/{passage.id}:prepare-context")
    )
    assert prepared.status_code == 200
    assert prepared.json()["context_query_status"] == "unavailable"
    assert prepared.json()["context_query_term_count"] == 0

    scheduled = asyncio.run(
        request(
            app,
            "POST",
            f"/v1/passages/{passage.id}/analysis",
            headers={"Idempotency-Key": "closed-book-without-search-terms"},
            json={"expected_revision": passage.revision},
        )
    )

    assert scheduled.status_code == 202
    assert scheduled.json()["status"] == "queued"
    assert scheduled.json()["stage"] == "prediction"
    assert scheduled.json()["prediction_basis"] == "model_only_v1"
    assert dispatched == [UUID(scheduled.json()["id"])]
