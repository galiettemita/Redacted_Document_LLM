from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Barrier
from uuid import UUID

import pytest
from sqlalchemy import event, func, select
from sqlalchemy.orm import Session

import redacted_backend.api.analysis as analysis_api
import redacted_backend.domain.analysis as analysis_domain
from redacted_backend.db.models import AnalysisEvent, AnalysisJob, Document
from redacted_backend.db.repositories.analysis import AnalysisJobRepository
from redacted_backend.db.repositories.passages import PassageRepository
from redacted_backend.domain.analysis import PredictionBasis
from redacted_backend.domain.passages import PassageRect, PassageSource
from redacted_backend.domain.readiness import ComponentReadiness
from tests.api.test_context import (
    create_test_app_and_passage,
    install_fake_preparation,
    request,
)


class RecordingDispatcher:
    def __init__(self) -> None:
        self.job_ids: list[UUID] = []

    def dispatch(self, job_id: UUID) -> None:
        self.job_ids.append(job_id)


def test_staged_limit_allows_only_first_prepared_passage_and_replay(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dispatcher = RecordingDispatcher()
    app, document, first, second = _staged_app(
        tmp_path,
        monkeypatch,
        dispatcher=dispatcher,
    )

    later_first = _schedule(app, second.id, second.revision, "later-first")
    allowed = _schedule(app, first.id, first.revision, "first-staged")
    replay = _schedule(app, first.id, first.revision, "first-staged")
    later_again = _schedule(app, second.id, second.revision, "later-again")

    assert later_first["prediction_basis"] == (
        PredictionBasis.BEST_AVAILABLE_FOIARCHIVE_BM25_V1.value
    )
    assert later_first["status"] == "blocked"
    assert later_first["blockers"] == [
        {
            "component": "staged_prediction_limit",
            "detail": "This passage is not enabled for staged prediction yet",
        }
    ]
    assert allowed["status"] == "queued"
    assert allowed["stage"] == "query"
    assert replay == allowed
    assert later_again["status"] == "blocked"
    assert dispatcher.job_ids == [UUID(allowed["id"])]
    with app.state.database.session_factory() as session:
        jobs = session.scalars(
            select(AnalysisJob).where(
                AnalysisJob.document_id == str(document.id)
            )
        ).all()
    assert len(jobs) == 3


def test_document_schedule_accepts_the_first_passage_and_defers_the_rest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dispatcher = RecordingDispatcher()
    app, document, first, second = _staged_app(
        tmp_path,
        monkeypatch,
        dispatcher=dispatcher,
    )

    response = asyncio.run(
        request(
            app,
            "POST",
            f"/v1/documents/{document.id}/analysis",
            headers={"Idempotency-Key": "document-run-one"},
        )
    )

    assert response.status_code == 202
    result = response.json()
    assert result["document_id"] == str(document.id)
    assert [job["passage_id"] for job in result["accepted_jobs"]] == [
        str(first.id)
    ]
    assert result["accepted_jobs"][0]["status"] == "queued"
    assert result["deferred_passage_ids"] == [str(second.id)]
    assert result["blockers"] == []
    assert dispatcher.job_ids == [UUID(result["accepted_jobs"][0]["id"])]
    with app.state.database.session_factory() as session:
        jobs = session.scalars(
            select(AnalysisJob).where(
                AnalysisJob.document_id == str(document.id)
            )
        ).all()
    assert len(jobs) == 1
    assert jobs[0].status == "queued"


def test_document_schedule_replays_the_same_job_with_safe_redispatch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dispatcher = RecordingDispatcher()
    app, document, _first, _second = _staged_app(
        tmp_path,
        monkeypatch,
        dispatcher=dispatcher,
    )
    headers = {"Idempotency-Key": "stable-document-run"}

    first = asyncio.run(
        request(app, "POST", f"/v1/documents/{document.id}/analysis", headers=headers)
    )
    replay = asyncio.run(
        request(app, "POST", f"/v1/documents/{document.id}/analysis", headers=headers)
    )

    assert first.status_code == replay.status_code == 202
    assert replay.json() == first.json()
    assert len(dispatcher.job_ids) == 2
    assert dispatcher.job_ids[0] == dispatcher.job_ids[1]


def test_document_schedule_retry_redispatches_a_committed_queued_job(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class FlakyDispatcher:
        def __init__(self) -> None:
            self.job_ids: list[UUID] = []

        def dispatch(self, job_id: UUID) -> None:
            self.job_ids.append(job_id)
            if len(self.job_ids) == 1:
                raise RuntimeError("broker unavailable")

    dispatcher = FlakyDispatcher()
    app, document, _first, _second = _staged_app(
        tmp_path,
        monkeypatch,
        dispatcher=dispatcher,
    )
    headers = {"Idempotency-Key": "retry-committed-document-run"}

    interrupted = asyncio.run(
        request(app, "POST", f"/v1/documents/{document.id}/analysis", headers=headers)
    )
    retried = asyncio.run(
        request(app, "POST", f"/v1/documents/{document.id}/analysis", headers=headers)
    )

    assert interrupted.status_code == 503
    assert interrupted.json()["detail"]["code"] == "worker_dispatch_failed"
    assert retried.status_code == 202
    assert len(dispatcher.job_ids) == 2
    assert dispatcher.job_ids[0] == dispatcher.job_ids[1]
    with app.state.database.session_factory() as session:
        assert session.scalar(select(func.count()).select_from(AnalysisJob)) == 1


def test_document_schedule_hashes_a_maximum_length_key_for_storage(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dispatcher = RecordingDispatcher()
    app, document, _first, _second = _staged_app(
        tmp_path,
        monkeypatch,
        dispatcher=dispatcher,
    )

    response = asyncio.run(
        request(
            app,
            "POST",
            f"/v1/documents/{document.id}/analysis",
            headers={"Idempotency-Key": "x" * 512},
        )
    )

    assert response.status_code == 202
    with app.state.database.session_factory() as session:
        job = session.scalar(select(AnalysisJob))
        assert job is not None
        assert len(job.idempotency_key) <= 512
        assert job.idempotency_key != "x" * 512


def test_document_analysis_state_recovers_only_current_nonfailed_jobs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dispatcher = RecordingDispatcher()
    app, document, first, second = _staged_app(
        tmp_path,
        monkeypatch,
        dispatcher=dispatcher,
        limit=2,
    )
    first_job = _schedule(app, first.id, first.revision, "recover-first")
    second_job = _schedule(app, second.id, second.revision, "recover-second")
    with app.state.database.session_factory() as session, session.begin():
        first_model = session.get(AnalysisJob, first_job["id"])
        assert first_model is not None
        first_model.status = "complete"
        first_model.stage = "complete"
        AnalysisJobRepository(session).fail(
            second_job["id"], code="test_failure", detail="test"
        )

    response = asyncio.run(
        request(app, "GET", f"/v1/documents/{document.id}/analysis")
    )

    assert response.status_code == 200
    assert response.json()["document_id"] == str(document.id)
    assert [job["id"] for job in response.json()["jobs"]] == [first_job["id"]]
    assert response.json()["jobs"][0]["status"] == "complete"


def test_document_schedule_returns_readiness_blockers_without_creating_jobs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    app, document, passage = create_test_app_and_passage(tmp_path)
    install_fake_preparation(monkeypatch, calls=[])
    assert asyncio.run(
        request(app, "POST", f"/v1/passages/{passage.id}:prepare-context")
    ).status_code == 200

    response = asyncio.run(
        request(
            app,
            "POST",
            f"/v1/documents/{document.id}/analysis",
            headers={"Idempotency-Key": "unavailable-document-run"},
        )
    )

    assert response.status_code == 202
    result = response.json()
    assert result["accepted_jobs"] == []
    assert result["deferred_passage_ids"] == [str(passage.id)]
    assert {blocker["component"] for blocker in result["blockers"]} == {
        "model",
        "worker",
    }
    with app.state.database.session_factory() as session:
        assert session.scalar(select(func.count()).select_from(AnalysisJob)) == 0


@pytest.mark.parametrize("terminal_status", ["failed", "cancelled"])
def test_failed_or_cancelled_staged_job_releases_the_document_limit(
    tmp_path: Path,
    monkeypatch,
    terminal_status: str,
) -> None:
    dispatcher = RecordingDispatcher()
    app, _document, first, _second = _staged_app(
        tmp_path,
        monkeypatch,
        dispatcher=dispatcher,
    )
    initial = _schedule(app, first.id, first.revision, "initial-staged")
    with app.state.database.session_factory() as session, session.begin():
        repository = AnalysisJobRepository(session)
        if terminal_status == "failed":
            repository.fail(initial["id"], code="test_failure", detail="test")
        else:
            repository.cancel(initial["id"], code="test_cancel", detail="test")

    retry = _schedule(app, first.id, first.revision, f"retry-{terminal_status}")

    assert retry["status"] == "queued"
    assert dispatcher.job_ids == [UUID(initial["id"]), UUID(retry["id"])]


def test_raising_staged_limit_to_two_enables_the_second_passage(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dispatcher = RecordingDispatcher()
    app, _document, first, second = _staged_app(
        tmp_path,
        monkeypatch,
        dispatcher=dispatcher,
        limit=2,
    )

    first_job = _schedule(app, first.id, first.revision, "limit-two-first")
    second_job = _schedule(app, second.id, second.revision, "limit-two-second")

    assert first_job["status"] == second_job["status"] == "queued"
    assert dispatcher.job_ids == [UUID(first_job["id"]), UUID(second_job["id"])]


def test_concurrent_staged_requests_cannot_bypass_the_document_limit(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dispatcher = RecordingDispatcher()
    app, _document, first, _second = _staged_app(
        tmp_path,
        monkeypatch,
        dispatcher=dispatcher,
    )
    barrier = Barrier(2)

    def schedule(key: str) -> dict[str, object]:
        barrier.wait(timeout=5)
        return _schedule(app, first.id, first.revision, key)

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(
            executor.map(schedule, ("concurrent-one", "concurrent-two"))
        )

    assert sorted(result["status"] for result in results) == ["blocked", "queued"]
    assert len(dispatcher.job_ids) == 1


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


def _staged_app(
    tmp_path: Path,
    monkeypatch,
    *,
    dispatcher: RecordingDispatcher,
    limit: int = 1,
):
    ready = ComponentReadiness(ready=True, detail="ready")
    unavailable = ComponentReadiness(ready=False, detail="not configured")
    monkeypatch.setattr(analysis_api, "resolve_worker_readiness", lambda _s: ready)
    monkeypatch.setattr(analysis_api, "resolve_corpus_readiness", lambda _s: unavailable)
    monkeypatch.setattr(
        analysis_api,
        "resolve_best_available_corpus_readiness",
        lambda _s: unavailable,
    )
    monkeypatch.setattr(
        analysis_api,
        "resolve_staged_bm25_corpus_readiness",
        lambda _s: ready,
        raising=False,
    )
    app, document, first = create_test_app_and_passage(
        tmp_path,
        analysis_dispatcher=dispatcher,
        settings_overrides={
            "redis_url": "redis://worker.test/0",
            "openai_api_key": "secret",
            "allow_paid_model_calls": True,
            "allow_model_only_calls": False,
            "allow_staged_bm25_calls": True,
            "allow_best_available_calls": False,
            "allow_paper_exact_calls": False,
            "staged_bm25_spend_cap_usd": "10.00",
            "bm25_stage_max_predictions_per_document": limit,
        },
    )
    with app.state.database.session_factory() as session, session.begin():
        second = PassageRepository(session).create(
            document_id=document.id,
            label="Passage 2",
            page_number=1,
            rects=[PassageRect(x=0.5, y=0.5, width=0.2, height=0.05)],
            source=PassageSource.MANUAL,
        )
    install_fake_preparation(monkeypatch, calls=[])
    for passage in (first, second):
        prepared = asyncio.run(
            request(app, "POST", f"/v1/passages/{passage.id}:prepare-context")
        )
        assert prepared.status_code == 200
    return app, document, first, second


def _schedule(
    app,
    passage_id,
    revision: int,
    key: str,
) -> dict[str, object]:
    response = asyncio.run(
        request(
            app,
            "POST",
            f"/v1/passages/{passage_id}/analysis",
            headers={"Idempotency-Key": key},
            json={"expected_revision": revision},
        )
    )
    assert response.status_code == 202
    return response.json()
