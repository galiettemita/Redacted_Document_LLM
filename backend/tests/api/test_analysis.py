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
from redacted_backend.db.models import (
    AnalysisEvent,
    AnalysisJob,
    Document,
    VisibleExampleArtifact,
)
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


def test_selected_staged_limit_allows_one_active_passage_at_a_time(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dispatcher = RecordingDispatcher()
    app, document, first, second = _staged_app(
        tmp_path,
        monkeypatch,
        dispatcher=dispatcher,
    )

    selected = _schedule(app, second.id, second.revision, "selected-second")
    replay = _schedule(app, second.id, second.revision, "selected-second")
    blocked_while_active = _schedule(
        app, first.id, first.revision, "first-while-second-active"
    )

    assert selected["prediction_basis"] == (
        PredictionBasis.BEST_AVAILABLE_FOIARCHIVE_BM25_V1.value
    )
    assert selected["status"] == "queued"
    assert selected["stage"] == "query"
    assert replay == selected
    assert blocked_while_active["status"] == "blocked"
    assert blocked_while_active["blockers"] == [
        {
            "component": "staged_prediction_limit",
            "detail": "This passage is not enabled for staged prediction yet",
        }
    ]
    with app.state.database.session_factory() as session, session.begin():
        completed = session.get(AnalysisJob, selected["id"])
        assert completed is not None
        completed.status = "complete"
        completed.stage = "complete"

    next_selected = _schedule(app, first.id, first.revision, "selected-first-next")

    assert next_selected["status"] == "queued"
    assert dispatcher.job_ids == [
        UUID(selected["id"]),
        UUID(selected["id"]),
        UUID(next_selected["id"]),
    ]
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


def test_document_schedule_reuses_a_completed_job_without_dispatch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dispatcher = RecordingDispatcher()
    app, document, first, second = _staged_app(
        tmp_path,
        monkeypatch,
        dispatcher=dispatcher,
    )
    completed = _schedule(app, first.id, first.revision, "completed-first")
    with app.state.database.session_factory() as session, session.begin():
        model = session.get(AnalysisJob, completed["id"])
        assert model is not None
        model.status = "complete"
        model.stage = "complete"

    response = asyncio.run(
        request(
            app,
            "POST",
            f"/v1/documents/{document.id}/analysis",
            headers={"Idempotency-Key": "after-completion"},
        )
    )

    assert response.status_code == 202
    result = response.json()
    assert [job["id"] for job in result["accepted_jobs"]] == [completed["id"]]
    assert result["accepted_jobs"][0]["status"] == "complete"
    assert result["deferred_passage_ids"] == [str(second.id)]
    assert dispatcher.job_ids == [UUID(completed["id"])]
    with app.state.database.session_factory() as session:
        assert session.scalar(select(func.count()).select_from(AnalysisJob)) == 1


def test_full_document_schedule_reuses_completed_and_queues_every_remaining_passage(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dispatcher = RecordingDispatcher()
    app, document, first, second = _full_app(
        tmp_path,
        monkeypatch,
        dispatcher=dispatcher,
    )
    completed = _schedule(app, first.id, first.revision, "completed-full-first")
    with app.state.database.session_factory() as session, session.begin():
        model = session.get(AnalysisJob, completed["id"])
        assert model is not None
        model.status = "complete"
        model.stage = "complete"

    response = asyncio.run(
        request(
            app,
            "POST",
            f"/v1/documents/{document.id}/analysis",
            headers={"Idempotency-Key": "full-document-run"},
        )
    )

    assert response.status_code == 202
    result = response.json()
    assert [job["passage_id"] for job in result["accepted_jobs"]] == [
        str(first.id),
        str(second.id),
    ]
    assert [job["status"] for job in result["accepted_jobs"]] == [
        "complete",
        "queued",
    ]
    assert {job["prediction_basis"] for job in result["accepted_jobs"]} == {
        PredictionBasis.BEST_AVAILABLE_FOIARCHIVE_V1.value,
    }
    assert result["deferred_passage_ids"] == []
    assert dispatcher.job_ids == [
        UUID(completed["id"]),
        UUID(result["accepted_jobs"][1]["id"]),
    ]


def test_full_document_schedule_replaces_staged_results_with_one_full_basis(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dispatcher = RecordingDispatcher()
    app, document, first, second = _full_app(
        tmp_path,
        monkeypatch,
        dispatcher=dispatcher,
    )
    with app.state.database.session_factory() as session, session.begin():
        staged = AnalysisJobRepository(session).schedule(
            passage_id=first.id,
            document_id=document.id,
            passage_revision=first.revision,
            idempotency_key="prior-staged-result",
            blockers=[],
            prediction_basis=PredictionBasis.BEST_AVAILABLE_FOIARCHIVE_BM25_V1,
        )
        model = session.get(AnalysisJob, str(staged.id))
        assert model is not None
        model.status = "complete"
        model.stage = "complete"

    response = asyncio.run(
        request(
            app,
            "POST",
            f"/v1/documents/{document.id}/analysis",
            headers={"Idempotency-Key": "upgrade-to-full-basis"},
        )
    )

    assert response.status_code == 202
    result = response.json()
    assert [job["passage_id"] for job in result["accepted_jobs"]] == [
        str(first.id),
        str(second.id),
    ]
    assert {job["prediction_basis"] for job in result["accepted_jobs"]} == {
        PredictionBasis.BEST_AVAILABLE_FOIARCHIVE_V1.value,
    }
    assert [job["status"] for job in result["accepted_jobs"]] == [
        "queued",
        "queued",
    ]
    assert result["deferred_passage_ids"] == []
    assert len(dispatcher.job_ids) == 2
    with app.state.database.session_factory() as session:
        assert session.scalar(select(func.count()).select_from(AnalysisJob)) == 3


def test_full_document_schedule_is_atomic_when_any_context_query_is_unavailable(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dispatcher = RecordingDispatcher()
    app, document, first, second = _full_app(
        tmp_path,
        monkeypatch,
        dispatcher=dispatcher,
    )
    with app.state.database.session_factory() as session, session.begin():
        artifact = session.scalar(
            select(VisibleExampleArtifact).where(
                VisibleExampleArtifact.passage_id == str(second.id),
                VisibleExampleArtifact.passage_revision == second.revision,
                VisibleExampleArtifact.invalidated_at.is_(None),
            )
        )
        assert artifact is not None
        artifact.context_query = None
        artifact.context_query_sha256 = None
        artifact.context_query_term_count = None
        artifact.context_query_version = None
        artifact.context_stopwords_version = None

    response = asyncio.run(
        request(
            app,
            "POST",
            f"/v1/documents/{document.id}/analysis",
            headers={"Idempotency-Key": "atomic-full-document-run"},
            json={
                "expected_prediction_basis": (
                    PredictionBasis.BEST_AVAILABLE_FOIARCHIVE_V1.value
                )
            },
        )
    )

    assert response.status_code == 409
    assert response.json()["detail"] == {
        "code": "full_document_context_incomplete",
        "message": "Every passage needs a visible context query before full-document prediction",
        "passage_ids": [str(second.id)],
    }
    assert dispatcher.job_ids == []
    with app.state.database.session_factory() as session:
        assert session.scalar(select(func.count()).select_from(AnalysisJob)) == 0
        assert PassageRepository(session).get_live(first.id) is not None


def test_selected_request_reuses_an_existing_current_full_basis_job(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dispatcher = RecordingDispatcher()
    app, document, first, _second = _full_app(
        tmp_path,
        monkeypatch,
        dispatcher=dispatcher,
    )
    bulk = asyncio.run(
        request(
            app,
            "POST",
            f"/v1/documents/{document.id}/analysis",
            headers={"Idempotency-Key": "full-document-before-stale-tab"},
        )
    )
    assert bulk.status_code == 202
    first_job = next(
        job
        for job in bulk.json()["accepted_jobs"]
        if job["passage_id"] == str(first.id)
    )

    selected = asyncio.run(
        request(
            app,
            "POST",
            f"/v1/passages/{first.id}/analysis",
            headers={"Idempotency-Key": "stale-selected-tab"},
            json={"expected_revision": first.revision},
        )
    )

    assert selected.status_code == 202
    assert selected.json() == first_job
    assert dispatcher.job_ids == [
        UUID(job["id"])
        for job in bulk.json()["accepted_jobs"]
    ] + [UUID(first_job["id"])]
    with app.state.database.session_factory() as session:
        assert session.scalar(select(func.count()).select_from(AnalysisJob)) == 2


def test_expected_full_basis_rejects_a_readiness_downgrade_before_dispatch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dispatcher = RecordingDispatcher()
    app, document, _first, _second = _staged_app(
        tmp_path,
        monkeypatch,
        dispatcher=dispatcher,
        limit=2,
    )

    response = asyncio.run(
        request(
            app,
            "POST",
            f"/v1/documents/{document.id}/analysis",
            headers={"Idempotency-Key": "full-run-after-readiness-downgrade"},
            json={
                "expected_prediction_basis": (
                    PredictionBasis.BEST_AVAILABLE_FOIARCHIVE_V1.value
                )
            },
        )
    )

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "prediction_basis_changed"
    assert dispatcher.job_ids == []
    with app.state.database.session_factory() as session:
        assert session.scalar(select(func.count()).select_from(AnalysisJob)) == 0


def test_full_document_schedule_resolves_live_readiness_once_for_all_passages(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dispatcher = RecordingDispatcher()
    app, document, _first, _second = _full_app(
        tmp_path,
        monkeypatch,
        dispatcher=dispatcher,
    )
    calls = {"worker": 0, "corpus": 0, "staged": 0, "full": 0}
    ready = ComponentReadiness(ready=True, detail="ready")
    unavailable = ComponentReadiness(ready=False, detail="not configured")

    def readiness(name: str, result: ComponentReadiness):
        def resolve(_settings):
            calls[name] += 1
            return result

        return resolve

    monkeypatch.setattr(
        analysis_api,
        "resolve_worker_readiness",
        readiness("worker", ready),
    )
    monkeypatch.setattr(
        analysis_api,
        "resolve_corpus_readiness",
        readiness("corpus", unavailable),
    )
    monkeypatch.setattr(
        analysis_api,
        "resolve_staged_bm25_corpus_readiness",
        readiness("staged", unavailable),
    )
    monkeypatch.setattr(
        analysis_api,
        "resolve_best_available_corpus_readiness",
        readiness("full", ready),
    )

    response = asyncio.run(
        request(
            app,
            "POST",
            f"/v1/documents/{document.id}/analysis",
            headers={"Idempotency-Key": "one-readiness-snapshot"},
            json={
                "expected_prediction_basis": (
                    PredictionBasis.BEST_AVAILABLE_FOIARCHIVE_V1.value
                )
            },
        )
    )

    assert response.status_code == 202
    assert calls == {"worker": 1, "corpus": 1, "staged": 1, "full": 1}


def test_full_document_schedule_reuses_matching_basis_behind_newer_staged_result(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dispatcher = RecordingDispatcher()
    app, document, first, second = _full_app(
        tmp_path,
        monkeypatch,
        dispatcher=dispatcher,
    )
    with app.state.database.session_factory() as session, session.begin():
        repository = AnalysisJobRepository(session)
        full = repository.schedule(
            passage_id=first.id,
            document_id=document.id,
            passage_revision=first.revision,
            idempotency_key="older-full-result",
            blockers=[],
            prediction_basis=PredictionBasis.BEST_AVAILABLE_FOIARCHIVE_V1,
        )
        full_model = session.get(AnalysisJob, str(full.id))
        assert full_model is not None
        full_model.status = "complete"
        full_model.stage = "complete"
        full_model.updated_at = datetime.now(UTC) - timedelta(seconds=1)
        staged = repository.schedule(
            passage_id=first.id,
            document_id=document.id,
            passage_revision=first.revision,
            idempotency_key="newer-staged-result",
            blockers=[],
            prediction_basis=PredictionBasis.BEST_AVAILABLE_FOIARCHIVE_BM25_V1,
        )
        staged_model = session.get(AnalysisJob, str(staged.id))
        assert staged_model is not None
        staged_model.status = "complete"
        staged_model.stage = "complete"
        staged_model.updated_at = datetime.now(UTC)

    response = asyncio.run(
        request(
            app,
            "POST",
            f"/v1/documents/{document.id}/analysis",
            headers={"Idempotency-Key": "reuse-hidden-full-result"},
            json={
                "expected_prediction_basis": (
                    PredictionBasis.BEST_AVAILABLE_FOIARCHIVE_V1.value
                )
            },
        )
    )

    assert response.status_code == 202
    jobs = response.json()["accepted_jobs"]
    assert [job["id"] for job in jobs if job["passage_id"] == str(first.id)] == [
        str(full.id)
    ]
    assert [job["passage_id"] for job in jobs] == [str(first.id), str(second.id)]
    with app.state.database.session_factory() as session:
        assert session.scalar(select(func.count()).select_from(AnalysisJob)) == 3


def test_selected_exact_key_replay_precedes_cross_key_full_basis_reuse(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dispatcher = RecordingDispatcher()
    app, document, first, _second = _full_app(
        tmp_path,
        monkeypatch,
        dispatcher=dispatcher,
    )
    with app.state.database.session_factory() as session, session.begin():
        repository = AnalysisJobRepository(session)
        staged = repository.schedule(
            passage_id=first.id,
            document_id=document.id,
            passage_revision=first.revision,
            idempotency_key="historical-selected-key",
            blockers=[],
            prediction_basis=PredictionBasis.BEST_AVAILABLE_FOIARCHIVE_BM25_V1,
        )
        staged_model = session.get(AnalysisJob, str(staged.id))
        assert staged_model is not None
        staged_model.status = "complete"
        staged_model.stage = "complete"
        full = repository.schedule(
            passage_id=first.id,
            document_id=document.id,
            passage_revision=first.revision,
            idempotency_key="new-full-key",
            blockers=[],
            prediction_basis=PredictionBasis.BEST_AVAILABLE_FOIARCHIVE_V1,
        )
        full_model = session.get(AnalysisJob, str(full.id))
        assert full_model is not None
        full_model.status = "complete"
        full_model.stage = "complete"

    response = asyncio.run(
        request(
            app,
            "POST",
            f"/v1/passages/{first.id}/analysis",
            headers={"Idempotency-Key": "historical-selected-key"},
            json={"expected_revision": first.revision},
        )
    )

    assert response.status_code == 202
    assert response.json()["id"] == str(staged.id)
    assert response.json()["prediction_basis"] == (
        PredictionBasis.BEST_AVAILABLE_FOIARCHIVE_BM25_V1.value
    )
    with app.state.database.session_factory() as session:
        assert session.scalar(select(func.count()).select_from(AnalysisJob)) == 2


def test_selected_retry_redelivers_a_recovered_queued_full_job(
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
    app, _document, first, _second = _full_app(
        tmp_path,
        monkeypatch,
        dispatcher=dispatcher,
    )

    failed = asyncio.run(
        request(
            app,
            "POST",
            f"/v1/passages/{first.id}/analysis",
            headers={"Idempotency-Key": "failed-full-dispatch"},
            json={"expected_revision": first.revision},
        )
    )
    retried = asyncio.run(
        request(
            app,
            "POST",
            f"/v1/passages/{first.id}/analysis",
            headers={"Idempotency-Key": "fresh-key-after-dispatch-failure"},
            json={"expected_revision": first.revision},
        )
    )

    assert failed.status_code == 503
    assert retried.status_code == 202
    assert dispatcher.job_ids == [
        UUID(retried.json()["id"]),
        UUID(retried.json()["id"]),
    ]
    with app.state.database.session_factory() as session:
        assert session.scalar(select(func.count()).select_from(AnalysisJob)) == 1


def test_full_document_schedule_does_not_duplicate_an_active_staged_job(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dispatcher = RecordingDispatcher()
    app, document, first, second = _full_app(
        tmp_path,
        monkeypatch,
        dispatcher=dispatcher,
    )
    with app.state.database.session_factory() as session, session.begin():
        staged = AnalysisJobRepository(session).schedule(
            passage_id=first.id,
            document_id=document.id,
            passage_revision=first.revision,
            idempotency_key="active-staged-result",
            blockers=[],
            prediction_basis=PredictionBasis.BEST_AVAILABLE_FOIARCHIVE_BM25_V1,
        )

    response = asyncio.run(
        request(
            app,
            "POST",
            f"/v1/documents/{document.id}/analysis",
            headers={"Idempotency-Key": "wait-for-active-staged"},
        )
    )

    assert response.status_code == 202
    result = response.json()
    assert [job["id"] for job in result["accepted_jobs"]] == [str(staged.id)]
    assert result["accepted_jobs"][0]["prediction_basis"] == (
        PredictionBasis.BEST_AVAILABLE_FOIARCHIVE_BM25_V1.value
    )
    assert result["deferred_passage_ids"] == [str(second.id)]
    assert dispatcher.job_ids == [UUID(str(staged.id))]
    with app.state.database.session_factory() as session:
        assert session.scalar(select(func.count()).select_from(AnalysisJob)) == 1


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


def _full_app(
    tmp_path: Path,
    monkeypatch,
    *,
    dispatcher: RecordingDispatcher,
):
    ready = ComponentReadiness(ready=True, detail="ready")
    unavailable = ComponentReadiness(ready=False, detail="not configured")
    monkeypatch.setattr(analysis_api, "resolve_worker_readiness", lambda _s: ready)
    monkeypatch.setattr(analysis_api, "resolve_corpus_readiness", lambda _s: unavailable)
    monkeypatch.setattr(
        analysis_api,
        "resolve_staged_bm25_corpus_readiness",
        lambda _s: unavailable,
    )
    monkeypatch.setattr(
        analysis_api,
        "resolve_best_available_corpus_readiness",
        lambda _s: ready,
    )
    app, document, first = create_test_app_and_passage(
        tmp_path,
        analysis_dispatcher=dispatcher,
        settings_overrides={
            "redis_url": "redis://worker.test/0",
            "openai_api_key": "secret",
            "allow_paid_model_calls": True,
            "allow_model_only_calls": False,
            "allow_staged_bm25_calls": False,
            "allow_best_available_calls": True,
            "allow_paper_exact_calls": False,
            "provider_spend_cap_usd": "10.00",
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
