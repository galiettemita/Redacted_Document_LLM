from __future__ import annotations

import asyncio
from pathlib import Path
from uuid import UUID

from sqlalchemy import func, select

import redacted_backend.api.analysis as analysis_api
from redacted_backend.db.models import AnalysisEvent, AnalysisJob
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
    assert {blocker["component"] for blocker in job["blockers"]} == {
        "corpus",
        "model",
        "worker",
    }
    assert "prediction" not in job
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

    monkeypatch.setattr(analysis_api, "_analysis_blockers", lambda _settings: [])
    app, _document, passage = create_test_app_and_passage(
        tmp_path,
        analysis_dispatcher=RecordingDispatcher(),
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
    assert dispatched == [UUID(response.json()["id"])]
