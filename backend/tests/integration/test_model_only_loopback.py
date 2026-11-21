from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from uuid import UUID

from sqlalchemy import select

import redacted_backend.api.analysis as analysis_api
import redacted_backend.api.passages as passages_api
from redacted_backend.config import Settings
from redacted_backend.db.base import Base
from redacted_backend.db.models import AnalysisJob
from redacted_backend.documents.detection import DetectedPassage, DetectionResult
from redacted_backend.domain.passages import PassageRect
from redacted_backend.domain.readiness import ComponentReadiness
from redacted_backend.jobs.generation_store import AnalysisGenerationStore
from redacted_backend.jobs.orchestrator import AnalysisOrchestrator
from redacted_backend.main import create_app
from redacted_backend.models.openai_gateway import (
    OpenAIGateway,
    ProviderHttpResponse,
)
from tests.api.test_context import install_fake_preparation, request
from tests.pdf_fixtures import make_pdf


class RecordingDispatcher:
    def __init__(self) -> None:
        self.job_ids: list[UUID] = []

    def dispatch(self, job_id: UUID) -> None:
        self.job_ids.append(job_id)


class FakeProviderTransport:
    def __init__(self) -> None:
        self.count_requests: list[dict[str, object]] = []
        self.create_requests: list[tuple[dict[str, object], str]] = []

    def count_input_tokens(self, request: dict[str, object]) -> int:
        self.count_requests.append(request)
        return 321

    def create_response(
        self,
        request: dict[str, object],
        *,
        client_request_id: str,
    ) -> ProviderHttpResponse:
        self.create_requests.append((request, client_request_id))
        return ProviderHttpResponse(
            body=_fake_response("The missing passage names the responsible officials."),
            request_id="fake-request-1",
        )


class RecordingGateway(OpenAIGateway):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.query_calls = 0
        self.retrieval_prediction_calls = 0

    def generate_query(self, *args: Any, **kwargs: Any):
        self.query_calls += 1
        return super().generate_query(*args, **kwargs)

    def generate_prediction(self, *args: Any, **kwargs: Any):
        self.retrieval_prediction_calls += 1
        return super().generate_prediction(*args, **kwargs)


class RecordingRetrieval:
    def __init__(self) -> None:
        self.calls = 0

    def retrieve_realistic(self, *args: Any, **kwargs: Any):
        self.calls += 1
        raise AssertionError("model-only execution must not retrieve")


def test_upload_to_model_only_sse_uses_only_the_injected_fake_transport(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dispatcher = RecordingDispatcher()
    monkeypatch.setattr(
        analysis_api,
        "resolve_worker_readiness",
        lambda _settings: ComponentReadiness(
            ready=True,
            detail="test worker ready",
        ),
    )
    monkeypatch.setattr(
        passages_api,
        "detect_passages",
        lambda _path: DetectionResult(
            passages=[
                DetectedPassage(
                    page_number=1,
                    rects=[PassageRect(x=0.1, y=0.2, width=0.2, height=0.03)],
                    kinds=["black"],
                    source_mask_ids=["mask-1"],
                    confidence=1.0,
                )
            ],
            pages_scanned=1,
            coverage_confident=True,
            uncertain_pages=[],
            warnings=[],
        ),
    )
    install_fake_preparation(monkeypatch, calls=[])
    app = create_app(
        Settings(
            database_url=f"sqlite+pysqlite:///{tmp_path / 'loopback.sqlite3'}",
            document_storage_root=tmp_path / "stored-documents",
            redis_url="redis://worker.test/0",
            openai_api_key="injected-fake-key",
            allow_paid_model_calls=True,
            allow_model_only_calls=True,
        ),
        analysis_dispatcher=dispatcher,
    )
    Base.metadata.create_all(app.state.database.engine)

    uploaded = asyncio.run(
        request(
            app,
            "POST",
            "/v1/documents",
            files={
                "file": (
                    "visible-bulletin.pdf",
                    make_pdf(text="Visible bulletin dated 1945-05-08"),
                    "application/pdf",
                )
            },
        )
    )
    assert uploaded.status_code == 201
    document_id = uploaded.json()["id"]

    detected = asyncio.run(
        request(app, "POST", f"/v1/documents/{document_id}/passages:detect")
    )
    assert detected.status_code == 200
    passage = detected.json()["passages"][0]

    context = asyncio.run(
        request(app, "POST", f"/v1/passages/{passage['id']}:prepare-context")
    )
    assert context.status_code == 200

    scheduled = asyncio.run(
        request(
            app,
            "POST",
            f"/v1/passages/{passage['id']}/analysis",
            headers={"Idempotency-Key": "loopback-model-only"},
            json={"expected_revision": passage["revision"]},
        )
    )
    assert scheduled.status_code == 202
    job = scheduled.json()
    assert job["status"] == "queued"
    assert job["stage"] == "prediction"
    assert job["prediction_basis"] == "model_only_v1"
    assert dispatcher.job_ids == [UUID(job["id"])]

    fake_transport = FakeProviderTransport()
    gateway = RecordingGateway(
        transport=fake_transport,
        store=AnalysisGenerationStore(app.state.database),
    )
    retrieval = RecordingRetrieval()
    AnalysisOrchestrator(
        database=app.state.database,
        model=gateway,
        retrieval=retrieval,
    ).run(job["id"])

    events = asyncio.run(
        request(app, "GET", f"/v1/documents/{document_id}/events")
    )
    assert events.status_code == 200
    assert "event: passage.prediction" in events.text
    assert '"prediction_basis":"model_only_v1"' in events.text
    assert "The missing passage names the responsible officials." in events.text

    with app.state.database.session_factory() as session:
        stored = session.scalar(select(AnalysisJob).where(AnalysisJob.id == job["id"]))
    assert stored is not None
    assert stored.status == "complete"
    assert stored.query_payload is None
    assert stored.retrieval_trace is None
    assert stored.prediction_payload["prediction_basis"] == "model_only_v1"
    assert stored.prediction_payload["retrieval_trace_sha256"] is None
    assert gateway.query_calls == 0
    assert gateway.retrieval_prediction_calls == 0
    assert retrieval.calls == 0
    assert len(fake_transport.count_requests) == 1
    assert len(fake_transport.create_requests) == 1


def _fake_response(text: str) -> dict[str, Any]:
    return {
        "id": "fake-response-1",
        "status": "completed",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text}],
            }
        ],
        "usage": {
            "input_tokens": 321,
            "output_tokens": 40,
            "total_tokens": 361,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 12},
        },
    }
