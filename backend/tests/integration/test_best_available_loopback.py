from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from typing import Any, Literal
from uuid import UUID

from sqlalchemy import select

import redacted_backend.api.analysis as analysis_api
import redacted_backend.api.passages as passages_api
from redacted_backend.config import Settings
from redacted_backend.db.base import Base
from redacted_backend.db.models import AnalysisJob
from redacted_backend.documents.detection import DetectedPassage, DetectionResult
from redacted_backend.domain.analysis import PredictionBasis
from redacted_backend.domain.passages import PassageRect
from redacted_backend.domain.readiness import ComponentReadiness
from redacted_backend.jobs.generation_store import AnalysisGenerationStore
from redacted_backend.jobs.orchestrator import AnalysisOrchestrator
from redacted_backend.main import create_app
from redacted_backend.models.openai_gateway import (
    OpenAIGateway,
    ProviderHttpResponse,
)
from redacted_backend.retrieval.service import RealisticRetrievalService
from redacted_backend.retrieval.types import RankedChunk
from tests.api.test_context import install_fake_preparation, request
from tests.pdf_fixtures import make_pdf


class RecordingDispatcher:
    def __init__(self) -> None:
        self.job_ids: list[UUID] = []

    def dispatch(self, job_id: UUID) -> None:
        self.job_ids.append(job_id)


class FakeTwoStageProviderTransport:
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
        number = len(self.create_requests)
        output = (
            "specific actors paris 1960 meeting"
            if number == 1
            else "The missing passage identifies the responsible officials."
        )
        return ProviderHttpResponse(
            body=_fake_response(output, number),
            request_id=f"fake-request-{number}",
        )


class RecordingRetriever:
    def __init__(self, retriever: Literal["bm25", "dense"]) -> None:
        self.retriever = retriever
        self.calls: list[tuple[str, int, str]] = []

    def search(
        self,
        query: str,
        limit: int,
        *,
        query_kind: Literal["context", "own"],
    ) -> list[RankedChunk]:
        self.calls.append((query, limit, query_kind))
        identity = f"{self.retriever}:{query_kind}"
        return [
            RankedChunk(
                chunk_id=hashlib.sha256(identity.encode("utf-8")).hexdigest(),
                document_id="public-archive-document",
                ordinal=0 if query_kind == "context" else 1,
                text=f"Public evidence from the {identity} ranked list.",
                title="public-archive-document.pdf",
                corpus="historylab-unlabeled",
                source_url="https://example.test/public-archive-document.pdf",
                classification="UNCLASSIFIED",
                authored="1960-01-01",
                doc_lang="en",
                rank=1,
                score=1.0,
                retriever=self.retriever,
                query_kind=query_kind,
                query_sha256=hashlib.sha256(query.encode("utf-8")).hexdigest(),
                index_manifest_hash=("1" if self.retriever == "bm25" else "2")
                * 64,
                corpus_manifest_hash="3" * 64,
            )
        ]


def test_upload_to_best_available_sse_runs_two_stage_four_list_prediction(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dispatcher = RecordingDispatcher()
    ready = ComponentReadiness(ready=True, detail="test ready")
    unavailable = ComponentReadiness(ready=False, detail="test unavailable")
    monkeypatch.setattr(analysis_api, "resolve_worker_readiness", lambda _settings: ready)
    monkeypatch.setattr(
        analysis_api,
        "resolve_best_available_corpus_readiness",
        lambda _settings: ready,
    )
    monkeypatch.setattr(
        analysis_api,
        "resolve_corpus_readiness",
        lambda _settings: unavailable,
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
            allow_model_only_calls=False,
            allow_best_available_calls=True,
            allow_paper_exact_calls=False,
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
            headers={"Idempotency-Key": "loopback-best-available"},
            json={"expected_revision": passage["revision"]},
        )
    )
    assert scheduled.status_code == 202
    job = scheduled.json()
    assert job["status"] == "queued"
    assert job["stage"] == "query"
    assert job["prediction_basis"] == "best_available_foiarchive_v1"
    assert dispatcher.job_ids == [UUID(job["id"])]

    fake_transport = FakeTwoStageProviderTransport()
    bm25 = RecordingRetriever("bm25")
    dense = RecordingRetriever("dense")
    AnalysisOrchestrator(
        database=app.state.database,
        model=OpenAIGateway(
            transport=fake_transport,
            store=AnalysisGenerationStore(app.state.database),
        ),
        retrieval=RealisticRetrievalService(bm25=bm25, dense=dense),
        prediction_basis=PredictionBasis.BEST_AVAILABLE_FOIARCHIVE_V1,
    ).run(job["id"])

    events = asyncio.run(request(app, "GET", f"/v1/documents/{document_id}/events"))
    assert events.status_code == 200
    assert "event: passage.prediction" in events.text
    assert '"prediction_basis":"best_available_foiarchive_v1"' in events.text
    assert "The missing passage identifies the responsible officials." in events.text

    with app.state.database.session_factory() as session:
        stored = session.scalar(select(AnalysisJob).where(AnalysisJob.id == job["id"]))
    assert stored is not None
    assert stored.status == "complete"
    assert stored.query_payload["query"] == "specific actors paris 1960 meeting"
    assert set(stored.retrieval_trace["lists"]) == {
        "context_bm25",
        "context_dense",
        "own_bm25",
        "own_dense",
    }
    assert all(stored.retrieval_trace["lists"].values())
    assert stored.retrieval_trace["documents"]
    assert stored.retrieval_trace["serialized_documents"].startswith("[DOCUMENT 1]")
    assert stored.prediction_payload["prediction_basis"] == (
        "best_available_foiarchive_v1"
    )
    assert stored.prediction_payload["retrieval_trace_sha256"] is not None
    assert [call[2] for call in bm25.calls] == ["context", "own"]
    assert [call[2] for call in dense.calls] == ["context", "own"]
    assert len(fake_transport.count_requests) == 2
    assert [item[1] for item in fake_transport.create_requests] == [
        f"{job['id']}:query",
        f"{job['id']}:prediction",
    ]


def _fake_response(text: str, number: int) -> dict[str, Any]:
    return {
        "id": f"fake-response-{number}",
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
