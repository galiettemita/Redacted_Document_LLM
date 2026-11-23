from __future__ import annotations

import asyncio
import hashlib
import re
from collections.abc import Sequence
from decimal import Decimal
from pathlib import Path
from typing import Any
from uuid import UUID

from sqlalchemy import select

import redacted_backend.api.analysis as analysis_api
import redacted_backend.api.passages as passages_api
from redacted_backend.config import Settings
from redacted_backend.corpus.best_available import (
    BEST_AVAILABLE_BM25_INDEX_NAME,
    BestAvailableCorpusManifest,
)
from redacted_backend.corpus.chunker import CorpusChunk
from redacted_backend.db.base import Base
from redacted_backend.db.models import (
    AnalysisJob,
    StagedSpendClaim,
    StagedSpendLedger,
)
from redacted_backend.documents.detection import DetectedPassage, DetectionResult
from redacted_backend.domain.analysis import PredictionBasis
from redacted_backend.domain.passages import PassageRect
from redacted_backend.domain.readiness import ComponentReadiness
from redacted_backend.jobs.generation_store import AnalysisGenerationStore
from redacted_backend.jobs.orchestrator import AnalysisOrchestrator
from redacted_backend.jobs.spend_budget import StagedSpendBudget
from redacted_backend.main import create_app
from redacted_backend.models.openai_gateway import (
    OpenAIGateway,
    ProviderHttpResponse,
)
from redacted_backend.retrieval.bm25 import Bm25Index
from redacted_backend.retrieval.service import Bm25RetrievalService
from redacted_backend.scientific_config import ScientificConfig
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
        text = (
            "specific actors paris 1960 meeting"
            if number == 1
            else "The missing passage identifies the responsible officials."
        )
        return ProviderHttpResponse(
            body={
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
            },
            request_id=f"fake-request-{number}",
        )


class TinyBm25Transport:
    def __init__(self) -> None:
        self.documents: list[dict[str, Any]] = []
        self.searches: list[str] = []

    def replace_index(self, _index_name: str, _definition: dict[str, Any]) -> None:
        self.documents = []

    def bulk_index(
        self,
        _index_name: str,
        documents: Sequence[dict[str, Any]],
    ) -> None:
        self.documents.extend(documents)

    def refresh(self, _index_name: str) -> None:
        return None

    def count(self, _index_name: str) -> int:
        return len(self.documents)

    def search(
        self,
        _index_name: str,
        request: dict[str, Any],
    ) -> list[dict[str, Any]]:
        query = str(request["query"]["match"]["text"]["query"])
        self.searches.append(query)
        terms = set(re.findall(r"[a-z0-9]+", query.casefold()))
        hits: list[dict[str, Any]] = []
        for document in self.documents:
            text_terms = set(
                re.findall(r"[a-z0-9]+", str(document["text"]).casefold())
            )
            score = len(terms & text_terms)
            if score:
                hits.append(
                    {
                        "_id": document["chunk_id"],
                        "_score": float(score),
                        "_source": document,
                    }
                )
        return sorted(hits, key=lambda hit: (-hit["_score"], hit["_id"]))


def test_upload_to_staged_sse_runs_two_bm25_lists_and_settles_spend(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dispatcher = RecordingDispatcher()
    ready = ComponentReadiness(ready=True, detail="test ready")
    unavailable = ComponentReadiness(ready=False, detail="test unavailable")
    monkeypatch.setattr(analysis_api, "resolve_worker_readiness", lambda _s: ready)
    monkeypatch.setattr(analysis_api, "resolve_corpus_readiness", lambda _s: unavailable)
    monkeypatch.setattr(
        analysis_api,
        "resolve_staged_bm25_corpus_readiness",
        lambda _s: ready,
    )
    monkeypatch.setattr(
        analysis_api,
        "resolve_best_available_corpus_readiness",
        lambda _s: unavailable,
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
                ),
                DetectedPassage(
                    page_number=1,
                    rects=[PassageRect(x=0.1, y=0.5, width=0.2, height=0.03)],
                    kinds=["black"],
                    source_mask_ids=["mask-2"],
                    confidence=1.0,
                ),
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
            allow_staged_bm25_calls=True,
            staged_bm25_spend_cap_usd="10.00",
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
    passages = detected.json()["passages"]
    assert len(passages) == 2
    for passage in passages:
        prepared = asyncio.run(
            request(app, "POST", f"/v1/passages/{passage['id']}:prepare-context")
        )
        assert prepared.status_code == 200

    first = _schedule(app, passages[0], "staged-first")
    second = _schedule(app, passages[1], "staged-second")
    assert first["status"] == "queued"
    assert second["status"] == "blocked"
    assert second["blockers"][0]["component"] == "staged_prediction_limit"
    assert dispatcher.job_ids == [UUID(first["id"])]

    scientific = ScientificConfig(top_chunks_per_list=3, top_documents=3)
    chunks = [
        _chunk(index, "visible before visible after specific actors paris meeting")
        for index in range(3)
    ]
    corpus = BestAvailableCorpusManifest(
        source_profile_hash="1" * 64,
        source_sha256="2" * 64,
        source_size_bytes=1,
        audit_manifest_hash="3" * 64,
        limitations_sha256="4" * 64,
        scientific_hash=scientific.manifest_hash(),
        chunker_version=scientific.corpus_chunker_version,
        chunk_words=scientific.chunk_words,
        chunk_overlap=scientific.chunk_overlap,
        parsed_row_count=3,
        included_row_count=3,
        empty_body_count=0,
        malformed_row_count=0,
        document_count=3,
        chunk_count=3,
        corpus_counts={"historylab-unlabeled": 3},
        chunks_sha256="5" * 64,
    )
    bm25_transport = TinyBm25Transport()
    bm25 = Bm25Index(
        bm25_transport,
        scientific=scientific,
        index_name=BEST_AVAILABLE_BM25_INDEX_NAME,
    )
    bm25.build(chunks, corpus)
    provider = FakeTwoStageProviderTransport()
    AnalysisOrchestrator(
        database=app.state.database,
        model=OpenAIGateway(
            transport=provider,
            store=AnalysisGenerationStore(app.state.database),
            spend_budget=StagedSpendBudget(
                app.state.database,
                cap_usd=Decimal("10.00"),
            ),
            scientific=scientific,
        ),
        retrieval=Bm25RetrievalService(bm25=bm25, scientific=scientific),
        prediction_basis=PredictionBasis.BEST_AVAILABLE_FOIARCHIVE_BM25_V1,
        scientific=scientific,
    ).run(first["id"])

    events = asyncio.run(request(app, "GET", f"/v1/documents/{document_id}/events"))
    assert events.status_code == 200
    assert "event: passage.prediction" in events.text
    assert "The missing passage identifies the responsible officials." in events.text
    assert "specific actors paris 1960 meeting" not in events.text
    assert len(provider.count_requests) == 2
    assert [key for _request, key in provider.create_requests] == [
        f"{first['id']}:query",
        f"{first['id']}:prediction",
    ]
    assert len(bm25_transport.searches) == 2
    assert bm25_transport.searches[0] != bm25_transport.searches[1]

    with app.state.database.session_factory() as session:
        stored = session.get(AnalysisJob, first["id"])
        ledger = session.get(
            StagedSpendLedger,
            PredictionBasis.BEST_AVAILABLE_FOIARCHIVE_BM25_V1.value,
        )
        claims = session.scalars(select(StagedSpendClaim)).all()
    assert stored is not None
    assert stored.status == "complete"
    assert stored.prediction_basis == (
        PredictionBasis.BEST_AVAILABLE_FOIARCHIVE_BM25_V1.value
    )
    assert set(stored.retrieval_trace["lists"]) == {
        "context_bm25",
        "own_bm25",
    }
    assert all(stored.retrieval_trace["lists"].values())
    assert ledger is not None
    assert ledger.committed_microusd == ledger.actual_microusd
    assert ledger.actual_microusd > 0
    assert len(claims) == 2
    assert all(claim.state == "settled" for claim in claims)


def _schedule(app, passage: dict[str, object], key: str) -> dict[str, object]:
    response = asyncio.run(
        request(
            app,
            "POST",
            f"/v1/passages/{passage['id']}/analysis",
            headers={"Idempotency-Key": key},
            json={"expected_revision": passage["revision"]},
        )
    )
    assert response.status_code == 202
    return response.json()


def _chunk(index: int, text: str) -> CorpusChunk:
    document_id = f"archive-document-{index}"
    return CorpusChunk(
        id=hashlib.sha256(f"{document_id}:0".encode()).hexdigest(),
        document_id=document_id,
        ordinal=0,
        text=f"{text} evidence {index}",
        word_count=len(text.split()) + 2,
        title=f"{document_id}.pdf",
        corpus="historylab-unlabeled",
        source_url=f"https://example.test/{document_id}.pdf",
        classification="UNCLASSIFIED",
        authored="1960-01-01",
        doc_lang="en",
    )
