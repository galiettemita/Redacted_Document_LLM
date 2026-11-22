from __future__ import annotations

import asyncio
import hashlib
import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from uuid import UUID

import numpy as np
from numpy.typing import NDArray
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
from redacted_backend.retrieval.bm25 import Bm25Index
from redacted_backend.retrieval.chunk_offsets import build_chunk_offsets
from redacted_backend.retrieval.dense_profile import (
    DenseArtifactProfile,
    dense_semantics_hash,
)
from redacted_backend.retrieval.dense_sharded import ShardedDenseIndex
from redacted_backend.retrieval.faiss_ivfpq import (
    FaissIvfPqIndex,
    train_ivfpq_template,
)
from redacted_backend.retrieval.service import RealisticRetrievalService
from redacted_backend.retrieval.types import (
    CompressedDenseShardManifest,
    CompressedShardedDenseIndexManifest,
)
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
        output = (
            "specific actors paris 1960 meeting"
            if number == 1
            else "The missing passage identifies the responsible officials."
        )
        return ProviderHttpResponse(
            body=_fake_response(output, number),
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
            if document["document_id"] == "dense-only-document":
                continue
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


class FixedDenseQueryEncoder:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def encode_query(self, text: str) -> NDArray[np.float32]:
        self.calls.append(text)
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)


def _build_real_retrievers(
    root: Path,
) -> tuple[Bm25Index, ShardedDenseIndex, TinyBm25Transport, FixedDenseQueryEncoder]:
    scientific = ScientificConfig(embedding_dimension=4)
    chunks = [
        _archive_chunk(
            "bm25-context-document",
            "visible before visible after archive bulletin",
        ),
        _archive_chunk(
            "bm25-own-document",
            "specific actors paris 1960 meeting public archive",
        ),
        _archive_chunk(
            "dense-only-document",
            "semantic vector evidence from a separate diplomatic cable",
        ),
    ]
    chunks_path = root / "real-retrieval-chunks.jsonl"
    chunks_path.write_text(
        "".join(
            json.dumps(
                chunk.model_dump(mode="json"),
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
            for chunk in chunks
        ),
        encoding="utf-8",
    )
    chunks_sha256 = hashlib.sha256(chunks_path.read_bytes()).hexdigest()
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
        parsed_row_count=len(chunks),
        included_row_count=len(chunks),
        empty_body_count=0,
        malformed_row_count=0,
        document_count=len(chunks),
        chunk_count=len(chunks),
        corpus_counts={"historylab-unlabeled": len(chunks)},
        chunks_sha256=chunks_sha256,
    )

    bm25_transport = TinyBm25Transport()
    bm25 = Bm25Index(
        bm25_transport,
        scientific=scientific,
        index_name=BEST_AVAILABLE_BM25_INDEX_NAME,
    )
    bm25.build(chunks, corpus)

    dense_root = root / "real-compressed-dense"
    dense_root.mkdir()
    profile = DenseArtifactProfile(
        embedding_dimension=4,
        ivf_nlist=1,
        pq_subvectors=1,
        pq_bits=1,
        nprobe=1,
        candidate_count_per_shard=3,
        training_sample_count=16,
        shard_size_limit=3,
        max_artifact_bytes=1024 * 1024,
        max_parallel_shards=1,
    )
    rng = np.random.default_rng(20261122)
    training = rng.normal(size=(profile.training_sample_count, 4)).astype(np.float32)
    training /= np.linalg.norm(training, axis=1, keepdims=True)
    template_path = dense_root / "candidate-template.faiss"
    template_sha256 = train_ivfpq_template(
        training,
        template_path,
        dimension=4,
        profile=profile,
    )
    vectors = np.asarray(
        [
            [0.8, 0.6, 0.0, 0.0],
            [0.6, 0.8, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    shard_root = dense_root / "shard-00000"
    shard_root.mkdir()
    embeddings_path = shard_root / "embeddings.f16"
    vectors.astype(np.float16).tofile(embeddings_path)
    candidate_path = shard_root / "candidate.faiss"
    candidate = FaissIvfPqIndex.create(template_path, profile=profile)
    candidate.add(vectors)
    candidate.save(candidate_path)
    offsets_path = dense_root / "chunks.offsets.u64"
    offsets = build_chunk_offsets(
        chunks_path,
        offsets_path,
        expected_count=len(chunks),
        expected_chunks_sha256=chunks_sha256,
    )
    recall_path = dense_root / "recall-report.json"
    recall_path.write_bytes(b"{}\n")
    shard = CompressedDenseShardManifest(
        artifact_profile_hash=profile.manifest_hash(),
        candidate_template_sha256=template_sha256,
        shard_id=0,
        path=shard_root.name,
        start_row=0,
        end_row=len(chunks),
        chunk_count=len(chunks),
        embeddings_size_bytes=embeddings_path.stat().st_size,
        embeddings_sha256=hashlib.sha256(embeddings_path.read_bytes()).hexdigest(),
        candidate_index_size_bytes=candidate_path.stat().st_size,
        candidate_index_sha256=hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
    )
    total_bytes = (
        offsets.offsets_size_bytes
        + template_path.stat().st_size
        + recall_path.stat().st_size
        + shard.embeddings_size_bytes
        + shard.candidate_index_size_bytes
    )
    manifest = CompressedShardedDenseIndexManifest(
        model_name=scientific.embedding_model,
        model_revision=scientific.embedding_model_revision,
        embedding_dimension=4,
        corpus_manifest_hash=corpus.manifest_hash(),
        dense_semantics_hash=dense_semantics_hash(scientific),
        chunks_sha256=chunks_sha256,
        chunk_count=len(chunks),
        artifact_profile=profile,
        artifact_profile_hash=profile.manifest_hash(),
        chunk_offsets_path=offsets_path.name,
        chunk_offsets_format=profile.metadata_lookup_version,
        chunk_offsets_size_bytes=offsets.offsets_size_bytes,
        chunk_offsets_sha256=offsets.offsets_sha256,
        candidate_template_path=template_path.name,
        candidate_template_size_bytes=template_path.stat().st_size,
        candidate_template_sha256=template_sha256,
        shard_size_limit=profile.shard_size_limit,
        shards=[shard],
        dtype="float16",
        metric="inner_product",
        candidate_index=profile.candidate_index,
        recall_report_path=recall_path.name,
        recall_report_size_bytes=recall_path.stat().st_size,
        recall_report_sha256=hashlib.sha256(recall_path.read_bytes()).hexdigest(),
        recall_at_300=1.0,
        recall_queries_sha256="5" * 64,
        recall_query_count=100,
        total_artifact_bytes=total_bytes,
    )
    encoder = FixedDenseQueryEncoder()
    dense = ShardedDenseIndex(
        root=dense_root,
        chunks_path=chunks_path,
        manifest=manifest,
        encoder=encoder,
        candidate_loader=FaissIvfPqIndex.load,
        scientific=scientific,
    )
    return bm25, dense, bm25_transport, encoder


def _archive_chunk(document_id: str, text: str) -> CorpusChunk:
    return CorpusChunk(
        id=hashlib.sha256(f"{document_id}:0".encode("utf-8")).hexdigest(),
        document_id=document_id,
        ordinal=0,
        text=text,
        word_count=len(text.split()),
        title=f"{document_id}.pdf",
        corpus="historylab-unlabeled",
        source_url=f"https://example.test/{document_id}.pdf",
        classification="UNCLASSIFIED",
        authored="1960-01-01",
        doc_lang="en",
    )


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
    bm25, dense, bm25_transport, dense_encoder = _build_real_retrievers(tmp_path)
    try:
        AnalysisOrchestrator(
            database=app.state.database,
            model=OpenAIGateway(
                transport=fake_transport,
                store=AnalysisGenerationStore(app.state.database),
            ),
            retrieval=RealisticRetrievalService(bm25=bm25, dense=dense),
            prediction_basis=PredictionBasis.BEST_AVAILABLE_FOIARCHIVE_V1,
        ).run(job["id"])
    finally:
        dense.close()

    events = asyncio.run(request(app, "GET", f"/v1/documents/{document_id}/events"))
    assert events.status_code == 200
    assert "event: passage.prediction" in events.text
    assert '"prediction_basis":"best_available_foiarchive_v1"' in events.text
    assert "The missing passage identifies the responsible officials." in events.text
    assert "specific actors paris 1960 meeting" not in events.text
    assert "semantic vector evidence" not in events.text.casefold()
    assert "visible before visible after archive bulletin" not in events.text.casefold()

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
    assert stored.retrieval_trace["fusion_version"] == "document-rrf-v1"
    assert all(
        document["rrf_score"] > 0
        for document in stored.retrieval_trace["documents"]
    )
    bm25_documents = {
        item["document_id"]
        for name in ("context_bm25", "own_bm25")
        for item in stored.retrieval_trace["lists"][name]
    }
    dense_documents = {
        item["document_id"]
        for name in ("context_dense", "own_dense")
        for item in stored.retrieval_trace["lists"][name]
    }
    assert "dense-only-document" in dense_documents
    assert "dense-only-document" not in bm25_documents
    bm25_manifest_hashes = {
        item["index_manifest_hash"]
        for name in ("context_bm25", "own_bm25")
        for item in stored.retrieval_trace["lists"][name]
    }
    dense_manifest_hashes = {
        item["index_manifest_hash"]
        for name in ("context_dense", "own_dense")
        for item in stored.retrieval_trace["lists"][name]
    }
    assert len(bm25_manifest_hashes) == len(dense_manifest_hashes) == 1
    assert bm25_manifest_hashes.isdisjoint(dense_manifest_hashes)
    assert stored.prediction_payload["prediction_basis"] == (
        "best_available_foiarchive_v1"
    )
    assert stored.prediction_payload["retrieval_trace_sha256"] is not None
    assert len(bm25_transport.searches) == 2
    assert bm25_transport.searches[1] == "specific actors paris 1960 meeting"
    assert len(dense_encoder.calls) == 2
    assert dense_encoder.calls == bm25_transport.searches
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
