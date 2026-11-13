import asyncio
import json
from pathlib import Path

from httpx import ASGITransport, AsyncClient

from redacted_backend.api.readiness import resolve_worker_readiness
from redacted_backend.config import Settings
from redacted_backend.corpus.manifest import CorpusManifest
from redacted_backend.domain.readiness import verify_corpus_readiness
from redacted_backend.main import create_app
from redacted_backend.retrieval.bm25 import index_definition_hash
from redacted_backend.retrieval.types import Bm25IndexManifest, DenseIndexManifest
from redacted_backend.scientific_config import ScientificConfig


def test_readiness_never_equates_http_health_with_prediction_readiness() -> None:
    settings = Settings(
        database_url=None,
        redis_url=None,
        corpus_manifest_path=None,
        openai_api_key=None,
    )

    async def request_readiness():
        transport = ASGITransport(app=create_app(settings))
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.get("/v1/readiness")

    response = asyncio.run(request_readiness())

    assert response.status_code == 200
    body = response.json()
    assert body["http"] == {"ready": True, "detail": "ready"}
    assert body["database"]["ready"] is False
    assert body["worker"]["ready"] is False
    assert body["corpus"]["ready"] is False
    assert body["model"]["ready"] is False
    assert body["prediction_ready"] is False
    assert set(body) == {
        "http",
        "database",
        "worker",
        "corpus",
        "model",
        "prediction_ready",
    }


def test_database_readiness_probes_the_configured_database(tmp_path: Path) -> None:
    settings = Settings(
        database_url=f"sqlite:///{tmp_path / 'readiness.db'}",
        document_storage_root=tmp_path / "documents",
    )

    async def request_readiness():
        transport = ASGITransport(app=create_app(settings))
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.get("/v1/readiness")

    body = asyncio.run(request_readiness()).json()

    assert body["database"] == {"ready": True, "detail": "ready"}
    assert body["prediction_ready"] is False


def test_worker_readiness_requires_a_live_expiring_heartbeat() -> None:
    class FakeRedis:
        def __init__(self, value: bytes | None) -> None:
            self.value = value

        def get(self, _key: str) -> bytes | None:
            return self.value

    settings = Settings(redis_url="redis://worker.test/0")

    ready = resolve_worker_readiness(
        settings,
        client_factory=lambda _url: FakeRedis(b"ready"),
    )
    stale = resolve_worker_readiness(
        settings,
        client_factory=lambda _url: FakeRedis(None),
    )

    assert ready.model_dump() == {"ready": True, "detail": "heartbeat current"}
    assert stale.model_dump() == {
        "ready": False,
        "detail": "heartbeat missing or expired",
    }


def test_model_readiness_requires_access_and_explicit_paid_call_approval() -> None:
    async def read(settings: Settings):
        transport = ASGITransport(app=create_app(settings))
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            return (await client.get("/v1/readiness")).json()

    unapproved = asyncio.run(
        read(Settings(openai_api_key="secret", allow_paid_model_calls=False))
    )
    approved = asyncio.run(
        read(Settings(openai_api_key="secret", allow_paid_model_calls=True))
    )

    assert unapproved["model"] == {
        "ready": False,
        "detail": "configured; paid calls not approved",
    }
    assert approved["model"] == {
        "ready": True,
        "detail": "configured and paid calls approved",
    }

    empty = asyncio.run(read(Settings(openai_api_key="", allow_paid_model_calls=True)))
    assert empty["model"] == {"ready": False, "detail": "not configured"}


def test_corpus_readiness_requires_matching_corpus_and_bm25_manifests(
    tmp_path: Path,
) -> None:
    scientific = ScientificConfig()
    corpus = CorpusManifest(
        schema_version="historylab-normalized-v1",
        source_manifest_hash="1" * 64,
        coverage_report_hash="4" * 64,
        exclusion_hash="2" * 64,
        scientific_hash=scientific.manifest_hash(),
        chunker_version=scientific.corpus_chunker_version,
        chunk_words=scientific.chunk_words,
        chunk_overlap=scientific.chunk_overlap,
        document_count=1,
        chunk_count=1,
        corpus_counts={"cfpf": 1},
        exclusion_counts={},
        coverage_ready=True,
        chunks_sha256="3" * 64,
    )
    index = Bm25IndexManifest(
        index_name=scientific.bm25_index_name,
        opensearch_version=scientific.opensearch_version,
        corpus_manifest_hash=corpus.manifest_hash(),
        scientific_hash=scientific.manifest_hash(),
        stopwords_hash=scientific.context_stopwords_hash(),
        index_definition_hash=index_definition_hash(scientific),
        document_count=1,
        chunk_count=1,
    )
    dense = _dense_manifest(scientific, corpus)
    corpus_path = tmp_path / "corpus.json"
    index_path = tmp_path / "bm25.json"
    dense_path = tmp_path / "dense.json"
    corpus_path.write_text(json.dumps(corpus.model_dump(mode="json")), encoding="utf-8")
    index_path.write_text(json.dumps(index.model_dump(mode="json")), encoding="utf-8")
    dense_path.write_text(json.dumps(dense.model_dump(mode="json")), encoding="utf-8")
    settings = Settings(
        database_url=None,
        redis_url=None,
        corpus_manifest_path=corpus_path,
        bm25_manifest_path=index_path,
        dense_manifest_path=dense_path,
        openai_api_key=None,
    )

    async def request_readiness():
        transport = ASGITransport(app=create_app(settings))
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.get("/v1/readiness")

    body = asyncio.run(request_readiness()).json()

    assert body["corpus"] == {
        "ready": True,
        "detail": "BM25 and dense indexes verified",
    }
    assert body["prediction_ready"] is False


def test_corpus_readiness_rejects_an_incompatible_bm25_definition_or_count() -> None:
    scientific = ScientificConfig()
    corpus = CorpusManifest(
        schema_version="historylab-normalized-v1",
        source_manifest_hash="1" * 64,
        coverage_report_hash="4" * 64,
        exclusion_hash="2" * 64,
        scientific_hash=scientific.manifest_hash(),
        chunker_version=scientific.corpus_chunker_version,
        chunk_words=scientific.chunk_words,
        chunk_overlap=scientific.chunk_overlap,
        document_count=1,
        chunk_count=1,
        corpus_counts={"cfpf": 1},
        exclusion_counts={},
        coverage_ready=True,
        chunks_sha256="3" * 64,
    )
    compatible = Bm25IndexManifest(
        index_name=scientific.bm25_index_name,
        opensearch_version=scientific.opensearch_version,
        corpus_manifest_hash=corpus.manifest_hash(),
        scientific_hash=scientific.manifest_hash(),
        stopwords_hash=scientific.context_stopwords_hash(),
        index_definition_hash=index_definition_hash(scientific),
        document_count=1,
        chunk_count=1,
    )
    dense = _dense_manifest(scientific, corpus)

    wrong_definition = compatible.model_copy(update={"index_definition_hash": "0" * 64})
    wrong_count = compatible.model_copy(update={"chunk_count": 2})

    assert verify_corpus_readiness(
        corpus, wrong_definition, dense, scientific
    ).model_dump() == {
        "ready": False,
        "detail": "BM25 index definition mismatch",
    }
    assert verify_corpus_readiness(
        corpus, wrong_count, dense, scientific
    ).model_dump() == {
        "ready": False,
        "detail": "BM25 corpus count mismatch",
    }


def test_corpus_readiness_rejects_dense_revision_and_recall_mismatches() -> None:
    scientific = ScientificConfig()
    corpus = CorpusManifest(
        schema_version="historylab-normalized-v1",
        source_manifest_hash="1" * 64,
        coverage_report_hash="4" * 64,
        exclusion_hash="2" * 64,
        scientific_hash=scientific.manifest_hash(),
        chunker_version=scientific.corpus_chunker_version,
        chunk_words=scientific.chunk_words,
        chunk_overlap=scientific.chunk_overlap,
        document_count=1,
        chunk_count=1,
        corpus_counts={"cfpf": 1},
        exclusion_counts={},
        coverage_ready=True,
        chunks_sha256="3" * 64,
    )
    bm25 = Bm25IndexManifest(
        index_name=scientific.bm25_index_name,
        opensearch_version=scientific.opensearch_version,
        corpus_manifest_hash=corpus.manifest_hash(),
        scientific_hash=scientific.manifest_hash(),
        stopwords_hash=scientific.context_stopwords_hash(),
        index_definition_hash=index_definition_hash(scientific),
        document_count=1,
        chunk_count=1,
    )
    dense = _dense_manifest(scientific, corpus)

    wrong_revision = dense.model_copy(update={"model_revision": "0" * 40})
    low_recall = dense.model_copy(update={"recall_at_300": 0.97})

    assert verify_corpus_readiness(
        corpus, bm25, wrong_revision, scientific
    ).model_dump() == {
        "ready": False,
        "detail": "dense model revision mismatch",
    }
    assert verify_corpus_readiness(
        corpus, bm25, low_recall, scientific
    ).model_dump() == {
        "ready": False,
        "detail": "dense recall@300 is below 0.9800",
    }


def _dense_manifest(
    scientific: ScientificConfig,
    corpus: CorpusManifest,
) -> DenseIndexManifest:
    return DenseIndexManifest(
        model_name=scientific.embedding_model,
        model_revision=scientific.embedding_model_revision,
        embedding_dimension=scientific.embedding_dimension,
        corpus_manifest_hash=corpus.manifest_hash(),
        scientific_hash=scientific.manifest_hash(),
        chunks_sha256=corpus.chunks_sha256,
        embeddings_sha256="5" * 64,
        metadata_sha256="6" * 64,
        candidate_index_sha256="7" * 64,
        chunk_count=corpus.chunk_count,
        dtype="float16",
        metric="inner_product",
        candidate_index="faiss-hnsw-flat-v1",
        hnsw_m=scientific.dense_hnsw_m,
        ef_construction=scientific.dense_ef_construction,
        ef_search=scientific.dense_ef_search,
        candidate_count=scientific.dense_candidate_count,
        recall_at_300=0.99,
        recall_sample_sha256="8" * 64,
    )
