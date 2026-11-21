import asyncio
import hashlib
import json
from pathlib import Path

from httpx import ASGITransport, AsyncClient

import redacted_backend.api.readiness as readiness_api
from redacted_backend.api.readiness import (
    resolve_best_available_corpus_readiness,
    resolve_corpus_readiness,
    resolve_worker_readiness,
)
from redacted_backend.config import Settings
from redacted_backend.corpus.best_available import (
    BEST_AVAILABLE_BM25_INDEX_NAME,
    BestAvailableCorpusManifest,
)
from redacted_backend.corpus.manifest import CorpusManifest
from redacted_backend.domain.readiness import (
    verify_best_available_corpus_readiness,
    verify_corpus_readiness,
)
from redacted_backend.domain.analysis import PredictionBasis
from redacted_backend.domain.readiness import ComponentReadiness
from redacted_backend.main import create_app
from redacted_backend.retrieval.bm25 import index_definition_hash
from redacted_backend.retrieval.types import (
    Bm25IndexManifest,
    DenseIndexManifest,
    DenseShardManifest,
    ShardedDenseIndexManifest,
)
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
        "prediction_bases",
        "strongest_ready_basis",
    }
    assert body["strongest_ready_basis"] is None
    assert set(body["prediction_bases"]) == {
        "model_only_v1",
        "best_available_foiarchive_v1",
        "paper_exact_v1",
    }


def test_http_health_does_not_probe_prediction_dependencies(monkeypatch) -> None:
    def unexpected_probe(_settings: Settings):
        raise AssertionError("liveness must not probe prediction dependencies")

    monkeypatch.setattr(
        "redacted_backend.api.readiness.resolve_corpus_readiness",
        unexpected_probe,
    )

    async def request_health():
        transport = ASGITransport(app=create_app(Settings()))
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.get("/v1/health")

    response = asyncio.run(request_health())

    assert response.status_code == 200
    assert response.json() == {"ready": True, "detail": "ready"}


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


def test_model_only_basis_is_operational_without_corpus_but_separately_approved() -> None:
    resolve = getattr(readiness_api, "resolve_prediction_basis_readiness", None)
    select = getattr(readiness_api, "select_prediction_basis", None)
    assert resolve is not None
    assert select is not None
    ready = ComponentReadiness(ready=True, detail="ready")
    corpus_unavailable = ComponentReadiness(ready=False, detail="not configured")
    settings = Settings(
        openai_api_key="secret",
        allow_paid_model_calls=True,
        allow_model_only_calls=True,
        allow_best_available_calls=True,
        allow_paper_exact_calls=True,
    )

    bases = resolve(
        settings,
        database=ready,
        worker=ready,
        corpus=corpus_unavailable,
        best_available_corpus=corpus_unavailable,
    )

    assert bases[PredictionBasis.MODEL_ONLY_V1].model_dump() == {
        "operational": True,
        "approved": True,
        "ready": True,
        "detail": "operational and approved",
    }
    assert bases[PredictionBasis.BEST_AVAILABLE_FOIARCHIVE_V1].model_dump() == {
        "operational": False,
        "approved": True,
        "ready": False,
        "detail": "corpus: not configured",
    }
    assert bases[PredictionBasis.PAPER_EXACT_V1].model_dump() == {
        "operational": False,
        "approved": True,
        "ready": False,
        "detail": "corpus: not configured",
    }
    assert select(bases) == PredictionBasis.MODEL_ONLY_V1


def test_model_only_basis_requires_master_and_its_own_spending_approval() -> None:
    resolve = getattr(readiness_api, "resolve_prediction_basis_readiness", None)
    assert resolve is not None
    ready = ComponentReadiness(ready=True, detail="ready")
    corpus_unavailable = ComponentReadiness(ready=False, detail="not configured")

    per_basis_unapproved = resolve(
        Settings(
            openai_api_key="secret",
            allow_paid_model_calls=True,
            allow_model_only_calls=False,
        ),
        database=ready,
        worker=ready,
        corpus=corpus_unavailable,
        best_available_corpus=corpus_unavailable,
    )[PredictionBasis.MODEL_ONLY_V1]
    master_unapproved = resolve(
        Settings(
            openai_api_key="secret",
            allow_paid_model_calls=False,
            allow_model_only_calls=True,
        ),
        database=ready,
        worker=ready,
        corpus=corpus_unavailable,
        best_available_corpus=corpus_unavailable,
    )[PredictionBasis.MODEL_ONLY_V1]

    assert per_basis_unapproved.model_dump() == {
        "operational": True,
        "approved": False,
        "ready": False,
        "detail": "operational; spending not approved",
    }
    assert master_unapproved == per_basis_unapproved


def test_best_available_basis_is_operational_without_contaminating_exact() -> None:
    ready = ComponentReadiness(ready=True, detail="ready")
    unavailable = ComponentReadiness(ready=False, detail="not configured")
    settings = Settings(
        openai_api_key="secret",
        allow_paid_model_calls=True,
        allow_best_available_calls=True,
        allow_paper_exact_calls=True,
    )

    bases = readiness_api.resolve_prediction_basis_readiness(
        settings,
        database=ready,
        worker=ready,
        corpus=unavailable,
        best_available_corpus=ready,
    )

    assert bases[PredictionBasis.BEST_AVAILABLE_FOIARCHIVE_V1].ready is True
    assert bases[PredictionBasis.PAPER_EXACT_V1].ready is False
    assert readiness_api.select_prediction_basis(bases) == (
        PredictionBasis.BEST_AVAILABLE_FOIARCHIVE_V1
    )


def test_corpus_readiness_requires_dense_artifact_files(
    tmp_path: Path,
) -> None:
    settings, _ = _compatible_readiness_settings(tmp_path)

    readiness = resolve_corpus_readiness(
        settings,
        index_counter=lambda _url, _index: 1,
    )

    assert readiness.model_dump() == {
        "ready": False,
        "detail": "dense artifacts not available",
    }


def test_corpus_readiness_requires_a_configured_retrieval_service(
    tmp_path: Path,
) -> None:
    settings, _ = _compatible_readiness_settings(tmp_path, create_dense=True)

    readiness = resolve_corpus_readiness(
        settings,
        index_counter=lambda _url, _index: 1,
    )

    assert readiness.model_dump() == {
        "ready": False,
        "detail": "retrieval service not configured",
    }


def test_corpus_readiness_fails_closed_when_the_live_index_probe_fails(
    tmp_path: Path,
) -> None:
    settings, _ = _compatible_readiness_settings(
        tmp_path,
        create_dense=True,
        opensearch_url="http://search.test:9200",
    )

    def unavailable(_url: str, _index: str) -> int:
        raise RuntimeError("credential-or-host-detail")

    readiness = resolve_corpus_readiness(settings, index_counter=unavailable)

    assert readiness.model_dump() == {
        "ready": False,
        "detail": "retrieval index probe failed",
    }


def test_corpus_readiness_rejects_a_live_index_count_mismatch(
    tmp_path: Path,
) -> None:
    settings, _ = _compatible_readiness_settings(
        tmp_path,
        create_dense=True,
        opensearch_url="http://search.test:9200",
    )

    readiness = resolve_corpus_readiness(
        settings,
        index_counter=lambda _url, _index: 2,
    )

    assert readiness.model_dump() == {
        "ready": False,
        "detail": "BM25 live index count mismatch",
    }


def test_corpus_readiness_requires_matching_manifests_artifacts_and_live_index(
    tmp_path: Path,
) -> None:
    settings, index = _compatible_readiness_settings(
        tmp_path,
        create_dense=True,
        opensearch_url="http://search.test:9200",
    )

    readiness = resolve_corpus_readiness(
        settings,
        index_counter=lambda url, name: (
            index.chunk_count
            if (url, name) == ("http://search.test:9200", index.index_name)
            else -1
        ),
    )

    assert readiness.model_dump() == {
        "ready": True,
        "detail": "corpus and retrieval indexes operational",
    }


def test_best_available_readiness_verifies_separate_shards_queries_and_live_index(
    tmp_path: Path,
) -> None:
    settings, corpus, bm25, dense = _compatible_best_available_settings(tmp_path)

    readiness = resolve_best_available_corpus_readiness(
        settings,
        index_counter=lambda url, name: (
            bm25.chunk_count
            if (url, name) == ("http://search.test:9200", bm25.index_name)
            else -1
        ),
    )

    assert readiness.model_dump() == {
        "ready": True,
        "detail": "best-available corpus and retrieval indexes operational",
    }
    assert verify_best_available_corpus_readiness(
        corpus,
        bm25,
        dense,
        ScientificConfig(),
    ).ready is True


def test_best_available_readiness_fails_closed_on_corruption_or_live_count(
    tmp_path: Path,
) -> None:
    settings, _corpus, bm25, dense = _compatible_best_available_settings(tmp_path)
    assert resolve_best_available_corpus_readiness(
        settings,
        index_counter=lambda _url, _name: bm25.chunk_count,
    ).ready is True
    shard_root = settings.best_available_dense_artifact_directory
    assert shard_root is not None
    (shard_root / dense.shards[0].path / "candidate.faiss").write_bytes(b"tampered")

    corrupt = resolve_best_available_corpus_readiness(
        settings,
        index_counter=lambda _url, _name: bm25.chunk_count,
    )
    assert corrupt.model_dump() == {
        "ready": False,
        "detail": "dense shard artifacts invalid",
    }

    settings, _corpus, bm25, _dense = _compatible_best_available_settings(
        tmp_path / "count"
    )
    mismatch = resolve_best_available_corpus_readiness(
        settings,
        index_counter=lambda _url, _name: bm25.chunk_count + 1,
    )
    assert mismatch.model_dump() == {
        "ready": False,
        "detail": "BM25 live index count mismatch",
    }


def _compatible_readiness_settings(
    tmp_path: Path,
    *,
    create_dense: bool = False,
    opensearch_url: str | None = None,
) -> tuple[Settings, Bm25IndexManifest]:
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
    dense_directory = tmp_path / "dense"
    corpus_path.write_text(json.dumps(corpus.model_dump(mode="json")), encoding="utf-8")
    index_path.write_text(json.dumps(index.model_dump(mode="json")), encoding="utf-8")
    dense_path.write_text(json.dumps(dense.model_dump(mode="json")), encoding="utf-8")
    if create_dense:
        dense_directory.mkdir()
        for name in ("embeddings.f16", "metadata.jsonl", "candidate.faiss"):
            (dense_directory / name).write_bytes(b"fixture")
    settings = Settings(
        database_url=None,
        redis_url=None,
        corpus_manifest_path=corpus_path,
        bm25_manifest_path=index_path,
        dense_manifest_path=dense_path,
        dense_artifact_directory=dense_directory,
        opensearch_url=opensearch_url,
        openai_api_key=None,
    )
    return settings, index


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


def _compatible_best_available_settings(
    tmp_path: Path,
) -> tuple[
    Settings,
    BestAvailableCorpusManifest,
    Bm25IndexManifest,
    ShardedDenseIndexManifest,
]:
    scientific = ScientificConfig()
    tmp_path.mkdir(parents=True, exist_ok=True)
    corpus = BestAvailableCorpusManifest(
        source_profile_hash="1" * 64,
        source_sha256="2" * 64,
        source_size_bytes=100,
        audit_manifest_hash="3" * 64,
        limitations_sha256="4" * 64,
        scientific_hash=scientific.manifest_hash(),
        chunker_version=scientific.corpus_chunker_version,
        chunk_words=scientific.chunk_words,
        chunk_overlap=scientific.chunk_overlap,
        parsed_row_count=1,
        included_row_count=1,
        empty_body_count=0,
        malformed_row_count=0,
        document_count=1,
        chunk_count=1,
        corpus_counts={"historylab-unlabeled": 1},
        chunks_sha256="5" * 64,
    )
    bm25 = Bm25IndexManifest(
        index_name=BEST_AVAILABLE_BM25_INDEX_NAME,
        retrieval_profile="best_available_foiarchive_v1",
        opensearch_version=scientific.opensearch_version,
        corpus_manifest_hash=corpus.manifest_hash(),
        scientific_hash=scientific.manifest_hash(),
        stopwords_hash=scientific.context_stopwords_hash(),
        index_definition_hash=index_definition_hash(scientific),
        document_count=1,
        chunk_count=1,
    )
    dense_root = tmp_path / "best-dense"
    shard_directory = dense_root / "shard-00000"
    shard_directory.mkdir(parents=True)
    artifacts = {
        "embeddings.f16": b"\0" * (scientific.embedding_dimension * 2),
        "metadata.jsonl": b'{}\n',
        "metadata.offsets.u64": b"\0" * 8,
        "candidate.faiss": b"candidate-fixture",
    }
    for name, value in artifacts.items():
        (shard_directory / name).write_bytes(value)
    shard = DenseShardManifest(
        shard_id=0,
        path=shard_directory.name,
        start_row=0,
        end_row=1,
        chunk_count=1,
        embeddings_sha256=hashlib.sha256(artifacts["embeddings.f16"]).hexdigest(),
        metadata_sha256=hashlib.sha256(artifacts["metadata.jsonl"]).hexdigest(),
        offsets_sha256=hashlib.sha256(
            artifacts["metadata.offsets.u64"]
        ).hexdigest(),
        candidate_index_sha256=hashlib.sha256(
            artifacts["candidate.faiss"]
        ).hexdigest(),
    )
    queries = "".join(f"public certification query {index}\n" for index in range(100))
    queries_path = tmp_path / "best-recall-queries.txt"
    queries_path.write_text(queries, encoding="utf-8")
    dense = ShardedDenseIndexManifest(
        model_name=scientific.embedding_model,
        model_revision=scientific.embedding_model_revision,
        embedding_dimension=scientific.embedding_dimension,
        corpus_manifest_hash=corpus.manifest_hash(),
        scientific_hash=scientific.manifest_hash(),
        chunks_sha256=corpus.chunks_sha256,
        chunk_count=1,
        shard_size_limit=1_000_000,
        shards=[shard],
        dtype="float16",
        metric="inner_product",
        candidate_index=scientific.dense_index_version,
        hnsw_m=scientific.dense_hnsw_m,
        ef_construction=scientific.dense_ef_construction,
        ef_search=scientific.dense_ef_search,
        candidate_count=scientific.dense_candidate_count,
        recall_at_300=0.99,
        recall_queries_sha256=hashlib.sha256(queries.encode("utf-8")).hexdigest(),
        recall_query_count=100,
    )
    corpus_path = tmp_path / "best-corpus.json"
    bm25_path = tmp_path / "best-bm25.json"
    dense_path = tmp_path / "best-dense.json"
    corpus_path.write_text(corpus.model_dump_json(), encoding="utf-8")
    bm25_path.write_text(bm25.model_dump_json(), encoding="utf-8")
    dense_path.write_text(dense.model_dump_json(), encoding="utf-8")
    settings = Settings(
        best_available_corpus_manifest_path=corpus_path,
        best_available_bm25_manifest_path=bm25_path,
        best_available_dense_manifest_path=dense_path,
        best_available_dense_artifact_directory=dense_root,
        best_available_recall_queries_path=queries_path,
        opensearch_url="http://search.test:9200",
    )
    return settings, corpus, bm25, dense
