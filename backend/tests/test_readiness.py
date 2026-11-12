import asyncio
import json
from pathlib import Path

from httpx import ASGITransport, AsyncClient

from redacted_backend.config import Settings
from redacted_backend.corpus.manifest import CorpusManifest
from redacted_backend.domain.readiness import verify_corpus_readiness
from redacted_backend.main import create_app
from redacted_backend.retrieval.bm25 import index_definition_hash
from redacted_backend.retrieval.types import Bm25IndexManifest
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


def test_corpus_readiness_requires_matching_corpus_and_bm25_manifests(
    tmp_path: Path,
) -> None:
    scientific = ScientificConfig()
    corpus = CorpusManifest(
        schema_version="foiarchive-nine-fields-v1",
        snapshot_hash="1" * 64,
        exclusion_hash="2" * 64,
        scientific_hash=scientific.manifest_hash(),
        chunker_version=scientific.corpus_chunker_version,
        chunk_words=scientific.chunk_words,
        chunk_overlap=scientific.chunk_overlap,
        document_count=1,
        chunk_count=1,
        source_counts={"cfpf": 1},
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
    corpus_path = tmp_path / "corpus.json"
    index_path = tmp_path / "bm25.json"
    corpus_path.write_text(json.dumps(corpus.model_dump(mode="json")), encoding="utf-8")
    index_path.write_text(json.dumps(index.model_dump(mode="json")), encoding="utf-8")
    settings = Settings(
        database_url=None,
        redis_url=None,
        corpus_manifest_path=corpus_path,
        bm25_manifest_path=index_path,
        openai_api_key=None,
    )

    async def request_readiness():
        transport = ASGITransport(app=create_app(settings))
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.get("/v1/readiness")

    body = asyncio.run(request_readiness()).json()

    assert body["corpus"] == {"ready": True, "detail": "BM25 index verified"}
    assert body["prediction_ready"] is False


def test_corpus_readiness_rejects_an_incompatible_bm25_definition_or_count() -> None:
    scientific = ScientificConfig()
    corpus = CorpusManifest(
        schema_version="foiarchive-nine-fields-v1",
        snapshot_hash="1" * 64,
        exclusion_hash="2" * 64,
        scientific_hash=scientific.manifest_hash(),
        chunker_version=scientific.corpus_chunker_version,
        chunk_words=scientific.chunk_words,
        chunk_overlap=scientific.chunk_overlap,
        document_count=1,
        chunk_count=1,
        source_counts={"cfpf": 1},
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

    wrong_definition = compatible.model_copy(update={"index_definition_hash": "0" * 64})
    wrong_count = compatible.model_copy(update={"chunk_count": 2})

    assert verify_corpus_readiness(
        corpus, wrong_definition, scientific
    ).model_dump() == {
        "ready": False,
        "detail": "BM25 index definition mismatch",
    }
    assert verify_corpus_readiness(corpus, wrong_count, scientific).model_dump() == {
        "ready": False,
        "detail": "BM25 corpus count mismatch",
    }
