from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from redacted_backend.config import Settings
from redacted_backend.corpus.build_dense import FaissHnswIndex
from redacted_backend.corpus.chunker import CorpusChunk
from redacted_backend.corpus.manifest import CorpusManifest
from redacted_backend.db.base import Database
from redacted_backend.jobs.generation_store import AnalysisGenerationStore
from redacted_backend.jobs.orchestrator import AnalysisOrchestrator
from redacted_backend.models.openai_gateway import OpenAIGateway, OpenAIHttpTransport
from redacted_backend.retrieval.bm25 import Bm25Index, OpenSearchHttpTransport
from redacted_backend.retrieval.dense import BgeEncoder, DenseIndex
from redacted_backend.retrieval.service import RealisticRetrievalService
from redacted_backend.retrieval.types import Bm25IndexManifest, DenseIndexManifest
from redacted_backend.scientific_config import ScientificConfig


def build_analysis_orchestrator(settings: Settings) -> AnalysisOrchestrator:
    if settings.database_url is None:
        raise RuntimeError("Analysis worker database is not configured")
    if settings.opensearch_url is None:
        raise RuntimeError("Analysis worker OpenSearch is not configured")
    if settings.openai_api_key is None or not settings.allow_paid_model_calls:
        raise RuntimeError("Analysis worker model access is not approved")
    required_paths = {
        "corpus manifest": settings.corpus_manifest_path,
        "BM25 manifest": settings.bm25_manifest_path,
        "dense manifest": settings.dense_manifest_path,
        "dense artifact directory": settings.dense_artifact_directory,
    }
    missing = [name for name, path in required_paths.items() if path is None]
    if missing:
        raise RuntimeError(f"Analysis worker is missing {', '.join(missing)}")

    scientific = ScientificConfig()
    corpus = CorpusManifest.model_validate_json(
        settings.corpus_manifest_path.read_text(encoding="utf-8")
    )
    bm25_manifest = Bm25IndexManifest.model_validate_json(
        settings.bm25_manifest_path.read_text(encoding="utf-8")
    )
    dense_manifest = DenseIndexManifest.model_validate_json(
        settings.dense_manifest_path.read_text(encoding="utf-8")
    )
    bm25 = Bm25Index(
        OpenSearchHttpTransport(settings.opensearch_url),
        scientific=scientific,
        index_manifest=bm25_manifest,
        corpus_manifest=corpus,
    )
    dense = _load_dense_index(
        settings.dense_artifact_directory,
        dense_manifest,
        scientific,
    )
    bm25.search("worker readiness probe", limit=1, query_kind="context")
    dense.search("worker readiness probe", limit=1, query_kind="context")
    database = Database.connect(settings.database_url)
    gateway = OpenAIGateway(
        transport=OpenAIHttpTransport(
            settings.openai_api_key.get_secret_value(),
        ),
        store=AnalysisGenerationStore(database),
        scientific=scientific,
    )
    return AnalysisOrchestrator(
        database=database,
        model=gateway,
        retrieval=RealisticRetrievalService(
            bm25=bm25,
            dense=dense,
            scientific=scientific,
        ),
        scientific=scientific,
    )


def _load_dense_index(
    directory: Path,
    manifest: DenseIndexManifest,
    scientific: ScientificConfig,
) -> DenseIndex:
    paths = {
        "embeddings": directory / "embeddings.f16",
        "metadata": directory / "metadata.jsonl",
        "candidate": directory / "candidate.faiss",
    }
    expected_hashes = {
        "embeddings": manifest.embeddings_sha256,
        "metadata": manifest.metadata_sha256,
        "candidate": manifest.candidate_index_sha256,
    }
    for name, path in paths.items():
        if not path.is_file():
            raise RuntimeError(f"Dense {name} artifact is missing")
        if _sha256_file(path) != expected_hashes[name]:
            raise RuntimeError(f"Dense {name} artifact checksum mismatch")
    chunks: list[CorpusChunk] = []
    with paths["metadata"].open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                chunks.append(CorpusChunk.model_validate_json(line))
            except ValueError as error:
                raise RuntimeError(
                    f"Dense metadata is invalid at line {line_number}"
                ) from error
    if len(chunks) != manifest.chunk_count:
        raise RuntimeError("Dense metadata count does not match its manifest")
    embeddings = np.memmap(
        paths["embeddings"],
        dtype=np.float16,
        mode="r",
        shape=(manifest.chunk_count, manifest.embedding_dimension),
    )
    candidate = FaissHnswIndex.load(
        paths["candidate"],
        ef_search=manifest.ef_search,
    )
    return DenseIndex(
        chunks=chunks,
        embeddings=embeddings,
        candidate_index=candidate,
        encoder=BgeEncoder(scientific=scientific),
        manifest=manifest,
        corpus_manifest_hash=manifest.corpus_manifest_hash,
        scientific=scientific,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()
