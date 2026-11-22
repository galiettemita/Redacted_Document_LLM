from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from redacted_backend.config import Settings
from redacted_backend.corpus.best_available import (
    BEST_AVAILABLE_BM25_INDEX_NAME,
    PRODUCTION_BEST_AVAILABLE_RETRIEVAL_CONTRACT,
    BestAvailableCorpusManifest,
    BestAvailableRetrievalContract,
)
from redacted_backend.corpus.build_dense import FaissHnswIndex
from redacted_backend.corpus.chunker import CorpusChunk
from redacted_backend.corpus.manifest import CorpusManifest
from redacted_backend.db.base import Database
from redacted_backend.domain.analysis import PredictionBasis
from redacted_backend.domain.readiness import (
    verify_best_available_corpus_readiness,
)
from redacted_backend.jobs.generation_store import AnalysisGenerationStore
from redacted_backend.jobs.orchestrator import AnalysisOrchestrator
from redacted_backend.models.openai_gateway import OpenAIGateway, OpenAIHttpTransport
from redacted_backend.retrieval.bm25 import Bm25Index, OpenSearchHttpTransport
from redacted_backend.retrieval.dense import BgeEncoder, DenseIndex
from redacted_backend.retrieval.dense_sharded import (
    ShardedDenseIndex,
    verify_compressed_dense_artifacts,
)
from redacted_backend.retrieval.faiss_ivfpq import FaissIvfPqIndex
from redacted_backend.retrieval.service import RealisticRetrievalService
from redacted_backend.retrieval.types import (
    Bm25IndexManifest,
    CompressedShardedDenseIndexManifest,
    DenseLatencyReport,
    DenseIndexManifest,
)
from redacted_backend.scientific_config import ScientificConfig


def build_analysis_orchestrator(
    settings: Settings,
    prediction_basis: PredictionBasis,
    *,
    database: Database | None = None,
    best_available_contract: BestAvailableRetrievalContract = (
        PRODUCTION_BEST_AVAILABLE_RETRIEVAL_CONTRACT
    ),
) -> AnalysisOrchestrator:
    if settings.database_url is None and database is None:
        raise RuntimeError("Analysis worker database is not configured")
    if settings.openai_api_key is None or not settings.allow_paid_model_calls:
        raise RuntimeError("Analysis worker model access is not approved")
    if database is None:
        assert settings.database_url is not None
        database = Database.connect(settings.database_url)
    scientific = ScientificConfig()
    gateway = OpenAIGateway(
        transport=OpenAIHttpTransport(
            settings.openai_api_key.get_secret_value(),
        ),
        store=AnalysisGenerationStore(database),
        scientific=scientific,
    )
    if prediction_basis == PredictionBasis.MODEL_ONLY_V1:
        if not settings.allow_model_only_calls:
            raise RuntimeError("Analysis worker model-only access is not approved")
        return AnalysisOrchestrator(
            database=database,
            model=gateway,
            retrieval=None,
            prediction_basis=prediction_basis,
            scientific=scientific,
        )
    if prediction_basis == PredictionBasis.BEST_AVAILABLE_FOIARCHIVE_V1:
        if not settings.allow_best_available_calls:
            raise RuntimeError("Analysis worker best-available access is not approved")
        return _build_best_available_orchestrator(
            settings,
            database=database,
            gateway=gateway,
            scientific=scientific,
            contract=best_available_contract,
        )
    if not settings.allow_paper_exact_calls:
        raise RuntimeError("Analysis worker paper-exact access is not approved")
    if settings.opensearch_url is None:
        raise RuntimeError("Analysis worker OpenSearch is not configured")
    required_paths = {
        "corpus manifest": settings.corpus_manifest_path,
        "BM25 manifest": settings.bm25_manifest_path,
        "dense manifest": settings.dense_manifest_path,
        "dense artifact directory": settings.dense_artifact_directory,
    }
    missing = [name for name, path in required_paths.items() if path is None]
    if missing:
        raise RuntimeError(f"Analysis worker is missing {', '.join(missing)}")

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
    return AnalysisOrchestrator(
        database=database,
        model=gateway,
        retrieval=RealisticRetrievalService(
            bm25=bm25,
            dense=dense,
            scientific=scientific,
        ),
        prediction_basis=prediction_basis,
        scientific=scientific,
    )


def _build_best_available_orchestrator(
    settings: Settings,
    *,
    database: Database,
    gateway: OpenAIGateway,
    scientific: ScientificConfig,
    contract: BestAvailableRetrievalContract,
) -> AnalysisOrchestrator:
    required_paths = {
        "best-available corpus manifest": (
            settings.best_available_corpus_manifest_path
        ),
        "best-available BM25 manifest": settings.best_available_bm25_manifest_path,
        "best-available dense manifest": settings.best_available_dense_manifest_path,
        "best-available dense artifact directory": (
            settings.best_available_dense_artifact_directory
        ),
        "best-available chunks": settings.best_available_chunks_path,
        "best-available recall queries": settings.best_available_recall_queries_path,
        "best-available latency report": (
            settings.best_available_latency_report_path
        ),
    }
    missing = [name for name, path in required_paths.items() if path is None]
    if missing:
        raise RuntimeError(f"Analysis worker is missing {', '.join(missing)}")
    if settings.opensearch_url is None:
        raise RuntimeError("Analysis worker OpenSearch is not configured")
    corpus_path = settings.best_available_corpus_manifest_path
    bm25_path = settings.best_available_bm25_manifest_path
    dense_path = settings.best_available_dense_manifest_path
    dense_root = settings.best_available_dense_artifact_directory
    chunks_path = settings.best_available_chunks_path
    queries_path = settings.best_available_recall_queries_path
    latency_path = settings.best_available_latency_report_path
    assert corpus_path is not None
    assert bm25_path is not None
    assert dense_path is not None
    assert dense_root is not None
    assert chunks_path is not None
    assert queries_path is not None
    assert latency_path is not None

    corpus = BestAvailableCorpusManifest.model_validate_json(
        corpus_path.read_text(encoding="utf-8")
    )
    bm25_manifest = Bm25IndexManifest.model_validate_json(
        bm25_path.read_text(encoding="utf-8")
    )
    dense_manifest = CompressedShardedDenseIndexManifest.model_validate_json(
        dense_path.read_text(encoding="utf-8")
    )
    manifest_readiness = verify_best_available_corpus_readiness(
        corpus,
        bm25_manifest,
        dense_manifest,
        scientific,
        contract=contract,
    )
    if not manifest_readiness.ready:
        raise RuntimeError(manifest_readiness.detail)
    try:
        query_bytes = queries_path.read_bytes()
        query_count = len(query_bytes.decode("utf-8").splitlines())
    except (OSError, UnicodeError) as error:
        raise RuntimeError("Dense recall query artifact is unavailable") from error
    if (
        hashlib.sha256(query_bytes).hexdigest()
        != dense_manifest.recall_queries_sha256
        or query_count != dense_manifest.recall_query_count
    ):
        raise RuntimeError("Dense recall query artifact mismatch")
    try:
        latency = DenseLatencyReport.model_validate_json(
            latency_path.read_text(encoding="utf-8")
        )
        latency.require_compatible(dense_manifest)
        if (
            latency.runtime_environment
            != contract.latency_runtime_environment
        ):
            raise ValueError("Dense latency runtime environment mismatch")
    except (OSError, ValueError) as error:
        raise RuntimeError("Dense latency report mismatch") from error
    try:
        verify_compressed_dense_artifacts(
            dense_root,
            chunks_path,
            dense_manifest,
        )
    except (OSError, ValueError) as error:
        raise RuntimeError("Dense artifacts failed verification") from error

    transport = OpenSearchHttpTransport(settings.opensearch_url)
    if transport.count(bm25_manifest.index_name) != bm25_manifest.chunk_count:
        raise RuntimeError("BM25 live index count mismatch")
    bm25 = Bm25Index(
        transport,
        scientific=scientific,
        index_manifest=bm25_manifest,
        corpus_manifest=corpus,
        index_name=BEST_AVAILABLE_BM25_INDEX_NAME,
    )
    dense = ShardedDenseIndex(
        root=dense_root,
        chunks_path=chunks_path,
        manifest=dense_manifest,
        encoder=BgeEncoder(scientific=scientific),
        candidate_loader=FaissIvfPqIndex.load,
        scientific=scientific,
    )
    bm25.search("worker readiness probe", limit=1, query_kind="context")
    dense.search("worker readiness probe", limit=1, query_kind="context")
    return AnalysisOrchestrator(
        database=database,
        model=gateway,
        retrieval=RealisticRetrievalService(
            bm25=bm25,
            dense=dense,
            scientific=scientific,
        ),
        prediction_basis=PredictionBasis.BEST_AVAILABLE_FOIARCHIVE_V1,
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
