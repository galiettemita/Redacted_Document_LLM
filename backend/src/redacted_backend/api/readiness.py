import hashlib
from collections.abc import Callable
from typing import Protocol

from fastapi import APIRouter
from pydantic import ValidationError
from redis import Redis
from redis.exceptions import RedisError
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from redacted_backend.config import Settings
from redacted_backend.corpus.best_available import (
    PRODUCTION_BEST_AVAILABLE_RETRIEVAL_CONTRACT,
    BestAvailableCorpusManifest,
    BestAvailableRetrievalContract,
)
from redacted_backend.corpus.manifest import CorpusManifest
from redacted_backend.db.base import Database
from redacted_backend.domain.analysis import PredictionBasis
from redacted_backend.domain.readiness import (
    BasisReadiness,
    ComponentReadiness,
    ReadinessResponse,
    verify_best_available_corpus_readiness,
    verify_corpus_readiness,
)
from redacted_backend.retrieval.bm25 import OpenSearchHttpTransport
from redacted_backend.retrieval.dense_sharded import (
    verify_compressed_dense_artifacts_cached,
)
from redacted_backend.retrieval.types import (
    Bm25IndexManifest,
    CompressedShardedDenseIndexManifest,
    DenseLatencyReport,
    DenseIndexManifest,
)
from redacted_backend.runtime_identity import WorkerHeartbeat, WorkerRuntimeIdentity
from redacted_backend.scientific_config import ScientificConfig


class HeartbeatClient(Protocol):
    def get(self, key: str) -> bytes | None: ...


def create_readiness_router(
    settings: Settings,
    database: Database | None,
) -> APIRouter:
    router = APIRouter()

    @router.get("/health", response_model=ComponentReadiness)
    def health() -> ComponentReadiness:
        return ComponentReadiness(ready=True, detail="ready")

    @router.get("/readiness", response_model=ReadinessResponse)
    def readiness() -> ReadinessResponse:
        database_readiness = resolve_database_readiness(database)
        worker = resolve_worker_readiness(settings)
        corpus = resolve_corpus_readiness(settings)
        best_available_corpus = resolve_best_available_corpus_readiness(settings)
        model = resolve_model_readiness(settings)
        prediction_bases = resolve_prediction_basis_readiness(
            settings,
            database=database_readiness,
            worker=worker,
            corpus=corpus,
            best_available_corpus=best_available_corpus,
        )
        strongest = select_prediction_basis(prediction_bases)
        return ReadinessResponse(
            http=ComponentReadiness(ready=True, detail="ready"),
            database=database_readiness,
            worker=worker,
            corpus=corpus,
            model=model,
            prediction_ready=strongest is not None,
            prediction_bases=prediction_bases,
            strongest_ready_basis=strongest,
        )

    return router


def resolve_model_readiness(settings: Settings) -> ComponentReadiness:
    return ComponentReadiness(
        ready=settings.openai_api_key is not None and settings.allow_paid_model_calls,
        detail=(
            "configured and paid calls approved"
            if settings.openai_api_key is not None and settings.allow_paid_model_calls
            else "configured; paid calls not approved"
            if settings.openai_api_key is not None
            else "not configured"
        ),
    )


def resolve_prediction_basis_readiness(
    settings: Settings,
    *,
    database: ComponentReadiness,
    worker: ComponentReadiness,
    corpus: ComponentReadiness,
    best_available_corpus: ComponentReadiness | None = None,
) -> dict[PredictionBasis, BasisReadiness]:
    model_only_operational, model_only_detail = _operational(
        ("database", database),
        ("worker", worker),
        model_configured=settings.openai_api_key is not None,
    )
    exact_operational, exact_detail = _operational(
        ("database", database),
        ("worker", worker),
        ("corpus", corpus),
        model_configured=settings.openai_api_key is not None,
    )
    best_operational, best_detail = _operational(
        ("database", database),
        ("worker", worker),
        (
            "corpus",
            best_available_corpus
            or ComponentReadiness(ready=False, detail="not configured"),
        ),
        model_configured=settings.openai_api_key is not None,
    )
    master = settings.allow_paid_model_calls
    return {
        PredictionBasis.MODEL_ONLY_V1: _basis_readiness(
            operational=model_only_operational,
            approved=master and settings.allow_model_only_calls,
            unavailable_detail=model_only_detail,
        ),
        PredictionBasis.BEST_AVAILABLE_FOIARCHIVE_V1: _basis_readiness(
            operational=best_operational,
            approved=master and settings.allow_best_available_calls,
            unavailable_detail=best_detail,
        ),
        PredictionBasis.PAPER_EXACT_V1: _basis_readiness(
            operational=exact_operational,
            approved=master and settings.allow_paper_exact_calls,
            unavailable_detail=exact_detail,
        ),
    }


def select_prediction_basis(
    readiness: dict[PredictionBasis, BasisReadiness],
) -> PredictionBasis | None:
    for basis in (
        PredictionBasis.PAPER_EXACT_V1,
        PredictionBasis.BEST_AVAILABLE_FOIARCHIVE_V1,
        PredictionBasis.MODEL_ONLY_V1,
    ):
        if readiness[basis].ready:
            return basis
    return None


def _operational(
    *components: tuple[str, ComponentReadiness],
    model_configured: bool,
) -> tuple[bool, str]:
    for name, component in components:
        if not component.ready:
            return False, f"{name}: {component.detail}"
    if not model_configured:
        return False, "model: not configured"
    return True, "operational"


def _basis_readiness(
    *,
    operational: bool,
    approved: bool,
    unavailable_detail: str,
) -> BasisReadiness:
    ready = operational and approved
    return BasisReadiness(
        operational=operational,
        approved=approved,
        ready=ready,
        detail=(
            "operational and approved"
            if ready
            else "operational; spending not approved"
            if operational
            else unavailable_detail
        ),
    )


def resolve_worker_readiness(
    settings: Settings,
    *,
    client_factory: Callable[[str], HeartbeatClient] | None = None,
) -> ComponentReadiness:
    heartbeat, detail = _read_worker_heartbeat(
        settings,
        client_factory=client_factory,
    )
    if heartbeat is None:
        if detail == "not configured":
            return ComponentReadiness(ready=False, detail=detail)
        if detail == "heartbeat probe failed":
            return ComponentReadiness(ready=False, detail=detail)
        return ComponentReadiness(
            ready=False,
            detail="heartbeat missing or expired",
        )
    return ComponentReadiness(ready=True, detail="heartbeat current")


def _read_worker_heartbeat(
    settings: Settings,
    *,
    client_factory: Callable[[str], HeartbeatClient] | None = None,
) -> tuple[WorkerHeartbeat | None, str]:
    if settings.redis_url is None:
        return None, "not configured"
    factory = client_factory or (lambda url: Redis.from_url(url))
    try:
        value = factory(settings.redis_url).get(settings.worker_heartbeat_key)
    except (RedisError, OSError, ValueError):
        return None, "heartbeat probe failed"
    if value is None:
        return None, "heartbeat missing or expired"
    try:
        heartbeat = WorkerHeartbeat.model_validate_json(value)
    except (ValidationError, ValueError):
        return None, "heartbeat missing or expired"
    return heartbeat, "heartbeat current"


def resolve_live_worker_identity(
    settings: Settings,
    *,
    client_factory: Callable[[str], HeartbeatClient] | None = None,
) -> WorkerRuntimeIdentity | None:
    heartbeat, _detail = _read_worker_heartbeat(
        settings,
        client_factory=client_factory,
    )
    return heartbeat.identity if heartbeat is not None else None


def resolve_database_readiness(database: Database | None) -> ComponentReadiness:
    if database is None:
        return ComponentReadiness(ready=False, detail="not configured")
    try:
        with database.engine.connect() as connection:
            connection.execute(text("SELECT 1")).scalar_one()
    except SQLAlchemyError:
        return ComponentReadiness(ready=False, detail="probe failed")
    return ComponentReadiness(ready=True, detail="ready")


def resolve_corpus_readiness(
    settings: Settings,
    *,
    index_counter: Callable[[str, str], int] | None = None,
) -> ComponentReadiness:
    if (
        settings.corpus_manifest_path is None
        or settings.bm25_manifest_path is None
        or settings.dense_manifest_path is None
    ):
        return ComponentReadiness(ready=False, detail="not configured")
    if (
        not settings.corpus_manifest_path.is_file()
        or not settings.bm25_manifest_path.is_file()
        or not settings.dense_manifest_path.is_file()
    ):
        return ComponentReadiness(ready=False, detail="manifest not verified")
    try:
        corpus = CorpusManifest.model_validate_json(
            settings.corpus_manifest_path.read_text(encoding="utf-8")
        )
        index = Bm25IndexManifest.model_validate_json(
            settings.bm25_manifest_path.read_text(encoding="utf-8")
        )
        dense = DenseIndexManifest.model_validate_json(
            settings.dense_manifest_path.read_text(encoding="utf-8")
        )
    except (OSError, ValidationError, ValueError):
        return ComponentReadiness(ready=False, detail="manifest invalid")
    manifest_readiness = verify_corpus_readiness(
        corpus,
        index,
        dense,
        ScientificConfig(),
    )
    if not manifest_readiness.ready:
        return manifest_readiness
    if settings.dense_artifact_directory is None or any(
        not (settings.dense_artifact_directory / name).is_file()
        for name in ("embeddings.f16", "metadata.jsonl", "candidate.faiss")
    ):
        return ComponentReadiness(
            ready=False,
            detail="dense artifacts not available",
        )
    if settings.opensearch_url is None:
        return ComponentReadiness(
            ready=False,
            detail="retrieval service not configured",
        )
    counter = index_counter or _count_opensearch_chunks
    try:
        live_count = counter(settings.opensearch_url, index.index_name)
    except Exception:
        return ComponentReadiness(
            ready=False,
            detail="retrieval index probe failed",
        )
    if live_count != index.chunk_count:
        return ComponentReadiness(
            ready=False,
            detail="BM25 live index count mismatch",
        )
    return ComponentReadiness(
        ready=True,
        detail="corpus and retrieval indexes operational",
    )


def resolve_best_available_corpus_readiness(
    settings: Settings,
    *,
    index_counter: Callable[[str, str], int] | None = None,
    contract: BestAvailableRetrievalContract = (
        PRODUCTION_BEST_AVAILABLE_RETRIEVAL_CONTRACT
    ),
    worker_identity: WorkerRuntimeIdentity | None = None,
) -> ComponentReadiness:
    required = (
        settings.best_available_corpus_manifest_path,
        settings.best_available_bm25_manifest_path,
        settings.best_available_dense_manifest_path,
        settings.best_available_dense_artifact_directory,
        settings.best_available_chunks_path,
        settings.best_available_recall_queries_path,
        settings.best_available_latency_report_path,
    )
    if any(path is None for path in required):
        return ComponentReadiness(ready=False, detail="not configured")
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
    if not all(path.is_file() for path in (corpus_path, bm25_path, dense_path)):
        return ComponentReadiness(ready=False, detail="manifest not verified")
    try:
        corpus = BestAvailableCorpusManifest.model_validate_json(
            corpus_path.read_text(encoding="utf-8")
        )
        index = Bm25IndexManifest.model_validate_json(
            bm25_path.read_text(encoding="utf-8")
        )
        dense = CompressedShardedDenseIndexManifest.model_validate_json(
            dense_path.read_text(encoding="utf-8")
        )
    except (OSError, ValidationError, ValueError):
        return ComponentReadiness(ready=False, detail="manifest invalid")
    manifest_readiness = verify_best_available_corpus_readiness(
        corpus,
        index,
        dense,
        ScientificConfig(),
        contract=contract,
    )
    if not manifest_readiness.ready:
        return manifest_readiness
    if not queries_path.is_file():
        return ComponentReadiness(
            ready=False,
            detail="dense recall queries not available",
        )
    try:
        query_bytes = queries_path.read_bytes()
        query_count = len(query_bytes.decode("utf-8").splitlines())
    except (OSError, UnicodeError):
        return ComponentReadiness(
            ready=False,
            detail="dense recall queries not available",
        )
    if (
        hashlib.sha256(query_bytes).hexdigest() != dense.recall_queries_sha256
        or query_count != dense.recall_query_count
    ):
        return ComponentReadiness(
            ready=False,
            detail="dense recall query artifact mismatch",
        )
    if not latency_path.is_file():
        return ComponentReadiness(
            ready=False,
            detail="dense latency report not available",
        )
    if worker_identity is None:
        worker_identity = resolve_live_worker_identity(settings)
    if worker_identity is None:
        return ComponentReadiness(
            ready=False,
            detail="serving worker identity unavailable",
        )
    try:
        latency = DenseLatencyReport.model_validate_json(
            latency_path.read_text(encoding="utf-8")
        )
        latency.require_runtime_compatible(
            dense,
            runtime_environment=contract.latency_runtime_environment,
            platform=worker_identity.platform,
            machine=worker_identity.machine,
            worker_build_id=worker_identity.worker_build_id,
            serving_host_id=worker_identity.serving_host_id,
        )
    except (OSError, ValidationError, ValueError):
        return ComponentReadiness(
            ready=False,
            detail="dense latency report invalid",
        )
    if not chunks_path.is_file():
        return ComponentReadiness(
            ready=False,
            detail="dense chunks not available",
        )
    try:
        verify_compressed_dense_artifacts_cached(
            dense_root,
            chunks_path,
            dense,
        )
    except (OSError, ValueError):
        return ComponentReadiness(
            ready=False,
            detail="dense shard artifacts invalid",
        )
    if settings.opensearch_url is None:
        return ComponentReadiness(
            ready=False,
            detail="retrieval service not configured",
        )
    counter = index_counter or _count_opensearch_chunks
    try:
        live_count = counter(settings.opensearch_url, index.index_name)
    except Exception:
        return ComponentReadiness(
            ready=False,
            detail="retrieval index probe failed",
        )
    if live_count != index.chunk_count or live_count != contract.chunk_count:
        return ComponentReadiness(
            ready=False,
            detail="BM25 live index count mismatch",
        )
    return ComponentReadiness(
        ready=True,
        detail="best-available corpus and retrieval indexes operational",
    )


def _count_opensearch_chunks(base_url: str, index_name: str) -> int:
    return OpenSearchHttpTransport(base_url, timeout_seconds=2.0).count(index_name)
