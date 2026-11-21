from collections.abc import Callable
from typing import Protocol

from fastapi import APIRouter
from pydantic import ValidationError
from redis import Redis
from redis.exceptions import RedisError
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from redacted_backend.config import Settings
from redacted_backend.corpus.manifest import CorpusManifest
from redacted_backend.db.base import Database
from redacted_backend.domain.analysis import PredictionBasis
from redacted_backend.domain.readiness import (
    BasisReadiness,
    ComponentReadiness,
    ReadinessResponse,
    verify_corpus_readiness,
)
from redacted_backend.retrieval.bm25 import OpenSearchHttpTransport
from redacted_backend.retrieval.types import Bm25IndexManifest, DenseIndexManifest
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
        model = resolve_model_readiness(settings)
        prediction_bases = resolve_prediction_basis_readiness(
            settings,
            database=database_readiness,
            worker=worker,
            corpus=corpus,
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
    master = settings.allow_paid_model_calls
    return {
        PredictionBasis.MODEL_ONLY_V1: _basis_readiness(
            operational=model_only_operational,
            approved=master and settings.allow_model_only_calls,
            unavailable_detail=model_only_detail,
        ),
        PredictionBasis.BEST_AVAILABLE_FOIARCHIVE_V1: BasisReadiness(
            operational=False,
            approved=master and settings.allow_best_available_calls,
            ready=False,
            detail="best-available retrieval is not implemented",
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
    if settings.redis_url is None:
        return ComponentReadiness(ready=False, detail="not configured")
    factory = client_factory or (lambda url: Redis.from_url(url))
    try:
        value = factory(settings.redis_url).get(settings.worker_heartbeat_key)
    except (RedisError, OSError, ValueError):
        return ComponentReadiness(ready=False, detail="heartbeat probe failed")
    if value != b"ready":
        return ComponentReadiness(
            ready=False,
            detail="heartbeat missing or expired",
        )
    return ComponentReadiness(ready=True, detail="heartbeat current")


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


def _count_opensearch_chunks(base_url: str, index_name: str) -> int:
    return OpenSearchHttpTransport(base_url, timeout_seconds=2.0).count(index_name)
