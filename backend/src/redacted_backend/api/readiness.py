from fastapi import APIRouter
from pydantic import ValidationError

from redacted_backend.config import Settings
from redacted_backend.corpus.manifest import CorpusManifest
from redacted_backend.domain.readiness import (
    ComponentReadiness,
    ReadinessResponse,
    verify_corpus_readiness,
)
from redacted_backend.retrieval.types import Bm25IndexManifest
from redacted_backend.scientific_config import ScientificConfig


def create_readiness_router(settings: Settings) -> APIRouter:
    router = APIRouter()

    @router.get("/readiness", response_model=ReadinessResponse)
    def readiness() -> ReadinessResponse:
        database = ComponentReadiness(
            ready=False,
            detail=(
                "not configured"
                if settings.database_url is None
                else "probe not connected"
            ),
        )
        worker = ComponentReadiness(
            ready=False,
            detail=(
                "not configured"
                if settings.redis_url is None
                else "heartbeat not connected"
            ),
        )
        corpus = _corpus_readiness(settings)
        model = ComponentReadiness(
            ready=settings.openai_api_key is not None,
            detail=(
                "configured"
                if settings.openai_api_key is not None
                else "not configured"
            ),
        )
        components = (database, worker, corpus, model)
        return ReadinessResponse(
            http=ComponentReadiness(ready=True, detail="ready"),
            database=database,
            worker=worker,
            corpus=corpus,
            model=model,
            prediction_ready=all(component.ready for component in components),
        )

    return router


def _corpus_readiness(settings: Settings) -> ComponentReadiness:
    if settings.corpus_manifest_path is None or settings.bm25_manifest_path is None:
        return ComponentReadiness(ready=False, detail="not configured")
    if (
        not settings.corpus_manifest_path.is_file()
        or not settings.bm25_manifest_path.is_file()
    ):
        return ComponentReadiness(ready=False, detail="manifest not verified")
    try:
        corpus = CorpusManifest.model_validate_json(
            settings.corpus_manifest_path.read_text(encoding="utf-8")
        )
        index = Bm25IndexManifest.model_validate_json(
            settings.bm25_manifest_path.read_text(encoding="utf-8")
        )
    except (OSError, ValidationError, ValueError):
        return ComponentReadiness(ready=False, detail="manifest invalid")
    return verify_corpus_readiness(corpus, index, ScientificConfig())
