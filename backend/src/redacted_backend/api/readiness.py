from fastapi import APIRouter

from redacted_backend.config import Settings
from redacted_backend.domain.readiness import ComponentReadiness, ReadinessResponse


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
        corpus = ComponentReadiness(
            ready=False,
            detail=(
                "not configured"
                if settings.corpus_manifest_path is None
                else "manifest not verified"
            ),
        )
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
