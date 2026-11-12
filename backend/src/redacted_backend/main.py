from fastapi import FastAPI

from redacted_backend.api.readiness import create_readiness_router
from redacted_backend.config import Settings


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved_settings = settings or Settings()
    app = FastAPI(title=resolved_settings.app_name)
    app.include_router(create_readiness_router(resolved_settings), prefix="/v1")
    return app


app = create_app()
