from fastapi import FastAPI

from redacted_backend.api.context import create_context_router
from redacted_backend.api.documents import create_documents_router
from redacted_backend.api.passages import create_passages_router
from redacted_backend.api.readiness import create_readiness_router
from redacted_backend.config import Settings
from redacted_backend.db.base import Database
from redacted_backend.storage.local import LocalDocumentStore


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved_settings = settings or Settings()
    app = FastAPI(title=resolved_settings.app_name)
    database = (
        Database.connect(resolved_settings.database_url)
        if resolved_settings.database_url is not None
        else None
    )
    document_store = (
        LocalDocumentStore(resolved_settings.document_storage_root)
        if database is not None
        else None
    )
    app.state.database = database
    app.state.document_store = document_store
    app.include_router(create_readiness_router(resolved_settings), prefix="/v1")
    app.include_router(
        create_documents_router(resolved_settings, database, document_store),
        prefix="/v1",
    )
    app.include_router(create_passages_router(database), prefix="/v1")
    app.include_router(
        create_context_router(database, document_store),
        prefix="/v1",
    )
    return app


app = create_app()
