from pathlib import Path
from typing import Any

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="REDACTED_",
        extra="ignore",
    )

    app_name: str = "Redacted Document LLM Backend"
    environment: str = "development"
    database_url: str | None = None
    redis_url: str | None = None
    corpus_manifest_path: Path | None = None
    bm25_manifest_path: Path | None = None
    dense_manifest_path: Path | None = None
    dense_artifact_directory: Path | None = None
    opensearch_url: str | None = None
    openai_api_key: SecretStr | None = None
    allow_paid_model_calls: bool = False
    worker_heartbeat_key: str = "redacted:analysis-worker:heartbeat"
    worker_heartbeat_ttl_seconds: int = 30
    document_storage_root: Path = Path("/var/lib/redacted/documents")
    document_retention_hours: int = 24
    pdf_max_file_bytes: int = 100 * 1024 * 1024
    pdf_max_pages: int = 500
    pdf_max_decompressed_bytes: int = 512 * 1024 * 1024
    pdf_max_rendered_page_pixels: int = 25_000_000
    pdf_parser_timeout_seconds: float = 10.0

    @field_validator("openai_api_key", mode="before")
    @classmethod
    def empty_model_key_is_unconfigured(cls, value: Any) -> Any:
        if isinstance(value, str) and not value.strip():
            return None
        return value
