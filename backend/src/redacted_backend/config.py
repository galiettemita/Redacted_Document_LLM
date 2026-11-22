from pathlib import Path
from typing import Any

from pydantic import Field, SecretStr, field_validator
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
    best_available_corpus_manifest_path: Path | None = None
    best_available_bm25_manifest_path: Path | None = None
    best_available_dense_manifest_path: Path | None = None
    best_available_dense_artifact_directory: Path | None = None
    best_available_chunks_path: Path | None = None
    best_available_recall_queries_path: Path | None = None
    best_available_latency_report_path: Path | None = None
    opensearch_url: str | None = None
    openai_api_key: SecretStr | None = None
    allow_paid_model_calls: bool = False
    allow_model_only_calls: bool = False
    allow_best_available_calls: bool = False
    allow_paper_exact_calls: bool = False
    worker_heartbeat_key: str = "redacted:analysis-worker:heartbeat"
    worker_heartbeat_ttl_seconds: int = 30
    document_storage_root: Path = Path("/var/lib/redacted/documents")
    document_retention_hours: int = Field(default=24, ge=1, le=24)
    document_retention_sweep_seconds: int = Field(default=300, ge=10, le=3_600)
    pdf_max_file_bytes: int = Field(
        default=100 * 1024 * 1024,
        ge=1,
        le=100 * 1024 * 1024,
    )
    pdf_max_pages: int = Field(default=500, ge=1, le=500)
    pdf_max_decompressed_bytes: int = Field(
        default=512 * 1024 * 1024,
        ge=1,
        le=512 * 1024 * 1024,
    )
    pdf_max_rendered_page_pixels: int = Field(
        default=25_000_000,
        ge=1,
        le=25_000_000,
    )
    pdf_parser_timeout_seconds: float = Field(default=10.0, gt=0, le=10.0)

    @field_validator("openai_api_key", mode="before")
    @classmethod
    def empty_model_key_is_unconfigured(cls, value: Any) -> Any:
        if isinstance(value, str) and not value.strip():
            return None
        return value
