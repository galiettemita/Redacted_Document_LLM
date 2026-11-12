from pathlib import Path

from pydantic import SecretStr
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
    openai_api_key: SecretStr | None = None
