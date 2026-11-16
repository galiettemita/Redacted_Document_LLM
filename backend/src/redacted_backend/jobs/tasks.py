from __future__ import annotations

import dramatiq
from dramatiq.brokers.redis import RedisBroker
from dramatiq.middleware import CurrentMessage

from redacted_backend.config import Settings
from redacted_backend.db.base import Database
from redacted_backend.db.repositories.analysis import AnalysisJobRepository
from redacted_backend.jobs.execution import execute_analysis_job
from redacted_backend.jobs.heartbeat import WorkerHeartbeatMiddleware
from redacted_backend.jobs.runtime import build_analysis_orchestrator
from redacted_backend.jobs.runtime_provider import AnalysisRuntimeProvider
from redacted_backend.observability import install_log_redaction

install_log_redaction()
_settings = Settings()
if _settings.redis_url is None:
    raise RuntimeError("REDACTED_REDIS_URL is required for the analysis worker")
if _settings.database_url is None:
    raise RuntimeError("REDACTED_DATABASE_URL is required for the analysis worker")
_broker = RedisBroker(url=_settings.redis_url)
_broker.add_middleware(CurrentMessage())
_broker.add_middleware(WorkerHeartbeatMiddleware(_settings))
dramatiq.set_broker(_broker)
_database = Database.connect(_settings.database_url)
_runtime_provider = AnalysisRuntimeProvider(
    lambda: build_analysis_orchestrator(_settings, database=_database)
)


@dramatiq.actor(max_retries=5, min_backoff=5_000, max_backoff=120_000)
def analyze_passage(job_id: str) -> None:
    execute_analysis_job(
        job_id,
        _runtime_provider,
        last_attempt=_last_attempt(),
        fail=_fail,
    )


def _last_attempt() -> bool:
    message = CurrentMessage.get_current_message()
    return bool(message and int(message.options.get("retries", 0)) >= 5)


def _fail(job_id: str, code: str, detail: str) -> None:
    with _database.session_factory() as session, session.begin():
        AnalysisJobRepository(session).fail(job_id, code=code, detail=detail)
