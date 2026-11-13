from __future__ import annotations

import dramatiq
from dramatiq.brokers.redis import RedisBroker
from dramatiq.middleware import CurrentMessage

from redacted_backend.config import Settings
from redacted_backend.db.repositories.analysis import AnalysisJobRepository
from redacted_backend.jobs.heartbeat import WorkerHeartbeatMiddleware
from redacted_backend.jobs.runtime import build_analysis_orchestrator
from redacted_backend.models.openai_gateway import ModelGatewayError
from redacted_backend.retrieval.service import RetrievalError

_settings = Settings()
if _settings.redis_url is None:
    raise RuntimeError("REDACTED_REDIS_URL is required for the analysis worker")
_broker = RedisBroker(url=_settings.redis_url)
_broker.add_middleware(CurrentMessage())
_broker.add_middleware(WorkerHeartbeatMiddleware(_settings))
dramatiq.set_broker(_broker)
_runtime = build_analysis_orchestrator(_settings)

_TRANSIENT_MODEL_ERRORS = {
    "provider_timeout",
    "provider_unavailable",
    "rate_limited",
}


@dramatiq.actor(max_retries=5, min_backoff=5_000, max_backoff=120_000)
def analyze_passage(job_id: str) -> None:
    try:
        _runtime.run(job_id)
    except ModelGatewayError as error:
        if error.code in _TRANSIENT_MODEL_ERRORS and not _last_attempt():
            raise
        _fail(job_id, error.code, str(error))
    except RetrievalError as error:
        _fail(job_id, error.code, str(error))
    except TimeoutError as error:
        if not _last_attempt():
            raise
        _fail(job_id, "provider_timeout", str(error))
    except Exception as error:
        if not _last_attempt():
            raise
        _fail(job_id, "analysis_worker_error", type(error).__name__)


def _last_attempt() -> bool:
    message = CurrentMessage.get_current_message()
    return bool(message and int(message.options.get("retries", 0)) >= 5)


def _fail(job_id: str, code: str, detail: str) -> None:
    with _runtime.database.session_factory() as session, session.begin():
        AnalysisJobRepository(session).fail(job_id, code=code, detail=detail)
