from __future__ import annotations

from collections.abc import Callable

from redacted_backend.jobs.runtime_provider import (
    AnalysisRuntimeProvider,
    RuntimeInitializationError,
)
from redacted_backend.models.openai_gateway import ModelGatewayError
from redacted_backend.retrieval.service import RetrievalError

FailureCallback = Callable[[str, str, str], None]

_TRANSIENT_MODEL_ERRORS = {
    "provider_timeout",
    "provider_unavailable",
    "rate_limited",
}


def execute_analysis_job(
    job_id: str,
    provider: AnalysisRuntimeProvider,
    *,
    last_attempt: bool,
    fail: FailureCallback,
) -> None:
    try:
        provider.get().run(job_id)
    except RuntimeInitializationError as error:
        if not last_attempt:
            raise
        fail(job_id, "analysis_runtime_unavailable", error.cause_name)
    except ModelGatewayError as error:
        if error.code in _TRANSIENT_MODEL_ERRORS and not last_attempt:
            raise
        fail(job_id, error.code, str(error))
    except RetrievalError as error:
        fail(job_id, error.code, str(error))
    except TimeoutError as error:
        if not last_attempt:
            raise
        fail(job_id, "provider_timeout", str(error))
    except Exception as error:
        if not last_attempt:
            raise
        fail(job_id, "analysis_worker_error", type(error).__name__)
