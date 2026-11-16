from __future__ import annotations

from typing import cast

import pytest

from redacted_backend.jobs.execution import execute_analysis_job
from redacted_backend.jobs.orchestrator import AnalysisOrchestrator
from redacted_backend.jobs.runtime_provider import AnalysisRuntimeProvider
from redacted_backend.models.openai_gateway import ModelGatewayError
from redacted_backend.retrieval.service import RetrievalError


class RaisingRuntime:
    def __init__(self, error: Exception) -> None:
        self.error = error

    def run(self, _job_id: str) -> None:
        raise self.error


def _provider_for(error: Exception) -> AnalysisRuntimeProvider:
    runtime = cast(AnalysisOrchestrator, RaisingRuntime(error))
    return AnalysisRuntimeProvider(lambda: runtime)


def test_non_final_runtime_initialization_failure_is_retried() -> None:
    failures: list[tuple[str, str, str]] = []

    def build() -> AnalysisOrchestrator:
        raise RuntimeError("secret-provider-token")

    with pytest.raises(RuntimeError, match="RuntimeError"):
        execute_analysis_job(
            "job-1",
            AnalysisRuntimeProvider(build),
            last_attempt=False,
            fail=lambda *value: failures.append(value),
        )

    assert failures == []


def test_final_runtime_initialization_failure_is_sanitized() -> None:
    failures: list[tuple[str, str, str]] = []

    def build() -> AnalysisOrchestrator:
        raise RuntimeError("secret-provider-token")

    execute_analysis_job(
        "job-2",
        AnalysisRuntimeProvider(build),
        last_attempt=True,
        fail=lambda *value: failures.append(value),
    )

    assert failures == [
        ("job-2", "analysis_runtime_unavailable", "RuntimeError")
    ]
    assert "secret-provider-token" not in repr(failures)


def test_retrieval_failure_after_initialization_retains_its_code() -> None:
    failures: list[tuple[str, str, str]] = []

    execute_analysis_job(
        "job-3",
        _provider_for(RetrievalError("self_document_unresolved", "ambiguous")),
        last_attempt=False,
        fail=lambda *value: failures.append(value),
    )

    assert failures == [
        (
            "job-3",
            "self_document_unresolved",
            "self_document_unresolved: ambiguous",
        )
    ]


def test_transient_model_failure_retries_then_retains_its_code() -> None:
    failures: list[tuple[str, str, str]] = []
    error = ModelGatewayError("provider_unavailable", "temporary outage")
    provider = _provider_for(error)

    with pytest.raises(ModelGatewayError, match="provider_unavailable"):
        execute_analysis_job(
            "job-4",
            provider,
            last_attempt=False,
            fail=lambda *value: failures.append(value),
        )
    assert failures == []

    execute_analysis_job(
        "job-4",
        provider,
        last_attempt=True,
        fail=lambda *value: failures.append(value),
    )

    assert failures == [
        (
            "job-4",
            "provider_unavailable",
            "provider_unavailable: temporary outage",
        )
    ]
