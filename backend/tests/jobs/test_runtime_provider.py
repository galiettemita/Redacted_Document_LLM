from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import cast

import pytest

from redacted_backend.jobs.orchestrator import AnalysisOrchestrator
from redacted_backend.jobs.runtime_provider import (
    AnalysisRuntimeProvider,
    RuntimeInitializationError,
)


def _runtime() -> AnalysisOrchestrator:
    return cast(AnalysisOrchestrator, object())


def test_provider_construction_does_not_build_the_runtime() -> None:
    calls = 0

    def build() -> AnalysisOrchestrator:
        nonlocal calls
        calls += 1
        return _runtime()

    AnalysisRuntimeProvider(build)

    assert calls == 0


def test_provider_caches_one_successful_runtime() -> None:
    calls = 0
    runtime = _runtime()

    def build() -> AnalysisOrchestrator:
        nonlocal calls
        calls += 1
        return runtime

    provider = AnalysisRuntimeProvider(build)

    assert provider.get() is runtime
    assert provider.get() is runtime
    assert calls == 1


def test_provider_retries_after_a_failed_initialization() -> None:
    calls = 0
    runtime = _runtime()

    def build() -> AnalysisOrchestrator:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("secret-provider-token")
        return runtime

    provider = AnalysisRuntimeProvider(build)

    with pytest.raises(RuntimeInitializationError) as captured:
        provider.get()

    assert captured.value.cause_name == "RuntimeError"
    assert str(captured.value) == "RuntimeError"
    assert "secret-provider-token" not in str(captured.value)
    assert provider.get() is runtime
    assert calls == 2


def test_provider_serializes_concurrent_first_initialization() -> None:
    calls = 0
    calls_lock = threading.Lock()
    entered = threading.Event()
    release = threading.Event()
    runtime = _runtime()

    def build() -> AnalysisOrchestrator:
        nonlocal calls
        with calls_lock:
            calls += 1
        entered.set()
        release.wait(timeout=2)
        return runtime

    provider = AnalysisRuntimeProvider(build)
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(provider.get) for _ in range(8)]
        assert entered.wait(timeout=1)
        try:
            assert calls == 1
        finally:
            release.set()
        results = [future.result(timeout=2) for future in futures]

    assert all(result is runtime for result in results)
    assert calls == 1
