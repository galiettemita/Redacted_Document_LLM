from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import cast

import pytest
from pydantic import SecretStr

import redacted_backend.jobs.runtime as runtime_module
from redacted_backend.config import Settings
from redacted_backend.db.base import Database
from redacted_backend.domain.analysis import PredictionBasis
from redacted_backend.jobs.orchestrator import AnalysisOrchestrator
from redacted_backend.jobs.runtime import build_analysis_orchestrator
from redacted_backend.jobs.runtime_provider import (
    AnalysisRuntimeProvider,
    RuntimeInitializationError,
)
from tests.test_readiness import _compatible_best_available_settings


def _runtime() -> AnalysisOrchestrator:
    return cast(AnalysisOrchestrator, object())


def test_provider_construction_does_not_build_the_runtime() -> None:
    calls = 0

    def build(_basis: PredictionBasis) -> AnalysisOrchestrator:
        nonlocal calls
        calls += 1
        return _runtime()

    AnalysisRuntimeProvider(build)

    assert calls == 0


def test_provider_caches_one_successful_runtime() -> None:
    calls = 0
    runtime = _runtime()

    def build(_basis: PredictionBasis) -> AnalysisOrchestrator:
        nonlocal calls
        calls += 1
        return runtime

    provider = AnalysisRuntimeProvider(build)

    assert provider.get(PredictionBasis.MODEL_ONLY_V1) is runtime
    assert provider.get(PredictionBasis.MODEL_ONLY_V1) is runtime
    assert calls == 1


def test_provider_retries_after_a_failed_initialization() -> None:
    calls = 0
    runtime = _runtime()

    def build(_basis: PredictionBasis) -> AnalysisOrchestrator:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("secret-provider-token")
        return runtime

    provider = AnalysisRuntimeProvider(build)

    with pytest.raises(RuntimeInitializationError) as captured:
        provider.get(PredictionBasis.MODEL_ONLY_V1)

    assert captured.value.cause_name == "RuntimeError"
    assert str(captured.value) == "RuntimeError"
    assert "secret-provider-token" not in str(captured.value)
    assert provider.get(PredictionBasis.MODEL_ONLY_V1) is runtime
    assert calls == 2


def test_provider_serializes_concurrent_first_initialization() -> None:
    calls = 0
    calls_lock = threading.Lock()
    entered = threading.Event()
    release = threading.Event()
    runtime = _runtime()

    def build(_basis: PredictionBasis) -> AnalysisOrchestrator:
        nonlocal calls
        with calls_lock:
            calls += 1
        entered.set()
        release.wait(timeout=2)
        return runtime

    provider = AnalysisRuntimeProvider(build)
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(provider.get, PredictionBasis.MODEL_ONLY_V1)
            for _ in range(8)
        ]
        assert entered.wait(timeout=1)
        try:
            assert calls == 1
        finally:
            release.set()
        results = [future.result(timeout=2) for future in futures]

    assert all(result is runtime for result in results)
    assert calls == 1


def test_provider_caches_each_prediction_basis_independently() -> None:
    calls: list[PredictionBasis] = []
    runtimes = {
        PredictionBasis.MODEL_ONLY_V1: _runtime(),
        PredictionBasis.PAPER_EXACT_V1: _runtime(),
    }

    def build(basis: PredictionBasis) -> AnalysisOrchestrator:
        calls.append(basis)
        return runtimes[basis]

    provider = AnalysisRuntimeProvider(build)

    assert provider.get(PredictionBasis.MODEL_ONLY_V1) is runtimes[
        PredictionBasis.MODEL_ONLY_V1
    ]
    assert provider.get(PredictionBasis.PAPER_EXACT_V1) is runtimes[
        PredictionBasis.PAPER_EXACT_V1
    ]
    assert provider.get(PredictionBasis.MODEL_ONLY_V1) is runtimes[
        PredictionBasis.MODEL_ONLY_V1
    ]
    assert calls == [
        PredictionBasis.MODEL_ONLY_V1,
        PredictionBasis.PAPER_EXACT_V1,
    ]


def test_model_only_runtime_requires_no_retrieval_configuration(tmp_path) -> None:
    database = Database.connect(f"sqlite+pysqlite:///{tmp_path / 'runtime.db'}")
    settings = Settings(
        openai_api_key="secret",
        allow_paid_model_calls=True,
        allow_model_only_calls=True,
    )

    runtime = build_analysis_orchestrator(
        settings,
        PredictionBasis.MODEL_ONLY_V1,
        database=database,
    )

    assert runtime.retrieval is None


def test_runtime_keeps_each_basis_fail_closed(tmp_path) -> None:
    database = Database.connect(f"sqlite+pysqlite:///{tmp_path / 'runtime.db'}")
    base = {
        "openai_api_key": "secret",
        "allow_paid_model_calls": True,
    }

    with pytest.raises(RuntimeError, match="model-only access is not approved"):
        build_analysis_orchestrator(
            Settings(**base),
            PredictionBasis.MODEL_ONLY_V1,
            database=database,
        )
    with pytest.raises(RuntimeError, match="best-available corpus manifest"):
        build_analysis_orchestrator(
            Settings(**base, allow_best_available_calls=True),
            PredictionBasis.BEST_AVAILABLE_FOIARCHIVE_V1,
            database=database,
        )
    with pytest.raises(RuntimeError, match="OpenSearch is not configured"):
        build_analysis_orchestrator(
            Settings(**base, allow_paper_exact_calls=True),
            PredictionBasis.PAPER_EXACT_V1,
            database=database,
        )


def test_best_available_runtime_loads_only_its_separate_verified_artifacts(
    tmp_path,
    monkeypatch,
) -> None:
    settings, _corpus, bm25_manifest, dense_manifest = (
        _compatible_best_available_settings(tmp_path)
    )
    settings = settings.model_copy(
        update={
            "openai_api_key": SecretStr("secret"),
            "allow_paid_model_calls": True,
            "allow_best_available_calls": True,
        }
    )
    database = Database.connect(f"sqlite+pysqlite:///{tmp_path / 'runtime.db'}")
    captured: dict[str, object] = {}

    class FakeBm25:
        def __init__(self, _transport, **kwargs) -> None:
            captured["bm25_manifest"] = kwargs["index_manifest"]
            captured["corpus_manifest"] = kwargs["corpus_manifest"]
            captured["index_name"] = kwargs["index_name"]

        def search(self, *_args, **_kwargs):
            captured["bm25_probe_count"] = int(
                captured.get("bm25_probe_count", 0)
            ) + 1
            return []

    class FakeDense:
        def __init__(self, **kwargs) -> None:
            captured["dense_root"] = kwargs["root"]
            captured["dense_chunks_path"] = kwargs["chunks_path"]
            captured["dense_manifest"] = kwargs["manifest"]
            captured["candidate_loader"] = kwargs["candidate_loader"]

        def search(self, *_args, **_kwargs):
            captured["dense_probe_count"] = int(
                captured.get("dense_probe_count", 0)
            ) + 1
            return []

    class FakeTransport:
        def __init__(self, url: str) -> None:
            captured["opensearch_url"] = url

        def count(self, index_name: str) -> int:
            assert index_name == bm25_manifest.index_name
            return bm25_manifest.chunk_count

    monkeypatch.setattr(runtime_module, "Bm25Index", FakeBm25)
    monkeypatch.setattr(runtime_module, "ShardedDenseIndex", FakeDense, raising=False)
    monkeypatch.setattr(runtime_module, "OpenSearchHttpTransport", FakeTransport)

    runtime = build_analysis_orchestrator(
        settings,
        PredictionBasis.BEST_AVAILABLE_FOIARCHIVE_V1,
        database=database,
    )

    assert runtime.retrieval is not None
    assert runtime.prediction_basis == PredictionBasis.BEST_AVAILABLE_FOIARCHIVE_V1
    assert captured["bm25_manifest"] == bm25_manifest
    assert captured["dense_manifest"] == dense_manifest
    assert captured["index_name"] == "foiarchive-best-available-bm25-v1"
    assert captured["dense_root"] == settings.best_available_dense_artifact_directory
    assert captured["dense_chunks_path"] == settings.best_available_chunks_path
    assert captured["candidate_loader"] == runtime_module.FaissIvfPqIndex.load
    assert captured["bm25_probe_count"] == 1
    assert captured["dense_probe_count"] == 1
