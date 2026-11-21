from __future__ import annotations

import threading
from collections.abc import Callable

from redacted_backend.domain.analysis import PredictionBasis
from redacted_backend.jobs.orchestrator import AnalysisOrchestrator


class RuntimeInitializationError(RuntimeError):
    def __init__(self, cause_name: str) -> None:
        super().__init__(cause_name)
        self.cause_name = cause_name


class AnalysisRuntimeProvider:
    def __init__(
        self,
        builder: Callable[[PredictionBasis], AnalysisOrchestrator],
    ) -> None:
        self._builder = builder
        self._runtimes: dict[PredictionBasis, AnalysisOrchestrator] = {}
        self._lock = threading.Lock()

    def get(self, basis: PredictionBasis) -> AnalysisOrchestrator:
        runtime = self._runtimes.get(basis)
        if runtime is not None:
            return runtime
        with self._lock:
            runtime = self._runtimes.get(basis)
            if runtime is not None:
                return runtime
            try:
                runtime = self._builder(basis)
            except Exception as error:
                raise RuntimeInitializationError(type(error).__name__) from None
            self._runtimes[basis] = runtime
            return runtime
