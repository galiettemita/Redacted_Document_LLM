from __future__ import annotations

import threading
from collections.abc import Callable

from redacted_backend.jobs.orchestrator import AnalysisOrchestrator


class RuntimeInitializationError(RuntimeError):
    def __init__(self, cause_name: str) -> None:
        super().__init__(cause_name)
        self.cause_name = cause_name


class AnalysisRuntimeProvider:
    def __init__(self, builder: Callable[[], AnalysisOrchestrator]) -> None:
        self._builder = builder
        self._runtime: AnalysisOrchestrator | None = None
        self._lock = threading.Lock()

    def get(self) -> AnalysisOrchestrator:
        runtime = self._runtime
        if runtime is not None:
            return runtime
        with self._lock:
            runtime = self._runtime
            if runtime is not None:
                return runtime
            try:
                runtime = self._builder()
            except Exception as error:
                raise RuntimeInitializationError(type(error).__name__) from None
            self._runtime = runtime
            return runtime
