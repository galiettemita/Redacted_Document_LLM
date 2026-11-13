from __future__ import annotations

import threading

from dramatiq import Middleware
from redis import Redis

from redacted_backend.config import Settings


class WorkerHeartbeatMiddleware(Middleware):
    """Maintain an expiring Redis key only while a Dramatiq worker is alive."""

    def __init__(self, settings: Settings) -> None:
        if settings.redis_url is None:
            raise ValueError("Redis is required for the analysis worker heartbeat")
        self._client = Redis.from_url(settings.redis_url)
        self._key = settings.worker_heartbeat_key
        self._ttl = settings.worker_heartbeat_ttl_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def after_worker_boot(self, broker, worker) -> None:
        del broker, worker
        self._stop.clear()
        self._write()
        self._thread = threading.Thread(
            target=self._run,
            name="analysis-worker-heartbeat",
            daemon=True,
        )
        self._thread.start()

    def before_worker_shutdown(self, broker, worker) -> None:
        del broker, worker
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._client.delete(self._key)

    def _run(self) -> None:
        interval = max(1.0, self._ttl / 3)
        while not self._stop.wait(interval):
            self._write()

    def _write(self) -> None:
        self._client.set(self._key, "ready", ex=self._ttl)
