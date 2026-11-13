from __future__ import annotations

from typing import Protocol
from uuid import UUID

from dramatiq import Message
from dramatiq.brokers.redis import RedisBroker

from redacted_backend.config import Settings


class AnalysisDispatcher(Protocol):
    def dispatch(self, job_id: UUID) -> None: ...


class DramatiqAnalysisDispatcher:
    def __init__(self, settings: Settings) -> None:
        if settings.redis_url is None:
            raise ValueError("Redis is required for analysis dispatch")
        self._broker = RedisBroker(url=settings.redis_url)

    def dispatch(self, job_id: UUID) -> None:
        self._broker.enqueue(
            Message(
                queue_name="default",
                actor_name="analyze_passage",
                args=(str(job_id),),
                kwargs={},
                options={},
            )
        )
