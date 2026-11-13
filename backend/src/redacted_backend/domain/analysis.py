from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class AnalysisStatus(StrEnum):
    BLOCKED = "blocked"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AnalysisStage(StrEnum):
    READINESS = "readiness"
    QUERY = "query"
    RETRIEVAL = "retrieval"
    PREDICTION = "prediction"
    COMPLETE = "complete"


class AnalysisBlocker(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    component: str = Field(min_length=1)
    detail: str = Field(min_length=1)


class AnalysisJobRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    id: UUID
    passage_id: UUID
    document_id: UUID
    passage_revision: int = Field(ge=1)
    status: AnalysisStatus
    stage: AnalysisStage
    blockers: list[AnalysisBlocker]
    error_code: str | None
    error_detail: str | None
    created_at: datetime
    updated_at: datetime


class AnalysisEventRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    id: int = Field(ge=1)
    job_id: UUID
    document_id: UUID
    passage_id: UUID
    sequence: int = Field(ge=1)
    event_type: str = Field(min_length=1)
    payload: dict[str, Any]
    created_at: datetime
