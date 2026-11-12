from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PassageSource(StrEnum):
    MANUAL = "manual"
    DETECTED = "detected"


class PassageStatus(StrEnum):
    MARKED = "marked"
    CONTEXT_READY = "context_ready"
    RETRIEVING = "retrieving"
    PREDICTING = "predicting"
    COMPLETE = "complete"
    FAILED = "failed"
    DELETED = "deleted"


class PassageRect(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)
    width: float = Field(gt=0, le=1)
    height: float = Field(gt=0, le=1)

    @model_validator(mode="after")
    def fits_on_page(self) -> "PassageRect":
        if self.x + self.width > 1 or self.y + self.height > 1:
            raise ValueError("rectangle must fit inside normalized page bounds")
        return self


class PassageRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: UUID
    document_id: UUID
    label: str
    page_number: int = Field(ge=1)
    rects: list[PassageRect] = Field(min_length=1)
    source: PassageSource
    status: PassageStatus
    revision: int = Field(ge=1)
    merged_into: UUID | None = None
    split_from: UUID | None = None
    created_at: datetime
    updated_at: datetime


class PassageMergeResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    passage: PassageRecord
    merged_ids: list[UUID]


class PassageSplitResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    passages: list[PassageRecord]
