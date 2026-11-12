from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class DocumentStatus(StrEnum):
    UPLOADED = "uploaded"
    EXTRACTING = "extracting"
    READY = "ready"
    FAILED = "failed"
    DELETED = "deleted"


class DateProvenance(StrEnum):
    VISIBLE_TEXT = "visible_text"
    PDF_METADATA_CORROBORATED = "pdf_metadata_corroborated"
    MANUAL_VISIBLE_DOCUMENT = "manual_visible_document"


class DocumentRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    id: UUID
    filename: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    media_type: str
    page_count: int = Field(ge=1)
    status: DocumentStatus
    storage_key: str
    visible_date: date | None = None
    date_provenance: DateProvenance | None = None
    created_at: datetime
    expires_at: datetime


class DocumentMetadataUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    visible_date: date
    date_provenance: Literal[DateProvenance.MANUAL_VISIBLE_DOCUMENT]
