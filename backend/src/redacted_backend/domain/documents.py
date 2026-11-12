from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from redacted_backend.domain.visible_examples import PageWord


class DocumentStatus(StrEnum):
    UPLOADED = "uploaded"
    EXTRACTING = "extracting"
    READY = "ready"
    FAILED = "failed"
    DELETED = "deleted"


class DateProvenance(StrEnum):
    VISIBLE_TEXT = "visible_text"
    PDF_METADATA_CORROBORATED = "pdf_metadata_corroborated"


class VisibleDateEvidence(BaseModel):
    """The visible PDF line that deterministically established the document date."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    value: date
    page_number: int = Field(ge=1)
    line_text: str = Field(min_length=1)
    words: list[PageWord] = Field(min_length=1)
    parser_version: Literal["visible-date-v1"] = "visible-date-v1"


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
    visible_date_evidence: VisibleDateEvidence | None = None
    created_at: datetime
    expires_at: datetime
