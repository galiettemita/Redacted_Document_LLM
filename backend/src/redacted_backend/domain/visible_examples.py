from __future__ import annotations

from datetime import date
from enum import StrEnum
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from redacted_backend.domain.passages import PassageRect


class WordSource(StrEnum):
    EMBEDDED = "embedded"
    OCR = "ocr"


class ExtractionSource(StrEnum):
    EMBEDDED = "embedded"
    OCR = "ocr"


class MaskKind(StrEnum):
    BLACK = "black"
    WHITE = "white"
    MANUAL = "manual"


class PageWord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str = Field(min_length=1)
    page_number: int = Field(ge=1)
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)
    width: float = Field(gt=0, le=1)
    height: float = Field(gt=0, le=1)
    block_id: int
    line_id: int
    source: WordSource
    confidence: float = Field(ge=0, le=1)

    @property
    def rect(self) -> PassageRect:
        return PassageRect(x=self.x, y=self.y, width=self.width, height=self.height)


class PageWords(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    page_number: int = Field(ge=1)
    width_points: float = Field(gt=0)
    height_points: float = Field(gt=0)
    extraction_source: ExtractionSource
    words: list[PageWord]


class MaskCandidate(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    page_number: int = Field(ge=1)
    rect: PassageRect
    kind: MaskKind
    confidence: float = Field(ge=0, le=1)
    confirmed: bool


class MaskDiscovery(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    page_number: int = Field(ge=1)
    masks: list[MaskCandidate]
    coverage_confident: bool
    warnings: list[str]


class VisibleDocument(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: UUID
    visible_date: date | None
    pages: list[PageWords] = Field(min_length=1)
    mask_discoveries: list[MaskDiscovery] = Field(min_length=1)


class SerializedTranscript(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str
    before_target: str
    after_target: str
    visible_words: list[PageWord]
    serialization_version: Literal["visible-transcript-v1"] = "visible-transcript-v1"


class LengthEstimate(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    estimated_words: int = Field(ge=1)
    bucket: Literal[
        "a few words",
        "part of a sentence",
        "about a sentence",
        "a paragraph",
    ]
    median_character_width: float = Field(gt=0)
    median_characters_per_word: float = Field(gt=0)
    line_capacities: list[int] = Field(min_length=1)


class VisibleExample(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    passage_id: UUID
    document_id: UUID
    document_date: date
    before_target: str
    after_target: str
    context_window: str
    length_cue: LengthEstimate
    visible_words: list[PageWord]
    serialization_version: Literal["visible-transcript-v1"]
