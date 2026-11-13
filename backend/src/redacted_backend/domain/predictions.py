from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class TokenUsage(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    cached_input_tokens: int = Field(ge=0)
    reasoning_tokens: int = Field(ge=0)


class QueryResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    passage_id: UUID
    query: str = Field(min_length=1)
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    provider_response_id: str = Field(min_length=1)
    provider_request_id: str | None
    input_tokens: int = Field(ge=0)
    usage: TokenUsage
    estimated_cost_usd: float = Field(ge=0)
    pricing_version: str


class PredictionRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    passage_id: UUID
    prediction: str = Field(min_length=1)
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    retrieval_trace_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    provider_response_id: str = Field(min_length=1)
    provider_request_id: str | None
    input_tokens: int = Field(ge=0)
    usage: TokenUsage
    estimated_cost_usd: float = Field(ge=0)
    pricing_version: str
