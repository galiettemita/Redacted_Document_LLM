from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from redacted_backend.domain.predictions import TokenUsage


class ModelPricing(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    version: str
    model: str
    source_url: str
    input_usd_per_million: float = Field(ge=0)
    cached_input_usd_per_million: float = Field(ge=0)
    output_usd_per_million: float = Field(ge=0)
    long_context_threshold: int = Field(ge=1)
    long_input_multiplier: float = Field(ge=1)
    long_output_multiplier: float = Field(ge=1)


OPENAI_GPT6_ASTRA_PRICING = ModelPricing(
    version="openai-gpt6-astra-2026-11-12",
    model="gpt-6-astra",
    source_url="https://developers.openai.com/api/docs/models/gpt-6-astra",
    input_usd_per_million=10.0,
    cached_input_usd_per_million=1.0,
    output_usd_per_million=50.0,
    long_context_threshold=272_000,
    long_input_multiplier=2.0,
    long_output_multiplier=1.5,
)


def estimate_maximum_cost(input_tokens: int, max_output_tokens: int) -> float:
    if input_tokens < 0 or max_output_tokens < 0:
        raise ValueError("Token counts must not be negative")
    input_rate, _cached_rate, output_rate = _rates(input_tokens)
    return (
        input_tokens * input_rate + max_output_tokens * output_rate
    ) / 1_000_000


def estimate_usage_cost(usage: TokenUsage) -> float:
    input_rate, cached_rate, output_rate = _rates(usage.input_tokens)
    uncached = max(usage.input_tokens - usage.cached_input_tokens, 0)
    return (
        uncached * input_rate
        + usage.cached_input_tokens * cached_rate
        + usage.output_tokens * output_rate
    ) / 1_000_000


def _rates(input_tokens: int) -> tuple[float, float, float]:
    pricing = OPENAI_GPT6_ASTRA_PRICING
    is_long = input_tokens > pricing.long_context_threshold
    input_multiplier = pricing.long_input_multiplier if is_long else 1.0
    output_multiplier = pricing.long_output_multiplier if is_long else 1.0
    return (
        pricing.input_usd_per_million * input_multiplier,
        pricing.cached_input_usd_per_million * input_multiplier,
        pricing.output_usd_per_million * output_multiplier,
    )
