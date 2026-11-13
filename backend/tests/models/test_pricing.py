from __future__ import annotations

import pytest

from redacted_backend.domain.predictions import TokenUsage
from redacted_backend.models.pricing import (
    OPENAI_GPT6_ASTRA_PRICING,
    estimate_maximum_cost,
    estimate_usage_cost,
)


def test_pricing_is_versioned_and_applies_the_long_context_multiplier() -> None:
    assert OPENAI_GPT6_ASTRA_PRICING.version == "openai-gpt6-astra-2026-11-12"
    assert estimate_maximum_cost(1_000, 4_096) == pytest.approx(0.2148)
    assert estimate_maximum_cost(300_000, 4_096) == pytest.approx(6.3072)


def test_usage_cost_separates_cached_input_and_includes_reasoning_output() -> None:
    usage = TokenUsage(
        input_tokens=1_000,
        output_tokens=100,
        total_tokens=1_100,
        cached_input_tokens=400,
        reasoning_tokens=80,
    )

    assert estimate_usage_cost(usage) == pytest.approx(0.0114)
