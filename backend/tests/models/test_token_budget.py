from __future__ import annotations

import pytest

from redacted_backend.models.token_budget import (
    PromptTooLargeError,
    check_token_budget,
)


class FixedCounter:
    def __init__(self, tokens: int) -> None:
        self.tokens = tokens
        self.requests: list[dict[str, object]] = []

    def count_input_tokens(self, request: dict[str, object]) -> int:
        self.requests.append(request)
        return self.tokens


def test_token_budget_uses_the_target_models_exact_request_counter() -> None:
    request = {"model": "gpt-6-astra", "input": "visible prompt"}
    counter = FixedCounter(1_000)

    report = check_token_budget(
        request,
        counter,
        context_window=1_050_000,
        max_output_tokens=4_096,
    )

    assert counter.requests == [request]
    assert report.input_tokens == 1_000
    assert report.max_output_tokens == 4_096
    assert report.total_reserved_tokens == 5_096


def test_token_budget_fails_without_truncating_or_dropping_evidence() -> None:
    with pytest.raises(PromptTooLargeError, match="prompt_too_large"):
        check_token_budget(
            {"model": "gpt-6-astra", "input": "complete evidence"},
            FixedCounter(1_049_000),
            context_window=1_050_000,
            max_output_tokens=4_096,
        )
