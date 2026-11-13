from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field


class InputTokenCounter(Protocol):
    def count_input_tokens(self, request: dict[str, object]) -> int: ...


class PromptTooLargeError(ValueError):
    code = "prompt_too_large"


class TokenBudgetReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    input_tokens: int = Field(ge=0)
    max_output_tokens: int = Field(ge=1)
    total_reserved_tokens: int = Field(ge=1)
    context_window: int = Field(ge=1)


def check_token_budget(
    request: dict[str, object],
    counter: InputTokenCounter,
    *,
    context_window: int,
    max_output_tokens: int,
) -> TokenBudgetReport:
    input_tokens = counter.count_input_tokens(request)
    if input_tokens < 0:
        raise ValueError("Input token counter returned a negative value")
    total = input_tokens + max_output_tokens
    if total > context_window:
        raise PromptTooLargeError(
            "prompt_too_large: complete request exceeds the target model context "
            "window; evidence was not truncated or reordered"
        )
    return TokenBudgetReport(
        input_tokens=input_tokens,
        max_output_tokens=max_output_tokens,
        total_reserved_tokens=total,
        context_window=context_window,
    )
