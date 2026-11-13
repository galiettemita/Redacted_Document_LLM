from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import httpx
import pytest

from redacted_backend.models.openai_gateway import (
    InMemoryGenerationStore,
    ModelGatewayError,
    OpenAIGateway,
    OpenAIHttpTransport,
    ProviderHttpResponse,
)
from tests.models.test_prompts import _example, _retrieval


class RecordingTransport:
    def __init__(self, outputs: Sequence[dict[str, Any]]) -> None:
        self.outputs = list(outputs)
        self.count_requests: list[dict[str, object]] = []
        self.create_requests: list[tuple[dict[str, object], str]] = []

    def count_input_tokens(self, request: dict[str, object]) -> int:
        self.count_requests.append(request)
        return 321

    def create_response(
        self,
        request: dict[str, object],
        *,
        client_request_id: str,
    ) -> ProviderHttpResponse:
        self.create_requests.append((request, client_request_id))
        return ProviderHttpResponse(
            body=self.outputs.pop(0),
            request_id="request-123",
        )


def test_gateway_sends_one_user_message_with_exact_frozen_settings() -> None:
    transport = RecordingTransport([_response("specific actors paris 1960 meeting")])
    gateway = OpenAIGateway(
        transport=transport,
        store=InMemoryGenerationStore(),
    )

    result = gateway.generate_query(_example(), "query-idempotency-key")

    request, client_request_id = transport.create_requests[0]
    assert request["model"] == "gpt-6-astra"
    assert request["reasoning"] == {"effort": "medium"}
    assert request["store"] is False
    assert request["tools"] == []
    assert request["truncation"] == "disabled"
    assert request["max_output_tokens"] == 4096
    assert "instructions" not in request
    assert request["input"] == [
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": request["input"][0]["content"][0]["text"],
                }
            ],
        }
    ]
    assert client_request_id == "query-idempotency-key"
    assert result.query == "specific actors paris 1960 meeting"
    assert result.provider_response_id == "resp-123"
    assert result.provider_request_id == "request-123"
    assert result.usage.input_tokens == 321
    assert result.usage.output_tokens == 40


def test_gateway_prediction_remains_linked_to_the_passage_and_trace() -> None:
    transport = RecordingTransport([_response("Predicted hidden passage")])
    gateway = OpenAIGateway(
        transport=transport,
        store=InMemoryGenerationStore(),
    )

    result = gateway.generate_prediction(
        _example(),
        _retrieval(),
        "prediction-idempotency-key",
    )

    assert result.passage_id == _example().passage_id
    assert result.prediction == "Predicted hidden passage"
    assert result.retrieval_trace_sha256


def test_gateway_rejects_invalid_query_empty_output_and_refusal_without_fallback() -> None:
    invalid = RecordingTransport([_response("bad, punctuation")])
    with pytest.raises(ModelGatewayError, match="invalid_query"):
        OpenAIGateway(
            transport=invalid,
            store=InMemoryGenerationStore(),
        ).generate_query(_example(), "invalid-query")
    assert len(invalid.create_requests) == 1

    empty = RecordingTransport([_response("   ")])
    with pytest.raises(ModelGatewayError, match="empty_output"):
        OpenAIGateway(
            transport=empty,
            store=InMemoryGenerationStore(),
        ).generate_prediction(_example(), _retrieval(), "empty-prediction")

    refusal = RecordingTransport([_refusal_response()])
    with pytest.raises(ModelGatewayError, match="provider_refusal"):
        OpenAIGateway(
            transport=refusal,
            store=InMemoryGenerationStore(),
        ).generate_prediction(_example(), _retrieval(), "refusal")


def test_gateway_idempotency_reuses_the_persisted_result_without_provider_calls() -> None:
    transport = RecordingTransport([_response("specific actors paris 1960 meeting")])
    store = InMemoryGenerationStore()
    gateway = OpenAIGateway(transport=transport, store=store)

    first = gateway.generate_query(_example(), "same-key")
    second = gateway.generate_query(_example(), "same-key")

    assert second == first
    assert len(transport.count_requests) == 1
    assert len(transport.create_requests) == 1


def test_http_transport_maps_rate_limits_without_retrying_or_falling_back() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(429, request=request, json={"error": {"message": "limit"}})

    client = httpx.Client(
        base_url="https://api.openai.test/v1",
        transport=httpx.MockTransport(handler),
    )
    transport = OpenAIHttpTransport("secret", client=client)

    with pytest.raises(ModelGatewayError, match="rate_limited"):
        transport.create_response(
            {"model": "gpt-6-astra", "input": "visible"},
            client_request_id="client-request-1",
        )

    assert len(requests) == 1
    assert requests[0].headers["X-Client-Request-Id"] == "client-request-1"


def _response(text: str) -> dict[str, Any]:
    return {
        "id": "resp-123",
        "status": "completed",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text}],
            }
        ],
        "usage": {
            "input_tokens": 321,
            "output_tokens": 40,
            "total_tokens": 361,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 12},
        },
    }


def _refusal_response() -> dict[str, Any]:
    return {
        "id": "resp-refusal",
        "status": "completed",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "refusal", "refusal": "Cannot comply"}],
            }
        ],
        "usage": {
            "input_tokens": 321,
            "output_tokens": 1,
            "total_tokens": 322,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
    }
