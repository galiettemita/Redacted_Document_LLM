from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import httpx
import pytest
from pydantic import ValidationError

from redacted_backend.domain.analysis import PredictionBasis
from redacted_backend.domain.predictions import PredictionRecord, TokenUsage
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


class AcceptThenTimeoutTransport(RecordingTransport):
    def __init__(self) -> None:
        super().__init__([])

    def create_response(
        self,
        request: dict[str, object],
        *,
        client_request_id: str,
    ) -> ProviderHttpResponse:
        self.create_requests.append((request, client_request_id))
        raise TimeoutError("response lost after provider acceptance")


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
    assert result.prediction_basis == PredictionBasis.PAPER_EXACT_V1
    assert result.retrieval_trace_sha256


def test_gateway_model_only_prediction_is_retrieval_free_and_idempotent() -> None:
    transport = RecordingTransport([_response("Predicted without archive evidence")])
    gateway = OpenAIGateway(
        transport=transport,
        store=InMemoryGenerationStore(),
    )
    generate = getattr(gateway, "generate_model_only_prediction", None)
    assert generate is not None

    first = generate(_example(), "model-only-idempotency-key")
    replay = generate(_example(), "model-only-idempotency-key")

    assert replay == first
    assert first.prediction_basis == PredictionBasis.MODEL_ONLY_V1
    assert first.retrieval_trace_sha256 is None
    assert len(transport.count_requests) == 1
    assert len(transport.create_requests) == 1
    request, client_request_id = transport.create_requests[0]
    assert client_request_id == "model-only-idempotency-key"
    assert request["model"] == "gpt-6-astra"
    assert request["reasoning"] == {"effort": "medium"}
    assert request["store"] is False
    assert request["tools"] == []
    assert request["truncation"] == "disabled"
    assert request["max_output_tokens"] == 4096


def test_model_only_gateway_never_resends_an_uncertain_generation() -> None:
    transport = AcceptThenTimeoutTransport()
    gateway = OpenAIGateway(
        transport=transport,
        store=InMemoryGenerationStore(),
    )

    with pytest.raises(ModelGatewayError, match="provider_outcome_unknown"):
        gateway.generate_model_only_prediction(_example(), "uncertain-generation")
    with pytest.raises(ModelGatewayError, match="provider_outcome_unknown"):
        gateway.generate_model_only_prediction(_example(), "uncertain-generation")

    assert len(transport.count_requests) == 1
    assert len(transport.create_requests) == 1


def test_model_only_gateway_rechecks_validity_after_token_count() -> None:
    valid = [True]

    class InvalidatingTransport(RecordingTransport):
        def count_input_tokens(self, request: dict[str, object]) -> int:
            valid[0] = False
            return super().count_input_tokens(request)

    transport = InvalidatingTransport([_response("must not be sent")])
    gateway = OpenAIGateway(
        transport=transport,
        store=InMemoryGenerationStore(),
    )

    with pytest.raises(ModelGatewayError, match="analysis_cancelled"):
        gateway.generate_model_only_prediction(
            _example(),
            "cancelled-after-count",
            before_generation=lambda: valid[0],
        )

    assert len(transport.count_requests) == 1
    assert transport.create_requests == []


def test_model_only_gateway_rejects_empty_refused_and_incomplete_outputs() -> None:
    cases = [
        (_response("   "), "empty_output"),
        (_refusal_response(), "provider_refusal"),
        ({**_response("prediction"), "status": "incomplete"}, "provider_incomplete"),
    ]
    for index, (response, code) in enumerate(cases):
        gateway = OpenAIGateway(
            transport=RecordingTransport([response]),
            store=InMemoryGenerationStore(),
        )
        generate = getattr(gateway, "generate_model_only_prediction", None)
        assert generate is not None
        with pytest.raises(ModelGatewayError, match=code):
            generate(_example(), f"model-only-invalid-{index}")


def test_prediction_record_requires_trace_only_for_retrieval_bases() -> None:
    values = {
        "passage_id": _example().passage_id,
        "prediction": "Predicted text",
        "prompt_sha256": "1" * 64,
        "request_sha256": "2" * 64,
        "provider_response_id": "resp-123",
        "provider_request_id": "req-123",
        "input_tokens": 10,
        "usage": TokenUsage(
            input_tokens=10,
            output_tokens=2,
            total_tokens=12,
            cached_input_tokens=0,
            reasoning_tokens=0,
        ),
        "estimated_cost_usd": 0.01,
        "pricing_version": "test",
    }
    model_only = PredictionRecord(
        **values,
        prediction_basis=PredictionBasis.MODEL_ONLY_V1,
        retrieval_trace_sha256=None,
    )
    assert model_only.retrieval_trace_sha256 is None

    for basis in (
        PredictionBasis.BEST_AVAILABLE_FOIARCHIVE_V1,
        PredictionBasis.PAPER_EXACT_V1,
    ):
        with pytest.raises(ValidationError):
            PredictionRecord(
                **values,
                prediction_basis=basis,
                retrieval_trace_sha256=None,
            )
    with pytest.raises(ValidationError):
        PredictionRecord(
            **values,
            prediction_basis=PredictionBasis.MODEL_ONLY_V1,
            retrieval_trace_sha256="3" * 64,
        )


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
