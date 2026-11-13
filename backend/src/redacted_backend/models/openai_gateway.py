from __future__ import annotations

import hashlib
import json
from typing import Any, Literal, Protocol, TypeVar

import httpx
from pydantic import BaseModel, ConfigDict, Field

from redacted_backend.domain.predictions import (
    PredictionRecord,
    QueryResult,
    TokenUsage,
)
from redacted_backend.domain.retrieval import RetrievalTrace
from redacted_backend.domain.visible_examples import VisibleExample
from redacted_backend.models.pricing import (
    OPENAI_GPT6_ASTRA_PRICING,
    estimate_usage_cost,
)
from redacted_backend.models.prompts import (
    RenderedPrompt,
    render_prediction_prompt,
    render_query_prompt,
)
from redacted_backend.models.token_budget import (
    InputTokenCounter,
    check_token_budget,
)
from redacted_backend.scientific_config import ScientificConfig

ResultT = TypeVar("ResultT", QueryResult, PredictionRecord)


class ModelGatewayError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code


class ProviderHttpResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    body: dict[str, Any]
    request_id: str | None


class ModelTransport(InputTokenCounter, Protocol):
    def create_response(
        self,
        request: dict[str, object],
        *,
        client_request_id: str,
    ) -> ProviderHttpResponse: ...


class StoredGeneration(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["query", "prediction"]
    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    payload: dict[str, Any]


class GenerationStore(Protocol):
    def get(self, scope: str, key: str) -> StoredGeneration | None: ...

    def save(self, scope: str, key: str, value: StoredGeneration) -> None: ...


class InMemoryGenerationStore:
    def __init__(self) -> None:
        self._values: dict[tuple[str, str], StoredGeneration] = {}

    def get(self, scope: str, key: str) -> StoredGeneration | None:
        return self._values.get((scope, key))

    def save(self, scope: str, key: str, value: StoredGeneration) -> None:
        existing = self._values.get((scope, key))
        if existing is not None and existing != value:
            raise ModelGatewayError(
                "idempotency_conflict",
                "Idempotency key already belongs to a different result",
            )
        self._values[(scope, key)] = value


class OpenAIGateway:
    def __init__(
        self,
        *,
        transport: ModelTransport,
        store: GenerationStore,
        scientific: ScientificConfig | None = None,
    ) -> None:
        self.transport = transport
        self.store = store
        self.scientific = scientific or ScientificConfig()

    def generate_query(
        self,
        example: VisibleExample,
        idempotency_key: str,
    ) -> QueryResult:
        prompt = render_query_prompt(example)
        request = self._request(prompt)
        request_hash = _canonical_hash(request)
        scope = f"query:{example.passage_id}"
        existing = self._existing(
            scope,
            idempotency_key,
            request_hash,
            "query",
            QueryResult,
        )
        if existing is not None:
            return existing
        input_tokens, response, output, usage = self._execute(
            request,
            idempotency_key,
        )
        query = _validate_query(output)
        result = QueryResult(
            passage_id=example.passage_id,
            query=query,
            prompt_sha256=prompt.sha256,
            request_sha256=request_hash,
            provider_response_id=_response_id(response.body),
            provider_request_id=response.request_id,
            input_tokens=input_tokens,
            usage=usage,
            estimated_cost_usd=estimate_usage_cost(usage),
            pricing_version=OPENAI_GPT6_ASTRA_PRICING.version,
        )
        self._save(scope, idempotency_key, "query", result)
        return result

    def generate_prediction(
        self,
        example: VisibleExample,
        retrieval: RetrievalTrace,
        idempotency_key: str,
    ) -> PredictionRecord:
        prompt = render_prediction_prompt(example, retrieval)
        request = self._request(prompt)
        request_hash = _canonical_hash(request)
        scope = f"prediction:{example.passage_id}"
        existing = self._existing(
            scope,
            idempotency_key,
            request_hash,
            "prediction",
            PredictionRecord,
        )
        if existing is not None:
            return existing
        input_tokens, response, output, usage = self._execute(
            request,
            idempotency_key,
        )
        prediction = output.strip()
        if not prediction:
            raise ModelGatewayError("empty_output", "Provider returned no prediction")
        result = PredictionRecord(
            passage_id=example.passage_id,
            prediction=prediction,
            prompt_sha256=prompt.sha256,
            request_sha256=request_hash,
            retrieval_trace_sha256=_canonical_hash(
                retrieval.model_dump(mode="json")
            ),
            provider_response_id=_response_id(response.body),
            provider_request_id=response.request_id,
            input_tokens=input_tokens,
            usage=usage,
            estimated_cost_usd=estimate_usage_cost(usage),
            pricing_version=OPENAI_GPT6_ASTRA_PRICING.version,
        )
        self._save(scope, idempotency_key, "prediction", result)
        return result

    def _request(self, prompt: RenderedPrompt) -> dict[str, object]:
        return {
            "model": self.scientific.model,
            "reasoning": {"effort": self.scientific.reasoning_effort},
            "input": [
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": prompt.text}],
                }
            ],
            "store": False,
            "tools": [],
            "truncation": "disabled",
            "max_output_tokens": self.scientific.model_max_output_tokens,
        }

    def _execute(
        self,
        request: dict[str, object],
        idempotency_key: str,
    ) -> tuple[int, ProviderHttpResponse, str, TokenUsage]:
        budget = check_token_budget(
            request,
            self.transport,
            context_window=self.scientific.model_context_window,
            max_output_tokens=self.scientific.model_max_output_tokens,
        )
        try:
            response = self.transport.create_response(
                request,
                client_request_id=idempotency_key,
            )
        except ModelGatewayError:
            raise
        except TimeoutError as error:
            raise ModelGatewayError("provider_timeout", str(error)) from error
        body = response.body
        if body.get("status") != "completed":
            raise ModelGatewayError(
                "provider_incomplete",
                "Provider response did not complete",
            )
        output = _output_text(body)
        usage = _usage(body)
        return budget.input_tokens, response, output, usage

    def _existing(
        self,
        scope: str,
        key: str,
        request_hash: str,
        kind: Literal["query", "prediction"],
        result_type: type[ResultT],
    ) -> ResultT | None:
        if not key or len(key) > 512 or not key.isascii():
            raise ModelGatewayError(
                "invalid_idempotency_key",
                "Idempotency key must be 1-512 ASCII characters",
            )
        existing = self.store.get(scope, key)
        if existing is None:
            return None
        if existing.request_sha256 != request_hash or existing.kind != kind:
            raise ModelGatewayError(
                "idempotency_conflict",
                "Idempotency key was reused for a different request",
            )
        return result_type.model_validate(existing.payload)

    def _save(
        self,
        scope: str,
        key: str,
        kind: Literal["query", "prediction"],
        result: QueryResult | PredictionRecord,
    ) -> None:
        self.store.save(
            scope,
            key,
            StoredGeneration(
                kind=kind,
                request_sha256=result.request_sha256,
                payload=result.model_dump(mode="json"),
            ),
        )


class OpenAIHttpTransport:
    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://api.openai.com/v1",
        timeout_seconds: float = 120.0,
        client: httpx.Client | None = None,
    ) -> None:
        self._client = client or httpx.Client(
            base_url=base_url,
            timeout=timeout_seconds,
            headers={"Authorization": f"Bearer {api_key}"},
        )

    def count_input_tokens(self, request: dict[str, object]) -> int:
        response = self._post("/responses/input_tokens", request)
        value = response.json().get("input_tokens")
        if not isinstance(value, int):
            raise ModelGatewayError(
                "token_count_invalid",
                "Provider token count response is missing input_tokens",
            )
        return value

    def create_response(
        self,
        request: dict[str, object],
        *,
        client_request_id: str,
    ) -> ProviderHttpResponse:
        response = self._post(
            "/responses",
            request,
            headers={"X-Client-Request-Id": client_request_id},
        )
        return ProviderHttpResponse(
            body=response.json(),
            request_id=response.headers.get("x-request-id"),
        )

    def _post(
        self,
        path: str,
        body: dict[str, object],
        *,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        try:
            response = self._client.post(path, json=body, headers=headers)
            response.raise_for_status()
            return response
        except httpx.TimeoutException as error:
            raise ModelGatewayError("provider_timeout", str(error)) from error
        except httpx.HTTPStatusError as error:
            code = "rate_limited" if error.response.status_code == 429 else "provider_error"
            raise ModelGatewayError(code, f"HTTP {error.response.status_code}") from error
        except httpx.HTTPError as error:
            raise ModelGatewayError("provider_unavailable", str(error)) from error


def _output_text(body: dict[str, Any]) -> str:
    pieces: list[str] = []
    for item in body.get("output", []):
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if not isinstance(content, dict):
                continue
            if content.get("type") == "refusal":
                raise ModelGatewayError(
                    "provider_refusal",
                    "Provider refused the requested generation",
                )
            if content.get("type") == "output_text":
                pieces.append(str(content.get("text") or ""))
    return "".join(pieces)


def _usage(body: dict[str, Any]) -> TokenUsage:
    usage = body.get("usage")
    if not isinstance(usage, dict):
        raise ModelGatewayError("usage_missing", "Provider response has no usage")
    input_details = usage.get("input_tokens_details") or {}
    output_details = usage.get("output_tokens_details") or {}
    try:
        return TokenUsage(
            input_tokens=int(usage["input_tokens"]),
            output_tokens=int(usage["output_tokens"]),
            total_tokens=int(usage["total_tokens"]),
            cached_input_tokens=int(input_details.get("cached_tokens") or 0),
            reasoning_tokens=int(output_details.get("reasoning_tokens") or 0),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ModelGatewayError("usage_invalid", "Provider usage is invalid") from error


def _validate_query(output: str) -> str:
    query = output.strip()
    if not query:
        raise ModelGatewayError("empty_output", "Provider returned no query")
    if "\n" in query or "\r" in query:
        raise ModelGatewayError("invalid_query", "Query must contain one line")
    if len(query.split()) > 30:
        raise ModelGatewayError("invalid_query", "Query exceeds 30 words")
    if any(not (character.isalnum() or character == " ") for character in query):
        raise ModelGatewayError(
            "invalid_query",
            "Query contains punctuation other than spaces",
        )
    return " ".join(query.split())


def _response_id(body: dict[str, Any]) -> str:
    value = body.get("id")
    if not isinstance(value, str) or not value:
        raise ModelGatewayError("response_id_missing", "Provider response has no ID")
    return value


def _canonical_hash(value: object) -> str:
    canonical = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


