import asyncio

from httpx import ASGITransport, AsyncClient

from redacted_backend.config import Settings
from redacted_backend.main import create_app


def test_readiness_never_equates_http_health_with_prediction_readiness() -> None:
    settings = Settings(
        database_url=None,
        redis_url=None,
        corpus_manifest_path=None,
        openai_api_key=None,
    )

    async def request_readiness():
        transport = ASGITransport(app=create_app(settings))
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.get("/v1/readiness")

    response = asyncio.run(request_readiness())

    assert response.status_code == 200
    body = response.json()
    assert body["http"] == {"ready": True, "detail": "ready"}
    assert body["database"]["ready"] is False
    assert body["worker"]["ready"] is False
    assert body["corpus"]["ready"] is False
    assert body["model"]["ready"] is False
    assert body["prediction_ready"] is False
    assert set(body) == {
        "http",
        "database",
        "worker",
        "corpus",
        "model",
        "prediction_ready",
    }
