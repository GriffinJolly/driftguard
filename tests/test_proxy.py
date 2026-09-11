import httpx
import pytest

from driftguard.ingest.log_schema import IntegrationPath
from driftguard.ingest.proxy import create_proxy_app


class FakeStore:
    def __init__(self):
        self.saved = []

    def write_log(self, entry):
        self.saved.append(entry)


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_proxy_forwards_and_captures_request():
    store = FakeStore()

    async def upstream_handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/openai/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer test"
        return httpx.Response(
            200,
            headers={"content-type": "application/json", "x-upstream": "yes"},
            json={
                "model": "test-model",
                "choices": [{"message": {"content": "hello"}}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1},
            },
        )

    app = create_proxy_app(
        store=store,
        provider_base_urls={"groq": "https://api.groq.com"},
        upstream_transports={"groq": httpx.MockTransport(upstream_handler)},
    )

    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://proxy") as client:
            response = await client.post(
                "/groq/openai/v1/chat/completions",
                headers={"Authorization": "Bearer test"},
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )

    assert response.status_code == 200
    assert response.headers["x-upstream"] == "yes"
    assert response.json()["choices"][0]["message"]["content"] == "hello"
    assert len(store.saved) == 1
    assert store.saved[0].integration_path == IntegrationPath.PROXY.value
    assert store.saved[0].provider == "groq"
    assert store.saved[0].model_id == "test-model"


@pytest.mark.anyio
async def test_proxy_health_and_unknown_provider():
    app = create_proxy_app(provider_base_urls={"groq": "https://api.groq.com"})

    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://proxy") as client:
            root = await client.get("/")
            health = await client.get("/health")
            unknown = await client.get("/unknown/status")

    assert root.status_code == 200
    assert root.json()["service"] == "driftguard-proxy"
    assert health.status_code == 200
    assert health.json() == {"status": "ok", "providers": ["groq"]}
    assert unknown.status_code == 404
    assert unknown.json()["providers"] == ["groq"]
