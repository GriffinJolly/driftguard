from __future__ import annotations

import argparse
from contextlib import asynccontextmanager
from typing import AsyncIterator, Mapping, Optional

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from driftguard.ingest.log_schema import IntegrationPath
from driftguard.ingest.transport_hook import AsyncDriftCaptureTransport
from driftguard.storage.store import DriftStore


DEFAULT_PROVIDER_BASE_URLS = {
    "groq": "https://api.groq.com",
    "openrouter": "https://openrouter.ai",
    "openai": "https://api.openai.com",
    "anthropic": "https://api.anthropic.com",
}

_HOP_BY_HOP_HEADERS = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailer", "transfer-encoding", "upgrade",
}


def _forward_headers(headers: Mapping[str, str]) -> dict[str, str]:
    return {
        name: value
        for name, value in headers.items()
        if name.lower() not in _HOP_BY_HOP_HEADERS and name.lower() != "host"
    }


def _response_headers(headers: Mapping[str, str]) -> dict[str, str]:
    return {
        name: value
        for name, value in headers.items()
        if name.lower() not in _HOP_BY_HOP_HEADERS
        and name.lower() not in {"content-length", "content-encoding"}
    }


def create_proxy_app(
    *,
    store: Optional[DriftStore] = None,
    provider_base_urls: Optional[Mapping[str, str]] = None,
    upstream_transports: Optional[Mapping[str, httpx.AsyncBaseTransport]] = None,
    timeout: float = 120.0,
) -> FastAPI:
    """Create a standalone reverse proxy at ``/{provider}/{path}``."""
    if timeout <= 0:
        raise ValueError("timeout must be positive")

    urls = {
        name.strip().lower(): url.rstrip("/")
        for name, url in (provider_base_urls or DEFAULT_PROVIDER_BASE_URLS).items()
    }
    if not urls or any(not name or not url for name, url in urls.items()):
        raise ValueError("provider_base_urls must contain non-empty names and URLs")

    proxy_store = store or DriftStore()
    clients: dict[str, httpx.AsyncClient] = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        for provider, base_url in urls.items():
            wrapped = (upstream_transports or {}).get(provider)
            transport = AsyncDriftCaptureTransport(
                store=proxy_store,
                wrapped=wrapped,
                provider_hosts={httpx.URL(base_url).host or ""},
                integration_path=IntegrationPath.PROXY,
            )
            clients[provider] = httpx.AsyncClient(
                transport=transport,
                base_url=base_url,
                timeout=timeout,
            )
        try:
            yield
        finally:
            for client in clients.values():
                await client.aclose()
            clients.clear()

    app = FastAPI(title="DriftGuard Proxy", lifespan=lifespan)

    @app.get("/")
    async def root() -> dict[str, object]:
        return {
            "service": "driftguard-proxy",
            "status": "ok",
            "providers": sorted(urls),
            "health": "/health",
            "docs": "/docs",
        }

    @app.get("/health")
    async def health() -> dict[str, object]:
        return {"status": "ok", "providers": sorted(urls)}

    @app.api_route(
        "/{provider}/{path:path}",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
    )
    async def forward(provider: str, path: str, request: Request) -> Response:
        client = clients.get(provider.lower())
        if client is None:
            return JSONResponse(
                status_code=404,
                content={"error": f"unknown provider '{provider}'", "providers": sorted(urls)},
            )

        upstream_path = f"/{path}" if path else "/"
        if request.url.query:
            upstream_path += f"?{request.url.query}"

        try:
            upstream_response = await client.request(
                request.method,
                upstream_path,
                headers=_forward_headers(request.headers),
                content=await request.body(),
            )
        except httpx.RequestError as exc:
            return JSONResponse(status_code=502, content={"error": str(exc)})

        return Response(
            content=upstream_response.content,
            status_code=upstream_response.status_code,
            headers=_response_headers(upstream_response.headers),
            media_type=upstream_response.headers.get("content-type"),
        )

    return app


def _main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description="Run the DriftGuard reverse proxy")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--db-path", default="driftguard.db")
    args = parser.parse_args()
    uvicorn.run(
        create_proxy_app(store=DriftStore(db_path=args.db_path)),
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    _main()


__all__ = ["DEFAULT_PROVIDER_BASE_URLS", "create_proxy_app"]
