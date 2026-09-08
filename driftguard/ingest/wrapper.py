from __future__ import annotations

from typing import Any, Optional

import httpx

from driftguard.ingest.log_schema import IntegrationPath
from driftguard.ingest.transport_hook import make_capturing_client
from driftguard.storage.store import DriftStore


DEFAULT_PROVIDER_BASE_URLS = {
    "groq": "https://api.groq.com/openai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "openai": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com",
}


class DriftGuardClient:
    """Thin wrapper around an httpx client with DriftGuard logging enabled.

    This is the one-line integration path for application code: instead of
    building a raw httpx.Client, app code can create a DriftGuardClient and
    keep using the same request API while the transport hook captures and stores
    provider traffic automatically.
    """

    def __init__(
        self,
        *,
        provider: str,
        store: Optional[DriftStore] = None,
        base_url: Optional[str] = None,
        provider_hosts: Optional[set[str]] = None,
        integration_path: IntegrationPath = IntegrationPath.WRAPPER,
        **client_kwargs: Any,
    ) -> None:
        self.provider = provider.strip().lower()
        self.store = store or DriftStore()
        self.base_url = base_url or DEFAULT_PROVIDER_BASE_URLS.get(self.provider)
        if not self.base_url:
            raise ValueError(
                f"Unknown provider '{provider}'. Supported providers: "
                f"{sorted(DEFAULT_PROVIDER_BASE_URLS)}"
            )

        self.client = make_capturing_client(
            store=self.store,
            provider_hosts=provider_hosts,
            integration_path=integration_path,
            base_url=self.base_url,
            **client_kwargs,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self.client, name)

    def request(self, *args: Any, **kwargs: Any) -> httpx.Response:
        return self.client.request(*args, **kwargs)

    def get(self, *args: Any, **kwargs: Any) -> httpx.Response:
        return self.client.get(*args, **kwargs)

    def post(self, *args: Any, **kwargs: Any) -> httpx.Response:
        return self.client.post(*args, **kwargs)

    def put(self, *args: Any, **kwargs: Any) -> httpx.Response:
        return self.client.put(*args, **kwargs)

    def patch(self, *args: Any, **kwargs: Any) -> httpx.Response:
        return self.client.patch(*args, **kwargs)

    def delete(self, *args: Any, **kwargs: Any) -> httpx.Response:
        return self.client.delete(*args, **kwargs)

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> "DriftGuardClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def make_wrapped_client(
    *,
    store: Optional[DriftStore] = None,
    provider: str = "groq",
    base_url: Optional[str] = None,
    provider_hosts: Optional[set[str]] = None,
    integration_path: IntegrationPath = IntegrationPath.WRAPPER,
    **client_kwargs: Any,
) -> httpx.Client:
    """Factory returning a pre-configured httpx client with DriftGuard capture enabled."""
    client = DriftGuardClient(
        provider=provider,
        store=store,
        base_url=base_url,
        provider_hosts=provider_hosts,
        integration_path=integration_path,
        **client_kwargs,
    )
    return client.client


__all__ = ["DriftGuardClient", "make_wrapped_client"]
