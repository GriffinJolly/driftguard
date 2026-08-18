"""
driftguard/tests/test_transport_hook.py

Fast, offline tests for transport_hook.py using httpx.MockTransport --
no real network calls, no API cost. Run with:

    pytest driftguard/tests/test_transport_hook.py -v

These verify:
  1. Requests to allowlisted provider hosts are captured (a DriftLogEntry
     is saved, tagged integration_path=WRAPPER by default) AND the
     response passed back to the caller is unmodified.
  2. Requests to non-provider hosts pass through with zero capture calls.
  3. Capture/logging failures never propagate -- the real response is
     still returned even if DriftStore.write_log raises.
  4. Error responses (4xx/5xx) and rate limits (429) map to the right
     CallOutcome and don't crash the transport.
  5. integration_path can be overridden (e.g. for future proxy.py use).
"""

import httpx
import pytest

from driftguard.ingest.log_schema import CallOutcome, IntegrationPath
from driftguard.ingest.transport_hook import (
    DriftCaptureTransport,
    make_capturing_client,
)


class FakeStore:
    """Minimal stand-in for DriftStore -- just records what it's given."""

    def __init__(self, raise_on_save: bool = False):
        self.saved = []
        self.raise_on_save = raise_on_save

    def write_log(self, entry):
        if self.raise_on_save:
            raise RuntimeError("simulated storage failure")
        self.saved.append(entry)


def _chat_completion_response(model="llama-3.3-70b-versatile", content="hello back"):
    return httpx.Response(
        200,
        json={
            "model": model,
            "choices": [{"message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        },
    )


def test_captures_request_to_allowlisted_provider():
    store = FakeStore()

    def handler(request: httpx.Request) -> httpx.Response:
        return _chat_completion_response()

    mock_transport = httpx.MockTransport(handler)
    capture_transport = DriftCaptureTransport(store=store, wrapped=mock_transport)
    client = httpx.Client(transport=capture_transport)

    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0.7,
    }
    resp = client.post("https://api.groq.com/openai/v1/chat/completions", json=payload)

    assert resp.status_code == 200
    assert resp.json()["choices"][0]["message"]["content"] == "hello back"

    assert len(store.saved) == 1
    entry = store.saved[0]
    assert entry.integration_path == IntegrationPath.WRAPPER.value
    assert entry.provider == "groq"
    assert entry.model_id == "llama-3.3-70b-versatile"
    assert entry.outcome == CallOutcome.OK.value
    assert entry.error_message is None
    assert entry.latency_ms is not None and entry.latency_ms >= 0
    assert entry.prompt_tokens == 5
    assert entry.completion_tokens == 3
    assert entry.response_text == "hello back"
    assert entry.request_params.get("temperature") == 0.7
    assert "model" not in entry.request_params
    assert "messages" not in entry.request_params


def test_non_provider_host_passes_through_uncaptured():
    store = FakeStore()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"ok": True})

    mock_transport = httpx.MockTransport(handler)
    capture_transport = DriftCaptureTransport(store=store, wrapped=mock_transport)
    client = httpx.Client(transport=capture_transport)

    resp = client.get("https://example.com/health")

    assert resp.status_code == 200
    assert store.saved == []


def test_capture_failure_does_not_break_real_response():
    store = FakeStore(raise_on_save=True)

    def handler(request: httpx.Request) -> httpx.Response:
        return _chat_completion_response()

    mock_transport = httpx.MockTransport(handler)

    captured_errors = []
    capture_transport = DriftCaptureTransport(
        store=store,
        wrapped=mock_transport,
        on_capture_error=lambda exc: captured_errors.append(exc),
    )
    client = httpx.Client(transport=capture_transport)

    payload = {"model": "llama-3.3-70b-versatile", "messages": [{"role": "user", "content": "hi"}]}
    resp = client.post("https://api.groq.com/openai/v1/chat/completions", json=payload)

    assert resp.status_code == 200
    assert resp.json()["choices"][0]["message"]["content"] == "hello back"

    assert len(captured_errors) == 1
    assert isinstance(captured_errors[0], RuntimeError)


def test_error_response_from_provider_is_logged_as_error_outcome():
    store = FakeStore()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": {"message": "internal error"}})

    mock_transport = httpx.MockTransport(handler)
    capture_transport = DriftCaptureTransport(store=store, wrapped=mock_transport)
    client = httpx.Client(transport=capture_transport)

    payload = {"model": "llama-3.3-70b-versatile", "messages": [{"role": "user", "content": "hi"}]}
    resp = client.post("https://api.groq.com/openai/v1/chat/completions", json=payload)

    assert resp.status_code == 500
    assert len(store.saved) == 1
    entry = store.saved[0]
    assert entry.outcome == CallOutcome.ERROR.value
    assert entry.error_message == "internal error"


def test_rate_limited_response_maps_to_rate_limited_outcome():
    store = FakeStore()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": {"message": "rate limit exceeded"}})

    mock_transport = httpx.MockTransport(handler)
    capture_transport = DriftCaptureTransport(store=store, wrapped=mock_transport)
    client = httpx.Client(transport=capture_transport)

    payload = {"model": "llama-3.3-70b-versatile", "messages": [{"role": "user", "content": "hi"}]}
    resp = client.post("https://api.groq.com/openai/v1/chat/completions", json=payload)

    assert resp.status_code == 429
    entry = store.saved[0]
    assert entry.outcome == CallOutcome.RATE_LIMITED.value


def test_timeout_is_logged_and_reraised():
    store = FakeStore()

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    mock_transport = httpx.MockTransport(handler)
    capture_transport = DriftCaptureTransport(store=store, wrapped=mock_transport)
    client = httpx.Client(transport=capture_transport)

    payload = {"model": "llama-3.3-70b-versatile", "messages": [{"role": "user", "content": "hi"}]}
    with pytest.raises(httpx.ReadTimeout):
        client.post("https://api.groq.com/openai/v1/chat/completions", json=payload)

    assert len(store.saved) == 1
    entry = store.saved[0]
    assert entry.outcome == CallOutcome.TIMEOUT.value
    assert entry.error_message is not None


def test_custom_integration_path_is_respected():
    store = FakeStore()

    def handler(request: httpx.Request) -> httpx.Response:
        return _chat_completion_response()

    mock_transport = httpx.MockTransport(handler)
    capture_transport = DriftCaptureTransport(
        store=store, wrapped=mock_transport, integration_path=IntegrationPath.PROXY
    )
    client = httpx.Client(transport=capture_transport)

    payload = {"model": "llama-3.3-70b-versatile", "messages": [{"role": "user", "content": "hi"}]}
    client.post("https://api.groq.com/openai/v1/chat/completions", json=payload)

    assert store.saved[0].integration_path == IntegrationPath.PROXY.value


def test_make_capturing_client_factory():
    store = FakeStore()
    client = make_capturing_client(store=store)
    assert isinstance(client, httpx.Client)
    client.close()


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))