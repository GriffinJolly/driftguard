"""Unit tests for driftguard/providers/clients.py.

call_model accepts an injected httpx.Client, so every provider response
(success / 429 / HTTP error / timeout) is exercised with an httpx.MockTransport
-- no real network, no real API key."""

from __future__ import annotations

import time

import httpx
import pytest

from driftguard.ingest.log_schema import CallOutcome
from driftguard.providers.clients import RateLimiter, call_model


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test")


def _ok_handler(request):
    return httpx.Response(200, json={
        "choices": [{"message": {"content": "B"}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 1},
    })


def test_unknown_provider_raises():
    with pytest.raises(ValueError):
        call_model("nope", "m", "hi")


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        call_model("groq", "m", "hi", client=_client(_ok_handler))


def test_call_ok(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "k")
    entry = call_model("groq", "m", "hi", client=_client(_ok_handler))
    assert entry.outcome == CallOutcome.OK
    assert entry.response_text == "B"
    assert entry.prompt_tokens == 5 and entry.completion_tokens == 1
    assert entry.latency_ms is not None and entry.latency_ms >= 0


def test_call_rate_limited(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "k")
    entry = call_model("groq", "m", "hi",
                       client=_client(lambda r: httpx.Response(429, text="slow down")))
    assert entry.outcome == CallOutcome.RATE_LIMITED
    assert entry.response_text is None
    assert "429" in entry.error_message


def test_call_http_error_becomes_error_outcome(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "k")
    entry = call_model("groq", "m", "hi",
                       client=_client(lambda r: httpx.Response(500, text="boom")))
    assert entry.outcome == CallOutcome.ERROR
    assert entry.error_message


def test_call_timeout(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "k")

    def handler(request):
        raise httpx.TimeoutException("timed out")

    entry = call_model("groq", "m", "hi", client=_client(handler))
    assert entry.outcome == CallOutcome.TIMEOUT


def test_request_params_forwarded(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    seen = {}

    def handler(request):
        import json
        seen.update(json.loads(request.content))
        return _ok_handler(request)

    call_model("openrouter", "m", "hi", request_params={"temperature": 0.7, "max_tokens": 16},
               client=_client(handler))
    assert seen["model"] == "m"
    assert seen["temperature"] == 0.7 and seen["max_tokens"] == 16
    assert seen["messages"] == [{"role": "user", "content": "hi"}]


def test_rate_limiter_blocks_over_window():
    rl = RateLimiter(max_calls=2, period_seconds=0.3)
    start = time.monotonic()
    rl.wait()
    rl.wait()          # two calls allowed immediately
    rl.wait()          # third must wait out the window
    elapsed = time.monotonic() - start
    assert elapsed >= 0.25
