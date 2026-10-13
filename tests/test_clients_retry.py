import httpx
import pytest

from driftguard.providers import clients
from driftguard.providers.clients import call_model
from driftguard.ingest.log_schema import CallOutcome

OK_BODY = {
    "choices": [{"message": {"content": "hi"}}],
    "usage": {"prompt_tokens": 3, "completion_tokens": 1},
}


@pytest.fixture
def sleeps(monkeypatch):
    """Fake the API key, record sleeps instead of sleeping, reset the limiter."""
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    recorded = []
    monkeypatch.setattr(clients.time, "sleep", lambda s: recorded.append(s))
    clients.PROVIDERS["groq"].rate_limiter._call_times.clear()
    return recorded


def make_client(sequence):
    """sequence: list of (status, headers); the last entry repeats forever."""
    calls = {"n": 0}

    def handler(request):
        status, headers = sequence[min(calls["n"], len(sequence) - 1)]
        calls["n"] += 1
        if status == 200:
            return httpx.Response(200, json=OK_BODY)
        return httpx.Response(status, headers=headers, text="rate limited")

    client = httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url=clients.PROVIDERS["groq"].base_url,
    )
    return client, calls


def run(client):
    return call_model("groq", "openai/gpt-oss-120b", "ping", client=client)


def test_success_first_try(sleeps):
    client, calls = make_client([(200, {})])
    assert run(client).outcome == CallOutcome.OK
    assert calls["n"] == 1 and sleeps == []


def test_recovers_after_two_429s(sleeps):
    client, calls = make_client([(429, {"retry-after": "1"})] * 2 + [(200, {})])
    entry = run(client)
    assert entry.outcome == CallOutcome.OK
    assert calls["n"] == 3
    assert sleeps == [1.0, 1.0]


def test_gives_up_after_max_retries(sleeps):
    client, calls = make_client([(429, {"retry-after": "1"})])
    assert run(client).outcome == CallOutcome.RATE_LIMITED
    assert calls["n"] == 4          # 1 try + 3 retries
    assert len(sleeps) == 3


def test_long_retry_after_gives_up_immediately(sleeps):
    client, calls = make_client([(429, {"retry-after": "3600"})])
    assert run(client).outcome == CallOutcome.RATE_LIMITED
    assert calls["n"] == 1 and sleeps == []


def test_missing_header_uses_backoff(sleeps):
    client, calls = make_client([(429, {})])
    run(client)
    assert sleeps == [5.0, 10.0, 20.0]