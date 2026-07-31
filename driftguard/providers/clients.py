"""
Provider clients for DriftGuard.

Both Groq and OpenRouter expose OpenAI-compatible chat completion endpoints,
so we talk to both over plain httpx rather than pulling in provider-specific
SDKs -- one code path, two base URLs. This also keeps things consistent with
the later httpx-transport-level wrapper (ingest/transport_hook.py), which
intercepts at the same layer.

Every call goes through `call_model(...)`, which:
  - applies a per-provider rate limiter (blocking sleep, since this runs as
    a scheduled batch job, not a high-concurrency service)
  - times the call
  - normalizes success/error/timeout into a DriftLogEntry
  - never raises on provider errors -- callers get an entry with
    outcome=ERROR/TIMEOUT/RATE_LIMITED instead, so one bad call in a batch
    of 20 doesn't kill the whole eval run

Free-tier limits this is sized against (verify current values periodically,
they do change):
  - Groq:        default tier ~30 requests/min for most models
  - OpenRouter:  free models ~20 requests/min

API keys are read from environment variables GROQ_API_KEY and
OPENROUTER_API_KEY, loaded via python-dotenv from a .env file at the
project root. Never hardcode keys here.
"""

from __future__ import annotations

import os
import time
import threading
from collections import deque
from dataclasses import dataclass
from typing import Any, Optional

import httpx
from dotenv import load_dotenv

from driftguard.ingest.log_schema import CallOutcome, DriftLogEntry, IntegrationPath

load_dotenv()


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

class RateLimiter:
    """
    Simple blocking sliding-window rate limiter: at most `max_calls` calls
    in any rolling `period_seconds` window. Blocking (time.sleep) is fine
    here since the eval runner makes calls sequentially, not concurrently.
    """

    def __init__(self, max_calls: int, period_seconds: float = 60.0):
        self.max_calls = max_calls
        self.period_seconds = period_seconds
        self._call_times: deque[float] = deque()
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            # drop timestamps outside the rolling window
            while self._call_times and now - self._call_times[0] > self.period_seconds:
                self._call_times.popleft()

            if len(self._call_times) >= self.max_calls:
                sleep_for = self.period_seconds - (now - self._call_times[0])
                if sleep_for > 0:
                    time.sleep(sleep_for)
                now = time.monotonic()
                while self._call_times and now - self._call_times[0] > self.period_seconds:
                    self._call_times.popleft()

            self._call_times.append(time.monotonic())


# ---------------------------------------------------------------------------
# Provider configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProviderConfig:
    name: str
    base_url: str
    api_key_env: str
    rate_limiter: RateLimiter
    extra_headers: dict[str, str]


PROVIDERS: dict[str, ProviderConfig] = {
    "groq": ProviderConfig(
        name="groq",
        base_url="https://api.groq.com/openai/v1",
        api_key_env="GROQ_API_KEY",
        rate_limiter=RateLimiter(max_calls=25, period_seconds=60.0),  # 5-call safety margin under the 30 RPM cap
        extra_headers={},
    ),
    "openrouter": ProviderConfig(
        name="openrouter",
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        rate_limiter=RateLimiter(max_calls=18, period_seconds=60.0),  # margin under the 20 RPM cap
        extra_headers={
            # optional but recommended by OpenRouter for attribution; harmless if unused
            "HTTP-Referer": "https://github.com/GriffinJolly/driftguard",
            "X-Title": "DriftGuard",
        },
    ),
}


def _get_api_key(config: ProviderConfig) -> str:
    key = os.getenv(config.api_key_env)
    if not key:
        raise RuntimeError(
            f"Missing {config.api_key_env} in environment. "
            f"Add it to your .env file at the project root."
        )
    return key


# ---------------------------------------------------------------------------
# Core call function
# ---------------------------------------------------------------------------

def call_model(
    provider: str,
    model_id: str,
    prompt: str,
    *,
    integration_path: IntegrationPath = IntegrationPath.EVAL_SUITE,
    request_params: Optional[dict[str, Any]] = None,
    eval_run_id: Optional[str] = None,
    eval_task_type: Optional[str] = None,
    eval_task_id: Optional[str] = None,
    timeout_seconds: float = 30.0,
    client: Optional[httpx.Client] = None,
) -> DriftLogEntry:
    """
    Make one chat-completion call to `provider` for `model_id`, and return
    a fully-populated DriftLogEntry. Never raises on provider-side failure
    -- errors are captured in the entry's outcome/error_message so a batch
    of calls (e.g. one eval-suite run) can continue past a single failure.

    `client` may be injected for testing (e.g. httpx.Client with a
    MockTransport); if omitted, a real client is created per call.
    """
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider '{provider}'. Known: {list(PROVIDERS)}")

    config = PROVIDERS[provider]
    request_params = request_params or {}
    api_key = _get_api_key(config)

    config.rate_limiter.wait()

    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        **request_params,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        **config.extra_headers,
    }

    owns_client = client is None
    if owns_client:
        client = httpx.Client(base_url=config.base_url, timeout=timeout_seconds)

    start = time.monotonic()
    try:
        resp = client.post("/chat/completions", json=payload, headers=headers)
        latency_ms = (time.monotonic() - start) * 1000

        if resp.status_code == 429:
            return DriftLogEntry(
                integration_path=integration_path,
                provider=provider,
                model_id=model_id,
                prompt=prompt,
                request_params=request_params,
                outcome=CallOutcome.RATE_LIMITED,
                error_message=f"HTTP 429: {resp.text[:500]}",
                latency_ms=latency_ms,
                eval_run_id=eval_run_id,
                eval_task_type=eval_task_type,
                eval_task_id=eval_task_id,
            )

        resp.raise_for_status()
        data = resp.json()
        choice = data["choices"][0]
        response_text = choice["message"]["content"]
        usage = data.get("usage", {})

        return DriftLogEntry(
            integration_path=integration_path,
            provider=provider,
            model_id=model_id,
            prompt=prompt,
            response_text=response_text,
            request_params=request_params,
            outcome=CallOutcome.OK,
            latency_ms=latency_ms,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            eval_run_id=eval_run_id,
            eval_task_type=eval_task_type,
            eval_task_id=eval_task_id,
        )

    except httpx.TimeoutException as e:
        latency_ms = (time.monotonic() - start) * 1000
        return DriftLogEntry(
            integration_path=integration_path,
            provider=provider,
            model_id=model_id,
            prompt=prompt,
            request_params=request_params,
            outcome=CallOutcome.TIMEOUT,
            error_message=str(e),
            latency_ms=latency_ms,
            eval_run_id=eval_run_id,
            eval_task_type=eval_task_type,
            eval_task_id=eval_task_id,
        )

    except Exception as e:  # noqa: BLE001 -- deliberately broad, see docstring
        latency_ms = (time.monotonic() - start) * 1000
        return DriftLogEntry(
            integration_path=integration_path,
            provider=provider,
            model_id=model_id,
            prompt=prompt,
            request_params=request_params,
            outcome=CallOutcome.ERROR,
            error_message=str(e)[:1000],
            latency_ms=latency_ms,
            eval_run_id=eval_run_id,
            eval_task_type=eval_task_type,
            eval_task_id=eval_task_id,
        )

    finally:
        if owns_client:
            client.close()