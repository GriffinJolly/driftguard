"""
driftguard/ingest/transport_hook.py

SDK-agnostic production traffic capture.

Intercepts outbound HTTP calls at the httpx transport layer (not the SDK
layer) so it works identically whether the caller is using the OpenAI SDK,
Anthropic SDK, Groq SDK, raw httpx, etc. -- all of them ultimately push
requests through httpx under the hood.

Design goals (locked in from planning):
  1. Fail-open: logging must NEVER break or delay the real API call. Any
     exception while parsing/logging is caught, routed to an error
     handler, and swallowed. The upstream response (or exception) is
     always returned/raised untouched.
  2. Host allowlist: only requests to known LLM provider hosts are
     captured; everything else passes through with zero overhead.
  3. Normalizes captured data into the real log_schema.py contract
     (DriftLogEntry, tagged with IntegrationPath.WRAPPER by default)
     and writes via DriftStore, so it lands in the exact same table/
     shape as eval_suite rows -- just distinguished by integration_path.

This module is the shared core. wrapper.py (one-line client wrap) passes
integration_path=IntegrationPath.WRAPPER (the default). proxy.py (reverse
-proxy mode, built later) will construct this transport with
integration_path=IntegrationPath.PROXY instead -- everything else about
this module is identical for both.
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from typing import Any, Callable, Optional

import httpx

from driftguard.ingest.log_schema import CallOutcome, DriftLogEntry, IntegrationPath
from driftguard.storage.store import DriftStore


# Known LLM provider hosts. Extend as needed -- kept as a simple set so
# wrapper.py/proxy.py can also import and extend it.
DEFAULT_PROVIDER_HOSTS = {
    "api.groq.com",
    "openrouter.ai",
    "api.openai.com",
    "api.anthropic.com",
}

# Request body keys that are the "content" of the call rather than a
# tunable parameter -- everything else in the request body is treated as
# request_params (temperature, max_tokens, top_p, stream, etc.).
_NON_PARAM_KEYS = {"model", "messages", "prompt"}


def _safe_json(data: bytes | str | None) -> Optional[dict]:
    """Best-effort JSON parse. Returns None rather than raising."""
    if not data:
        return None
    try:
        if isinstance(data, bytes):
            data = data.decode("utf-8", errors="replace")
        return json.loads(data)
    except Exception:
        return None


def _extract_model_id(request_body: Optional[dict], response_body: Optional[dict]) -> str:
    if request_body and request_body.get("model"):
        return request_body["model"]
    if response_body and response_body.get("model"):
        return response_body["model"]
    return "unknown"


def _extract_provider(host: str) -> str:
    if "groq" in host:
        return "groq"
    if "openrouter" in host:
        return "openrouter"
    if "openai" in host:
        return "openai"
    if "anthropic" in host:
        return "anthropic"
    return host


def _extract_prompt(request_body: Optional[dict]) -> str:
    """Required field on DriftLogEntry -- always returns a string, using a
    safe fallback if the request body couldn't be parsed at all."""
    if not request_body:
        return ""
    messages = request_body.get("messages")
    if messages:
        try:
            return json.dumps(messages)[:4000]
        except Exception:
            return str(messages)[:4000]
    prompt = request_body.get("prompt")
    if prompt:
        return str(prompt)[:4000]
    return ""


def _extract_request_params(request_body: Optional[dict]) -> dict:
    if not request_body:
        return {}
    return {k: v for k, v in request_body.items() if k not in _NON_PARAM_KEYS}


def _extract_response_text(response_body: Optional[dict]) -> Optional[str]:
    if not response_body:
        return None
    try:
        choices = response_body.get("choices")
        if choices:
            first = choices[0]
            if "message" in first:
                return str(first["message"].get("content", ""))[:4000]
            if "text" in first:
                return str(first["text"])[:4000]
    except Exception:
        pass
    return None


def _extract_token_counts(response_body: Optional[dict]) -> tuple[Optional[int], Optional[int]]:
    if not response_body:
        return None, None
    usage = response_body.get("usage")
    if not usage:
        return None, None
    return usage.get("prompt_tokens"), usage.get("completion_tokens")


def _extract_error_message(response_body: Optional[dict], fallback: Optional[str]) -> Optional[str]:
    if response_body:
        err = response_body.get("error")
        if isinstance(err, dict):
            return err.get("message") or str(err)
        if err:
            return str(err)
    return fallback


def _outcome_from_status(status_code: int) -> CallOutcome:
    if 200 <= status_code < 300:
        return CallOutcome.OK
    if status_code == 429:
        return CallOutcome.RATE_LIMITED
    return CallOutcome.ERROR


def _build_log_entry(
    *,
    integration_path: IntegrationPath,
    provider: str,
    request_body: Optional[dict],
    response_body: Optional[dict],
    outcome: CallOutcome,
    error_message: Optional[str],
    latency_ms: float,
) -> DriftLogEntry:
    prompt_tokens, completion_tokens = _extract_token_counts(response_body)
    return DriftLogEntry(
        integration_path=integration_path,
        provider=provider,
        model_id=_extract_model_id(request_body, response_body),
        prompt=_extract_prompt(request_body),
        response_text=_extract_response_text(response_body),
        request_params=_extract_request_params(request_body),
        outcome=outcome,
        error_message=error_message,
        latency_ms=latency_ms,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


def _default_error_handler(exc: Exception) -> None:
    # Never raise. Just surface to stderr so it's visible during dev
    # without ever affecting the real request/response path.
    print(f"[driftguard.transport_hook] capture failed: {exc}", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)


class DriftCaptureTransport(httpx.BaseTransport):
    """
    Synchronous httpx transport wrapper. Wraps an existing transport
    (defaults to a fresh httpx.HTTPTransport) and logs matching requests
    to the DriftStore after letting the real request through.
    """

    def __init__(
        self,
        store: DriftStore,
        wrapped: Optional[httpx.BaseTransport] = None,
        provider_hosts: Optional[set[str]] = None,
        integration_path: IntegrationPath = IntegrationPath.WRAPPER,
        on_capture_error: Optional[Callable[[Exception], None]] = None,
    ):
        self._store = store
        self._wrapped = wrapped or httpx.HTTPTransport()
        self._provider_hosts = provider_hosts or DEFAULT_PROVIDER_HOSTS
        self._integration_path = integration_path
        self._on_capture_error = on_capture_error or _default_error_handler

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        host = request.url.host or ""
        should_capture = any(h in host for h in self._provider_hosts)

        if not should_capture:
            return self._wrapped.handle_request(request)

        start = time.monotonic()
        request_body = _safe_json(request.content) if request.content else None

        try:
            response = self._wrapped.handle_request(request)
        except httpx.TimeoutException as exc:
            latency_ms = (time.monotonic() - start) * 1000
            self._safe_capture(
                host=host,
                request_body=request_body,
                response_body=None,
                outcome=CallOutcome.TIMEOUT,
                error_message=str(exc),
                latency_ms=latency_ms,
            )
            raise
        except Exception as exc:
            # Real request itself failed (network error, etc). Log what we
            # can, then re-raise -- capture must never swallow real errors.
            latency_ms = (time.monotonic() - start) * 1000
            self._safe_capture(
                host=host,
                request_body=request_body,
                response_body=None,
                outcome=CallOutcome.ERROR,
                error_message=str(exc),
                latency_ms=latency_ms,
            )
            raise

        latency_ms = (time.monotonic() - start) * 1000

        # Read response body for logging WITHOUT losing it for the caller.
        # httpx.Response.read() buffers content; after that, normal access
        # (.json(), .text, iter_bytes()) still works on the buffered bytes.
        response_body = None
        try:
            response.read()
            response_body = _safe_json(response.content)
        except Exception as exc:
            self._on_capture_error(exc)

        outcome = _outcome_from_status(response.status_code)
        error_message = (
            _extract_error_message(response_body, None) if outcome != CallOutcome.OK else None
        )

        self._safe_capture(
            host=host,
            request_body=request_body,
            response_body=response_body,
            outcome=outcome,
            error_message=error_message,
            latency_ms=latency_ms,
        )

        return response

    def _safe_capture(self, **kwargs) -> None:
        try:
            self._capture(**kwargs)
        except Exception as capture_exc:
            self._on_capture_error(capture_exc)

    def _capture(
        self,
        *,
        host: str,
        request_body: Optional[dict],
        response_body: Optional[dict],
        outcome: CallOutcome,
        error_message: Optional[str],
        latency_ms: float,
    ) -> None:
        entry = _build_log_entry(
            integration_path=self._integration_path,
            provider=_extract_provider(host),
            request_body=request_body,
            response_body=response_body,
            outcome=outcome,
            error_message=error_message,
            latency_ms=latency_ms,
        )
        self._store.write_log(entry)

    def close(self) -> None:
        self._wrapped.close()


class AsyncDriftCaptureTransport(httpx.AsyncBaseTransport):
    """Async counterpart, for httpx.AsyncClient / async SDK usage."""

    def __init__(
        self,
        store: DriftStore,
        wrapped: Optional[httpx.AsyncBaseTransport] = None,
        provider_hosts: Optional[set[str]] = None,
        integration_path: IntegrationPath = IntegrationPath.WRAPPER,
        on_capture_error: Optional[Callable[[Exception], None]] = None,
    ):
        self._store = store
        self._wrapped = wrapped or httpx.AsyncHTTPTransport()
        self._provider_hosts = provider_hosts or DEFAULT_PROVIDER_HOSTS
        self._integration_path = integration_path
        self._on_capture_error = on_capture_error or _default_error_handler

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        host = request.url.host or ""
        should_capture = any(h in host for h in self._provider_hosts)

        if not should_capture:
            return await self._wrapped.handle_async_request(request)

        start = time.monotonic()
        request_body = _safe_json(request.content) if request.content else None

        try:
            response = await self._wrapped.handle_async_request(request)
        except httpx.TimeoutException as exc:
            latency_ms = (time.monotonic() - start) * 1000
            self._safe_capture(
                host=host,
                request_body=request_body,
                response_body=None,
                outcome=CallOutcome.TIMEOUT,
                error_message=str(exc),
                latency_ms=latency_ms,
            )
            raise
        except Exception as exc:
            latency_ms = (time.monotonic() - start) * 1000
            self._safe_capture(
                host=host,
                request_body=request_body,
                response_body=None,
                outcome=CallOutcome.ERROR,
                error_message=str(exc),
                latency_ms=latency_ms,
            )
            raise

        latency_ms = (time.monotonic() - start) * 1000

        response_body = None
        try:
            await response.aread()
            response_body = _safe_json(response.content)
        except Exception as exc:
            self._on_capture_error(exc)

        outcome = _outcome_from_status(response.status_code)
        error_message = (
            _extract_error_message(response_body, None) if outcome != CallOutcome.OK else None
        )

        self._safe_capture(
            host=host,
            request_body=request_body,
            response_body=response_body,
            outcome=outcome,
            error_message=error_message,
            latency_ms=latency_ms,
        )

        return response

    def _safe_capture(self, **kwargs) -> None:
        try:
            self._capture(**kwargs)
        except Exception as capture_exc:
            self._on_capture_error(capture_exc)

    def _capture(
        self,
        *,
        host: str,
        request_body: Optional[dict],
        response_body: Optional[dict],
        outcome: CallOutcome,
        error_message: Optional[str],
        latency_ms: float,
    ) -> None:
        entry = _build_log_entry(
            integration_path=self._integration_path,
            provider=_extract_provider(host),
            request_body=request_body,
            response_body=response_body,
            outcome=outcome,
            error_message=error_message,
            latency_ms=latency_ms,
        )
        self._store.write_log(entry)

    async def aclose(self) -> None:
        await self._wrapped.aclose()


def make_capturing_client(
    store: DriftStore,
    provider_hosts: Optional[set[str]] = None,
    integration_path: IntegrationPath = IntegrationPath.WRAPPER,
    **client_kwargs: Any,
) -> httpx.Client:
    """Convenience factory: httpx.Client pre-wired with capture."""
    transport = DriftCaptureTransport(
        store=store, provider_hosts=provider_hosts, integration_path=integration_path
    )
    return httpx.Client(transport=transport, **client_kwargs)


def make_capturing_async_client(
    store: DriftStore,
    provider_hosts: Optional[set[str]] = None,
    integration_path: IntegrationPath = IntegrationPath.WRAPPER,
    **client_kwargs: Any,
) -> httpx.AsyncClient:
    """Convenience factory: httpx.AsyncClient pre-wired with capture."""
    transport = AsyncDriftCaptureTransport(
        store=store, provider_hosts=provider_hosts, integration_path=integration_path
    )
    return httpx.AsyncClient(transport=transport, **client_kwargs)