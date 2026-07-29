"""
Shared log schema for DriftGuard.

Both ingestion paths (wrapper mode and proxy mode) must produce entries
conforming to this exact schema. This is the single contract that lets
the downstream pipeline (eval suite runner, detectors, dashboard) treat
both integration paths identically.

Design notes:
- `model_id` is the *nominal* model string the caller asked for (e.g.
  "llama-3.3-70b-versatile"). It is intentionally NOT assumed to uniquely
  identify the actual serving weights/infra behind it -- that is the
  entire premise of this project.
- `provider` distinguishes which backend actually served the request
  (groq, openrouter, openai, anthropic, ...). Combined with `model_id`,
  (provider, model_id) is the practical identity key the eval suite
  tracks a metric time series against.
- Token counts and latency are captured for every call, not just eval
  suite calls, so production traffic logged via wrapper/proxy also
  contributes passive signal if the operator wants it later. The eval
  suite runner is the *primary* source of the metric time series that
  feeds detection, since it uses controlled, versioned prompts -- raw
  production traffic has uncontrolled prompt distribution and is not
  used for change-point detection directly in v1.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


class IntegrationPath(str, Enum):
    WRAPPER = "wrapper"
    PROXY = "proxy"
    EVAL_SUITE = "eval_suite"  # direct calls made by the eval runner itself


class CallOutcome(str, Enum):
    OK = "ok"
    ERROR = "error"
    TIMEOUT = "timeout"
    RATE_LIMITED = "rate_limited"


class DriftLogEntry(BaseModel):
    """One row per LLM API call, regardless of ingestion path."""

    # --- identity -----------------------------------------------------
    log_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    integration_path: IntegrationPath

    provider: str  # "groq" | "openrouter" | "openai" | "anthropic" | ...
    model_id: str  # the model string exactly as requested by the caller

    # --- request/response content -------------------------------------
    # Stored for exemplar surfacing in the dashboard. Truncated at write
    # time by the caller if very large; this schema does not itself cap
    # length so callers can decide their own retention/size policy.
    prompt: str
    response_text: Optional[str] = None
    request_params: dict[str, Any] = Field(default_factory=dict)  # temperature, max_tokens, etc.

    # --- outcome --------------------------------------------------------
    outcome: CallOutcome = CallOutcome.OK
    error_message: Optional[str] = None

    # --- timing -----------------------------------------------------
    latency_ms: Optional[float] = None

    # --- token accounting ---------------------------------------------
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None

    # --- eval-suite-specific tagging (None for raw production traffic) --
    eval_run_id: Optional[str] = None       # groups all calls from one eval_suite run
    eval_task_type: Optional[str] = None    # "accuracy" | "format" | "refusal" | "latency"
    eval_task_id: Optional[str] = None      # id of the specific fixed task/prompt used
    eval_score: Optional[float] = None      # per-call score, if applicable (e.g. 1.0/0.0 for accuracy)

    @field_validator("provider")
    @classmethod
    def _lower_provider(cls, v: str) -> str:
        return v.strip().lower()

    model_config = {
        "use_enum_values": True,
    }


class MetricPoint(BaseModel):
    """
    One aggregated point in a metric time series, produced by the eval
    suite runner from a batch of DriftLogEntry rows sharing an
    eval_run_id. This -- not raw DriftLogEntry rows -- is what the
    detection stage consumes.
    """

    run_id: str
    timestamp: datetime
    provider: str
    model_id: str

    accuracy: Optional[float] = None          # fraction correct on held-out tasks
    format_adherence: Optional[float] = None  # fraction of responses passing schema check
    refusal_rate: Optional[float] = None       # fraction of borderline prompts refused
    latency_p50_ms: Optional[float] = None
    latency_p95_ms: Optional[float] = None

    n_calls: int = 0  # number of DriftLogEntry rows aggregated into this point