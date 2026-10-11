"""
driftguard/remediation/mapper.py

Maps a categorised shift to a recommended first action for the operator.
DriftGuard never acts on its own: these are suggestions shown next to an
alert, and adapting a production dependency stays a human decision.
"""

from __future__ import annotations

REMEDIATION: dict[str, str] = {
    "accuracy_error": (
        "Re-run the frozen suite to confirm, then pin a dated model snapshot if the "
        "provider offers one, or route affected traffic to a fallback model. Check the "
        "provider changelog for a model update around the shift."
    ),
    "format_error": (
        "Turn on the provider's JSON / structured-output mode or tighten the schema "
        "instruction, and add a parse-and-retry guard in the client."
    ),
    "safe_refusal": (
        "Inspect the refused prompts in the call explorer. Adjust the system prompt, "
        "or move the affected flows to a model with a less aggressive safety layer."
    ),
    "unsafe_compliance": (
        "Escalate to the safety owner. Put an output moderation check in front of users "
        "until the model's refusal behaviour is confirmed."
    ),
    "call_failure": (
        "Check the provider status page and your rate-limit headroom. Add backoff with "
        "retries, and fail over to a second provider if errors persist."
    ),
    "quota_exhausted": (
        "Client-side, not a provider change: slow the call pacing, spread runs across "
        "the day, or move to a plan with a higher quota."
    ),
    "slow_call": (
        "Check the provider status page, raise client timeouts, and route "
        "latency-sensitive requests to a fallback endpoint."
    ),
}


def remediation_for(signal_key: str, worsening: bool = True) -> str:
    """Recommended action for a shift on `signal_key`. Improving shifts need none."""
    if not worsening:
        return "No action needed: the signal improved. Confirm the new level is expected."
    return REMEDIATION.get(signal_key, "Review the calls around the shift in the call explorer.")
