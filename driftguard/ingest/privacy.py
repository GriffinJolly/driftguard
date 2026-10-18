"""
driftguard/ingest/privacy.py

What the capture layer keeps of application traffic.

Detection only needs per-call outcomes, timings and, for the eval suite, a score.
The text of a production prompt or reply is useful for an operator reading the
call explorer, but it can contain personal data. The capture transports therefore
take a text policy:

  "full"    keep prompt and reply as sent (the default, unchanged behaviour)
  "redact"  keep them with e-mail addresses, phone numbers, card-like numbers,
            IP addresses and API keys replaced by placeholders
  "hash"    keep only a SHA-256 digest, so identical prompts can still be grouped
            (for example to drop repeats) but the text cannot be read back
  "none"    keep no text at all

Eval-suite calls are not affected: their prompts are the frozen public benchmark
questions, which the flip-set evidence needs. Retention is handled separately by
DriftStore.purge_logs.

The redaction is pattern-based. It removes common identifiers, not all personal
information (names and free-text details remain), so "hash" or "none" is the
safer choice when prompts may be sensitive.
"""

from __future__ import annotations

import hashlib
import re
from typing import Optional

TEXT_POLICIES = ("full", "redact", "hash", "none")

_PATTERNS = [
    # API keys and bearer tokens first, so their digits are not read as phone numbers
    (re.compile(r"\b(?:sk|gsk|nvapi|pk|rk)[-_][A-Za-z0-9_\-]{16,}\b"), "[SECRET]"),
    (re.compile(r"\bBearer\s+[A-Za-z0-9._\-]{16,}\b"), "Bearer [SECRET]"),
    (re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"), "[EMAIL]"),
    (re.compile(r"\b(?:\d[ \-]?){13,19}\b"), "[CARD]"),
    (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), "[IP]"),
    (re.compile(r"(?<![\w])\+?\d[\d ().\-]{7,}\d\b"), "[PHONE]"),
]


def redact_text(text: str) -> str:
    """Replace common identifiers in `text` with placeholders."""
    for pattern, placeholder in _PATTERNS:
        text = pattern.sub(placeholder, text)
    return text


def apply_text_policy(text: Optional[str], policy: str) -> Optional[str]:
    """What to store for one prompt or reply under `policy`."""
    if policy not in TEXT_POLICIES:
        raise ValueError(f"Unknown text policy {policy!r}; use one of {TEXT_POLICIES}")
    if text is None or policy == "full":
        return text
    if policy == "redact":
        return redact_text(text)
    if policy == "hash":
        return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()
    return ""
