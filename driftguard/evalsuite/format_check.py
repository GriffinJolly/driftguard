"""
Format-adherence task for the DriftGuard eval suite.

Unlike accuracy (MMLU) and refusal (XSTest), this task isn't backed by an
external dataset -- it's testing instruction-following against a schema WE
define, not knowledge, so there's nothing to pull from Hugging Face. The
prompts and schemas below are the frozen, versioned spec directly (no
separate freeze-to-disk step needed, since there's no sampling involved --
every prompt here is used every run).

Task coverage (6 tasks, chosen to span the shapes real structured-output
requests actually take, while staying inside the free-tier rate budget):
  1. flat object, required string/int/string fields
  2. array of strings, fixed length
  3. nested object (one level deep)
  4. enum-constrained string field
  5. array of objects -- the most common real-world shape not covered above
  6. boolean + nullable field -- a distinct, common failure mode
     (models often emit "true"/"false" as strings, or use "" instead of
     null for an absent value)

Two things are checked per response:
  1. LENIENT check (eval_score): does the model's output contain valid
     JSON matching the schema, even if wrapped in markdown fences or
     prose? This is what feeds the accuracy/format drift TIME SERIES and
     what gets stored -- it isolates "did the model get the content
     right" from "did it also add unwanted wrapper text."
  2. STRICT check (strict adherence, computed separately, no extra API
     calls): does the RAW response parse as JSON with nothing else
     around it at all? Tracked as a secondary diagnostic signal -- a
     model that used to answer with bare JSON and starts wrapping
     everything in explanatory prose is a real form of format drift the
     lenient check alone would miss, even though its lenient score
     might stay at 1.0 throughout.
"""

from __future__ import annotations

import json
import re
from typing import Optional

import jsonschema

from driftguard.ingest.log_schema import DriftLogEntry
from driftguard.providers.clients import call_model


# ---------------------------------------------------------------------------
# Fixed, versioned task set -- varied structure (flat, nested, array,
# array-of-objects, enum, boolean/nullable) so both subtle and major
# format regressions would show up.
# ---------------------------------------------------------------------------

FORMAT_TASKS: list[dict] = [
    {
        "task_id": "format_flat_object",
        "prompt": (
            'Return ONLY a JSON object (no other text) describing a person named "Ada Lovelace" '
            'who is 28 years old and works as an "engineer". Use exactly the keys: '
            '"name" (string), "age" (integer), "occupation" (string).'
        ),
        "schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "occupation": {"type": "string"},
            },
            "required": ["name", "age", "occupation"],
        },
    },
    {
        "task_id": "format_array_of_strings",
        "prompt": (
            'Return ONLY a JSON array (no other text) containing exactly 3 primary colors '
            'as lowercase strings, e.g. ["red", "green", "blue"].'
        ),
        "schema": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 3,
            "maxItems": 3,
        },
    },
    {
        "task_id": "format_nested_object",
        "prompt": (
            'Return ONLY a JSON object (no other text) describing a book with keys: '
            '"title" (string), "author" (string), and "publication" (a nested object with '
            'keys "year" (integer) and "publisher" (string)). Make up any plausible book.'
        ),
        "schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "author": {"type": "string"},
                "publication": {
                    "type": "object",
                    "properties": {
                        "year": {"type": "integer"},
                        "publisher": {"type": "string"},
                    },
                    "required": ["year", "publisher"],
                },
            },
            "required": ["title", "author", "publication"],
        },
    },
    {
        "task_id": "format_enum_field",
        "prompt": (
            'Return ONLY a JSON object (no other text) with a single key "status" whose value '
            'is exactly one of: "active", "inactive", "pending". Pick any one of these three.'
        ),
        "schema": {
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["active", "inactive", "pending"]},
            },
            "required": ["status"],
        },
    },
    {
        "task_id": "format_array_of_objects",
        "prompt": (
            'Return ONLY a JSON array (no other text) containing exactly 3 objects, each '
            'representing a fruit with keys "name" (string) and "color" (string). Make up any '
            'plausible fruits, e.g. [{"name": "apple", "color": "red"}, ...].'
        ),
        "schema": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "color": {"type": "string"},
                },
                "required": ["name", "color"],
            },
            "minItems": 3,
            "maxItems": 3,
        },
    },
    {
        "task_id": "format_boolean_nullable",
        "prompt": (
            'Return ONLY a JSON object (no other text) describing a task with keys: '
            '"title" (string), "completed" (boolean -- use true or false, not a string), '
            'and "due_date" (either a date string like "2026-01-01", or null if there is no '
            'due date). Make up a task that is NOT completed and has NO due date, so '
            '"completed" should be false and "due_date" should be null.'
        ),
        "schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "completed": {"type": "boolean"},
                "due_date": {"type": ["string", "null"]},
            },
            "required": ["title", "completed", "due_date"],
        },
    },
]


def _extract_json(response_text: Optional[str]) -> Optional[object]:
    """
    Try to parse JSON out of a raw model response. Models often wrap JSON
    in markdown code fences (```json ... ```) or add a sentence before/
    after it -- this recovers the JSON payload in those common cases
    before giving up, since we want to measure "did it produce correct
    JSON content", not "did it skip all prose", which would conflate two
    different kinds of format drift.
    """
    if not response_text:
        return None

    stripped = response_text.strip()

    # direct parse first
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    # try stripping markdown code fences
    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", stripped, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass

    # try grabbing the largest {...} or [...] substring
    for open_ch, close_ch in [("{", "}"), ("[", "]")]:
        start = stripped.find(open_ch)
        end = stripped.rfind(close_ch)
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(stripped[start : end + 1])
            except json.JSONDecodeError:
                continue

    return None


def _is_strict_json(response_text: Optional[str]) -> bool:
    """
    True only if the RAW response is valid JSON with nothing else around
    it -- no markdown fences, no leading/trailing prose. This is the
    zero-tolerance check, computed from the same response already
    fetched for the lenient check, so it costs no extra API calls.
    """
    if not response_text:
        return False
    try:
        json.loads(response_text.strip())
        return True
    except json.JSONDecodeError:
        return False


def _validate_against_schema(parsed: object, schema: dict) -> bool:
    try:
        jsonschema.validate(instance=parsed, schema=schema)
        return True
    except jsonschema.ValidationError:
        return False


def run_format_tasks(
    provider: str,
    model_id: str,
    eval_run_id: str,
) -> list[DriftLogEntry]:
    """
    Run the fixed format-adherence task set against one (provider, model_id).
    Same pattern as run_accuracy_tasks: returns scored DriftLogEntry rows
    (lenient schema-adherence score in eval_score), saving to the store is
    the caller's responsibility. Use strict_format_adherence_rate()
    separately on the same entries for the no-extra-cost strict signal.
    """
    entries: list[DriftLogEntry] = []

    for task in FORMAT_TASKS:
        entry = call_model(
            provider=provider,
            model_id=model_id,
            prompt=task["prompt"],
            eval_run_id=eval_run_id,
            eval_task_type="format",
            eval_task_id=task["task_id"],
        )

        if entry.outcome == "ok":
            parsed = _extract_json(entry.response_text)
            if parsed is None:
                entry.eval_score = 0.0
            else:
                entry.eval_score = 1.0 if _validate_against_schema(parsed, task["schema"]) else 0.0
        else:
            entry.eval_score = None

        entries.append(entry)

    return entries


def format_adherence_rate(entries: list[DriftLogEntry]) -> Optional[float]:
    """Fraction of responses that passed their schema check (lenient), among successfully-answered calls."""
    scored = [e.eval_score for e in entries if e.eval_score is not None]
    if not scored:
        return None
    return sum(scored) / len(scored)


def strict_format_adherence_rate(entries: list[DriftLogEntry]) -> Optional[float]:
    """
    Fraction of responses that were raw, unwrapped JSON with no
    surrounding prose or markdown fences -- computed from response_text
    already present on each entry, no extra API calls. Tracked as a
    secondary diagnostic: a drop here while format_adherence_rate stays
    flat is itself a signal (the model started wrapping valid JSON in
    more chatter than it used to), distinct from schema-content drift.
    """
    ok_entries = [e for e in entries if e.outcome == "ok"]
    if not ok_entries:
        return None
    strict_flags = [_is_strict_json(e.response_text) for e in ok_entries]
    return sum(strict_flags) / len(strict_flags)