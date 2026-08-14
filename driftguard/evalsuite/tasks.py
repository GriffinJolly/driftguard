"""
Accuracy task for the DriftGuard eval suite, backed by MMLU.

Design:
- The very first time this runs, it pulls MMLU via Hugging Face's
  `datasets` library, picks a small, seeded, stratified sample, and
  freezes that exact sample to `configs/mmlu_frozen.json`.
- Every run after that reads ONLY the frozen file -- it never re-queries
  Hugging Face again. This is what makes the eval suite "fixed and
  versioned": the questions we ask in week 3 are byte-for-byte the same
  ones we asked in week 1, regardless of whether the upstream dataset
  gets updated in the meantime.
- Grading is exact-match on the answer letter (A/B/C/D) -- deterministic,
  no free-form parsing, so accuracy noise doesn't get contaminated by
  formatting noise (that's format_check.py's job).

The frozen file is meant to be committed to git -- it's part of the
versioned eval spec, same spirit as configs/eval_suite.yaml.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional

from driftguard.ingest.log_schema import DriftLogEntry
from driftguard.providers.clients import call_model

FROZEN_PATH = Path(__file__).resolve().parent.parent.parent / "configs" / "mmlu_frozen.json"

# Subjects chosen to mix STEM and humanities, so a subject-specific
# capability regression doesn't get diluted into "overall accuracy looks
# fine". 5 questions per subject, 4 subjects = 20 total -- sized to cut
# down single-question noise while still comfortably fitting inside the
# per-minute free-tier rate caps alongside the format/refusal tasks.
DEFAULT_SUBJECTS = [
    "high_school_mathematics",
    "college_computer_science",
    "high_school_us_history",
    "philosophy",
]
QUESTIONS_PER_SUBJECT = 5
SEED = 42  # fixed -- part of what makes the subset reproducible/versioned

ANSWER_LETTERS = ["A", "B", "C", "D"]


def _build_frozen_subset(
    subjects: list[str] = DEFAULT_SUBJECTS,
    n_per_subject: int = QUESTIONS_PER_SUBJECT,
    seed: int = SEED,
) -> list[dict]:
    """
    Pull MMLU from Hugging Face and select a small, seeded, stratified
    sample. Only ever called once -- the result gets written to
    FROZEN_PATH and every subsequent call reads that file instead.
    """
    from datasets import load_dataset  # imported here so this is only required when actually building

    rng = random.Random(seed)
    frozen: list[dict] = []

    for subject in subjects:
        ds = load_dataset("cais/mmlu", subject, split="test")
        indices = list(range(len(ds)))
        rng.shuffle(indices)
        chosen = indices[:n_per_subject]

        for idx in chosen:
            row = ds[idx]
            frozen.append(
                {
                    "task_id": f"mmlu_{subject}_{idx}",
                    "subject": subject,
                    "question": row["question"],
                    "choices": row["choices"],  # list of 4 strings
                    "correct_index": row["answer"],  # 0-3
                }
            )

    return frozen


def _ensure_frozen_subset() -> list[dict]:
    """Load the frozen subset from disk, building + saving it if it doesn't exist yet."""
    if FROZEN_PATH.exists():
        with open(FROZEN_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    frozen = _build_frozen_subset()
    FROZEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FROZEN_PATH, "w", encoding="utf-8") as f:
        json.dump(frozen, f, indent=2, ensure_ascii=False)
    return frozen


def _format_prompt(task: dict) -> str:
    """Render one MMLU task as a prompt asking for a single-letter answer."""
    lines = [task["question"], ""]
    for letter, choice in zip(ANSWER_LETTERS, task["choices"]):
        lines.append(f"{letter}. {choice}")
    lines.append("")
    lines.append("Answer with only the letter of the correct choice (A, B, C, or D). No explanation.")
    return "\n".join(lines)


def _extract_answer_letter(response_text: Optional[str]) -> Optional[str]:
    """
    Pull the first standalone A/B/C/D out of a response. Models sometimes
    answer as e.g. "A" or "The answer is A." -- this handles both without
    being so lenient it accepts a letter that just happens to appear
    inside a word.
    """
    if not response_text:
        return None
    for token in response_text.strip().replace(".", " ").replace(",", " ").split():
        token_clean = token.strip("()[]:").upper()
        if token_clean in ANSWER_LETTERS:
            return token_clean
    # fallback: single-character response with no surrounding text
    stripped = response_text.strip().upper()
    if stripped in ANSWER_LETTERS:
        return stripped
    return None


def run_accuracy_tasks(
    provider: str,
    model_id: str,
    eval_run_id: str,
) -> list[DriftLogEntry]:
    """
    Run the full frozen MMLU subset against one (provider, model_id),
    tagging every call with eval_run_id so it's grouped for this run.
    Returns the list of DriftLogEntry rows -- grading (eval_score) is
    filled in on each entry before it's returned; saving to the store is
    the caller's responsibility (same pattern as clients.call_model).
    """
    tasks = _ensure_frozen_subset()
    entries: list[DriftLogEntry] = []

    for task in tasks:
        prompt = _format_prompt(task)
        entry = call_model(
            provider=provider,
            model_id=model_id,
            prompt=prompt,
            eval_run_id=eval_run_id,
            eval_task_type="accuracy",
            eval_task_id=task["task_id"],
        )

        if entry.outcome == "ok":
            given_letter = _extract_answer_letter(entry.response_text)
            correct_letter = ANSWER_LETTERS[task["correct_index"]]
            entry.eval_score = 1.0 if given_letter == correct_letter else 0.0
        else:
            entry.eval_score = None  # failed calls don't count toward accuracy either way

        entries.append(entry)

    return entries


def accuracy_rate(entries: list[DriftLogEntry]) -> Optional[float]:
    """Fraction correct among entries that actually got a scored response."""
    scored = [e.eval_score for e in entries if e.eval_score is not None]
    if not scored:
        return None
    return sum(scored) / len(scored)