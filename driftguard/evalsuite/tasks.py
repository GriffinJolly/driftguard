from __future__ import annotations
import json
import random
from pathlib import Path
from typing import Optional
from driftguard.providers.clients import call_model
from driftguard.ingest.log_schema import DriftLogEntry

FROZEN_PATH = Path(__file__).resolve().parent.parent.parent / "configs" / "mmlu_frozen.json"


DEFAULT_SUBJECTS = [
    "high_school_mathematics",
    "college_computer_science",
    "high_school_us_history",
    "philosophy",
]
QUESTIONS_PER_SUBJECT = 5
SEED = 42

ANSWER_LETTERS=["A", "B", "C", "D"]

def _build_frozen_subset(
        subjects:list[str] = DEFAULT_SUBJECTS,
        n_per_subject:int = QUESTIONS_PER_SUBJECT,
        seed:int = SEED,
)->list[dict]:
    from datasets import load_dataset
    rng=random.Random(seed)
    frozen: list[dict] = []
    for subject in subjects:
        ds=load_dataset("cais/mmlu",subject,split="test")
        indices=list(range(len(ds)))
        rng.shuffle(indices)
        chosen=indices[:n_per_subject]

        for idx in chosen:
            row=ds[idx]
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

def _ensure_frozen_subset()->list[dict]:
    if FROZEN_PATH.exists():
        with open(FROZEN_PATH,"r",encoding="utf-8") as f:
            frozen=json.load(f)
    frozen=_build_frozen_subset()
    FROZEN_PATH.parent.mkdir(parents=True,exist_ok=True)
    with open(FROZEN_PATH,"w",encoding="utf-8") as f:
        json.dump(frozen,f,indent=2,ensure_ascii=False)
    return frozen

def _format_prompt(task: dict) -> str:
    """Render one MMLU task as a prompt asking for a single-letter answer."""
    lines = [task["question"], ""]
    for letter, choice in zip(ANSWER_LETTERS, task["choices"]):
        lines.append(f"{letter}. {choice}")
    lines.append("")
    lines.append("Answer with only the letter of the correct choice (A, B, C, or D). No explanation.")
    return "\n".join(lines)

def _extract_answer_letter(response_text: Optional[str])->Optional[str]:
    """Return the first letter in A-D found in the response text, or None if not found."""
    if response_text is None:
        return None
    for token in response_text.strip().replace("."," ").replace(","," ").split():
        token_clean=token.strip("()[]:").upper()
        if token_clean in ANSWER_LETTERS:
            return token_clean
    stripped=response_text.strip().upper()
    if stripped in ANSWER_LETTERS:
        return stripped
    return None

def run_accuracy_tasks(
        provider:str, model_id:str, eval_run_id:str
)->list[DriftLogEntry]:
    tasks=_ensure_frozen_subset()
    entries: list[DriftLogEntry] = []
    for task in tasks:
        prompt=_format_prompt(task)
        entry=call_model(
            provider=provider,
            model_id=model_id,
            prompt=prompt,
            eval_run_id=eval_run_id,
            eval_task_type="accuracy",
            eval_task_id=task["task_id"],
        )
        if entry.outcome=="ok":
            given_letter=_extract_answer_letter(entry.response_text)
            correct_letter=ANSWER_LETTERS[task["correct_index"]]
            entry.eval_score=1.0 if given_letter==correct_letter else 0.0
        else:
            entry.eval_score=None
        entries.append(entry)
    return entries


def accuracy_rate(entries: list[DriftLogEntry]) -> Optional[float]:
    """Return the fraction of entries with score=1.0, or None if no scored entries."""
    scored=[e.eval_score for e in entries if e.eval_score is not None]
    if not scored:
        return None
    return sum(scored)/len(scored)