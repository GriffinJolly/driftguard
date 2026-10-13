"""
validation/lmsys_model_swap.py

A silent model swap, built from REAL responses.

LMSYS-Chat-1M (lmsys/lmsys-chat-1m, gated; accept its terms on Hugging Face and
log in once with `hf auth login`) holds 1M real conversations with 25 models.
It has no timestamps, so it cannot show drift over time on its own. What it can
do is stand in for a provider that silently replaces the model behind one name:

    calls 1..N    answered by model A   (e.g. vicuna-13b)
    calls N+1..   answered by model B   (e.g. llama-2-13b-chat)

Every reply is a genuine response from that model to a genuine user prompt; only
the ORDER is constructed. The swap point is ground truth, stored next to the
data and never shown to the detectors. A control stream (A then A, different
conversations) checks that nothing fires when nothing changed.

English conversations only; first user prompt and first reply. Refusal is the
eval suite's keyword heuristic (driftguard/evalsuite/refusal_probe.py).

  * prompts NOT flagged by OpenAI moderation -> over-refusal (refusal_rate,
    labelled a safety_shift by classification/rules.py when it rises)
  * prompts flagged by moderation           -> under-refusal (outside the
    taxonomy: detected and shown for context, never labelled)

Each stream is also written to a SQLite database in the normal log_rows schema,
so the dashboard can open it:  set DRIFTGUARD_DB to the .db path and run
`streamlit run driftguard/dashboard/app.py`.

Usage:
    python -m validation.lmsys_model_swap
    python -m validation.lmsys_model_swap --model-a vicuna-13b --model-b llama-2-13b-chat --per-model 3000
"""

from __future__ import annotations

import argparse
import json
import uuid
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from driftguard.dashboard import data as D
from driftguard.detection.signals import SIGNALS
from driftguard.evalsuite.refusal_probe import _is_refusal

REPO_ID = "lmsys/lmsys-chat-1m"
CACHE = Path("validation/data/lmsys_english_first_turns.parquet")
OUT = Path("validation/results/lmsys_model_swap")
DB_DIR = Path("validation/data")
TEXT_CAP = 2000
PROVIDER = "lmsys-replay"


def extract() -> pd.DataFrame:
    """First user prompt + first reply of every English conversation (cached after the first run)."""
    if CACHE.exists():
        return pd.read_parquet(CACHE)
    import pyarrow.parquet as pq
    from huggingface_hub import snapshot_download

    root = Path(snapshot_download(REPO_ID, repo_type="dataset", allow_patterns=["data/*.parquet"]))
    rows = []
    for path in sorted((root / "data").glob("*.parquet")):
        pf = pq.ParquetFile(path)
        for rg in range(pf.num_row_groups):
            t = pf.read_row_group(rg, columns=["conversation_id", "model", "language", "conversation",
                                               "openai_moderation"]).to_pandas()
            t = t[t["language"] == "English"]
            for r in t.itertuples(index=False):
                conv = list(r.conversation)
                ui = next((i for i, m in enumerate(conv) if m["role"] == "user"), None)
                bot = next((m for m in conv if m["role"] == "assistant"), None)
                mod = list(r.openai_moderation) if r.openai_moderation is not None else []
                if ui is None or bot is None or ui >= len(mod):
                    continue
                rows.append(dict(conversation_id=r.conversation_id, model=r.model,
                                 prompt=conv[ui]["content"][:TEXT_CAP], reply=(bot["content"] or "")[:TEXT_CAP],
                                 flagged=bool(mod[ui]["flagged"]), refused=_is_refusal(bot["content"])))
        print(f"  {path.name}: {len(rows):,} English conversations so far", flush=True)
    df = pd.DataFrame(rows)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CACHE, index=False)
    return df


def model_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Per model: how many English conversations, and how often it refuses."""
    g = df.groupby("model")
    out = pd.DataFrame({
        "conversations": g.size(),
        "flagged_share": g["flagged"].mean(),
        "refusal_unflagged": df[~df["flagged"]].groupby("model")["refused"].mean(),
        "refusal_flagged": df[df["flagged"]].groupby("model")["refused"].mean(),
    })
    return out.sort_values("conversations", ascending=False)


def build_stream(df: pd.DataFrame, model_a: str, model_b: str, per_model: int, seed: int) -> tuple[pd.DataFrame, int]:
    """per_model conversations from A, then per_model from B (each side shuffled). Returns rows + swap index."""
    rng = np.random.default_rng(seed)
    a = df[df["model"] == model_a]
    b = df[df["model"] == model_b]
    if model_a == model_b:                      # control: two disjoint samples of the same model
        idx = rng.permutation(len(a))[: 2 * per_model]
        first, second = a.iloc[idx[:per_model]], a.iloc[idx[per_model:]]
    else:
        first = a.iloc[rng.permutation(len(a))[:per_model]]
        second = b.iloc[rng.permutation(len(b))[:per_model]]
    return pd.concat([first, second], ignore_index=True), len(first)


def write_db(stream: pd.DataFrame, served_name: str, db: Path, swap_at: int, model_a: str, model_b: str) -> None:
    """The stream as log_rows (refusal tasks) so the dashboard can open it. Replayed one call per
    minute: the dataset has no timestamps, and incidents.py groups events by time."""
    from driftguard.ingest.log_schema import CallOutcome, DriftLogEntry, IntegrationPath
    from driftguard.storage.store import DriftStore

    if db.exists():
        db.unlink()
    store = DriftStore(db_path=str(db))
    t0 = pd.Timestamp("2023-08-01T00:00:00Z")
    run_id = f"lmsys_swap_{db.stem}"
    for i, r in enumerate(stream.itertuples(index=False)):
        e = DriftLogEntry(
            integration_path=IntegrationPath.EVAL_SUITE, provider=PROVIDER, model_id=served_name,
            prompt=r.prompt, response_text=r.reply,
            request_params={"served_by": r.model, "conversation_id": r.conversation_id},
            outcome=CallOutcome.OK, eval_run_id=run_id, eval_task_type="refusal",
            eval_task_id=f"lmsys_{'unsafe' if r.flagged else 'safe'}_{i}",
            eval_score=1.0 if r.refused else 0.0,
            timestamp=(t0 + timedelta(minutes=i)).to_pydatetime(), log_id=str(uuid.uuid4()),
        )
        store.write_log(e)
    db.with_suffix(".ground_truth.json").write_text(json.dumps({
        "swap_at_call": swap_at + 1, "swap_time_utc": str(t0 + timedelta(minutes=swap_at)),
        "model_a": model_a, "model_b": model_b, "served_name": served_name}, indent=2))


def evaluate(db: Path, served_name: str, swap_at: int) -> tuple[dict, list[str]]:
    calls = D.load_calls(db)
    a = D.analyse_scope(calls, PROVIDER, served_name, D.load_settings())
    results = a.results
    swap_time = calls["timestamp"].iloc[swap_at]
    out = {}
    for key, r in results.items():
        st = r.stream
        onset = int(st.timestamp.searchsorted(swap_time))
        conf = [t for t in r.tested if t["significant"]]
        out[key] = dict(
            signal=SIGNALS[key].label, calls=len(st.values), onset=onset,
            rate_before=float(st.values[:onset].mean()) if onset else np.nan,
            rate_after=float(st.values[onset:].mean()) if onset < len(st.values) else np.nan,
            confirmed=[(t["index"] - onset, round(t["pre_mean"], 3), round(t["post_mean"], 3)) for t in conf],
            ph_alerts=[i - onset for i in r.alerts],
            status=D.signal_status(key, a))
    lines = [f"[{p.urgency}] {p.headline}" for p in a.plans]
    return out, lines


def main() -> None:
    ap = argparse.ArgumentParser(description="Silent model swap built from real LMSYS-Chat-1M responses")
    ap.add_argument("--model-a", default="vicuna-13b")
    ap.add_argument("--model-b", default="llama-2-13b-chat")
    ap.add_argument("--per-model", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    print("Extracting English first turns from LMSYS-Chat-1M (cached after the first run)...")
    df = extract()
    prof = model_profile(df)
    prof.to_csv(OUT / "model_profile.csv")
    pd.set_option("display.width", 200)
    print(f"\n{len(df):,} English conversations. Refusal profile by model:")
    print(prof.round(3).to_string())

    served = "served-model"
    report = [f"# Silent model swap from real LMSYS-Chat-1M responses", "",
              f"English first turns only. Refusal = keyword heuristic of the eval suite.", ""]
    for name, (ma, mb) in {"swap": (args.model_a, args.model_b), "control": (args.model_a, args.model_a)}.items():
        stream, swap_at = build_stream(df, ma, mb, args.per_model, args.seed)
        db = DB_DIR / f"lmsys_{name}.db"
        write_db(stream, served, db, swap_at, ma, mb)
        res, incidents = evaluate(db, served, swap_at)
        print(f"\n=== {name}: {ma} -> {mb}, swap after call {swap_at} ({db})")
        report += [f"## {name}: {ma} → {mb} (swap after call {swap_at:,}; `{db}`)", ""]
        for v in res.values():
            line = (f"{v['signal']:<14} calls={v['calls']:>5} rate {v['rate_before']:.3f} -> {v['rate_after']:.3f} "
                    f"| confirmed (offset, before, after) {v['confirmed']} | PH alert offsets {v['ph_alerts']} "
                    f"| status {v['status']}")
            print("  " + line)
            report.append(f"- {line}")
        print("  incidents:", incidents or "none")
        report += [f"- incident: {x}" for x in incidents] or ["- incidents: none"]
        report.append("")
    (OUT / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"\nWrote {OUT}/REPORT.md and model_profile.csv")


if __name__ == "__main__":
    main()
