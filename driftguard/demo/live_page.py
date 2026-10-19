"""A simple page for demonstrating DriftGuard on a real LLM API.

You write the prompts and the answer each one should contain. The page sends them to your
real provider (Groq, NVIDIA or OpenRouter) round after round and scores every answer. DriftGuard's
own detectors (Page-Hinkley, PELT with a permutation test and Benjamini-Hochberg correction, then
the classification and remediation rules) run on the result.

To make drift happen on demand, the page can SIMULATE a silent provider update: while the box is
ticked, calls are secretly sent to a different model, or with a tiny output limit, while the log
still says they went to the model you are watching. Everything else is real.

Run:   driftguard live
"""
from __future__ import annotations

import os
import re
import sqlite3
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

DEFAULT_WATCHLIST = [
    ("What is 47 * 38 + 129? Reply with only the number.", "1915"),
    ("What is the capital of Australia? Reply with one word.", "Canberra"),
    ("How many times does the letter r appear in the word 'strawberry'? Reply with only the number.", "3"),
    ("What is the chemical symbol for tungsten? Reply with only the symbol.", "W"),
    ("In what year did the Berlin Wall fall? Reply with only the year.", "1989"),
    ("What is the next prime number after 89? Reply with only the number.", "97"),
    ("What is the square root of 1764? Reply with only the number.", "42"),
    ("Who developed the theory of general relativity? Reply with the surname only.", "Einstein"),
    ("What is 2 to the power of 12? Reply with only the number.", "4096"),
    ("What is the largest ocean on Earth? Reply with one word.", "Pacific"),
]

DEFAULT_MODELS = {
    "groq": ("openai/gpt-oss-120b", "openai/gpt-oss-20b"),
    "nvidia": ("nvidia/nemotron-3-super-120b-a12b", "openai/gpt-oss-20b"),
    "openrouter": ("", ""),
}
NORMAL_MAX_TOKENS = 2048


# ------------------------------------------------------------------------- logic ----
@dataclass
class Change:
    """The simulated silent provider update."""
    active: bool = False
    lever: str = "cap"            # "cap" = tiny output limit, "model" = a different model
    swap_model: str = ""
    cap_tokens: int = 8


def is_correct(answer: Optional[str], expected: str) -> bool:
    """True if the expected text appears in the answer (case-insensitive, commas in numbers ignored)."""
    if not answer or not expected.strip():
        return False
    a = re.sub(r"(?<=\d),(?=\d)", "", answer).lower()
    e = expected.strip().lower()
    return re.search(rf"(?<![\w.]){re.escape(e)}(?![\w])", a) is not None


def run_round(store, provider: str, model_id: str, watchlist: list[tuple[str, str]], change: Change,
              call_fn: Optional[Callable] = None, on_call: Optional[Callable] = None) -> list:
    """Send every watch prompt once. Returns the logged entries; stops early if a call fails."""
    if call_fn is None:
        from driftguard.providers.clients import call_model as call_fn
    run_id = uuid.uuid4().hex[:8]
    entries = []
    for i, (prompt, expected) in enumerate(watchlist):
        params = {"temperature": 0, "max_tokens": NORMAL_MAX_TOKENS}
        served = model_id
        if change.active and change.lever == "model":
            served = change.swap_model
        elif change.active:
            params["max_tokens"] = change.cap_tokens
        entry = call_fn(provider=provider, model_id=served, prompt=prompt, request_params=params,
                        eval_run_id=run_id, eval_task_type="accuracy", eval_task_id=f"w{i + 1}")
        entry.model_id = model_id  # the model the app thinks it is talking to (what DriftGuard monitors)
        entry.eval_score = (1.0 if is_correct(entry.response_text, expected) else 0.0) if entry.outcome == "ok" else None
        entry.request_params = {**params, "served_model": served, "silent_change": bool(change.active)}
        store.write_log(entry)
        entries.append(entry)
        if on_call:
            on_call(i + 1, len(watchlist), entry)
        if entry.outcome != "ok":
            break
    return entries


def analyse(db_path: str, provider: str, model_id: str):
    """Run DriftGuard's real pipeline on everything logged for this target. None if nothing yet."""
    from driftguard.dashboard import data as D
    from driftguard.detection import pipeline
    from driftguard.storage.store import DriftStore

    calls = pipeline.load_calls(DriftStore(db_path), provider, model_id)
    if calls.empty:
        return None
    return D.analyse_scope(calls, provider, model_id)


def change_position(db_path: str, provider: str, model_id: str, seq_ids) -> Optional[int]:
    """Index (in the accuracy stream) of the first call logged while the silent change was on."""
    con = sqlite3.connect(db_path)
    try:
        rows = dict(con.execute("SELECT rowid, request_params_json FROM log_rows WHERE provider=? AND model_id=?",
                                (provider, model_id)).fetchall())
    finally:
        con.close()
    import json
    for i, sid in enumerate(seq_ids):
        if json.loads(rows.get(int(sid)) or "{}").get("silent_change"):
            return i
    return None


def recent_calls(db_path: str, provider: str, model_id: str, n: int = 25) -> pd.DataFrame:
    con = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(
            "SELECT rowid AS call_no, timestamp, prompt, response_text, outcome, latency_ms, eval_score, "
            "request_params_json FROM log_rows WHERE provider=? AND model_id=? ORDER BY rowid DESC LIMIT ?",
            con, params=(provider, model_id, n))
    finally:
        con.close()
    import json
    df["served_by"] = df.request_params_json.map(lambda s: json.loads(s or "{}").get("served_model", ""))
    df["max_tokens"] = df.request_params_json.map(lambda s: json.loads(s or "{}").get("max_tokens"))
    df["correct"] = df.eval_score.map(lambda v: "yes" if v == 1 else ("no" if v == 0 else "-"))
    df["latency_s"] = (df.latency_ms / 1000).round(2)
    return df[["call_no", "prompt", "response_text", "correct", "latency_s", "served_by", "max_tokens", "outcome"]]


# --------------------------------------------------------------------------- page ----
def main() -> None:
    import altair as alt
    import streamlit as st
    from driftguard.providers.clients import PROVIDERS

    st.set_page_config(page_title="DriftGuard live demo", layout="wide")
    st.title("DriftGuard: live drift demo")
    st.caption("Your prompts go to a real LLM API. DriftGuard's detectors look for a change in how it answers.")

    # ---- sidebar
    with st.sidebar:
        st.subheader("Model to watch")
        provider = st.selectbox("Provider", list(DEFAULT_MODELS))
        default_model, default_swap = DEFAULT_MODELS[provider]
        model_id = st.text_input("Model", default_model, key=f"model_{provider}")
        env = PROVIDERS[provider].api_key_env
        key = st.text_input(f"{env}", type="password", help="Used for this session only. Never stored.")
        if key:
            os.environ[env] = key
        elif os.environ.get(env):
            st.caption(f"Using {env} from the environment.")
        db_path = st.text_input("Database file", "live_demo.db")

        st.subheader("Simulated silent update")
        lever = st.radio("What changes?", ["cap", "model"], horizontal=True,
                         format_func=lambda v: {"cap": "Tiny output limit", "model": "A different model"}[v])
        if lever == "cap":
            cap = st.number_input("max_tokens while changed", 1, 200, 8)
            swap = default_swap
        else:
            swap = st.text_input("Secretly use this model", default_swap, key=f"swap_{provider}")
            cap = 8
        st.caption("If the other model still answers correctly there is nothing to detect. "
                   "The output limit always causes a visible change.")

    if not model_id:
        st.info("Enter a model id in the sidebar.")
        return

    # ---- watch prompts
    st.subheader("1. Your watch prompts")
    st.caption("Each prompt is sent every round. A reply is correct if it contains the expected answer.")
    table = st.data_editor(
        pd.DataFrame(DEFAULT_WATCHLIST, columns=["Prompt", "Expected answer"]),
        num_rows="dynamic", width="stretch", hide_index=True, key="watchlist")
    watchlist = [(str(p), str(e)) for p, e in zip(table["Prompt"], table["Expected answer"])
                 if p and str(p).strip() and e and str(e).strip()]

    # ---- run
    st.subheader("2. Run rounds")
    col1, col2, col3 = st.columns([2, 2, 1])
    active = col1.toggle("The provider silently changes the model now", key="active",
                         help="Simulated. Tick this after you have built a baseline.")
    rounds = col2.number_input("Rounds to run", 1, 30, 5, help="One round = every watch prompt once.")
    go = col3.button("Run", type="primary", width="stretch")
    if st.button("Delete all data and start again"):
        p = Path(db_path)
        if p.exists() and p.name.endswith("_demo.db"):
            p.unlink()
            st.rerun()
        else:
            st.warning("Only files whose name ends in _demo.db can be deleted from here.")
    st.caption("Build a baseline first: at least 100 calls with the box unticked (for 10 prompts, 10 rounds). "
               "Then tick the box and run more rounds.")

    if go:
        if not os.environ.get(env):
            st.error(f"Enter {env} in the sidebar first.")
        elif not watchlist:
            st.error("Add at least one prompt with an expected answer.")
        else:
            from driftguard.storage.store import DriftStore
            store = DriftStore(db_path)
            change = Change(active=bool(active), lever=lever, swap_model=swap, cap_tokens=int(cap))
            bar = st.progress(0.0, text="Starting")
            total = int(rounds) * len(watchlist)
            done = 0
            failed = None
            for r in range(int(rounds)):
                def on_call(i, n, entry, r=r):
                    nonlocal done
                    done += 1
                    bar.progress(done / total, text=f"Round {r + 1}/{int(rounds)}, call {i}/{n}")
                entries = run_round(store, provider, model_id, watchlist, change, on_call=on_call)
                if entries and entries[-1].outcome != "ok":
                    failed = entries[-1]
                    break
            bar.empty()
            if failed is not None:
                st.error(f"The API call failed ({failed.outcome}): {failed.error_message}. "
                         "Check the key, the model id and your rate limit.")

    # ---- results
    st.subheader("3. What DriftGuard sees")
    if not Path(db_path).exists():
        st.info("No calls yet. Enter your key and press Run.")
        return
    analysis = analyse(db_path, provider, model_id)
    if analysis is None:
        st.info("No calls logged for this model yet.")
        return
    stream = analysis.results["accuracy_error"].stream if "accuracy_error" in analysis.results else None
    if stream is None or len(stream.values) == 0:
        st.warning("No scored calls yet (failed calls are not scored).")
        return

    n = len(stream.values)
    change_idx = change_position(db_path, provider, model_id, stream.seq_id)
    acc_events = [e for e in analysis.events if e.signal == "accuracy_error"]

    if analysis.incidents:
        plan = analysis.plans[0]
        lag = ""
        if change_idx is not None:
            lag = f" That is {n - change_idx} scored calls after the silent change began."
        st.error(f"**{plan.urgency.upper()}: {plan.headline}**{lag}")
        with st.expander("What DriftGuard recommends", expanded=True):
            for line in plan.meanings:
                st.markdown(f"- {line}")
            st.markdown("**Plan**")
            for i, step in enumerate(plan.steps, 1):
                st.markdown(f"{i}. {step}")
    elif analysis.events:
        st.warning("Something moved, but it is outside DriftGuard's drift taxonomy. See the detections below.")
    else:
        st.success(f"No drift detected in {n} scored calls." +
                   (" Build a baseline of at least 100 calls before the change." if n < 100 else ""))

    m1, m2, m3 = st.columns(3)
    passes = 1 - stream.values
    m1.metric("Scored calls", n)
    m2.metric("Correct, all calls", f"{passes.mean():.0%}")
    m3.metric("Correct, last 20", f"{passes[-20:].mean():.0%}")

    df = pd.DataFrame({"call": range(1, n + 1), "correct": passes})
    df["rolling"] = df.correct.rolling(20, min_periods=5).mean()
    layers = [alt.Chart(df).mark_line(color="#2a78d6", strokeWidth=2).encode(
        x=alt.X("call:Q", title="Scored call"),
        y=alt.Y("rolling:Q", title="Correct (rolling 20)", scale=alt.Scale(domain=[0, 1.05]), axis=alt.Axis(format="%")))]
    if change_idx is not None:
        layers.append(alt.Chart(pd.DataFrame({"call": [change_idx + 1], "what": ["silent change began"]}))
                      .mark_rule(color="#e74c3c", strokeDash=[5, 4], strokeWidth=2).encode(x="call:Q", tooltip="what"))
    marks = [(e.index + 1, "Page-Hinkley alert" if e.event.detector != "pelt" else "PELT confirmed shift")
             for e in acc_events]
    if marks:
        layers.append(alt.Chart(pd.DataFrame(marks, columns=["call", "detector"]).assign(y=1.03))
                      .mark_point(shape="triangle-down", size=220, filled=True, color="#f39c12")
                      .encode(x="call:Q", y="y:Q", tooltip="detector"))
    st.altair_chart(alt.layer(*layers).properties(height=280), width="stretch")
    st.caption("Blue: share of correct answers. Red dashed line: when the simulated change began. "
               "Orange triangles: where DriftGuard's detectors fired.")

    from driftguard.dashboard import data as D
    det = D.shift_table(analysis)
    if len(det):
        st.markdown("**Every detection**")
        view = det[["signal_label", "label", "detector", "tier", "index", "before", "after", "urgency", "reason"]].copy()
        view["index"] = view["index"] + 1
        st.dataframe(view.rename(columns={"signal_label": "signal", "label": "drift type", "tier": "evidence",
                                          "index": "call #"}), width="stretch", hide_index=True)

    st.markdown("**Latest calls**")
    st.dataframe(recent_calls(db_path, provider, model_id), width="stretch", hide_index=True)
    st.caption(f"Full dashboard on the same data:  driftguard dashboard --db-path {db_path}")


if __name__ == "__main__":
    main()
