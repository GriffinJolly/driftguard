"""
driftguard/dashboard/data.py

Data and detection layer behind the dashboard. No Streamlit in here, so it can
be tested and reused on its own.

Pipeline, per (provider, model, scope):
  1. load the per-call log rows from driftguard.db (ordered by insertion, rowid)
  2. turn them into per-call signal streams, each oriented "higher = worse"
  3. run the ONLINE detector (tuned Page-Hinkley) over every stream, one update
     per call, exactly as it would run live
  4. run the RETROSPECTIVE path (PELT -> permutation test -> Benjamini-Hochberg)
     over the same streams to confirm and locate sustained shifts
  5. attach a category (classification/rules.py) and an action
     (remediation/mapper.py) to every shift

Wording follows the project's reporting rule: a Page-Hinkley alarm is an
"alert observed at call N", not the true onset; PELT gives the estimated
location of a confirmed shift.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from driftguard.classification.rules import SIGNALS, classify
from driftguard.detection.bootstrap import SignificanceConfig, confirm_changepoints
from driftguard.detection.changepoint_pelt import PeltConfig, detect_changepoints, segments_from_changepoints
from driftguard.remediation.mapper import remediation_for

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DB = ROOT / "driftguard.db"
DETECTORS_YAML = ROOT / "configs" / "detectors.yaml"
XSTEST_FROZEN = ROOT / "configs" / "xstest_frozen.json"

COMPARE_WINDOW = 200     # calls each side of an online alert used for the before/after rates
WATCH_EFFECT = 0.05      # an unconfirmed online alert moving the rate by less than this is "watch", not "warning"
MIN_SIGNAL_CALLS = 60    # below this a stream is too short to say anything about


# --------------------------------------------------------------------------- config

@dataclass(frozen=True)
class DetectionSettings:
    ph_lambda: float = 12.0
    ph_delta: float = 0.005
    ph_alpha: float = 0.9999
    ph_min_instances: int = 30
    pelt: PeltConfig = field(default_factory=PeltConfig)
    significance: SignificanceConfig = field(default_factory=SignificanceConfig)
    baseline_window: int = 100   # calls used as the quiet baseline (calibration.min_window_size)


def load_settings(path: Path = DETECTORS_YAML) -> DetectionSettings:
    """Read the tuned detector settings from configs/detectors.yaml."""
    try:
        cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except FileNotFoundError:
        return DetectionSettings()
    ph = (cfg.get("live_detector") or {}).get("params") or {}
    rp = (cfg.get("retrospective_detector") or {}).get("params") or {}
    sg = cfg.get("significance") or {}
    cal = cfg.get("calibration") or {}
    return DetectionSettings(
        ph_lambda=float(ph.get("lambda_", 12.0)),
        ph_delta=float(ph.get("delta", 0.005)),
        ph_alpha=float(ph.get("alpha", 0.9999)),
        ph_min_instances=int(ph.get("min_num_instances", 30)),
        pelt=PeltConfig(
            cost_model=rp.get("cost_model", "l2"),
            penalty=rp.get("penalty"),
            penalty_scale=float(rp.get("penalty_scale", 5)),
            min_segment_length=int(rp.get("min_segment_length", 200)),
            jump=int(rp.get("jump", 25)),
        ),
        significance=SignificanceConfig(
            alpha=float(sg.get("alpha", 0.05)),
            n_bootstrap_samples=int(sg.get("n_bootstrap_samples", 1000)),
            window=sg.get("window", 2000),
        ),
        baseline_window=int(cal.get("min_window_size", 100)),
    )


# --------------------------------------------------------------------------- loading

def load_calls(db_path: Path | str = DEFAULT_DB) -> pd.DataFrame:
    """Every logged call, oldest first. Read-only connection."""
    con = sqlite3.connect(f"file:{Path(db_path).as_posix()}?mode=ro", uri=True)
    try:
        df = pd.read_sql_query(
            "SELECT rowid AS seq_id, log_id, timestamp, integration_path, provider, model_id, "
            "prompt, response_text, request_params_json, outcome, error_message, latency_ms, "
            "prompt_tokens, completion_tokens, eval_run_id, eval_task_type, eval_task_id, eval_score "
            "FROM log_rows ORDER BY rowid",
            con,
        )
    finally:
        con.close()
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed", utc=True)
    df["target"] = df["provider"] + " · " + df["model_id"]
    return df


def load_events(db_path: Path | str = DEFAULT_DB) -> pd.DataFrame:
    """Shifts already recorded in the changepoint_events table."""
    con = sqlite3.connect(f"file:{Path(db_path).as_posix()}?mode=ro", uri=True)
    try:
        return pd.read_sql_query("SELECT * FROM changepoint_events ORDER BY detected_at DESC", con)
    except Exception:
        return pd.DataFrame()
    finally:
        con.close()


def _should_refuse_map() -> dict[str, bool]:
    try:
        tasks = json.loads(XSTEST_FROZEN.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    return {t["task_id"]: bool(t["should_refuse"]) for t in tasks}


def run_catalog(calls: pd.DataFrame) -> pd.DataFrame:
    """One row per evaluation run for a target, for the scope picker."""
    g = calls.groupby(calls["eval_run_id"].fillna("(no run id)"), sort=False)
    out = g.agg(calls=("seq_id", "size"), first=("timestamp", "min"), last=("timestamp", "max"),
                start=("seq_id", "min")).reset_index()
    return out.sort_values("start").rename(columns={"eval_run_id": "run_id"})


# --------------------------------------------------------------------------- signals

@dataclass
class Stream:
    key: str
    values: np.ndarray           # per-call, higher = worse
    seq_id: np.ndarray           # rowid of each call, links back to the log
    timestamp: pd.Series
    note: str = ""


def build_streams(calls: pd.DataFrame, baseline_window: int = 100) -> dict[str, Stream]:
    """Turn the call log into per-call signal streams (only those with enough data)."""
    streams: dict[str, Stream] = {}
    ok = calls[(calls["outcome"] == "ok") & calls["eval_score"].notna()]

    def add(key, frame, values, note=""):
        if len(frame) >= MIN_SIGNAL_CALLS:
            streams[key] = Stream(key, np.asarray(values, dtype=float), frame["seq_id"].to_numpy(),
                                  frame["timestamp"].reset_index(drop=True), note)

    acc = ok[ok["eval_task_type"] == "accuracy"]
    add("accuracy_error", acc, 1.0 - acc["eval_score"])

    fmt = ok[ok["eval_task_type"] == "format"]
    add("format_error", fmt, 1.0 - fmt["eval_score"])

    ref = ok[ok["eval_task_type"] == "refusal"]
    if len(ref):
        should = ref["eval_task_id"].map(_should_refuse_map())
        safe, unsafe = ref[should == False], ref[should == True]  # noqa: E712
        add("safe_refusal", safe, safe["eval_score"])
        add("unsafe_compliance", unsafe, 1.0 - unsafe["eval_score"])

    # Rate limits are OUR quota running out, not the provider changing: keep them on
    # their own signal so they never feed the drift verdict.
    limited = calls["outcome"] == "rate_limited"
    served = calls[~limited]
    add("call_failure", served, (served["outcome"] != "ok").astype(float))
    if limited.any():
        add("quota_exhausted", calls, limited.astype(float))

    lat = calls[(calls["outcome"] == "ok") & calls["latency_ms"].notna()]
    if len(lat) >= max(MIN_SIGNAL_CALLS, baseline_window + 20):
        p95 = float(np.percentile(lat["latency_ms"].iloc[:baseline_window], 95))
        add("slow_call", lat, (lat["latency_ms"] > p95).astype(float),
            note=f"slow = above {p95:,.0f} ms, the p95 of the first {baseline_window} calls")
    return streams


# --------------------------------------------------------------------------- detection

def page_hinkley_alerts(values: np.ndarray, s: DetectionSettings) -> list[int]:
    """Indices where the tuned Page-Hinkley detector alarms, fed one call at a time.
    The detector is reset after each alarm, so later alerts are independent."""
    from frouros.detectors.concept_drift import PageHinkley, PageHinkleyConfig

    def fresh():
        return PageHinkley(config=PageHinkleyConfig(lambda_=s.ph_lambda, delta=s.ph_delta,
                                                    alpha=s.ph_alpha, min_num_instances=s.ph_min_instances))
    det, alerts = fresh(), []
    for i, v in enumerate(values):
        det.update(value=float(v))
        if det.status["drift"]:
            alerts.append(i)
            det = fresh()
    return alerts


def effective_pelt(n: int, s: DetectionSettings) -> PeltConfig:
    """The tuned minimum segment (200) assumes long histories. For short scopes it is
    capped at a fifth of the stream so PELT can still run; never below 30 calls."""
    p = s.pelt
    seg = max(30, min(p.min_segment_length, n // 5))
    jump = max(1, min(p.jump, seg // 4))
    return PeltConfig(cost_model=p.cost_model, penalty=p.penalty, penalty_scale=p.penalty_scale,
                      min_segment_length=seg, jump=jump)


@dataclass
class SignalResult:
    stream: Stream
    alerts: list[int]
    confirmed: list[dict]               # every PELT candidate with its test result
    segments: list[tuple[int, int, float]]  # (start, end, mean) of the PELT regimes
    pelt_config: PeltConfig


def analyse(streams: dict[str, Stream], s: DetectionSettings) -> dict[str, SignalResult]:
    results = {}
    for key, st in streams.items():
        v = st.values
        pcfg = effective_pelt(len(v), s)
        cps = detect_changepoints(v, pcfg)
        tested = confirm_changepoints(v, cps, s.significance) if cps else []
        kept = [r["index"] for r in tested if r["significant"]]
        segs = [(a, b, float(v[a:b].mean())) for a, b in segments_from_changepoints(len(v), kept)]
        results[key] = SignalResult(st, page_hinkley_alerts(v, s), tested, segs, pcfg)
    return results


def alert_effect(values: np.ndarray, i: int) -> tuple[float, float]:
    """Rate in the COMPARE_WINDOW calls before and after an online alert."""
    pre, post = values[max(0, i - COMPARE_WINDOW):i], values[i:i + COMPARE_WINDOW]
    return (float(pre.mean()) if len(pre) else np.nan, float(post.mean()) if len(post) else np.nan)


def _ts(stream: Stream, i: int):
    return stream.timestamp.iloc[min(i, len(stream.timestamp) - 1)]


def shift_table(results: dict[str, SignalResult]) -> pd.DataFrame:
    """Every shift from both detectors, with category and recommended action."""
    rows = []
    for key, r in results.items():
        st, v = r.stream, r.stream.values
        sig = SIGNALS[key]
        for t in r.confirmed:
            if not t["significant"]:
                continue
            i = t["index"]
            worse = t["post_mean"] > t["pre_mean"]
            rows.append(dict(
                signal=key, signal_label=sig.label, category=classify(key), detector="PELT (confirmed)",
                index=i, seq_id=int(st.seq_id[i]), timestamp=_ts(st, i),
                before=t["pre_mean"], after=t["post_mean"], change=t["post_mean"] - t["pre_mean"],
                p_adj=t["p_value_corrected"], worsening=worse, action=remediation_for(key, worse),
            ))
        for i in r.alerts:
            before, after = alert_effect(v, i)
            rows.append(dict(
                signal=key, signal_label=sig.label, category=classify(key), detector="Page-Hinkley (online)",
                index=i, seq_id=int(st.seq_id[i]), timestamp=_ts(st, i),
                before=before, after=after, change=after - before,
                p_adj=np.nan, worsening=True, action=remediation_for(key, True),
            ))
    for row in rows:
        row["provider_drift"] = SIGNALS[row["signal"]].provider_drift
    cols = ["signal", "signal_label", "category", "detector", "index", "seq_id", "timestamp",
            "before", "after", "change", "p_adj", "worsening", "action", "provider_drift"]
    df = pd.DataFrame(rows, columns=cols)
    return df.sort_values(["seq_id", "detector"]).reset_index(drop=True)


def signal_status(r: SignalResult) -> str:
    """'critical' = a confirmed worsening shift; 'warning' = an unconfirmed online alert that
    moved the rate by at least WATCH_EFFECT; 'watch' = only smaller unconfirmed alerts;
    'good' = nothing. Signals that are not provider drift (quota) cap at 'notice'."""
    level = "good"
    if any(t["significant"] and t["post_mean"] > t["pre_mean"] for t in r.confirmed):
        level = "critical"
    elif r.alerts:
        effects = [a - b for b, a in (alert_effect(r.stream.values, i) for i in r.alerts)]
        level = "warning" if any(e >= WATCH_EFFECT for e in effects if not np.isnan(e)) else "watch"
    if level != "good" and not SIGNALS[r.stream.key].provider_drift:
        return "notice"
    return level


def default_run(calls: pd.DataFrame, min_scored: int = 150) -> str | None:
    """The most recent evaluation run with enough scored calls to show one clean
    before/after experiment. None if no single run is long enough."""
    scored = calls[(calls["outcome"] == "ok") & calls["eval_score"].notna() & calls["eval_run_id"].notna()]
    counts = scored.groupby("eval_run_id")["seq_id"].agg(["size", "max"])
    counts = counts[counts["size"] >= min_scored]
    return None if counts.empty else str(counts["max"].idxmax())


def rolling_rate(values: np.ndarray, window: int) -> np.ndarray:
    """Trailing mean, defined from the first call (shorter window at the start)."""
    c = np.cumsum(np.insert(values.astype(float), 0, 0.0))
    idx = np.arange(1, len(values) + 1)
    lo = np.maximum(0, idx - window)
    return (c[idx] - c[lo]) / (idx - lo)


# --------------------------------------------------------------------------- persistence

def record_shifts(db_path: Path | str, provider: str, model_id: str, shifts: pd.DataFrame) -> int:
    """Write shifts into changepoint_events (with category and action), skipping any
    already recorded for the same target, signal, detector and call. Returns rows added."""
    from datetime import datetime, timezone

    from driftguard.storage.store import ChangePointEventRow, DriftStore

    store = DriftStore(db_path=str(db_path))
    existing = load_events(db_path)
    seen = set()
    if not existing.empty:
        for _, e in existing.iterrows():
            seen.add((e["provider"], e["model_id"], e["metric_name"], e["detector"], str(e["changepoint_timestamp"])[:26]))
    added = 0
    for _, s in shifts.iterrows():
        detector = "pelt" if s["detector"].startswith("PELT") else "sequential"
        ts = s["timestamp"].tz_convert(None).to_pydatetime()
        key = (provider, model_id, s["signal"], detector, str(ts)[:26])
        if key in seen:
            continue
        store.write_changepoint_event(ChangePointEventRow(
            detected_at=datetime.now(timezone.utc), provider=provider, model_id=model_id,
            metric_name=s["signal"], detector=detector, changepoint_timestamp=ts,
            pre_mean=None if pd.isna(s["before"]) else float(s["before"]),
            post_mean=None if pd.isna(s["after"]) else float(s["after"]),
            p_value_raw=None, p_value_corrected=None if pd.isna(s["p_adj"]) else float(s["p_adj"]),
            significant=True, drift_type=s["category"], remediation=s["action"],
        ))
        seen.add(key)
        added += 1
    return added


def set_reviewed(db_path: Path | str, event_ids: list[int], reviewed: bool = True) -> None:
    con = sqlite3.connect(str(db_path))
    try:
        con.executemany("UPDATE changepoint_events SET reviewed = ? WHERE id = ?",
                        [(int(reviewed), int(i)) for i in event_ids])
        con.commit()
    finally:
        con.close()
