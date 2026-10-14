"""
driftguard/detection/pipeline.py

The detection pipeline: stored calls in, classified drift events out.

Detection is per call -- the tuned Page-Hinkley detector only works on
per-call input (changepoint_sequential.py), and on real data the per-call
retrospective path placed changes correctly where a per-run one did not
(validation/results/classification_validation/). Classification is left
entirely to the classification layer, unchanged:

    log_rows (one row per call)
      -> signals.py          per-call 0/1 streams, "higher = worse"
      -> changepoint_sequential   Page-Hinkley, one update per call, reset after each alarm
      -> changepoint_pelt + bootstrap + multiple_testing
                             candidate change points, permutation test on the two
                             adjacent segments, Benjamini-Hochberg
      -> ChangePointEventRow in the classification layer's vocabulary
                             (accuracy / format_adherence / refusal_rate / latency_p95_ms,
                             each in its own units and direction -- see signals.py)
      -> classification/rules.py, incidents.py, remediation/mapper.py
                             label, evidence tier, incident, urgency, plan

Entry points:
    detect(calls, provider, model_id, settings)   -> Detection   (nothing written)
    run_detection(store, provider, model_id)       -> RunReport   (writes new events,
                                                                   then mapper.apply_remediation)
    call_signals(...)                              per-call signal values, shared with the
                                                   live monitor (live.py) so both agree
The eval runner calls both paths automatically: live.py on every score during a
run, run_detection at the end of it.

A Page-Hinkley alarm marks where the change was NOTICED, not where it began;
PELT gives the estimated location of a confirmed change.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

from driftguard.classification import rules
from driftguard.detection.bootstrap import SignificanceConfig, confirm_changepoints
from driftguard.detection.changepoint_pelt import PeltConfig, detect_changepoints, segments_from_changepoints
from driftguard.detection.changepoint_sequential import PageHinkleyDetector
from driftguard.detection.signals import SIGNALS, to_metric_value
from driftguard.storage.store import ChangePointEventRow, DriftStore

ROOT = Path(__file__).resolve().parent.parent.parent
DETECTORS_YAML = ROOT / "configs" / "detectors.yaml"
XSTEST_FROZEN = ROOT / "configs" / "xstest_frozen.json"

COMPARE_WINDOW = 200     # calls each side of an alarm used for its before/after values
MIN_SIGNAL_CALLS = 60    # below this a signal is too short to test


# --------------------------------------------------------------------------- settings

@dataclass(frozen=True)
class DetectionSettings:
    ph_lambda: float = 12.0
    ph_delta: float = 0.005
    ph_alpha: float = 0.9999
    ph_min_instances: int = 30
    ph_noise_ref: Optional[float] = None   # bad-call rate lambda was tuned at; None = no noise scaling
    pelt: PeltConfig = field(default_factory=PeltConfig)
    significance: SignificanceConfig = field(default_factory=SignificanceConfig)
    baseline_window: int = 100   # calls used as the quiet baseline (calibration.min_window_size)


def load_settings(path: Path = DETECTORS_YAML) -> DetectionSettings:
    """The tuned settings from configs/detectors.yaml."""
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
        ph_noise_ref=(float(ph["noise_reference_rate"]) if ph.get("noise_reference_rate") is not None else None),
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


# --------------------------------------------------------------------------- per-call signals

@dataclass
class Stream:
    key: str
    values: np.ndarray                         # per-call, higher = worse
    seq_id: np.ndarray                         # row id of each call in log_rows
    timestamp: pd.Series
    note: str = ""
    latency_ms: Optional[np.ndarray] = None    # measured latency per call (slow_call only)


def load_calls(store: DriftStore, provider: Optional[str] = None, model_id: Optional[str] = None) -> pd.DataFrame:
    """Every logged call for one target, in insertion order."""
    sql = ("SELECT rowid AS seq_id, timestamp, provider, model_id, outcome, latency_ms, eval_run_id, "
           "eval_task_type, eval_task_id, eval_score FROM log_rows")
    where, params = [], []
    if provider is not None:
        where.append("provider = ?"); params.append(provider)
    if model_id is not None:
        where.append("model_id = ?"); params.append(model_id)
    if where:
        sql += " WHERE " + " AND ".join(where)
    df = pd.read_sql_query(sql + " ORDER BY rowid", store.engine, params=tuple(params))
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed", utc=True)
    return df


def _should_refuse_map() -> dict[str, bool]:
    try:
        tasks = json.loads(XSTEST_FROZEN.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    return {t["task_id"]: bool(t["should_refuse"]) for t in tasks}


def slow_threshold(calls: pd.DataFrame, baseline_window: int = 100) -> Optional[float]:
    """Latency above which a call counts as slow: the p95 of the first `baseline_window`
    successful calls. None until there are enough successful calls to fix it."""
    lat = calls.loc[(calls["outcome"] == "ok") & calls["latency_ms"].notna(), "latency_ms"]
    if len(lat) < max(MIN_SIGNAL_CALLS, baseline_window + 20):
        return None
    return float(np.percentile(lat.iloc[:baseline_window], 95))


def call_signals(outcome: str, task_type: Optional[str], task_id: Optional[str], score, latency_ms,
                 should_refuse: dict[str, bool], slow_ms: Optional[float]) -> dict[str, float]:
    """The per-call signal values (1 = bad outcome) one call contributes. The single definition
    used by both the batch path (build_streams) and the live monitor (live.py), so the two always
    agree on where an alarm falls."""
    out: dict[str, float] = {}
    ok = outcome == "ok"
    scored = ok and score is not None and not (isinstance(score, float) and np.isnan(score))
    if scored and task_type == "accuracy":
        out["accuracy_error"] = 1.0 - float(score)
    elif scored and task_type == "format":
        out["format_error"] = 1.0 - float(score)
    elif scored and task_type == "refusal":
        should = should_refuse.get(task_id or "")
        if should is None:
            # prompts from outside the frozen XSTest file carry the "_safe_" / "_unsafe_"
            # task-id convention that remediation/evidence.py also reads
            should = True if "_unsafe_" in (task_id or "") else False if "_safe_" in (task_id or "") else None
        if should is False:
            out["safe_refusal"] = float(score)
        elif should is True:
            out["unsafe_compliance"] = 1.0 - float(score)
    # rate limits are our own quota running out, not a change in the provider
    out["quota_exhausted"] = 1.0 if outcome == "rate_limited" else 0.0
    if outcome != "rate_limited":
        out["call_failure"] = 0.0 if ok else 1.0
    if ok and slow_ms is not None and latency_ms is not None and not np.isnan(latency_ms):
        out["slow_call"] = 1.0 if latency_ms > slow_ms else 0.0
    return out


def build_streams(calls: pd.DataFrame, baseline_window: int = 100) -> dict[str, Stream]:
    """Turn the call log into per-call signal streams (only those with enough calls)."""
    should = _should_refuse_map()
    slow_ms = slow_threshold(calls, baseline_window)
    cols: dict[str, dict[str, list]] = {}
    for r in calls.itertuples(index=False):
        latency = None if pd.isna(r.latency_ms) else float(r.latency_ms)
        score = None if pd.isna(r.eval_score) else float(r.eval_score)
        for key, value in call_signals(r.outcome, r.eval_task_type, r.eval_task_id, score, latency,
                                       should, slow_ms).items():
            c = cols.setdefault(key, {"v": [], "seq": [], "ts": [], "lat": []})
            c["v"].append(value); c["seq"].append(r.seq_id); c["ts"].append(r.timestamp); c["lat"].append(latency)
    streams: dict[str, Stream] = {}
    has_quota = bool(cols.get("quota_exhausted", {}).get("v")) and max(cols["quota_exhausted"]["v"]) > 0
    for key, c in cols.items():
        if len(c["v"]) < MIN_SIGNAL_CALLS or (key == "quota_exhausted" and not has_quota):
            continue
        note = (f"slow = above {slow_ms:,.0f} ms, the p95 of the first {baseline_window} calls"
                if key == "slow_call" else "")
        streams[key] = Stream(key, np.asarray(c["v"], dtype=float), np.asarray(c["seq"]),
                              pd.Series(c["ts"]).reset_index(drop=True), note,
                              np.asarray(c["lat"], dtype=float) if key == "slow_call" else None)
    return streams


# --------------------------------------------------------------------------- the two detectors

def ph_noise_scale(baseline: np.ndarray, s: DetectionSettings) -> float:
    """How much noisier this signal's quiet baseline is than the stream lambda was tuned on.

    lambda = 12 was tuned on 0/1 streams with a 5% bad-call rate. A signal that is wrong 30% of
    the time wanders much further by chance, so with the same threshold it alarms on noise. The
    threshold and drift allowance are scaled by the ratio of standard deviations, sqrt(p(1-p)),
    and only ever scaled up, so quiet signals behave exactly as tuned."""
    if s.ph_noise_ref is None or len(baseline) == 0:
        return 1.0
    p, ref = float(np.mean(baseline)), s.ph_noise_ref
    return max(1.0, float(np.sqrt(p * (1 - p) / (ref * (1 - ref)))))


def new_page_hinkley(s: DetectionSettings, scale: float = 1.0) -> PageHinkleyDetector:
    return PageHinkleyDetector("signal", lambda_=s.ph_lambda * scale, delta=s.ph_delta * scale,
                               alpha=s.ph_alpha, min_num_instances=s.ph_min_instances)


def ph_start(s: DetectionSettings) -> int:
    """With noise scaling on, alarms start after the baseline window the scale is taken from."""
    return s.baseline_window if s.ph_noise_ref is not None else 0


def warmed_page_hinkley(baseline: np.ndarray, s: DetectionSettings) -> tuple[PageHinkleyDetector, float]:
    """A scaled detector that has already seen the baseline calls, so its running mean starts
    from the quiet level instead of being learned from calls that may include the change.
    Alarms during the warm-up are ignored (the detector is restarted)."""
    scale = ph_noise_scale(baseline, s)
    det = new_page_hinkley(s, scale)
    for v in baseline:
        if det.update(float(v)) is not None:
            det = new_page_hinkley(s, scale)
    return det, scale


def page_hinkley_alerts(values: np.ndarray, s: DetectionSettings) -> list[int]:
    """Where the tuned Page-Hinkley detector alarms, fed one call at a time. It is reset after
    each alarm, so a sustained change gives one alarm rather than a permanent alarm state."""
    start = ph_start(s)
    if len(values) <= start:
        return []
    if start:
        det, scale = warmed_page_hinkley(np.asarray(values[:start], dtype=float), s)
    else:
        det, scale = new_page_hinkley(s), 1.0
    alerts = []
    for i in range(start, len(values)):
        if det.update(float(values[i])) is not None:
            alerts.append(i)
            det = new_page_hinkley(s, scale)
    return alerts


def effective_pelt(n: int, s: DetectionSettings) -> PeltConfig:
    """The tuned minimum segment (200 calls) assumes long histories. For a short history it is
    capped at a fifth of the stream so PELT can still run, never below 30 calls."""
    p = s.pelt
    seg = max(30, min(p.min_segment_length, n // 5))
    jump = max(1, min(p.jump, seg // 4))
    return PeltConfig(cost_model=p.cost_model, penalty=p.penalty, penalty_scale=p.penalty_scale,
                      min_segment_length=seg, jump=jump)


@dataclass
class SignalResult:
    stream: Stream
    alerts: list[int]                          # Page-Hinkley alarm positions
    tested: list[dict]                         # every PELT candidate with its significance result
    segments: list[tuple[int, int, float]]     # (start, end, mean) of the confirmed regimes
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


# --------------------------------------------------------------------------- events for the classification layer

@dataclass
class DetectedEvent:
    signal: str                       # signals.py key
    index: int                        # position in that signal's stream
    seq_id: int                       # log row of the call where it was placed
    event: ChangePointEventRow        # what rules.py / incidents.py read (not yet saved)


@dataclass
class Detection:
    results: dict[str, SignalResult]
    events: list[DetectedEvent]


def _p95(x: np.ndarray) -> float:
    return float(np.percentile(x, 95)) if len(x) else float("nan")


def _metric_values(st: Stream, lo_pre: int, cut: int, hi_post: int, rate_pre: float, rate_post: float):
    """Before/after values in the metric's own units."""
    if SIGNALS[st.key].metric == "latency_p95_ms" and st.latency_ms is not None:
        return _p95(st.latency_ms[lo_pre:cut]), _p95(st.latency_ms[cut:hi_post])
    return to_metric_value(st.key, rate_pre), to_metric_value(st.key, rate_post)


def _ts(st: Stream, i: int) -> datetime:
    """Timestamp of call i as a timezone-aware UTC datetime (newer sqlmodel refuses naive ones)."""
    t = pd.Timestamp(st.timestamp.iloc[min(i, len(st.timestamp) - 1)])
    t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    return t.to_pydatetime()


def detect(calls: pd.DataFrame, provider: str, model_id: str,
           s: Optional[DetectionSettings] = None) -> Detection:
    """Run both detectors over one target's calls and express every detection as a
    ChangePointEventRow in the classification layer's vocabulary. Writes nothing.
    PELT candidates that fail the significance test are included with significant=False
    (rules.py calls them 'unconfirmed')."""
    s = s or load_settings()
    results = analyse(build_streams(calls, s.baseline_window), s)
    now = datetime.now(timezone.utc)
    win = s.significance.window or 10**9
    events: list[DetectedEvent] = []
    for key, r in results.items():
        st, v, n = r.stream, r.stream.values, len(r.stream.values)
        metric = SIGNALS[key].metric or key
        bounds = [0, *[t["index"] for t in r.tested], n]
        for j, t in enumerate(r.tested):
            i = t["index"]
            pre, post = _metric_values(st, max(bounds[j], i - win), i, min(bounds[j + 2], i + win),
                                       t["pre_mean"], t["post_mean"])
            events.append(DetectedEvent(key, i, int(st.seq_id[i]), ChangePointEventRow(
                detected_at=now, provider=provider, model_id=model_id, metric_name=metric, detector="pelt",
                changepoint_timestamp=_ts(st, i), pre_mean=pre, post_mean=post,
                p_value_raw=t["p_value_raw"], p_value_corrected=t.get("p_value_corrected"),
                significant=bool(t.get("significant")))))
        for i in r.alerts:
            pre_v, post_v = v[max(0, i - COMPARE_WINDOW):i], v[i:i + COMPARE_WINDOW]
            rb = float(pre_v.mean()) if len(pre_v) else float("nan")
            ra = float(post_v.mean()) if len(post_v) else float("nan")
            pre, post = _metric_values(st, max(0, i - COMPARE_WINDOW), i, i + COMPARE_WINDOW, rb, ra)
            events.append(DetectedEvent(key, i, int(st.seq_id[i]), ChangePointEventRow(
                detected_at=now, provider=provider, model_id=model_id, metric_name=metric,
                detector="sequential", changepoint_timestamp=_ts(st, i), pre_mean=pre, post_mean=post,
                significant=True)))
    events.sort(key=lambda d: (d.seq_id, d.event.detector))
    return Detection(results, events)


# --------------------------------------------------------------------------- persistence + classification

@dataclass
class RunReport:
    detection: Detection
    written: int        # new events saved to changepoint_events
    labelled: int       # events the classification layer updated (label and/or plan)


def _event_key(e) -> tuple:
    return (e.provider, e.model_id, e.metric_name, e.detector, str(e.changepoint_timestamp)[:19])


def save_events(store: DriftStore, events: list[ChangePointEventRow]) -> tuple[int, int]:
    """Save events not already stored (same target, metric, detector and time), then let the
    classification layer label and plan them (remediation/mapper.apply_remediation, which runs
    rules.py and incidents.py). Returns (events written, events labelled/planned)."""
    from driftguard.remediation import mapper

    existing = pd.read_sql_query("SELECT provider, model_id, metric_name, detector, changepoint_timestamp "
                                 "FROM changepoint_events", store.engine)
    seen = {(r.provider, r.model_id, r.metric_name, r.detector, str(r.changepoint_timestamp)[:19])
            for r in existing.itertuples(index=False)}
    written = 0
    for e in events:
        k = _event_key(e)
        if k in seen:
            continue
        store.write_changepoint_event(ChangePointEventRow(**e.model_dump(exclude={"id"})))
        seen.add(k)
        written += 1
    labelled = mapper.apply_remediation(store) if written else 0
    return written, labelled


def run_detection(store: DriftStore, provider: str, model_id: str,
                  s: Optional[DetectionSettings] = None) -> RunReport:
    """Detect drift for one target from its stored calls, save every new event, then let the
    classification layer label and plan them. Re-running over the same calls adds nothing twice."""
    det = detect(load_calls(store, provider, model_id), provider, model_id, s)
    written, labelled = save_events(store, [d.event for d in det.events])
    return RunReport(det, written, labelled)
