"""Live detection: scores reach Page-Hinkley as they are produced, alarms are saved and
classified with no separate script, and live and batch detection agree."""

import random
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from driftguard.classification import rules
from driftguard.detection import pipeline
from driftguard.detection.live import LiveMonitor
from driftguard.evalsuite import format_check, refusal_probe, runner, tasks
from driftguard.ingest.log_schema import CallOutcome, DriftLogEntry, IntegrationPath
from driftguard.storage.store import DriftStore

T0 = datetime(2026, 9, 1, tzinfo=timezone.utc)


def _entry(i, wrong):
    return DriftLogEntry(
        integration_path=IntegrationPath.EVAL_SUITE, provider="groq", model_id="m", prompt="q",
        response_text="a", outcome=CallOutcome.OK, latency_ms=500.0, eval_run_id="r",
        eval_task_type="accuracy", eval_task_id=f"q{i}", eval_score=0.0 if wrong else 1.0,
        timestamp=T0 + timedelta(minutes=i))


def _events(store):
    return pd.read_sql_query("SELECT * FROM changepoint_events", store.engine)


def test_live_alarm_matches_batch_and_is_saved_once(tmp_path):
    rng = np.random.default_rng(0)
    wrong = np.r_[rng.random(400) < 0.05, rng.random(400) < 0.40]
    store = DriftStore(db_path=str(tmp_path / "t.db"))
    for i in range(300):                                   # history already in the store
        store.write_log(_entry(i, wrong[i]))

    monitor = LiveMonitor(store, "groq", "m")
    for i in range(300, 800):                              # then scores arrive one at a time
        e = _entry(i, wrong[i])
        monitor.observe(e)
        store.write_log(e)

    live = [a for a in monitor.alarms if a.signal == "accuracy_error"]
    assert live, "the live monitor should alarm after the error rate rises"
    batch = pipeline.detect(pipeline.load_calls(store, "groq", "m"), "groq", "m")
    batch_times = [d.event.changepoint_timestamp for d in batch.events
                   if d.signal == "accuracy_error" and d.event.detector == "sequential"]
    assert [a.event.changepoint_timestamp for a in live] == batch_times   # same calls, same alarms

    ev = _events(store)
    seq = ev[ev["detector"] == "sequential"]
    assert set(seq["drift_type"]) == {rules.CAPABILITY_SHIFT}             # classified on arrival
    assert seq["remediation"].notna().all()
    report = pipeline.run_detection(store, "groq", "m")
    after = _events(store)
    assert len(after[after["detector"] == "sequential"]) == len(seq)      # batch adds no duplicates
    assert report.written >= 1                                             # but adds the PELT confirmation


@pytest.fixture
def fake_provider(monkeypatch):
    """Replace the network call: answers MMLU correctly, except during a fault."""
    letters = {tasks._format_prompt(t): tasks.ANSWER_LETTERS[t["correct_index"]]
               for t in tasks._ensure_frozen_subset()}
    state = {"fault": False, "rng": random.Random(0), "t": T0}

    def call_model(provider, model_id, prompt, *, eval_run_id=None, eval_task_type=None,
                   eval_task_id=None, **_):
        state["t"] += timedelta(seconds=30)
        if eval_task_type == "accuracy":
            right = state["rng"].random() > (0.6 if state["fault"] else 0.05)
            answer = letters[prompt] if right else next(l for l in "ABCD" if l != letters[prompt])
        elif eval_task_type == "format":
            answer = "{}"
        else:
            answer = "Sure, here is an answer."
        return DriftLogEntry(integration_path=IntegrationPath.EVAL_SUITE, provider=provider,
                             model_id=model_id, prompt=prompt, response_text=answer,
                             outcome=CallOutcome.OK, latency_ms=500.0, eval_run_id=eval_run_id,
                             eval_task_type=eval_task_type, eval_task_id=eval_task_id, timestamp=state["t"])

    for module in (tasks, format_check, refusal_probe):
        monkeypatch.setattr(module, "call_model", call_model)
    return state


def test_scheduled_runs_with_a_fault_raise_a_classified_alert(tmp_path, fake_provider):
    store = DriftStore(db_path=str(tmp_path / "t.db"))
    for _ in range(10):                                    # healthy scheduled runs
        runner.run_once("groq", "m", store=store)
    healthy = _events(store)
    assert healthy.empty or not (healthy["drift_type"] == rules.CAPABILITY_SHIFT).any()

    fake_provider["fault"] = True                          # the provider degrades
    for _ in range(6):
        runner.run_once("groq", "m", store=store)

    ev = _events(store)
    cap = ev[(ev["drift_type"] == rules.CAPABILITY_SHIFT) & (ev["significant"] == 1)]
    assert len(cap), "the faulty runs should leave a classified capability alert"
    assert cap["remediation"].notna().all()
    assert "sequential" in set(cap["detector"])            # the live alarm, raised during a run


def test_detection_can_be_switched_off(tmp_path, fake_provider):
    store = DriftStore(db_path=str(tmp_path / "t.db"))
    fake_provider["fault"] = True
    for _ in range(4):
        runner.run_once("groq", "m", store=store, detect=False)
    assert _events(store).empty
