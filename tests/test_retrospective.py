"""Unit tests for the retrospective drift path:
driftguard/detection/changepoint_pelt.py (PELT) and
driftguard/detection/bootstrap.py (bootstrap significance + Benjamini-Hochberg).

Uses small synthetic streams with a known change point so results are checkable
and fast."""

from __future__ import annotations

import numpy as np
import pytest

from driftguard.detection.bootstrap import (
    SignificanceConfig,
    benjamini_hochberg,
    changepoint_pvalue,
    confirm_changepoints,
)
from driftguard.detection.changepoint_pelt import (
    PeltConfig,
    detect_changepoints,
    segments_from_changepoints,
)


def _step_stream(n_each=300, p0=0.1, p1=0.6, seed=0):
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.binomial(1, p0, n_each), rng.binomial(1, p1, n_each)]).astype(float)


# --- PELT ------------------------------------------------------------------

def test_pelt_finds_the_step_near_the_true_changepoint():
    stream = _step_stream(n_each=300)  # true change at index 300
    cfg = PeltConfig(cost_model="l2", penalty_scale=2.0, min_segment_length=30, jump=5)
    cps = detect_changepoints(stream, cfg)
    assert cps, "PELT should find at least one change point"
    assert min(abs(c - 300) for c in cps) <= 40  # a change point lands near the real one


def test_pelt_flat_signal_finds_few_or_none():
    rng = np.random.default_rng(1)
    flat = rng.binomial(1, 0.3, 600).astype(float)
    cfg = PeltConfig(cost_model="l2", penalty_scale=3.0, min_segment_length=30, jump=5)
    # a stationary stream should not be chopped into many pieces
    assert len(detect_changepoints(flat, cfg)) <= 2


def test_pelt_short_signal_returns_empty():
    assert detect_changepoints(np.zeros(10), PeltConfig(min_segment_length=200)) == []


def test_segments_from_changepoints():
    assert segments_from_changepoints(100, [30, 70]) == [(0, 30), (30, 70), (70, 100)]
    assert segments_from_changepoints(50, []) == [(0, 50)]


# --- bootstrap significance ------------------------------------------------

def test_changepoint_pvalue_detects_real_jump():
    stream = np.concatenate([np.zeros(200), np.ones(200)]).astype(float)
    r = changepoint_pvalue(stream, cp=200, left_bound=0, right_bound=400, n_bootstrap=200, window=None)
    assert r["pre_mean"] == 0.0 and r["post_mean"] == 1.0
    assert r["delta"] == pytest.approx(1.0)
    assert r["p_value_raw"] < 0.05  # a real, sharp jump is significant


def test_changepoint_pvalue_noise_is_not_significant():
    rng = np.random.default_rng(2)
    stream = rng.binomial(1, 0.4, 400).astype(float)  # no real change
    r = changepoint_pvalue(stream, cp=200, left_bound=0, right_bound=400, n_bootstrap=500, window=None, seed=2)
    assert r["p_value_raw"] > 0.05  # random split of homogeneous data is not significant


def test_benjamini_hochberg_basic():
    sig, adj = benjamini_hochberg([0.001, 0.2, 0.9, 0.04], alpha=0.05)
    assert sig[0] and not sig[2]           # tiny p rejected, large p not
    assert adj.shape == (4,)
    assert (adj >= 0).all() and (adj <= 1).all()
    assert benjamini_hochberg([], 0.05)[0].size == 0


def test_confirm_changepoints_integration():
    stream = _step_stream(n_each=250, p0=0.1, p1=0.7, seed=3)  # true change ~250
    results = confirm_changepoints(stream, [250], SignificanceConfig(n_bootstrap_samples=200, window=None))
    assert len(results) == 1
    r = results[0]
    assert {"pre_mean", "post_mean", "delta", "p_value_raw", "p_value_corrected", "significant"} <= set(r)
    assert r["significant"] is True and r["delta"] > 0.3
