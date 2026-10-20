"""Recovery of known simulated-responder parameters (N, g, s)."""

import dataclasses

import numpy as np
import pytest

from changedet.fit import fit_speed, fit_threshold
from changedet.responders import ThresholdResponder, simulate
from changedet.scoring import build_trials, summarize
from changedet.validation import (DISSOCIATED, FAST, N_RATE_MIN, SLOW, recover,
                                  recovery_study, responder_grid)

SEED = 11


def test_simulated_text_always_parses(items):
    r = ThresholdResponder("x", N=2, g=0.3, s=FAST)
    s = summarize(build_trials(items, simulate(r, items, SEED)))
    assert s["unparseable_rate"] == 0.0


@pytest.mark.parametrize("resp", responder_grid(), ids=lambda r: r.name)
def test_grid_recovery(items, resp):
    rec = recover(resp, items, SEED)
    assert rec.N_ok, rec
    assert rec.g_ok, rec
    assert rec.s_ok, rec


@pytest.mark.parametrize("name", list(DISSOCIATED))
def test_dissociated_recovery(items, name):
    rec = recover(DISSOCIATED[name], items, SEED)
    assert rec.ok, rec


def test_dissociation_high_N_slow_vs_low_N_fast(items):
    a = recover(DISSOCIATED["high-N/slow"], items, SEED)
    b = recover(DISSOCIATED["low-N/fast"], items, SEED)
    assert a.N_hat > b.N_hat          # more capacity...
    assert a.s_hat > b.s_hat + 3 * np.hypot(a.se_s, b.se_s)  # ...but slower


@pytest.mark.parametrize("name", ["N4-g0.4-slow", "N1-g0.05-fast", "N3-g0.2-fast"])
def test_recovery_rate_over_replicates(items, name):
    resp = next(r for r in responder_grid() if r.name == name)
    st = recovery_study(resp, items, n_reps=30, seed=5)
    assert st["N_rate"] >= N_RATE_MIN, st
    # 3-SE bands: expect ~99.7% coverage; allow a little Monte Carlo slack.
    assert st["g_rate"] >= 0.9 and st["s_rate"] >= 0.9, st


def test_changing_speed_only_leaves_N_g_unchanged(items):
    """Decisions use their own random stream, so s cannot leak into (N, g)."""
    base = ThresholdResponder("a", N=3, g=0.2, s=FAST)
    slow = dataclasses.replace(base, s=SLOW)
    fa = fit_threshold(build_trials(items, simulate(base, items, SEED)))
    fb = fit_threshold(build_trials(items, simulate(slow, items, SEED)))
    assert (fa.N, fa.g) == (fb.N, fb.g)


def test_changing_N_g_only_leaves_speed_unchanged(items):
    a = ThresholdResponder("a", N=1, g=0.05, s=SLOW)
    b = ThresholdResponder("b", N=4, g=0.4, s=SLOW)
    sa = fit_speed(build_trials(items, simulate(a, items, SEED)))
    sb = fit_speed(build_trials(items, simulate(b, items, SEED)))
    assert sa.s == sb.s


def test_speed_estimate_does_not_track_N_or_g(items):
    """Across the grid (different seeds per cell), s_hat errors are unrelated to N and g."""
    rows = []
    for i, r in enumerate(responder_grid()):
        rec = recover(r, items, 1000 + i)
        rows.append((r.N, r.g, (rec.s_hat - r.s) / rec.se_s))
    rows = np.array(rows)
    assert np.all(np.abs(rows[:, 2]) <= 3)
    for col in (0, 1):
        assert abs(np.corrcoef(rows[:, col], rows[:, 2])[0, 1]) < 0.6


def test_dprime_shifts_with_g_alone(items):
    """Model caveat: under the threshold rule, d' moves when only g changes.

    So d' is descriptive, not a capacity measure; the (N, g) fit still
    recovers N = 2 in both conditions.
    """
    reps = 15
    dps = {}
    for g in (0.05, 0.4):
        r = ThresholdResponder(f"g{g}", N=2, g=g, s=FAST)
        vals, Ns = [], []
        for k in range(reps):
            t = build_trials(items, simulate(r, items, 300 + k))
            vals.append(summarize(t)["dprime"])
            Ns.append(fit_threshold(t).N)
        assert all(n == 2 for n in Ns)
        dps[g] = np.array(vals)
    diff = dps[0.05].mean() - dps[0.4].mean()
    sd = np.hypot(dps[0.05].std(ddof=1), dps[0.4].std(ddof=1))
    assert abs(diff) > 3 * sd, (diff, sd)


def test_N_not_identifiable_beyond_max_n_changed(items):
    small = [it for it in items if it.n_changed <= 2]
    r = ThresholdResponder("x", N=4, g=0.2, s=FAST)
    f = fit_threshold(build_trials(small, simulate(r, small, SEED)))
    assert f.N == 2 and f.n_censored
