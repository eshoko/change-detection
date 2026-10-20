"""Validation: recover known simulated-responder parameters.

Tolerances (see README, "Validation and tolerances"):

* N: the estimate must equal the true N in >= 95% of independent replicate
  datasets (recovery study). N is discrete, and the likelihood separates
  adjacent N sharply, so we require near-certain identification.
* g: |g_hat - g| <= 3 * SE, with SE = sqrt(g_hat (1 - g_hat) / T), where T
  is the number of guess trials. The closed-form estimator is a binomial
  proportion, so a 3-SE band has ~99.7% coverage.
* s: |s_hat - s| <= 3 * SE_OLS (classical OLS standard error; the RT noise is
  Gaussian and homoscedastic by construction, so the OLS SE is exact).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .fit import fit_speed, fit_threshold
from .generator import Item
from .responders import ThresholdResponder, simulate
from .scoring import build_trials

FAST, SLOW = 0.05, 0.20  # seconds per comparison
TOL_SE = 3.0
N_RATE_MIN = 0.95


def responder_grid() -> list[ThresholdResponder]:
    out = []
    for N in (1, 2, 3, 4):
        for g in (0.05, 0.2, 0.4):
            for label, s in (("fast", FAST), ("slow", SLOW)):
                out.append(ThresholdResponder(f"N{N}-g{g}-{label}", N=N, g=g, s=s))
    return out


# Dissociated responders: capacity and speed vary in opposite directions.
DISSOCIATED: dict[str, ThresholdResponder] = {
    "high-N/slow": ThresholdResponder("high-N/slow", N=4, g=0.2, s=SLOW),
    "low-N/fast": ThresholdResponder("low-N/fast", N=1, g=0.2, s=FAST),
    "low-N/slow": ThresholdResponder("low-N/slow", N=1, g=0.2, s=SLOW),
    "high-N/fast": ThresholdResponder("high-N/fast", N=4, g=0.2, s=FAST),
}


@dataclass(frozen=True)
class Recovery:
    name: str
    N: int
    g: float
    s: float
    N_hat: int
    g_hat: float
    se_g: float
    s_hat: float
    se_s: float
    t0_hat: float
    n_censored: bool

    @property
    def N_ok(self) -> bool:
        return self.N_hat == self.N

    @property
    def g_ok(self) -> bool:
        return abs(self.g_hat - self.g) <= TOL_SE * self.se_g

    @property
    def s_ok(self) -> bool:
        return abs(self.s_hat - self.s) <= TOL_SE * self.se_s

    @property
    def ok(self) -> bool:
        return self.N_ok and self.g_ok and self.s_ok

    def as_row(self) -> dict:
        d = asdict(self)
        d.update(N_ok=self.N_ok, g_ok=self.g_ok, s_ok=self.s_ok, ok=self.ok)
        return d


def recover(responder: ThresholdResponder, items: list[Item], seed: int) -> Recovery:
    table = build_trials(items, simulate(responder, items, seed))
    tf, sf = fit_threshold(table), fit_speed(table)
    return Recovery(responder.name, responder.N, responder.g, responder.s,
                    tf.N, tf.g, tf.se_g, sf.s, sf.se_s, sf.t0, tf.n_censored)


def recovery_study(responder: ThresholdResponder, items: list[Item], n_reps: int,
                   seed: int) -> dict[str, float]:
    """Fraction of independent replicate datasets that pass each criterion."""
    seeds = np.random.SeedSequence(seed).generate_state(n_reps)
    recs = [recover(responder, items, int(s)) for s in seeds]
    return dict(
        name=responder.name,
        n_reps=n_reps,
        N_rate=float(np.mean([r.N_ok for r in recs])),
        g_rate=float(np.mean([r.g_ok for r in recs])),
        s_rate=float(np.mean([r.s_ok for r in recs])),
        g_bias=float(np.mean([r.g_hat for r in recs]) - responder.g),
        s_bias=float(np.mean([r.s_hat for r in recs]) - responder.s),
    )


def parametric_bootstrap(items: list[Item], rec: Recovery, n_boot: int, seed: int,
                         t0: float = 1.0, sigma: float = 0.3) -> dict[str, object]:
    """Simulate from the fitted parameters and refit. Returns percentile CIs."""
    fitted = ThresholdResponder("boot", N=rec.N_hat, g=rec.g_hat, s=max(rec.s_hat, 0.0),
                                t0=t0, sigma=sigma)
    seeds = np.random.SeedSequence(seed).generate_state(n_boot)
    reps = [recover(fitted, items, int(s)) for s in seeds]
    Ns = np.array([r.N_hat for r in reps])
    gs = np.array([r.g_hat for r in reps])
    ss = np.array([r.s_hat for r in reps])
    return dict(
        N_mode_rate=float(np.mean(Ns == rec.N_hat)),
        g_ci=tuple(float(x) for x in np.percentile(gs, [2.5, 97.5])),
        s_ci=tuple(float(x) for x in np.percentile(ss, [2.5, 97.5])),
    )
