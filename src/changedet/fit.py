"""Parameter recovery: threshold-model (N, g) and speed (s, t0).

(N, g): exact per-item likelihood
---------------------------------
For an item with k = n_changed changing and m unchanged queried attributes:

* the number of hits is d + Binomial(k - d, g), where d = min(N, k), so any
  item with fewer than d hits has likelihood 0 under that N;
* the number of false alarms is Binomial(m, g);
* items with k = 0 contribute false-alarm trials only.

For a fixed N, the maximum-likelihood g has a closed form: the share of
"guess" trials (undetected changed plus unchanged) that were reported.
N is chosen by grid search over 0..max(k). Ties go to the smallest N.

N is only identifiable up to the largest n_changed in the data: any N >=
max(k) predicts the same data, so an estimate equal to max(k) means
"at least max(k)" (``n_censored``).

Speed: OLS of RT on comparison load gives the slope s and intercept t0,
with classical standard errors.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import gammaln, xlogy

from .scoring import TrialTable


@dataclass(frozen=True)
class ThresholdFit:
    N: int
    g: float
    se_g: float
    loglik: float
    n_guess_trials: int
    n_censored: bool
    loglik_by_N: tuple[float, ...]


@dataclass(frozen=True)
class SpeedFit:
    s: float
    t0: float
    se_s: float
    se_t0: float
    resid_sd: float


def _log_comb(n: np.ndarray, k: np.ndarray) -> np.ndarray:
    return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)


def threshold_loglik(N: int, k: np.ndarray, m: np.ndarray, hits: np.ndarray,
                     fas: np.ndarray, g: float | None = None) -> tuple[float, float, int]:
    """Return (log-likelihood, g used, number of guess trials) for capacity N.

    If g is None, the closed-form MLE for this N is used.
    """
    d = np.minimum(N, k)
    if np.any(hits < d):
        return -np.inf, float("nan"), 0
    guess_yes = int(np.sum(hits - d) + np.sum(fas))
    guess_n = int(np.sum(k - d) + np.sum(m))
    if g is None:
        g = guess_yes / guess_n if guess_n else 0.0
    ll = (np.sum(_log_comb(k - d, hits - d)) + np.sum(_log_comb(m, fas))
          + xlogy(guess_yes, g) + xlogy(guess_n - guess_yes, 1.0 - g))
    return float(ll), float(g), guess_n


def fit_threshold(table: TrialTable) -> ThresholdFit:
    c = table.per_item_counts()
    k, m, hits, fas = c["n_changed"], c["n_unchanged"], c["hits"], c["fas"]
    k_max = int(k.max())
    results = [threshold_loglik(N, k, m, hits, fas) for N in range(k_max + 1)]
    lls = [r[0] for r in results]
    best = int(np.argmax(lls))  # first maximum, i.e. smallest N on ties
    ll, g, n_guess = results[best]
    se_g = float(np.sqrt(g * (1 - g) / n_guess)) if n_guess else float("nan")
    return ThresholdFit(N=best, g=g, se_g=se_g, loglik=ll, n_guess_trials=n_guess,
                        n_censored=best == k_max, loglik_by_N=tuple(lls))


def fit_speed(table: TrialTable) -> SpeedFit:
    x, y = table.item_load, table.item_rt
    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = len(y) - 2
    sigma2 = float(resid @ resid / dof)
    cov = sigma2 * np.linalg.inv(X.T @ X)
    return SpeedFit(s=float(beta[1]), t0=float(beta[0]), se_s=float(np.sqrt(cov[1, 1])),
                    se_t0=float(np.sqrt(cov[0, 0])), resid_sd=float(np.sqrt(sigma2)))
