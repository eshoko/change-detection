"""Scoring: per-attribute signal detection, accuracy and response time.

Conventions
-----------
* One trial = one queried attribute of one item. Signal = the attribute
  changed. "Yes" = the response reports it as changed.
* Hit rate H and false-alarm rate F use the log-linear correction
  (Hautus, 1995): H = (hits + 0.5) / (n_signal + 1), and likewise for F.
  It is applied to every cell, so d' and c are always finite.
* d' = z(H) - z(F) and c = -(z(H) + z(F)) / 2. A positive c means a
  conservative responder; a negative c means a bias toward reporting change.
* d' and c are *descriptive*. They assume an equal-variance Gaussian model,
  but the simulated responders follow a high-threshold rule, under which d'
  also shifts when only the guess rate changes. Parameter recovery relies on
  the (N, g) fit in ``fit.py``.
* Accuracy (per trial, and exact match per item) and RT (seconds) are
  reported separately.
* An unparseable response counts as "nothing reported" and is counted in
  ``unparseable_rate``. Reported attributes that were not queried
  (e.g. distractors) are ignored and counted in ``extraneous_reports``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from .generator import Item
from .parse import parse_response
from .responders import Response


@dataclass
class TrialTable:
    # per trial
    item_index: np.ndarray
    attr: np.ndarray
    changed: np.ndarray
    reported: np.ndarray
    # per item
    item_level: np.ndarray
    item_n_changed: np.ndarray
    item_n_attrs: np.ndarray
    item_load: np.ndarray
    item_rt: np.ndarray
    item_unparseable: np.ndarray
    item_extraneous: np.ndarray

    def per_item_counts(self) -> dict[str, np.ndarray]:
        """Hits and false alarms per item (for the threshold-model fit)."""
        n = len(self.item_level)
        hits = np.bincount(self.item_index, weights=self.changed & self.reported, minlength=n)
        fas = np.bincount(self.item_index, weights=~self.changed & self.reported, minlength=n)
        return dict(
            n_changed=self.item_n_changed,
            n_unchanged=self.item_n_attrs - self.item_n_changed,
            hits=hits.astype(int),
            fas=fas.astype(int),
        )

    def subset_items(self, mask: np.ndarray) -> "TrialTable":
        keep = np.flatnonzero(mask)
        remap = -np.ones(len(mask), dtype=int)
        remap[keep] = np.arange(len(keep))
        t = np.isin(self.item_index, keep)
        return TrialTable(
            item_index=remap[self.item_index[t]], attr=self.attr[t],
            changed=self.changed[t], reported=self.reported[t],
            item_level=self.item_level[keep], item_n_changed=self.item_n_changed[keep],
            item_n_attrs=self.item_n_attrs[keep], item_load=self.item_load[keep],
            item_rt=self.item_rt[keep], item_unparseable=self.item_unparseable[keep],
            item_extraneous=self.item_extraneous[keep],
        )


def build_trials(items: list[Item], responses: list[Response]) -> TrialTable:
    if len(items) != len(responses):
        raise ValueError("items and responses differ in length")
    idx, attrs, changed, reported = [], [], [], []
    unparse, extra = [], []
    for i, (it, r) in enumerate(zip(items, responses)):
        if it.item_id != r.item_id:
            raise ValueError(f"response {r.item_id} does not match item {it.item_id}")
        parsed = parse_response(r.text)
        unparse.append(parsed is None)
        parsed = parsed or frozenset()
        extra.append(len(parsed - set(it.queried)))
        for a in it.queried:
            idx.append(i)
            attrs.append(a)
            changed.append(it.ground_truth[a])
            reported.append(a in parsed)
    return TrialTable(
        item_index=np.array(idx, dtype=int), attr=np.array(attrs),
        changed=np.array(changed, dtype=bool), reported=np.array(reported, dtype=bool),
        item_level=np.array([it.level if it.level is not None else -1 for it in items]),
        item_n_changed=np.array([it.n_changed for it in items]),
        item_n_attrs=np.array([len(it.queried) for it in items]),
        item_load=np.array([it.load for it in items], dtype=float),
        item_rt=np.array([r.rt for r in responses], dtype=float),
        item_unparseable=np.array(unparse, dtype=bool),
        item_extraneous=np.array(extra, dtype=int),
    )


def sdt(hits: int, n_signal: int, fas: int, n_noise: int) -> dict[str, float]:
    """Log-linear-corrected d' and criterion c."""
    H = (hits + 0.5) / (n_signal + 1.0)
    F = (fas + 0.5) / (n_noise + 1.0)
    zH, zF = norm.ppf(H), norm.ppf(F)
    return dict(H=float(H), F=float(F), dprime=float(zH - zF), c=float(-(zH + zF) / 2))


def summarize(table: TrialTable) -> dict[str, float]:
    sig, rep = table.changed, table.reported
    hits, fas = int(np.sum(sig & rep)), int(np.sum(~sig & rep))
    n_sig, n_noise = int(sig.sum()), int((~sig).sum())
    out = sdt(hits, n_sig, fas, n_noise)
    correct = sig == rep
    n_items = len(table.item_level)
    item_wrong = np.bincount(table.item_index, weights=~correct, minlength=n_items)
    out.update(
        n_items=n_items,
        n_signal=n_sig,
        n_noise=n_noise,
        hit_rate_raw=hits / n_sig if n_sig else float("nan"),
        fa_rate_raw=fas / n_noise if n_noise else float("nan"),
        accuracy=float(correct.mean()),
        exact_accuracy=float(np.mean(item_wrong == 0)),
        mean_rt=float(table.item_rt.mean()),
        sd_rt=float(table.item_rt.std(ddof=1)) if n_items > 1 else float("nan"),
        unparseable_rate=float(table.item_unparseable.mean()),
        extraneous_reports=int(table.item_extraneous.sum()),
    )
    return out


def summarize_by_level(table: TrialTable) -> dict[int, dict[str, float]]:
    return {
        int(lvl): summarize(table.subset_items(table.item_level == lvl))
        for lvl in np.unique(table.item_level)
    }
