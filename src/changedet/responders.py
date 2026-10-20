"""Simulated responders with known, independent parameters.

Each responder follows a high-threshold rule with a speed component:

* capacity ``N``: detects a random subset of ``min(N, n_changed)`` of the
  changing queried attributes;
* guess rate ``g`` (bias toward reporting change): every queried attribute
  that was not detected (changed or not) is reported as changed with
  probability ``g``;
* optional ``lapse`` (default 0): each detected change is lost with this
  probability and falls back to guessing. The (N, g) fit has no lapse
  parameter; this option exists to show how fragile the fit is without one.
* speed ``s`` (seconds per comparison): ``RT = t0 + s * load + noise``, where
  ``load = n_objects * n_attrs * (n_panels - 1)`` and noise ~ N(0, sigma).
  RT is floored at ``rt_floor``.

Decisions and RT noise use separate random streams. So, for a given seed,
changing ``s`` leaves every decision unchanged, and changing ``N`` or ``g``
leaves every RT unchanged. This independence is built into the simulation:
recovering the three parameters separately validates the *analysis*, not a
claim about real responders.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .attributes import NAME_ALIASES
from .generator import Item

_PREAMBLES = (
    "I compared each attribute across consecutive panels.",
    "Comparing the panels one transition at a time:",
    "After checking every object group in each panel,",
    "",
)


@dataclass(frozen=True)
class Response:
    item_id: str
    text: str
    rt: float


@dataclass(frozen=True)
class ThresholdResponder:
    name: str
    N: int
    g: float
    s: float
    t0: float = 1.0
    sigma: float = 0.3
    rt_floor: float = 0.05
    lapse: float = 0.0

    def __post_init__(self):
        if self.N < 0:
            raise ValueError("N must be >= 0")
        if not 0.0 <= self.g <= 1.0:
            raise ValueError("g must be in [0, 1]")
        if self.s < 0:
            raise ValueError("s must be >= 0")
        if not 0.0 <= self.lapse <= 1.0:
            raise ValueError("lapse must be in [0, 1]")

    def decide(self, item: Item, rng: np.random.Generator) -> set[str]:
        changed = [a for a in item.queried if item.ground_truth[a]]
        n_det = min(self.N, len(changed))
        detected = {changed[int(i)] for i in rng.choice(len(changed), size=n_det, replace=False)}
        if self.lapse > 0:  # draws only when lapses are on, so lapse=0 streams are unchanged
            detected = {a for a in sorted(detected) if rng.random() >= self.lapse}
        reported = set(detected)
        for a in item.queried:
            if a not in detected and rng.random() < self.g:
                reported.add(a)
        return reported

    def write(self, reported: set[str], item: Item, rng: np.random.Generator) -> str:
        names = [a for a in item.queried if a in reported]
        names = [names[int(i)] for i in rng.permutation(len(names))]
        # Occasionally use an alias (e.g. "colour") to exercise the parser.
        words = []
        for a in names:
            aliases = NAME_ALIASES[a]
            words.append(aliases[int(rng.integers(len(aliases)))] if rng.random() < 0.2 else a)
        answer = ", ".join(words) if words else "none"
        pre = _PREAMBLES[int(rng.integers(len(_PREAMBLES)))]
        return f"{pre}\nCHANGED: {answer}".strip()

    def rt(self, item: Item, rng: np.random.Generator) -> float:
        return max(self.rt_floor, self.t0 + self.s * item.load + self.sigma * rng.standard_normal())


def simulate(responder: ThresholdResponder, items: list[Item], seed: int) -> list[Response]:
    dec_seq, rt_seq = np.random.SeedSequence(seed).spawn(2)
    dec_rng, rt_rng = np.random.default_rng(dec_seq), np.random.default_rng(rt_seq)
    out = []
    for it in items:
        reported = responder.decide(it, dec_rng)
        text = responder.write(reported, it, dec_rng)
        out.append(Response(it.item_id, text, responder.rt(it, rt_rng)))
    return out
