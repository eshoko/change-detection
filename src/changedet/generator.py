"""Procedural generator of text-described panel sequences with known ground truth.

Conventions
-----------
* An item is a sequence of ``n_panels`` panels. Each panel holds ``n_objects``
  object groups (e.g. "3 large red circles at the top-left").
* Each object is described by its ``n_attrs`` queried attributes plus
  ``n_distractors`` distractor attributes. Attributes that are not described
  do not exist for that item.
* A queried attribute is *changed* if its value differs for at least one object
  between any two consecutive panels. Exactly ``n_changed`` queried attributes
  change. Every distractor attribute changes, but distractors are never queried.
* At each panel transition, a changing attribute changes for exactly one
  object. Swaps therefore cannot happen, and a change is always visible as a
  difference in the multiset of that attribute's values between the two panels.
  So the ground truth can be recovered from the text even though object order
  is shuffled and no object IDs are shown.
* No two objects in a panel share all described values, so no panel contains
  two identical descriptions (up to synonyms). The initial objects are drawn
  distinct, and a change may not make an object identical to another one.
  If the random choices reach a state with no valid change, the item is redrawn
  (deterministically, from the same generator).
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass, field

import numpy as np

from .attributes import DISTRACTORS, QUERYABLE, VALUES
from .render import MAX_SURFACE_NOISE, render_panel

INSTRUCTIONS = (
    "Each panel below describes groups of objects. An attribute counts as changed "
    "if, for at least one object group, its value differs between two consecutive "
    "panels. Objects may be listed in a different order, and the same value may be "
    "described with different but equivalent words; neither counts as a change."
)
ANSWER_FORMAT = (
    "End your answer with one line of the form "
    "'CHANGED: <attributes separated by commas>' or 'CHANGED: none'."
)


@dataclass(frozen=True)
class Knobs:
    """Difficulty knobs for a single item."""

    n_objects: int
    n_attrs: int
    n_changed: int
    surface_noise: int
    n_distractors: int
    n_panels: int = 3

    def validate(self) -> None:
        if not 1 <= self.n_objects <= 6:
            raise ValueError("n_objects must be in 1..6")
        if not 1 <= self.n_attrs <= len(QUERYABLE):
            raise ValueError(f"n_attrs must be in 1..{len(QUERYABLE)}")
        if not 0 <= self.n_changed <= self.n_attrs:
            raise ValueError("n_changed must be in 0..n_attrs")
        if not 0 <= self.surface_noise <= MAX_SURFACE_NOISE:
            raise ValueError(f"surface_noise must be in 0..{MAX_SURFACE_NOISE}")
        if not 0 <= self.n_distractors <= len(DISTRACTORS):
            raise ValueError(f"n_distractors must be in 0..{len(DISTRACTORS)}")
        if self.n_panels < 2:
            raise ValueError("n_panels must be >= 2")
        if self.n_attrs + self.n_distractors == 1 and self.n_objects > 2:
            # One described attribute with as few as 3 values: distinct objects
            # leave room to change only if n_objects <= 2.
            raise ValueError("with a single described attribute, n_objects must be <= 2")


# Difficulty presets. Within a level, n_changed is drawn uniformly from
# 0..n_attrs for each item, so its mean rises with the level.
LEVELS: dict[int, dict[str, int]] = {
    1: dict(n_objects=2, n_attrs=2, surface_noise=0, n_distractors=0),
    2: dict(n_objects=3, n_attrs=3, surface_noise=1, n_distractors=1),
    3: dict(n_objects=4, n_attrs=4, surface_noise=2, n_distractors=1),
    4: dict(n_objects=5, n_attrs=5, surface_noise=3, n_distractors=2),
}


@dataclass
class Item:
    item_id: str
    level: int | None
    knobs: Knobs
    queried: tuple[str, ...]
    distractors: tuple[str, ...]
    panels: list[list[dict]]
    panel_texts: list[str]
    ground_truth: dict[str, bool]
    prompt: str = field(default="")

    @property
    def n_changed(self) -> int:
        return sum(self.ground_truth.values())

    @property
    def load(self) -> int:
        """Comparison load: object x queried-attribute comparisons over all transitions."""
        k = self.knobs
        return k.n_objects * k.n_attrs * (k.n_panels - 1)

    @property
    def question(self) -> str:
        return f"Which of these attributes change across panels: {', '.join(self.queried)}?"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["knobs"] = asdict(self.knobs)
        return d


def compute_ground_truth(panels: list[list[dict]], attrs) -> dict[str, bool]:
    """Changed = the value differs for some object between consecutive panels."""
    return {
        a: any(
            p[j][a] != q[j][a]
            for p, q in zip(panels[:-1], panels[1:])
            for j in range(len(p))
        )
        for a in attrs
    }


def _pick(rng: np.random.Generator, seq, k: int) -> list:
    idx = sorted(int(i) for i in rng.choice(len(seq), size=k, replace=False))
    return [seq[i] for i in idx]


def _random_value(rng: np.random.Generator, attr: str):
    vals = VALUES[attr]
    return vals[int(rng.integers(len(vals)))]


def _valid_new_values(panel: list[dict], j: int, attr: str) -> list:
    """Values of attr that object j can take without duplicating another object."""
    taken = {o[attr] for i, o in enumerate(panel)
             if i != j and all(o[a] == panel[j][a] for a in o if a != attr)}
    return [v for v in VALUES[attr] if v != panel[j][attr] and v not in taken]


def _draw_objects(rng: np.random.Generator, described, n: int) -> list[dict]:
    objs: list[dict] = []
    while len(objs) < n:
        cand = {a: _random_value(rng, a) for a in described}
        if cand not in objs:
            objs.append(cand)
    return objs


_MAX_REDRAWS = 100


def _draw_panels(rng, knobs: Knobs, described, changing) -> list[list[dict]] | None:
    first = _draw_objects(rng, described, knobs.n_objects)
    n_trans = knobs.n_panels - 1
    # Decide, for each changing attribute, the transitions at which it changes.
    schedule: dict[int, list[str]] = {t: [] for t in range(n_trans)}
    for attr in changing:
        k = int(rng.integers(1, n_trans + 1))
        for t in rng.choice(n_trans, size=k, replace=False):
            schedule[int(t)].append(attr)
    panels = [first]
    for t in range(n_trans):
        nxt = copy.deepcopy(panels[-1])
        for attr in schedule[t]:
            options = {j: _valid_new_values(nxt, j, attr) for j in range(knobs.n_objects)}
            movable = [j for j, vals in options.items() if vals]
            if not movable:
                return None
            j = movable[int(rng.integers(len(movable)))]
            vals = options[j]
            nxt[j][attr] = vals[int(rng.integers(len(vals)))]
        panels.append(nxt)
    return panels


def generate_item(knobs: Knobs, rng: np.random.Generator, item_id: str = "",
                  level: int | None = None) -> Item:
    knobs.validate()
    queried = tuple(_pick(rng, QUERYABLE, knobs.n_attrs))
    distractors = tuple(_pick(rng, DISTRACTORS, knobs.n_distractors))
    changed = _pick(rng, list(queried), knobs.n_changed)
    described = queried + distractors

    for _ in range(_MAX_REDRAWS):
        panels = _draw_panels(rng, knobs, described, list(changed) + list(distractors))
        if panels is not None:
            break
    else:
        raise RuntimeError(f"could not draw distinct objects for {knobs}")

    truth = compute_ground_truth(panels, queried)
    assert truth == {a: a in changed for a in queried}  # internal consistency

    texts = [render_panel(p, i + 1, knobs.surface_noise, rng) for i, p in enumerate(panels)]
    item = Item(item_id=item_id, level=level, knobs=knobs, queried=queried,
                distractors=distractors, panels=panels, panel_texts=texts,
                ground_truth=truth)
    item.prompt = "\n".join([INSTRUCTIONS, "", *texts, "", f"Question: {item.question}",
                             ANSWER_FORMAT])
    return item


def generate_items(knobs: Knobs, n: int, seed: int) -> list[Item]:
    """n items with identical knobs (for analysing one knob at a time)."""
    rng = np.random.default_rng(seed)
    return [generate_item(knobs, rng, item_id=f"K{seed}-{i:04d}") for i in range(n)]


def generate_set(level: int, n: int, seed: int) -> list[Item]:
    """n items at a preset difficulty level; n_changed uniform over 0..n_attrs."""
    preset = LEVELS[level]
    rng = np.random.default_rng([seed, level])
    items = []
    for i in range(n):
        n_changed = int(rng.integers(0, preset["n_attrs"] + 1))
        knobs = Knobs(n_changed=n_changed, **preset)
        items.append(generate_item(knobs, rng, item_id=f"L{level}-{i:04d}", level=level))
    return items


def generate_all_levels(n_per_level: int, seed: int) -> list[Item]:
    return [it for lvl in sorted(LEVELS) for it in generate_set(lvl, n_per_level, seed)]


def items_digest(items: list[Item]) -> str:
    """Stable SHA-256 of an item list (used by regression tests)."""
    payload = json.dumps([it.to_dict() for it in items], sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()
