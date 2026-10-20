"""Render canonical panels as text, with optional surface noise.

Surface noise never changes an attribute value. Its levels add up:

* 0: fixed object order, canonical words, one sentence template
* 1: object order shuffled independently in each panel
* 2: plus synonyms (e.g. large -> big, 3 -> three), chosen for each mention
* 3: plus varied sentence templates
"""

from __future__ import annotations

import numpy as np

from .attributes import SYNONYMS

SHUFFLE_LEVEL = 1
SYNONYM_LEVEL = 2
TEMPLATE_LEVEL = 3
MAX_SURFACE_NOISE = 3

# Adjective order inside a noun phrase.
_ADJ_ORDER = ("size", "orientation", "texture", "color")


def _form(attr: str, value, surface_noise: int, rng: np.random.Generator) -> str:
    forms = SYNONYMS[attr][value]
    if surface_noise >= SYNONYM_LEVEL and len(forms) > 1:
        return forms[int(rng.integers(len(forms)))]
    return forms[0]


def _article(word: str) -> str:
    return "an" if word[0].lower() in "aeiou" else "a"


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:]


def noun_phrase(obj: dict, surface_noise: int, rng: np.random.Generator) -> tuple[str, bool]:
    """Return (noun phrase, is_plural) for one object group."""
    plural = False
    head: list[str] = []
    if "count" in obj:
        plural = obj["count"] > 1
        head.append(_form("count", obj["count"], surface_noise, rng))
    adjs = [_form(a, obj[a], surface_noise, rng) for a in _ADJ_ORDER if a in obj]
    noun = _form("shape", obj["shape"], surface_noise, rng) if "shape" in obj else "object"
    if plural:
        noun += "s"
    words = adjs + [noun]
    if not head:
        head.append(_article(words[0]))
    return " ".join(head + words), plural


def render_object(obj: dict, surface_noise: int, rng: np.random.Generator) -> str:
    np_text, plural = noun_phrase(obj, surface_noise, rng)
    be = "are" if plural else "is"
    pos = _form("position", obj["position"], surface_noise, rng) if "position" in obj else ""
    template = 0
    if surface_noise >= TEMPLATE_LEVEL:
        choices = [0, 1, 2, 3] if pos else [0, 2]
        template = choices[int(rng.integers(len(choices)))]
    tail = f" {pos}" if pos else ""
    if template == 0:
        return f"There {be} {np_text}{tail}."
    if template == 1:
        return f"{_cap(pos)}, there {be} {np_text}."
    if template == 2:
        return f"You can see {np_text}{tail}."
    return f"{_cap(np_text)} {be} {pos}."


def render_panel(objects: list[dict], index: int, surface_noise: int,
                 rng: np.random.Generator) -> str:
    """Render one panel. `objects` hold canonical values for the described attributes only."""
    order = list(range(len(objects)))
    if surface_noise >= SHUFFLE_LEVEL:
        order = [int(i) for i in rng.permutation(len(objects))]
    sentences = [render_object(objects[i], surface_noise, rng) for i in order]
    return f"Panel {index}: " + " ".join(sentences)
