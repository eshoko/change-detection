"""Attribute vocabularies, surface synonyms and name aliases.

Every attribute has a fixed list of canonical values. Surface synonyms are
alternative wordings of the *same* canonical value; they are a source of
irrelevant variation and never change the ground truth.
"""

from __future__ import annotations

# Attributes that can be queried ("Which of these attributes change?").
QUERYABLE: tuple[str, ...] = ("shape", "size", "color", "position", "count")

# Distractor attributes: may change across panels but are never queried.
DISTRACTORS: tuple[str, ...] = ("texture", "orientation")

ALL_ATTRIBUTES: tuple[str, ...] = QUERYABLE + DISTRACTORS

VALUES: dict[str, tuple] = {
    "shape": ("circle", "square", "triangle", "star", "hexagon"),
    "size": ("small", "medium", "large"),
    "color": ("red", "blue", "green", "yellow", "purple"),
    "position": ("top-left", "top-right", "center", "bottom-left", "bottom-right"),
    "count": (1, 2, 3, 4),
    "texture": ("striped", "dotted", "solid", "checkered"),
    "orientation": ("upright", "tilted", "sideways", "upside-down"),
}

# Surface forms per canonical value. Index 0 is the canonical wording used
# when surface noise is below the synonym threshold.
SYNONYMS: dict[str, dict] = {
    "shape": {
        "circle": ("circle", "disc"),
        "square": ("square",),
        "triangle": ("triangle",),
        "star": ("star",),
        "hexagon": ("hexagon",),
    },
    "size": {
        "small": ("small", "little"),
        "medium": ("medium", "medium-sized"),
        "large": ("large", "big"),
    },
    "color": {
        "red": ("red", "scarlet"),
        "blue": ("blue",),
        "green": ("green",),
        "yellow": ("yellow",),
        "purple": ("purple", "violet"),
    },
    "position": {
        "top-left": ("at the top-left", "in the upper-left corner"),
        "top-right": ("at the top-right", "in the upper-right corner"),
        "center": ("in the center", "in the middle"),
        "bottom-left": ("at the bottom-left", "in the lower-left corner"),
        "bottom-right": ("at the bottom-right", "in the lower-right corner"),
    },
    "count": {
        1: ("1", "one"),
        2: ("2", "two"),
        3: ("3", "three"),
        4: ("4", "four"),
    },
    "texture": {
        "striped": ("striped", "banded"),
        "dotted": ("dotted", "spotted"),
        "solid": ("solid", "plain"),
        "checkered": ("checkered", "chequered"),
    },
    "orientation": {
        "upright": ("upright",),
        "tilted": ("tilted", "slanted"),
        "sideways": ("sideways",),
        "upside-down": ("upside-down", "inverted"),
    },
}

# Words a respondent may use to name an attribute (lower case).
NAME_ALIASES: dict[str, tuple[str, ...]] = {
    "shape": ("shape", "shapes", "form", "forms"),
    "size": ("size", "sizes"),
    "color": ("color", "colors", "colour", "colours"),
    "position": ("position", "positions", "location", "locations", "place", "places"),
    "count": ("count", "counts", "number", "numbers", "quantity", "quantities"),
    "texture": ("texture", "textures", "pattern", "patterns"),
    "orientation": ("orientation", "orientations", "rotation", "rotations"),
}

ALIAS_TO_NAME: dict[str, str] = {
    alias: name for name, aliases in NAME_ALIASES.items() for alias in aliases
}


def _check_tables() -> None:
    for attr in ALL_ATTRIBUTES:
        assert set(SYNONYMS[attr]) == set(VALUES[attr]), attr
    # A surface form must identify a single canonical value within its attribute.
    for attr, table in SYNONYMS.items():
        forms = [f for fs in table.values() for f in fs]
        assert len(forms) == len(set(forms)), attr


_check_tables()
