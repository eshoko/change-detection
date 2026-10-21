"""Parse a free-text response into the set of attributes reported as changed.

Only the last line that starts with ``CHANGED:`` (case-insensitive) is used.
Attribute names may use aliases (e.g. colour, location, number). ``none``
means no attribute was reported as changed. If no ``CHANGED:`` line exists,
or a token names no attribute, the response is unparseable and
``parse_response`` returns ``None``; scoring treats that as "nothing reported"
and counts it.
"""

from __future__ import annotations

import re

from .attributes import ALIAS_TO_NAME

_LINE = re.compile(r"^\s*\**\s*changed\s*\**\s*:\s*(.*)$", re.IGNORECASE)
_SPLIT = re.compile(r",|;|/|\band\b|&|\+")
_NONE = {"none", "nothing", "no change", "no changes", "-", ""}


def parse_response(text: str) -> frozenset[str] | None:
    answer = None
    for line in text.splitlines():
        m = _LINE.match(line)
        if m:
            answer = m.group(1)
    if answer is None:
        return None
    answer = answer.strip().strip(".*`'\"").strip().lower()
    if answer in _NONE:
        return frozenset()
    names = set()
    for tok in _SPLIT.split(answer):
        tok = tok.strip().strip(".*`'\"()[]").strip()
        if not tok:
            continue
        if tok not in ALIAS_TO_NAME:
            return None
        names.add(ALIAS_TO_NAME[tok])
    return frozenset(names)
