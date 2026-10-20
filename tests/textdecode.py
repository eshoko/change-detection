"""Test helper: decode a rendered panel back into per-attribute value multisets.

Independent of the renderer's internals: it only uses the synonym table.
Used to check that the text itself carries the ground truth.
"""

import re
from collections import Counter

from changedet.attributes import SYNONYMS


def decode_panel(text: str, attrs) -> dict[str, Counter]:
    body = re.sub(r"^Panel \d+:\s*", "", text)
    out = {}
    for attr in attrs:
        forms = [(f, v) for v, fs in SYNONYMS[attr].items() for f in fs]
        forms.sort(key=lambda fv: -len(fv[0]))
        remaining = body.lower()
        found = Counter()
        for form, value in forms:
            pat = re.compile(r"(?<![\w-])" + re.escape(form.lower()) + r"s?(?![\w-])")
            n = len(pat.findall(remaining))
            if n:
                found[value] += n
                remaining = pat.sub(" ", remaining)
        out[attr] = found
    return out
