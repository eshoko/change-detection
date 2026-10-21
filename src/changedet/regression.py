"""Fixed-seed snapshot used by the regression tests (tests/golden/snapshot.json).

Regenerate with ``python scripts/update_golden.py`` only after an intended
change, and say why in the commit message.
"""

from __future__ import annotations

from .fit import fit_speed, fit_threshold
from .generator import LEVELS, generate_all_levels, generate_set, items_digest
from .responders import simulate
from .scoring import build_trials, summarize_by_level
from .validation import DISSOCIATED

ITEM_SEED = 12345
RESPONSE_SEED = 99


def snapshot() -> dict:
    out: dict = {"item_digests": {str(lvl): items_digest(generate_set(lvl, 25, ITEM_SEED))
                                  for lvl in LEVELS}}
    items = generate_all_levels(100, ITEM_SEED)
    out["first_prompt"] = items[0].prompt
    out["responders"] = {}
    for name in ("high-N/slow", "low-N/fast"):
        table = build_trials(items, simulate(DISSOCIATED[name], items, RESPONSE_SEED))
        tf, sf = fit_threshold(table), fit_speed(table)
        out["responders"][name] = {
            "by_level": {str(k): v for k, v in summarize_by_level(table).items()},
            "fit": {"N": tf.N, "g": tf.g, "se_g": tf.se_g, "loglik": tf.loglik,
                    "s": sf.s, "t0": sf.t0, "se_s": sf.se_s},
        }
    return out
