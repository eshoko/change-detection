"""Rewrite tests/golden/snapshot.json from the current code (intended changes only)."""

import json
from pathlib import Path

from changedet.regression import snapshot

path = Path(__file__).resolve().parents[1] / "tests" / "golden" / "snapshot.json"
path.write_text(json.dumps(snapshot(), indent=1, sort_keys=True) + "\n")
print(f"wrote {path}")
