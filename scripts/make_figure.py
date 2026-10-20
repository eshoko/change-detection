"""Write results/figure_dprime_rt.png (d' and RT vs difficulty level)."""

from pathlib import Path

from changedet.figure import compute_curves, plot
from changedet.generator import generate_all_levels

ITEM_SEED, CURVE_SEED, N_REPS = 2026, 31, 30

out = Path(__file__).resolve().parents[1] / "results" / "figure_dprime_rt.png"
out.parent.mkdir(exist_ok=True)
items = generate_all_levels(400, ITEM_SEED)
plot(compute_curves(items, N_REPS, CURVE_SEED), str(out), N_REPS)
print(f"wrote {out}")
