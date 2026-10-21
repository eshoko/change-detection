"""Run the full validation and write results/validation.json and results/validation.md.

Usage:  python scripts/run_validation.py [--reps 50]

Every random step is seeded; the output is identical on every run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from changedet.fit import fit_threshold
from changedet.generator import generate_all_levels
from changedet.responders import ThresholdResponder, simulate
from changedet.scoring import build_trials, summarize, summarize_by_level
from changedet.validation import (DISSOCIATED, FAST, N_RATE_MIN, parametric_bootstrap,
                                  recover, recovery_study, responder_grid)

ITEM_SEED = 2026
N_PER_LEVEL = 400
OUT = Path(__file__).resolve().parents[1] / "results"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=50, help="replicate datasets per grid cell")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    items = generate_all_levels(N_PER_LEVEL, ITEM_SEED)
    res: dict = {"item_seed": ITEM_SEED, "n_items": len(items), "reps": args.reps}

    # 1. Grid: replicate recovery study (N x g x speed).
    grid = [recovery_study(r, items, args.reps, seed=100 + i)
            for i, r in enumerate(responder_grid())]
    res["grid"] = grid

    # 2. Dissociated responders: one fixed-seed fit + parametric bootstrap CIs.
    diss = {}
    for i, (name, r) in enumerate(DISSOCIATED.items()):
        rec = recover(r, items, seed=500 + i)
        boot = parametric_bootstrap(items, rec, n_boot=200, seed=600 + i)
        diss[name] = {**rec.as_row(), **boot}
    res["dissociated"] = diss

    # 3. Model caveat: d' moves with g alone (N fixed at 2).
    cav = {}
    for g in (0.05, 0.2, 0.4):
        r = ThresholdResponder(f"g{g}", N=2, g=g, s=FAST)
        dps, cs, Ns = [], [], []
        for k in range(args.reps):
            t = build_trials(items, simulate(r, items, 700 + k))
            s = summarize(t)
            dps.append(s["dprime"]); cs.append(s["c"]); Ns.append(fit_threshold(t).N)
        cav[str(g)] = dict(dprime_mean=float(np.mean(dps)), dprime_sd=float(np.std(dps, ddof=1)),
                           c_mean=float(np.mean(cs)), N_hat_rate=float(np.mean(np.array(Ns) == 2)))
    res["dprime_caveat_N2"] = cav

    # 4. Sample-size sweep: how many items are needed for N recovery (hardest cell).
    hard = ThresholdResponder("N4-g0.4", N=4, g=0.4, s=FAST)
    sweep = []
    for n in (3, 5, 10, 25, 100, 400):
        its = generate_all_levels(n, ITEM_SEED + n)
        st = recovery_study(hard, its, args.reps, seed=800 + n)
        sweep.append(dict(n_per_level=n, **{k: st[k] for k in ("N_rate", "g_rate", "s_rate")}))
    res["sample_size_sweep_N4_g0.4"] = sweep

    # 5. Lapse sensitivity: the fit has no lapse parameter.
    lapse_res = {}
    for lapse in (0.0, 0.001, 0.01):
        Ns = {}
        for N in (1, 2, 3, 4):
            for g in (0.05, 0.2, 0.4):
                r = ThresholdResponder("x", N=N, g=g, s=FAST, lapse=lapse)
                Ns[f"{N}"] = Ns.get(f"{N}", []) + [
                    fit_threshold(build_trials(items, simulate(r, items, 1000 + k))).N
                    for k in range(20)]
        allN = np.concatenate([np.array(v) for v in Ns.values()])
        lapse_res[str(lapse)] = dict(
            N_hat_share={str(n): float(np.mean(allN == n)) for n in range(6)},
            correct_by_true_N={n: float(np.mean(np.array(v) == int(n))) for n, v in Ns.items()},
            share_0_or_1=float(np.mean(allN <= 1)), n_fits=int(len(allN)))
    res["lapse_sensitivity"] = lapse_res

    # 6. By-level summaries for the dissociated responders (one dataset each).
    res["by_level"] = {
        name: {str(k): v for k, v in summarize_by_level(
            build_trials(items, simulate(r, items, 900 + i))).items()}
        for i, (name, r) in enumerate(DISSOCIATED.items())
    }

    (OUT / "validation.json").write_text(json.dumps(res, indent=1) + "\n")
    md = render_markdown(res)
    (OUT / "validation.md").write_text(md)
    print(md)
    failed = [g["name"] for g in grid if g["N_rate"] < N_RATE_MIN]
    failed += [n for n, d in diss.items() if not d["ok"]]
    if failed:
        raise SystemExit(f"FAILED: {failed}")
    print("ALL VALIDATION CHECKS PASSED")


def render_markdown(res: dict) -> str:
    L = [f"# Validation results\n",
         f"Items: {res['n_items']} (400 per level, seed {res['item_seed']}); "
         f"{res['reps']} replicate datasets per grid cell.\n",
         "## Recovery over replicates (share of datasets within tolerance)\n",
         "| responder | N̂ = N | ĝ within 3 SE | ŝ within 3 SE | mean ĝ − g | mean ŝ − s |",
         "|---|---|---|---|---|---|"]
    for g in res["grid"]:
        L.append(f"| {g['name']} | {g['N_rate']:.2f} | {g['g_rate']:.2f} | {g['s_rate']:.2f} "
                 f"| {g['g_bias']:+.4f} | {g['s_bias']:+.5f} |")
    L += ["\n## Dissociated responders (one dataset; 95% parametric-bootstrap CIs)\n",
          "| responder | N | N̂ | g | ĝ [95% CI] | s (s/comparison) | ŝ [95% CI] | pass |",
          "|---|---|---|---|---|---|---|---|"]
    for name, d in res["dissociated"].items():
        L.append(f"| {name} | {d['N']} | {d['N_hat']} | {d['g']} | {d['g_hat']:.3f} "
                 f"[{d['g_ci'][0]:.3f}, {d['g_ci'][1]:.3f}] | {d['s']} | {d['s_hat']:.4f} "
                 f"[{d['s_ci'][0]:.4f}, {d['s_ci'][1]:.4f}] | {'yes' if d['ok'] else 'NO'} |")
    L += ["\n## d′ is descriptive: it shifts with g alone (N = 2 fixed)\n",
          "| g | mean d′ (SD) | mean c | N̂ = 2 rate |", "|---|---|---|---|"]
    for g, d in res["dprime_caveat_N2"].items():
        L.append(f"| {g} | {d['dprime_mean']:.3f} ({d['dprime_sd']:.3f}) | {d['c_mean']:+.3f} "
                 f"| {d['N_hat_rate']:.2f} |")
    L += ["\n## Sample size needed (hardest cell: N = 4, g = 0.4)\n",
          "| items per level | N̂ = N | ĝ ok | ŝ ok |", "|---|---|---|---|"]
    for s in res["sample_size_sweep_N4_g0.4"]:
        L.append(f"| {s['n_per_level']} | {s['N_rate']:.2f} | {s['g_rate']:.2f} | {s['s_rate']:.2f} |")
    L += ["\n## Lapse sensitivity (fit has no lapse parameter; 12 grid cells x 20 datasets)\n",
          "| lapse | N̂ = 0 or 1 | N̂ = N (true N = 1 / 2 / 3 / 4) |", "|---|---|---|"]
    for lapse, d in res["lapse_sensitivity"].items():
        cb = d["correct_by_true_N"]
        L.append(f"| {lapse} | {d['share_0_or_1']:.2f} | "
                 + " / ".join(f"{cb[n]:.2f}" for n in ("1", "2", "3", "4")) + " |")
    L += ["\n## By level (one dataset per dissociated responder)\n",
          "| responder | level | d′ | c | accuracy | exact | mean RT (s) |",
          "|---|---|---|---|---|---|---|"]
    for name, by in res["by_level"].items():
        for lvl, s in by.items():
            L.append(f"| {name} | L{lvl} | {s['dprime']:.2f} | {s['c']:+.2f} | {s['accuracy']:.3f} "
                     f"| {s['exact_accuracy']:.3f} | {s['mean_rt']:.2f} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
