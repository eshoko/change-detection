"""The project figure: d' and mean RT vs difficulty level for the dissociated responders.

Two panels on separate axes (never a dual axis). Hue encodes capacity
(N = 4 blue, N = 1 orange); line style encodes speed (solid fast, dashed slow).
If capacity and speed are separable, the d' curves group by hue and the RT
curves group by line style.
"""

from __future__ import annotations

import numpy as np

from .generator import Item
from .responders import simulate
from .scoring import build_trials, summarize_by_level
from .validation import DISSOCIATED

# Validated palette (dataviz reference slots 1-2, light surface).
SURFACE = "#fcfcfb"
TEXT, TEXT_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
HUE = {4: "#2a78d6", 1: "#eb6834"}
STYLE = {"fast": "-", "slow": (0, (4, 2.5))}
# Small horizontal offsets so series that coincide by construction stay visible.
DODGE = (-0.09, -0.03, 0.03, 0.09)


def compute_curves(items: list[Item], n_reps: int, seed: int) -> dict:
    """Mean and SD (over replicate datasets) of d' and RT per level per responder."""
    seeds = np.random.SeedSequence(seed).generate_state(n_reps)
    out = {}
    for name, r in DISSOCIATED.items():
        dp, rt = [], []
        for s in seeds:
            by = summarize_by_level(build_trials(items, simulate(r, items, int(s))))
            levels = sorted(by)
            dp.append([by[l]["dprime"] for l in levels])
            rt.append([by[l]["mean_rt"] for l in levels])
        dp, rt = np.array(dp), np.array(rt)
        out[name] = dict(levels=levels, N=r.N, speed="slow" if r.s > 0.1 else "fast",
                         dprime=dp.mean(0), dprime_sd=dp.std(0, ddof=1),
                         rt=rt.mean(0), rt_sd=rt.std(0, ddof=1))
    return out


def plot(curves: dict, path: str, n_reps: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2,
                         "xtick.color": TEXT_2, "ytick.color": TEXT_2, "text.color": TEXT})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), facecolor=SURFACE)
    panels = [("dprime", "d′ (descriptive)", "Sensitivity d′ by level"),
              ("rt", "Mean response time (s)", "Response time by level")]
    for ax, (key, ylabel, title) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        for i, (name, c) in enumerate(curves.items()):
            x = np.array(c["levels"])
            ax.errorbar(x + DODGE[i], c[key], yerr=c[key + "_sd"], color=HUE[c["N"]],
                        linestyle=STYLE[c["speed"]], linewidth=2, marker="o", markersize=6,
                        markeredgecolor=SURFACE, markeredgewidth=1.5, capsize=0,
                        elinewidth=1, label=f"{name}  (N = {c['N']}, {c['speed']})")
        ax.set_xticks(x, [f"L{l}" for l in x])
        ax.set_xlabel("Difficulty level")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontsize=11, color=TEXT)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_ylim(bottom=0)

    # Direct labels: capacity groups on the d' panel, speed groups on the RT panel.
    first = next(iter(curves.values()))
    xe = first["levels"][-1]
    for N in (4, 1):
        y = np.mean([c["dprime"][-1] for c in curves.values() if c["N"] == N])
        axes[0].annotate(f"N = {N}", (xe, y), xytext=(16, 0), textcoords="offset points",
                         va="center", color=TEXT, fontsize=9)
    for sp in ("slow", "fast"):
        y = np.mean([c["rt"][-1] for c in curves.values() if c["speed"] == sp])
        axes[1].annotate(sp, (xe, y), xytext=(16, 0), textcoords="offset points",
                         va="center", color=TEXT, fontsize=9)
    for ax in axes:
        ax.set_xlim(first["levels"][0] - 0.3, xe + 0.7)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Capacity drives d′, speed drives RT (simulated responders at fixed bias, g = 0.2)",
                 x=0.01, ha="left", fontsize=12, color=TEXT)
    fig.text(0.01, 0.875, f"Points: mean over {n_reps} replicate datasets; bars: ±1 SD. "
             "Equal-capacity d′ curves and equal-speed RT curves coincide by construction;\n"
             "points are offset slightly so all four show. d′ uses the log-linear correction "
             "and is descriptive under the threshold model.",
             fontsize=8.5, color=TEXT_2)
    fig.tight_layout(rect=(0, 0.08, 1, 0.86))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
