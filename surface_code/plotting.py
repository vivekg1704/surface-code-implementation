"""Figures for the threshold sweep and the decoding-time benchmark."""

import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import sinter

# Categorical slots in fixed order; every series also gets its own marker so
# identity never rests on colour alone.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
MARKERS = ["o", "s", "^"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb",
    "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "lines.linewidth": 2,
    "lines.markersize": 6,
    "legend.frameon": False,
    "font.size": 11,
})


def load_threshold(path):
    """Return {d: arrays of p, rate, low, high} for per-shot and per-round rates."""
    stats = sinter.read_stats_from_csv_files(path)
    by_d = defaultdict(list)
    for s in stats:
        by_d[s.json_metadata["d"]].append(s)

    out = {}
    for d, group in sorted(by_d.items()):
        group.sort(key=lambda s: s.json_metadata["p"])
        rows = []
        for s in group:
            fit = sinter.fit_binomial(num_shots=s.shots, num_hits=s.errors, max_likelihood_factor=1000)
            rounds = s.json_metadata["rounds"]
            rows.append((
                s.json_metadata["p"],
                fit.best, fit.low, fit.high,
                *(sinter.shot_error_rate_to_piece_error_rate(x, pieces=rounds) for x in (fit.best, fit.low, fit.high)),
            ))
        a = np.array(rows).T
        out[d] = {"p": a[0], "shot": a[1:4], "round": a[4:7]}
    return out


def crossing(p, y1, y2):
    """Physical error rate where two logical-error curves cross, interpolated in log-log space."""
    diff = np.log(y1) - np.log(y2)
    for i in range(len(p) - 1):
        if diff[i] > 0 >= diff[i + 1] or diff[i] < 0 <= diff[i + 1]:
            lp = np.log(p[i : i + 2])
            t = diff[i] / (diff[i] - diff[i + 1])
            return float(np.exp(lp[0] + t * (lp[1] - lp[0])))
    return None


def plot_threshold(csv_path, path):
    data = load_threshold(csv_path)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    distances = sorted(data)
    for ax, kind, ylabel in [
        (axes[0], "shot", "Logical error rate per shot (d rounds)"),
        (axes[1], "round", "Logical error rate per round"),
    ]:
        for i, d in enumerate(distances):
            p, (best, low, high) = data[d]["p"], data[d][kind]
            ax.errorbar(
                p * 100, best, yerr=[best - low, high - best],
                color=COLORS[i], marker=MARKERS[i], capsize=0, elinewidth=1.2, label=f"d = {d}",
            )
            ax.annotate(f"d={d}", (p[0] * 100, best[0]), xytext=(-6, 0), textcoords="offset points",
                        ha="right", va="center", color=INK, fontsize=10)

        crossings = []
        for d1, d2 in zip(distances, distances[1:]):
            pc = crossing(data[d1]["p"], data[d1][kind][0], data[d2][kind][0])
            if pc is not None:
                crossings.append((d1, d2, pc))
                ax.axvline(pc * 100, color=MUTED, linestyle=":", linewidth=1)
        text = "\n".join(f"d={d1}/{d2} cross at p ≈ {pc * 100:.2f}%" for d1, d2, pc in crossings)
        ax.text(0.97, 0.04, text, transform=ax.transAxes, ha="right", va="bottom", color=MUTED, fontsize=9,
                bbox={"facecolor": "#fcfcfb", "edgecolor": "none", "pad": 2})
        print(f"[{kind}] " + "; ".join(f"d={d1}/{d2}: {pc * 100:.3f}%" for d1, d2, pc in crossings))

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Physical error rate p (%)")
        ax.set_ylabel(ylabel)
        ax.set_xticks([0.2, 0.3, 0.5, 0.7, 1.0, 1.5], labels=["0.2", "0.3", "0.5", "0.7", "1.0", "1.5"])
        ax.minorticks_off()
        ax.set_xlim(0.15, 1.7)
        ax.legend(loc="upper left")

    fig.suptitle("Rotated surface code memory, circuit-level depolarising noise, MWPM (PyMatching)",
                 color=INK, fontsize=12)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=160)
    print(f"wrote {path}")


def load_timing(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return [{k: float(v) for k, v in r.items()} for r in rows]


def plot_timing(csv_path, path):
    rows = load_timing(csv_path)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    ps = sorted({r["p"] for r in rows})

    for i, p in enumerate(ps):
        sub = sorted((r for r in rows if r["p"] == p), key=lambda r: r["d"])
        d = np.array([r["d"] for r in sub])
        t_round = np.array([r["us_per_round"] for r in sub])
        t_shot = np.array([r["us_per_shot"] for r in sub])
        defects = np.array([r["mean_defects"] for r in sub])

        # Power-law fit on the larger distances, where fixed per-shot overheads no longer dominate.
        big = d >= 7
        slope = np.polyfit(np.log(d[big]), np.log(t_round[big]), 1)[0]
        print(f"p={p:.3f}: time per round ~ d^{slope:.2f} (d >= 7);  time per shot ~ d^{slope + 1:.2f}")

        axes[0].plot(d, t_round, color=COLORS[i], marker=MARKERS[i], label=f"p = {p * 100:.1f}%  (∝ $d^{{{slope:.1f}}}$)")
        axes[1].plot(defects, t_shot, color=COLORS[i], marker=MARKERS[i], label=f"p = {p * 100:.1f}%")

    ax = axes[0]
    ax.axhline(1.0, color=MUTED, linestyle="--", linewidth=1, label="~1 µs superconducting syndrome cycle")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ds = sorted({int(r["d"]) for r in rows})
    ax.set_xticks(ds, labels=[str(x) for x in ds])
    ax.minorticks_off()
    ax.set_xlabel("Code distance d (d rounds per shot)")
    ax.set_ylabel("Decoding time per round (µs)")
    ax.set_title("Single-threaded PyMatching, batch decoding", color=INK, fontsize=11)
    ax.set_ylim(top=ax.get_ylim()[1] * 30)  # headroom so the legend clears the data
    ax.legend(loc="upper left")

    ax = axes[1]
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Mean detection events per shot")
    ax.set_ylabel("Decoding time per shot (µs)")
    ax.set_title("Cost tracks the number of defects to match", color=INK, fontsize=11)
    ax.legend(loc="upper left")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=160)
    print(f"wrote {path}")

