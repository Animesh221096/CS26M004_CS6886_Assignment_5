"""
generate_time_plots.py -- creates plots/time_N<N>_d<d>.png for every input.

Reads results.csv and produces one grouped bar chart per (N, d) combination,
with the time in milliseconds for every variant. OMP thread-sweep rows are
excluded (only the default 8-thread run per input is plotted).
"""

import os
import csv
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

RESULTS_CSV = "results.csv"
OUT_DIR = "plots"

# Columns in results.csv
# variant,N,d,threads,time_ms,gflops,arithmetic_intensity_flop_per_byte,make_check

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Collect data: {(N, d): {variant: time_ms}}
    data = defaultdict(dict)
    with open(RESULTS_CSV, newline="") as f:
        for row in csv.DictReader(f):
            try:
                t = float(row["time_ms"])
            except (ValueError, TypeError):
                continue  # skip empty rows
            key = (int(row["N"]), int(row["d"]))
            variant = row["variant"]
            threads = int(row["threads"] or 1)
            # For omp, keep only one entry per input: the fastest of the runs,
            # or just the first encountered 8-thread run. Simplest: keep min.
            if variant in data[key] and t >= data[key][variant]:
                continue
            data[key][variant] = t

    # Consistent variant ordering and colors across plots
    ORDER = ["naive", "loops", "tiled", "fused", "best", "omp"]
    COLORS = {
        "naive": "#9e9e9e",
        "loops": "#5c9ce6",
        "tiled": "#f2a33c",
        "fused": "#c66fd1",
        "best":  "#3fbf7f",
        "omp":   "#e05656",
    }

    for (N, d), variants in sorted(data.items()):
        # Keep only known variants, in fixed order
        names = [v for v in ORDER if v in variants]
        times = [variants[v] for v in names]

        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(names))
        bars = ax.bar(x, times, color=[COLORS[v] for v in names],
                      edgecolor="black", linewidth=0.8)

        # Value labels on top of each bar
        for bar, t in zip(bars, times):
            ax.annotate(f"{t:.1f}",
                        (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        textcoords="offset points", xytext=(0, 4),
                        ha="center", fontsize=10, fontweight="bold")

        # Speedup annotation over naive
        if "naive" in variants and variants["naive"] > 0:
            for bar, t, v in zip(bars, times, names):
                sp = variants["naive"] / t
                ax.annotate(f"{sp:.2f}x",
                            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                            textcoords="offset points", xytext=(0, 18),
                            ha="center", fontsize=9, color="#333333")

        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=11)
        ax.set_ylabel("Time (ms)", fontsize=12)
        ax.set_title(f"Attention variant timing — N={N}, d={d}", fontsize=13)
        ax.grid(axis="y", alpha=0.4)
        ax.set_axisbelow(True)

        plt.tight_layout()
        out = os.path.join(OUT_DIR, f"time_N{N}_d{d}.png")
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out}")

if __name__ == "__main__":
    main()