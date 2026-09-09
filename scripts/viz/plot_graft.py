import re
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SRC = sys.argv[1] if len(sys.argv) > 1 else "outputs/graft_post_evals.txt"
OUT = sys.argv[2] if len(sys.argv) > 2 else "outputs/graft_curves_post.pdf"
TAG = sys.argv[3] if len(sys.argv) > 3 else "post0.40"
BASE = [float(v) for v in sys.argv[4:6]] if len(sys.argv) > 5 else [0.062, 0.449]
STEPS = [1000, 2000, 5000, 10000, 20000]
COLOR = {"detaug": "#4477aa", "demogen": "#bbbbbb"}
LABEL = {"detaug": "ours", "demogen": "DemoGen"}
REGIME = {"frozen": "adapter-only", "full": "full fine-tune"}
INK = "black"
plt.rcParams.update(
    {
        "font.size": 9,
        "pdf.fonttype": 42,
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
    }
)

run_of, cells = {}, {}
for line in open(SRC):
    job = line.split(":")[0].split()[1]
    if m := re.search(r"train outputs/franka/graft/(\S+):", line):
        run_of[job] = m.group(1)
    if m := re.search(rf"{TAG}_(\d+) overall: success ([\d.]+) failure ([\d.]+)", line):
        cells[run_of[job], int(m.group(1))] = (float(m.group(2)), float(m.group(3)))

series = defaultdict(lambda: defaultdict(dict))
for (run, step), v in cells.items():
    method, regime, seed = run.split("-")
    series[regime, method][seed][step] = v

fig, axes = plt.subplots(2, 2, figsize=(3.3, 3.0), sharex=True, sharey="row")
for col, regime in enumerate(["frozen", "full"]):
    for row, (ylabel, idx) in enumerate([("success rate", 0), ("collision rate", 1)]):
        ax = axes[row, col]
        ax.axhline(
            BASE[idx],
            color="#444444",
            ls=(0, (3, 2)),
            lw=0.9,
            zorder=3,
            label="base policy",
        )
        for method in ["detaug", "demogen"]:
            seeds = series[regime, method]
            if not seeds:
                continue
            nan = (np.nan, np.nan)
            per_seed = np.array(
                [[d.get(k, nan)[idx] for k in STEPS] for d in seeds.values()]
            )
            if len(seeds) > 1:
                ax.fill_between(
                    STEPS,
                    np.nanmin(per_seed, 0),
                    np.nanmax(per_seed, 0),
                    color=COLOR[method],
                    alpha=0.25,
                    lw=0,
                )
            ax.plot(
                STEPS,
                np.nanmean(per_seed, 0),
                color=COLOR[method],
                lw=1.5,
                marker="o",
                ms=3,
                label=LABEL[method],
            )
        ax.set_xscale("log")
        ax.set_xticks([1000, 5000, 20000])
        ax.set_xticklabels(["1k", "5k", "20k"])
        ax.minorticks_off()
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.5, 1])
        ax.tick_params(labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        if col == 0:
            ax.set_ylabel(ylabel)
        if row == 0:
            ax.set_title(REGIME[regime], fontsize=9, pad=4)
        if row == 1:
            ax.set_xlabel("fine-tuning steps")
fig.legend(
    *axes[0, 0].get_legend_handles_labels(),
    frameon=False,
    fontsize=8,
    ncol=3,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.02),
    columnspacing=1.2,
    handlelength=1.5,
)
fig.tight_layout(w_pad=0.8, h_pad=0.5, rect=(0, 0, 1, 0.93))
fig.savefig(OUT)
print(OUT, len(cells), "cells")
