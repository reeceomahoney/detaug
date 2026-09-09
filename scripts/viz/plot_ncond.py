import glob
import re
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SRC = sys.argv[1] if len(sys.argv) > 1 else "outputs/ncond"
OUT = sys.argv[2] if len(sys.argv) > 2 else "outputs/ncond/ncond.pdf"
KS = [1, 4, 8, 16, 32, 64]
COLOR = {"nc": "#4477aa", "nz": "#bbbbbb"}
LABEL = {"nc": "random labels", "nz": "zero labels"}
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

cells = defaultdict(dict)
for f in glob.glob(f"{SRC}/n[cz]*_s?.log"):
    name = re.match(r".*/(n[cz])(\d+)_(\d_I+)_s(\d)\.log", f)
    m = re.search(
        r"overall: success ([\d.]+) failure ([\d.]+)", open(f, errors="ignore").read()
    )
    assert name and m, f
    arm, k, cell, seed = name.groups()
    cells[arm, seed, int(k)][cell] = (float(m[1]), float(m[2]))

fig, axes = plt.subplots(1, 2, figsize=(3.3, 1.9))
for ax, (ylabel, idx) in zip(axes, [("success rate", 0), ("collision rate", 1)]):
    for arm in ["nc", "nz"]:
        seeds = sorted({s for a, s, _ in cells if a == arm})
        per_seed = np.array(
            [
                [np.mean([v[idx] for v in cells[arm, s, k].values()]) for k in KS]
                for s in seeds
            ]
        )
        ax.fill_between(
            KS, per_seed.min(0), per_seed.max(0), color=COLOR[arm], alpha=0.25, lw=0
        )
        ax.plot(
            KS,
            per_seed.mean(0),
            color=COLOR[arm],
            lw=1.5,
            marker="o",
            ms=3,
            label=LABEL[arm],
        )
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 4, 16, 64])
    ax.set_xticklabels(["1", "4", "16", "64"])
    ax.minorticks_off()
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.5, 1])
    ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("$K$")
fig.legend(
    *axes[0].get_legend_handles_labels(),
    frameon=False,
    fontsize=8,
    ncol=2,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.02),
    columnspacing=1.2,
    handlelength=1.5,
)
fig.tight_layout(w_pad=0.5, rect=(0, 0, 1, 0.9))
fig.savefig(OUT)
print(OUT)
