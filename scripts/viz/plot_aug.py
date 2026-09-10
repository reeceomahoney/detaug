import sys
from pathlib import Path

import matplotlib
import numpy as np
import pyarrow.parquet as pq

from detaug.bend import EE, grasp_segments

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d import proj3d  # noqa: E402

REPO = sys.argv[1] if len(sys.argv) > 1 else "reece-omahoney/spatial_viz"
OUT = sys.argv[2] if len(sys.argv) > 2 else f"outputs/{REPO.split('/')[-1]}.pdf"
DEMOS = [int(v) for v in sys.argv[3].split(",")] if len(sys.argv) > 3 else None
COLOR = {"orig": "#444444", "aug": "#4477aa"}
MARK = dict(ls="", mec=COLOR["orig"], mew=0.8, ms=4)
MARKERS = [
    dict(marker="o", mfc="white", label="start", **MARK),
    dict(marker="^", mfc="white", label="grasp", **{**MARK, "ms": 4.5}),
    dict(marker="s", mfc="white", label="end", **MARK),
]
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

t = pq.read_table(
    Path.home() / ".cache/huggingface/lerobot" / REPO / "data",
    columns=["episode_index", "task_index", "observation.state", "action"],
).to_pandas()
groups = t.groupby("episode_index")
first = groups.first()
s0 = np.stack(first["observation.state"].to_numpy())
key = np.concatenate([first.task_index.to_numpy()[:, None], s0[:, :7].round(4)], 1)
_, src = np.unique(key, axis=0, return_inverse=True)
orig = np.array([np.flatnonzero(src == g).min() for g in range(src.max() + 1)])
tasks = first.task_index.to_numpy()[orig]
if DEMOS is None:
    DEMOS = [int(np.flatnonzero(tasks == k)[0]) for k in (0, 2)]


def ee(e):
    return np.stack(groups.get_group(e)["observation.state"].to_numpy())[:, EE]


def grasp(e):
    return grasp_segments(np.stack(groups.get_group(e).action.to_numpy())[:, 7])[0][0]


rows = int(np.ceil(len(DEMOS) / 2))
fig, axes = plt.subplots(
    rows, 2, figsize=(3.3, 1.7 * rows + 0.4), subplot_kw={"projection": "3d"}
)
axes = np.atleast_1d(axes).ravel()
for ax in axes[len(DEMOS) :]:
    ax.set_axis_off()
trajs = {}
for d in DEMOS:
    e0 = orig[d]
    o = ee(e0)
    copies = [ee(e) for e in np.flatnonzero(src == src[e0]) if e != e0]
    trajs[d] = o, [c for c in copies if c.shape != o.shape or not np.allclose(c, o)]
span = np.exp(
    np.mean([np.log(np.ptp(np.concatenate([o, *c]), 0)) for o, c in trajs.values()], 0)
)
for ax, d in zip(axes, DEMOS):
    ax.view_init(elev=30, azim=-60)
    ax.set_proj_type("ortho")
    unit = np.array([[a, b, c] for a in (0, 1) for b in (0, 1) for c in (0, 1)], float)
    ux, uy, uz = unit.T
    depth = proj3d.proj_transform(ux, uy, uz, ax.get_proj())[2]
    near_hi = unit[depth.argmin()] == 1
    e0 = orig[d]
    o, copies = trajs[d]
    print(f"demo {d}: episode {e0}, {len(copies)} copies")
    for c in copies:
        ax.plot(*c.T, color=COLOR["aug"], lw=0.8, alpha=0.7, clip_on=False)
    ax.plot(*o.T, color=COLOR["orig"], lw=1.2, clip_on=False)
    for p, m in zip((o[0], o[grasp(e0)], o[-1]), MARKERS):
        ax.plot(*p, zorder=10, clip_on=False, **m)
    pts = np.concatenate([o, *copies])
    full = span * (np.ptp(pts, 0) / span).max()
    mid = (pts.min(0) + pts.max(0)) / 2
    lo, hi = mid - full / 2, mid + full / 2
    ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), zlim=(lo[2], hi[2]))
    ax.set_box_aspect(hi - lo, zoom=1.0)

    def screen(q):
        sx, sy, _ = proj3d.proj_transform(q[:, 0], q[:, 1], q[:, 2], ax.get_proj())
        return np.stack([sx, sy], 1)

    A = screen(mid + np.eye(3)) - screen(mid[None])
    sp = screen(pts)
    delta = (sp.min(0) + sp.max(0)) / 2 - screen(mid[None])[0]
    shift = np.clip(np.linalg.pinv(A.T) @ delta, pts.max(0) - hi, pts.min(0) - lo)
    lo, hi = lo + shift, hi + shift
    ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), zlim=(lo[2], hi[2]))
    ax.set(xticks=[], yticks=[], zticks=[])
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor("none")
        axis.line.set_color("none")
    near = np.where(near_hi, hi, lo)
    for i in range(3):
        for a, b in [(lo, lo), (lo, hi), (hi, lo), (hi, hi)]:
            e = np.array([lo, hi])
            e[:, (i + 1) % 3] = a[(i + 1) % 3]
            e[:, (i + 2) % 3] = b[(i + 2) % 3]
            if (e == near).all(1).any():
                continue
            ax.plot(*e.T, color="#dddddd", lw=0.5, zorder=0, clip_on=False)
fig.legend(
    handles=[
        plt.Line2D([], [], color=COLOR["orig"], lw=1.2, label="demo"),
        plt.Line2D([], [], color=COLOR["aug"], lw=0.8, label="augmented"),
        *axes[0].get_legend_handles_labels()[0][-3:],
    ],
    frameon=False,
    fontsize=8,
    ncol=5,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.0),
    columnspacing=0.8,
    handlelength=1.2,
    handletextpad=0.4,
)
fig.subplots_adjust(left=0, right=1, bottom=0, top=0.84, wspace=0)
fig.savefig(OUT)
fig.savefig(OUT.replace(".pdf", ".png"), dpi=200)
print(OUT)
