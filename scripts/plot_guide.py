import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
ARMS = {  # 6-cell mean per seed
    "selection": (0.587, 0.557),
    "selection\n+ guidance": (0.430, 0.440),
    "null token\n+ selection": (0.330, 0.320),
    "null token\n+ guidance": (0.323, 0.347),
}
MEAN = [sum(v) / 2 for v in ARMS.values()]

fig, ax = plt.subplots(figsize=(3.3, 2.4))
colors = ["#4477aa"] + ["#bbbbbb"] * (len(ARMS) - 1)
ax.bar(list(ARMS), MEAN, width=0.6, color=colors)
for i, m in enumerate(MEAN):
    ax.text(i, m + 0.01, f"{round(m, 3):.2f}", ha="center", va="bottom")
ax.tick_params(axis="x", labelsize=8)
ax.set_ylabel("success rate")
ax.set_ylim(0, 0.7)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("outputs/guide/bars.pdf")
fig.savefig("outputs/guide/bars.png", dpi=200)
