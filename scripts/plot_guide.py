import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARMS = {
    "selection\n(ours)": 0.572,
    "selection\n+ guidance": 0.435,
    "null token\n+ selection": 0.325,
    "null token\n+ guidance": 0.335,
}

fig, ax = plt.subplots(figsize=(4.2, 2.8))
colors = ["#2a78d6"] + ["#9a9a94"] * (len(ARMS) - 1)
ax.bar(list(ARMS), list(ARMS.values()), width=0.6, color=colors)
for i, m in enumerate(ARMS.values()):
    ax.text(i, m + 0.01, f"{m:.2f}", ha="center", va="bottom", fontsize=9)
ax.set_ylabel("success (6-cell mean)")
ax.set_ylim(0, 0.7)
ax.spines[["top", "right"]].set_visible(False)
ax.yaxis.grid(True, color="#e6e6e2", lw=0.8)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig("outputs/guide/bars.png", dpi=200)
