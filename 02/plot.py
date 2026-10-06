import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 300,
    "font.family": "serif", "font.size": 11,
    "axes.grid": True, "grid.alpha": 0.3,
    "axes.spines.top": False, "axes.spines.right": False,
})

def save(fig, name):
    fig.savefig(f"{name}.pdf", bbox_inches="tight")
    fig.savefig(f"{name}.png", bbox_inches="tight")

# ---- ejemplo: datos + ajuste ----
x = np.linspace(0, 10, 30)
y = 2.0 * x + 1.0 + np.random.normal(0, 1.5, x.size)
yerr = np.full_like(y, 1.5)

fig, ax = plt.subplots(figsize=(5.5, 4))
ax.errorbar(x, y, yerr=yerr, fmt="o", ms=4, capsize=2,
            color="k", ecolor="gray", label="datos")
ax.plot(x, 2.0 * x + 1.0, "r-", lw=1.5, label=r"$y = 2x + 1$")

ax.set(xlabel="x", ylabel="y", title="Título")
ax.legend(frameon=False)
fig.tight_layout()
save(fig, "fig")
plt.show()
