"""Figures for ML I Lecture 2. Run from the lecture directory: python figures/make_figures.py"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path(__file__).resolve().parent
BLUE, RED, GREEN, GREY, YELLOW = "#0072B2", "#D55E00", "#009E73", "#7f7f7f", "#E69F00"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

# ---------- fig_margin: hyperplane, margin band, support vectors ----------
rng = np.random.default_rng(2)
pos = rng.normal([2.4, 2.4], 0.55, (12, 2)); neg = rng.normal([0.4, 0.4], 0.55, (12, 2))
w = np.array([1.0, 1.0]); b = -2.8
fig, ax = plt.subplots(figsize=(5.2, 3.6))
ax.scatter(*pos.T, color=BLUE, s=32, label="$y=+1$"); ax.scatter(*neg.T, color=RED, s=32, marker="s", label="$y=-1$")
xx = np.linspace(-0.8, 3.8, 10)
# functional margin 1 lines: w.x + b = 0, +1, -1
for c, ls, lw in [(0, "-", 2), (1, "--", 1), (-1, "--", 1)]:
    ax.plot(xx, (c - b - w[0] * xx) / w[1], color="black", ls=ls, lw=lw)
# support vectors: points closest to the boundary on each side
d = (np.vstack([pos, neg]) @ w + b) / np.linalg.norm(w)
sv_pos = pos[np.argmin(pos @ w + b)]; sv_neg = neg[np.argmax(neg @ w + b)]
ax.scatter(*sv_pos, s=160, facecolors="none", edgecolors=GREEN, lw=2); ax.scatter(*sv_neg, s=160, facecolors="none", edgecolors=GREEN, lw=2)
ax.annotate("", xy=(1.9, 0.9), xytext=(1.4, 1.4), arrowprops=dict(arrowstyle="<->", color=GREEN, lw=1.6))
ax.text(1.85, 1.25, r"$\gamma$", color=GREEN, fontsize=13)
ax.annotate("", xy=(0.9 + 0.55, 3.0 + 0.55), xytext=(0.9, 3.0), arrowprops=dict(arrowstyle="->", color="black", lw=1.4))
ax.text(1.5, 3.35, r"$\mathbf{w}$", fontsize=12)
ax.set_xlim(-0.8, 3.8); ax.set_ylim(-0.8, 3.8); ax.set_aspect("equal"); ax.legend(frameon=False, loc="lower right", fontsize=9)
ax.set_xticks([]); ax.set_yticks([]); fig.tight_layout(); fig.savefig(OUT / "fig_margin.pdf"); plt.close(fig)

# ---------- fig_perceptron: the four-point trace ----------
pts = np.array([[1, 1], [3, 3], [0, 2], [4, 1]]); ys = np.array([-1, 1, -1, 1])
fig, ax = plt.subplots(figsize=(4.8, 3.8))
for p, y in zip(pts, ys):
    ax.scatter(*p, color=BLUE if y > 0 else RED, marker="o" if y > 0 else "s", s=70, zorder=5)
    ax.text(p[0] + 0.12, p[1] + 0.12, f"({p[0]},{p[1]})", fontsize=9)
xx = np.linspace(-0.5, 5, 10)
# intermediate separators (after mistakes 3, 6) and the final one: w = (w1, w2, b)
for wv, ls, lab in [((2, 0, -1), ":", "after 3 mistakes"), ((4, 0, -2), "-.", "after 6 mistakes"), ((3, -1, -3), "-", "final: $3x_1-x_2-3=0$")]:
    w1, w2, bb = wv
    if abs(w2) > 1e-9:
        ax.plot(xx, (-bb - w1 * xx) / w2, color=GREY if ls != "-" else "black", ls=ls, lw=1.2 if ls != "-" else 2, label=lab)
    else:
        ax.axvline(-bb / w1, color=GREY, ls=ls, lw=1.2, label=lab)
ax.set_xlim(-0.5, 5); ax.set_ylim(-0.5, 4.2); ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$"); ax.legend(frameon=False, fontsize=8, loc="upper left")
fig.tight_layout(); fig.savefig(OUT / "fig_perceptron.pdf"); plt.close(fig)

# ---------- fig_sigmoid ----------
z = np.linspace(-6, 6, 300); s = 1 / (1 + np.exp(-z))
fig, ax = plt.subplots(figsize=(4, 2.8))
ax.plot(z, s, color=BLUE, lw=2, label=r"$\sigma(z)$"); ax.plot(z, s * (1 - s), color=RED, lw=1.6, ls="--", label=r"$\sigma'(z)$")
ax.axhline(0.5, color=GREY, lw=0.6, ls=":"); ax.axhline(0.25, color=GREY, lw=0.6, ls=":")
ax.set_xlabel("$z$"); ax.legend(frameon=False); fig.tight_layout(); fig.savefig(OUT / "fig_sigmoid.pdf"); plt.close(fig)

# ---------- fig_losses: as functions of the margin ----------
m = np.linspace(-2.5, 3, 400)
fig, ax = plt.subplots(figsize=(7, 3.8))
ax.plot(m, (m <= 0).astype(float), color="black", lw=2.2, label="0/1")
ax.plot(m, np.maximum(0, -m), color=GREY, lw=1.6, ls=":", label="perceptron")
ax.plot(m, np.maximum(0, 1 - m), color=BLUE, lw=2, label="hinge")
ax.plot(m, np.log2(1 + np.exp(-m)), color=RED, lw=2, label=r"log  (scaled by $1/\ln 2$)")
ax.plot(m, (1 - m) ** 2, color=GREEN, lw=1.6, ls="--", label="squared")
ax.set_ylim(-0.1, 4); ax.set_xlabel(r"margin  $m = y\,f(\mathbf{x})$"); ax.set_ylabel(r"$\ell(m)$")
ax.axvline(0, color=GREY, lw=0.6); ax.legend(frameon=False, fontsize=9); fig.tight_layout(); fig.savefig(OUT / "fig_losses.pdf"); plt.close(fig)
print("figures written:", sorted(p.name for p in OUT.glob("*.pdf")))
