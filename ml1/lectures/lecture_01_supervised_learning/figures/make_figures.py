"""Figures for ML I Lecture 1. Run from the lecture directory: python figures/make_figures.py"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path(__file__).resolve().parent
BLUE, RED, GREEN, GREY = "#0072B2", "#D55E00", "#009E73", "#7f7f7f"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

# ---------- fig_ls_fit: the numerical example ----------
x = np.array([1, 2, 3, 4, 5.0]); y = np.array([2.2, 2.8, 4.5, 3.7, 5.5])
X = np.column_stack([np.ones(5), x])
beta = np.linalg.solve(X.T @ X, X.T @ y)
fig, ax = plt.subplots(figsize=(6, 3.6))
xx = np.linspace(0.5, 5.5, 50)
ax.plot(xx, beta[0] + beta[1] * xx, color=BLUE, lw=2, label=rf"$\hat y = {beta[0]:.2f} + {beta[1]:.2f}\,x$")
for xi, yi in zip(x, y):
    ax.plot([xi, xi], [yi, beta[0] + beta[1] * xi], color=RED, lw=1.2, ls="--")
ax.scatter(x, y, color="black", zorder=5, s=40)
ax.set_xlabel("$x$"); ax.set_ylabel("$y$"); ax.legend(frameon=False, loc="upper left")
ax.set_xlim(0.5, 5.5); fig.tight_layout(); fig.savefig(OUT / "fig_ls_fit.pdf"); plt.close(fig)

# ---------- fig_projection: y, its projection onto col(X), and the residual ----------
fig, ax = plt.subplots(figsize=(5, 3.6)); ax.set_axis_off()
# a parallelogram standing in for the plane col(X)
plane = np.array([[0, 0], [4.2, 0.9], [5.2, 2.6], [1.0, 1.7], [0, 0]])
ax.fill(plane[:, 0], plane[:, 1], color=BLUE, alpha=0.08); ax.plot(plane[:, 0], plane[:, 1], color=BLUE, lw=0.8)
o = np.array([0.4, 0.35]); yv = np.array([3.0, 3.4]); yh = np.array([3.0, 1.35])
ax.annotate("", xy=yv, xytext=o, arrowprops=dict(arrowstyle="->", lw=2, color="black"))
ax.annotate("", xy=yh, xytext=o, arrowprops=dict(arrowstyle="->", lw=2, color=BLUE))
ax.annotate("", xy=yv, xytext=yh, arrowprops=dict(arrowstyle="->", lw=1.6, color=RED, ls="--"))
ax.plot([yh[0] - 0.18, yh[0] - 0.18, yh[0]], [yh[1], yh[1] + 0.18, yh[1] + 0.18], color=GREY, lw=0.8)
ax.text(*(yv + [0.1, 0.05]), r"$\mathbf{y}$", fontsize=14)
ax.text(*(yh + [0.12, -0.32]), r"$\hat{\mathbf{y}} = \mathbf{H}\mathbf{y}$", fontsize=13, color=BLUE)
ax.text(*((yv + yh) / 2 + [0.15, 0]), r"$\mathbf{r} = \mathbf{y}-\hat{\mathbf{y}}$", fontsize=12, color=RED)
ax.text(3.6, 0.15, r"$\mathrm{col}(\mathbf{X})$", fontsize=12, color=BLUE)
ax.set_xlim(-0.2, 5.6); ax.set_ylim(-0.2, 3.9); fig.tight_layout(); fig.savefig(OUT / "fig_projection.pdf"); plt.close(fig)

# ---------- fig_gd_steps: three step sizes on L = 1/2 a theta^2 ----------
a = 1.0; th = np.linspace(-3.2, 3.2, 200)
fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), sharey=True)
for ax, eta, title in zip(axes, [0.3, 1.5, 2.2], [r"$\eta=0.3<1/a$", r"$1/a<\eta=1.5<2/a$", r"$\eta=2.2>2/a$"]):
    ax.plot(th, 0.5 * a * th**2, color=GREY, lw=1.5)
    t = 2.5; pts = [t]
    for _ in range(6):
        t = (1 - eta * a) * t; pts.append(t)
    pts = np.array(pts); pts = pts[np.abs(pts) < 3.2]
    ax.plot(pts, 0.5 * a * pts**2, "o-", color=RED if eta > 2 else BLUE, ms=5, lw=1.2)
    ax.set_title(title, fontsize=11); ax.set_xlabel(r"$\theta$")
axes[0].set_ylabel(r"$L(\theta)$"); fig.tight_layout(); fig.savefig(OUT / "fig_gd_steps.pdf"); plt.close(fig)

# ---------- fig_poly_fits: overfitting ----------
rng = np.random.default_rng(155)
xs = np.sort(rng.uniform(0, 1, 12)); f = lambda t: np.sin(2 * np.pi * t); ys = f(xs) + rng.normal(0, 0.25, 12)
grid = np.linspace(0, 1, 300)
fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), sharey=True)
for ax, deg in zip(axes, [1, 3, 9]):
    c = np.polyfit(xs, ys, deg); pred = np.polyval(c, grid)
    tr = np.mean((np.polyval(c, xs) - ys) ** 2)
    ax.plot(grid, f(grid), color=GREY, lw=1, ls="--", label="truth")
    ax.plot(grid, np.clip(pred, -2.2, 2.2), color=BLUE, lw=2, label=f"degree {deg}")
    ax.scatter(xs, ys, color="black", s=22, zorder=5)
    ax.set_title(f"degree {deg}   train MSE = {tr:.3f}", fontsize=10); ax.set_ylim(-2.2, 2.2); ax.set_xlabel("$x$")
axes[0].set_ylabel("$y$"); fig.tight_layout(); fig.savefig(OUT / "fig_poly_fits.pdf"); plt.close(fig)

# ---------- fig_bias_variance: schematic ----------
c = np.linspace(0.05, 1, 200); bias2 = 1.0 / (1 + 12 * c) ** 1.2; var = 0.9 * c**2; noise = 0.15
fig, ax = plt.subplots(figsize=(6, 3.3))
ax.plot(c, bias2, color=BLUE, lw=2, label=r"Bias$^2$"); ax.plot(c, var, color=RED, lw=2, label="Variance")
ax.plot(c, bias2 + var + noise, color="black", lw=2.4, label="Expected test error")
ax.axhline(noise, color=GREY, lw=1, ls=":"); ax.text(0.02, noise + 0.03, r"$\sigma^2$", color=GREY)
k = np.argmin(bias2 + var + noise); ax.axvline(c[k], color=GREY, lw=0.8, ls="--")
ax.set_xlabel(r"model complexity  (size of $\mathcal{H}$)  $\longrightarrow$"); ax.set_ylabel("error")
ax.set_xticks([]); ax.set_yticks([]); ax.legend(frameon=False); fig.tight_layout(); fig.savefig(OUT / "fig_bias_variance.pdf"); plt.close(fig)
print("figures written:", sorted(p.name for p in OUT.glob("*.pdf")))
