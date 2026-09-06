"""Figures for ML I Lecture 3. Run from the lecture directory: python figures/make_figures.py"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path(__file__).resolve().parent
BLUE, RED, GREEN, GREY, YELLOW = "#0072B2", "#D55E00", "#009E73", "#7f7f7f", "#E69F00"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

# ---------- fig_ridge_shrinkage: d^2/(d^2+lambda) for several d ----------
lam = np.logspace(-2, 3, 300)
fig, ax = plt.subplots(figsize=(6, 3.2))
for d, c in zip([10, 3, 1, 0.3], [BLUE, GREEN, YELLOW, RED]):
    ax.plot(lam, d**2 / (d**2 + lam), color=c, lw=2, label=f"$d_j={d}$")
ax.set_xscale("log"); ax.set_xlabel(r"$\lambda$"); ax.set_ylabel(r"shrinkage factor  $d_j^2/(d_j^2+\lambda)$")
ax.legend(frameon=False); fig.tight_layout(); fig.savefig(OUT / "fig_ridge_shrinkage.pdf"); plt.close(fig)

# ---------- fig_l1_l2_geometry: RSS contours meeting the l2 ball and the l1 ball ----------
b_ols = np.array([1.6, 1.0]); A = np.array([[1.0, 0.55], [0.55, 0.5]])   # elongated ellipses
g1, g2 = np.meshgrid(np.linspace(-1.6, 2.6, 300), np.linspace(-1.6, 2.4, 300))
D = np.stack([g1 - b_ols[0], g2 - b_ols[1]], -1)
rss = np.einsum("...i,ij,...j->...", D, A, D)
fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.9))
t = 1.0
for ax, kind in zip(axes, ["l2", "l1"]):
    ax.contour(g1, g2, rss, levels=[0.08, 0.3, 0.7, 1.3, 2.1], colors=GREY, linewidths=0.9)
    if kind == "l2":
        th = np.linspace(0, 2 * np.pi, 200); ax.fill(t * np.cos(th), t * np.sin(th), color=BLUE, alpha=0.15); ax.plot(t * np.cos(th), t * np.sin(th), color=BLUE, lw=1.8)
        title = r"$\|\beta\|_2 \leq t$  (ridge)"
    else:
        dm = np.array([[t, 0], [0, t], [-t, 0], [0, -t], [t, 0]]); ax.fill(dm[:, 0], dm[:, 1], color=RED, alpha=0.15); ax.plot(dm[:, 0], dm[:, 1], color=RED, lw=1.8)
        title = r"$\|\beta\|_1 \leq t$  (lasso)"
    # constrained minimizer: minimize rss over the set numerically
    mask = (np.hypot(g1, g2) <= t) if kind == "l2" else (np.abs(g1) + np.abs(g2) <= t)
    k = np.argmin(np.where(mask, rss, np.inf)); ax.plot(g1.flat[k], g2.flat[k], "o", color="black", ms=7)
    ax.plot(*b_ols, "x", color="black", ms=8, mew=2); ax.text(b_ols[0] + 0.08, b_ols[1] + 0.08, r"$\hat\beta_{\rm OLS}$")
    ax.axhline(0, color="black", lw=0.5); ax.axvline(0, color="black", lw=0.5)
    ax.set_aspect("equal"); ax.set_title(title, fontsize=11); ax.set_xlabel(r"$\beta_1$"); ax.set_ylabel(r"$\beta_2$")
    ax.set_xlim(-1.6, 2.6); ax.set_ylim(-1.6, 2.4)
fig.tight_layout(); fig.savefig(OUT / "fig_l1_l2_geometry.pdf"); plt.close(fig)

# ---------- fig_soft_threshold ----------
b = np.linspace(-4, 4, 400); lam1 = 1.0
fig, ax = plt.subplots(figsize=(5.2, 3.4))
ax.plot(b, b, color=GREY, lw=1, ls=":", label="OLS (identity)")
ax.plot(b, b / (1 + lam1), color=BLUE, lw=2, label=r"ridge  $b/(1+\lambda)$")
ax.plot(b, np.sign(b) * np.maximum(np.abs(b) - lam1, 0), color=RED, lw=2, label=r"lasso  $S_\lambda(b)$")
ax.axvspan(-lam1, lam1, color=RED, alpha=0.08); ax.text(0, 2.6, r"$|b|\leq\lambda$", ha="center", color=RED)
ax.set_xlabel(r"$b_j = \hat\beta_j^{\rm OLS}$"); ax.set_ylabel(r"$\hat\beta_j$"); ax.set_aspect("equal"); ax.legend(frameon=False, fontsize=9, loc="upper left")
fig.tight_layout(); fig.savefig(OUT / "fig_soft_threshold.pdf"); plt.close(fig)

# ---------- fig_lasso_path: diabetes data ----------
from sklearn.datasets import load_diabetes
from sklearn.linear_model import lasso_path
X, y = load_diabetes(return_X_y=True)
X = (X - X.mean(0)) / X.std(0); y = y - y.mean()
alphas, coefs, _ = lasso_path(X, y, n_alphas=100)
fig, ax = plt.subplots(figsize=(7, 3.6))
names = load_diabetes().feature_names
for j in range(coefs.shape[0]):
    ax.plot(alphas, coefs[j], lw=1.6, label=names[j])
ax.set_xscale("log"); ax.invert_xaxis(); ax.set_xlabel(r"$\lambda$   (decreasing $\rightarrow$)"); ax.set_ylabel(r"$\hat\beta_j(\lambda)$")
ax.axhline(0, color="black", lw=0.5); ax.legend(frameon=False, fontsize=7, ncol=5, loc="upper left")
fig.tight_layout(); fig.savefig(OUT / "fig_lasso_path.pdf"); plt.close(fig)

# ---------- fig_roc ----------
rng = np.random.default_rng(3)
s_pos = rng.normal(1.2, 1, 300); s_neg = rng.normal(0, 1, 700)
scores = np.concatenate([s_pos, s_neg]); labels = np.concatenate([np.ones(300), np.zeros(700)])
order = np.argsort(-scores); tp = np.cumsum(labels[order]); fp = np.cumsum(1 - labels[order])
tpr = tp / 300; fpr = fp / 700
auc = np.mean(s_pos[:, None] > s_neg[None, :])
fig, ax = plt.subplots(figsize=(4, 3.6))
ax.plot(fpr, tpr, color=BLUE, lw=2, label=f"AUC = {auc:.3f}"); ax.plot([0, 1], [0, 1], color=GREY, lw=1, ls="--", label="random")
ax.fill_between(fpr, tpr, alpha=0.08, color=BLUE)
ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate"); ax.set_aspect("equal"); ax.legend(frameon=False, loc="lower right")
fig.tight_layout(); fig.savefig(OUT / "fig_roc.pdf"); plt.close(fig)
print("figures written:", sorted(p.name for p in OUT.glob("*.pdf")))
