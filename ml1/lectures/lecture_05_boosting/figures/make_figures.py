"""Figures for ML I Lecture 5. Run from the lecture directory: python figures/make_figures.py"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

OUT = Path(__file__).resolve().parent
BLUE, RED, GREEN, GREY, YELLOW = "#0072B2", "#D55E00", "#009E73", "#7f7f7f", "#E69F00"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
rng = np.random.default_rng(155)

# ---------- fig_stagewise: stumps fitted to residuals ----------
x = np.sort(rng.uniform(0, 1, 80)); f = lambda t: np.sin(2 * np.pi * t) + 0.5 * t; y = f(x) + rng.normal(0, 0.15, 80)
grid = np.linspace(0, 1, 400)
fig, axes = plt.subplots(1, 3, figsize=(10, 3), sharey=True)
F = np.full_like(y, y.mean()); Fg = np.full_like(grid, y.mean())
for m in range(1, 51):
    stump = DecisionTreeRegressor(max_depth=1).fit(x[:, None], y - F)
    F += stump.predict(x[:, None]); Fg += stump.predict(grid[:, None])
    if m in (1, 5, 50):
        ax = axes[[1, 5, 50].index(m)]
        ax.scatter(x, y, s=10, color="black"); ax.plot(grid, f(grid), color=GREY, lw=1, ls="--")
        ax.plot(grid, Fg, color=BLUE, lw=2); ax.set_title(f"$M={m}$ stumps", fontsize=10); ax.set_xlabel("$x$")
fig.tight_layout(); fig.savefig(OUT / "fig_stagewise.pdf"); plt.close(fig)

# ---------- fig_boosting_curves & fig_margins: AdaBoost with stumps on breast cancer ----------
X, yb = load_breast_cancer(return_X_y=True); yb = 2 * yb - 1
Xtr, Xte, ytr, yte = train_test_split(X, yb, test_size=0.4, random_state=155)
M = 1000
ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=M, learning_rate=1.0, random_state=0).fit(Xtr, ytr)
tr_err = [1 - np.mean(p == ytr) for p in ada.staged_predict(Xtr)]
te_err = [1 - np.mean(p == yte) for p in ada.staged_predict(Xte)]
fig, ax = plt.subplots(figsize=(6, 3.4))
ax.plot(np.arange(1, M + 1), tr_err, color=RED, lw=1.6, label="training error")
ax.plot(np.arange(1, M + 1), te_err, color=BLUE, lw=1.6, label="test error")
ax.set_xscale("log"); ax.set_xlabel("rounds $M$"); ax.set_ylabel("error"); ax.legend(frameon=False)
first_zero = next((i + 1 for i, e in enumerate(tr_err) if e == 0), None)
if first_zero:
    ax.axvline(first_zero, color=GREY, lw=0.8, ls=":"); ax.text(first_zero * 1.1, max(te_err) * 0.8, f"train error = 0\nat $M={first_zero}$", fontsize=8, color=GREY)
fig.tight_layout(); fig.savefig(OUT / "fig_boosting_curves.pdf"); plt.close(fig)

# margins: y * sum(alpha_m h_m) / sum(alpha_m), using staged decision_function
alphas = ada.estimator_weights_
fig, ax = plt.subplots(figsize=(5.6, 3.4))
for m, c in zip([5, 100, 1000], [RED, YELLOW, BLUE]):
    m = min(m, len(alphas))
    votes = sum(a * est.predict(Xtr) for a, est in zip(alphas[:m], ada.estimators_[:m]))
    margins = np.sort(ytr * votes / alphas[:m].sum())
    ax.plot(margins, np.arange(1, len(margins) + 1) / len(margins), color=c, lw=2, label=f"$M={m}$")
ax.axvline(0, color=GREY, lw=0.8); ax.set_xlabel("margin  $\\mu_i$"); ax.set_ylabel("cumulative fraction of training points")
ax.set_xlim(-0.6, 1); ax.legend(frameon=False, loc="upper left"); fig.tight_layout(); fig.savefig(OUT / "fig_margins.pdf"); plt.close(fig)
print("figures written:", sorted(p.name for p in OUT.glob("*.pdf")), "| train err hits 0 at M =", first_zero)
