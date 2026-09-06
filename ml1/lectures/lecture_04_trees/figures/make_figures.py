"""Figures for ML I Lecture 4. Run from the lecture directory: python figures/make_figures.py"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.datasets import make_classification, load_breast_cancer
from sklearn.model_selection import train_test_split

OUT = Path(__file__).resolve().parent
BLUE, RED, GREEN, GREY, YELLOW = "#0072B2", "#D55E00", "#009E73", "#7f7f7f", "#E69F00"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
rng = np.random.default_rng(155)


def regions(ax, clf, xlim, ylim):
    xx, yy = np.meshgrid(np.linspace(*xlim, 300), np.linspace(*ylim, 300))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5], colors=[RED, BLUE], alpha=0.15)


# ---------- fig_tree_vs_linear: XOR-like data ----------
n = 160
X = rng.uniform(-1, 1, (n, 2)); y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)
flip = rng.random(n) < 0.08; y[flip] = 1 - y[flip]
fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
for ax, clf, title in zip(axes, [LogisticRegression().fit(X, y), DecisionTreeClassifier(max_depth=3, random_state=0).fit(X, y)],
                          ["linear model", "tree, depth 3"]):
    regions(ax, clf, (-1, 1), (-1, 1))
    ax.scatter(*X[y == 1].T, color=BLUE, s=16); ax.scatter(*X[y == 0].T, color=RED, s=16, marker="s")
    ax.set_title(f"{title}   train acc = {clf.score(X, y):.2f}", fontsize=10); ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
fig.tight_layout(); fig.savefig(OUT / "fig_tree_vs_linear.pdf"); plt.close(fig)

# ---------- fig_impurity ----------
p = np.linspace(1e-4, 1 - 1e-4, 400)
fig, ax = plt.subplots(figsize=(5.4, 3.3))
ax.plot(p, np.minimum(p, 1 - p), color=GREY, lw=2, label="misclassification")
ax.plot(p, 2 * p * (1 - p), color=BLUE, lw=2, label="Gini")
ax.plot(p, -p * np.log2(p) - (1 - p) * np.log2(1 - p), color=RED, lw=2, label="entropy (bits)")
ax.set_xlabel("$p$ = proportion of class 1"); ax.set_ylabel("$I(p)$"); ax.legend(frameon=False, fontsize=9)
fig.tight_layout(); fig.savefig(OUT / "fig_impurity.pdf"); plt.close(fig)

# ---------- fig_tree_instability: two bootstraps, two trees ----------
Xs, ys = make_classification(n_samples=120, n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=1,
                             class_sep=1.0, flip_y=0.1, random_state=7)
fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
lim = (Xs.min(0) - 0.4, Xs.max(0) + 0.4)
for ax, seed in zip(axes, [1, 2]):
    idx = rng.integers(0, len(ys), len(ys))
    t = DecisionTreeClassifier(max_depth=4, random_state=seed).fit(Xs[idx], ys[idx])
    regions(ax, t, (lim[0][0], lim[1][0]), (lim[0][1], lim[1][1]))
    ax.scatter(*Xs[ys == 1].T, color=BLUE, s=14); ax.scatter(*Xs[ys == 0].T, color=RED, s=14, marker="s")
    f, s = t.tree_.feature[0], t.tree_.threshold[0]
    ax.set_title(f"bootstrap {seed}: root split $x_{f+1} \\leq {s:.2f}$", fontsize=10); ax.set_xticks([]); ax.set_yticks([])
fig.tight_layout(); fig.savefig(OUT / "fig_tree_instability.pdf"); plt.close(fig)

# ---------- fig_bagging_variance ----------
B = np.arange(1, 201)
fig, ax = plt.subplots(figsize=(5.6, 3.3))
for rho, c in zip([1.0, 0.6, 0.3, 0.0], [GREY, RED, YELLOW, BLUE]):
    ax.plot(B, rho + (1 - rho) / B, color=c, lw=2, label=rf"$\rho={rho}$")
ax.set_xlabel("$B$  (number of predictors averaged)"); ax.set_ylabel(r"variance / $\sigma^2$"); ax.set_ylim(0, 1.05)
ax.legend(frameon=False); fig.tight_layout(); fig.savefig(OUT / "fig_bagging_variance.pdf"); plt.close(fig)

# ---------- fig_ensemble_error: breast cancer, test error vs B ----------
Xb, yb = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(Xb, yb, test_size=0.4, random_state=155)
Bs = [1, 2, 5, 10, 20, 50, 100, 200]
tree_err = 1 - DecisionTreeClassifier(random_state=0).fit(Xtr, ytr).score(Xte, yte)
bag_err = [1 - BaggingClassifier(DecisionTreeClassifier(), n_estimators=b, random_state=0).fit(Xtr, ytr).score(Xte, yte) for b in Bs]
rf_err = [1 - RandomForestClassifier(n_estimators=b, random_state=0).fit(Xtr, ytr).score(Xte, yte) for b in Bs]
fig, ax = plt.subplots(figsize=(5.8, 3.4))
ax.axhline(tree_err, color=GREY, ls="--", lw=1.4, label="single tree")
ax.plot(Bs, bag_err, "o-", color=RED, lw=2, label="bagging")
ax.plot(Bs, rf_err, "s-", color=BLUE, lw=2, label=r"random forest ($m=\sqrt{p}$)")
ax.set_xscale("log"); ax.set_xlabel("$B$"); ax.set_ylabel("test error"); ax.legend(frameon=False, fontsize=9)
fig.tight_layout(); fig.savefig(OUT / "fig_ensemble_error.pdf"); plt.close(fig)
print("figures written:", sorted(p.name for p in OUT.glob("*.pdf")))
