"""Figures for ML I Lecture 6. Run from the lecture directory: python figures/make_figures.py"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs, load_digits
from scipy.cluster.hierarchy import linkage, dendrogram

OUT = Path(__file__).resolve().parent
BLUE, RED, GREEN, GREY, YELLOW = "#0072B2", "#D55E00", "#009E73", "#7f7f7f", "#E69F00"
COLS = [BLUE, RED, GREEN, YELLOW]
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
rng = np.random.default_rng(155)

# ---------- fig_kmeans_init: same data, random init vs k-means++ ----------
Xb, _ = make_blobs(300, centers=[[0, 0], [4, 0], [8, 0], [4, 4]], cluster_std=0.7, random_state=3)
fig, axes = plt.subplots(1, 2, figsize=(8, 2.6))
for ax, init, seed, title in zip(axes, ["random", "k-means++"], [7, 0], ["random initialization", "k-means++"]):
    km = KMeans(4, init=init, n_init=1, random_state=seed).fit(Xb)
    for k in range(4):
        ax.scatter(*Xb[km.labels_ == k].T, s=8, color=COLS[k])
    ax.scatter(*km.cluster_centers_.T, marker="X", s=80, color="black")
    ax.set_title(f"{title}   $J={km.inertia_:.0f}$", fontsize=10); ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
fig.tight_layout(); fig.savefig(OUT / "fig_kmeans_init.pdf"); plt.close(fig)

# ---------- fig_kmeans_fails: elongated clusters and unequal sizes ----------
fig, axes = plt.subplots(1, 2, figsize=(8, 2.8))
t = rng.uniform(0, 1, 150); Xe = np.vstack([np.c_[6 * t, 0.3 * rng.normal(size=150)], np.c_[6 * t, 2 + 0.3 * rng.normal(size=150)]])
Xu = np.vstack([rng.normal([0, 0], 1.6, (400, 2)), rng.normal([5, 0], 0.4, (40, 2))])
for ax, X_, title in zip(axes, [Xe, Xu], ["elongated clusters", "unequal sizes and spreads"]):
    km = KMeans(2, n_init=10, random_state=0).fit(X_)
    for k in range(2):
        ax.scatter(*X_[km.labels_ == k].T, s=8, color=COLS[k])
    ax.scatter(*km.cluster_centers_.T, marker="X", s=80, color="black"); ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
fig.tight_layout(); fig.savefig(OUT / "fig_kmeans_fails.pdf"); plt.close(fig)

# ---------- fig_dendrogram: Ward on 30 points from 3 groups ----------
Xd, yd = make_blobs(30, centers=3, cluster_std=0.9, random_state=155)
Z = linkage(Xd, method="ward")
fig, ax = plt.subplots(figsize=(7, 3.6))
dendrogram(Z, ax=ax, color_threshold=Z[-3, 2] + 1e-9, above_threshold_color=GREY, no_labels=True)
ax.axhline((Z[-3, 2] + Z[-2, 2]) / 2, color=RED, ls="--", lw=1.2)
ax.set_ylabel("merge cost (Ward)"); ax.set_xlabel("30 points"); fig.tight_layout(); fig.savefig(OUT / "fig_dendrogram.pdf"); plt.close(fig)

# ---------- fig_pca_2d ----------
Xp = rng.multivariate_normal([0, 0], [[3, 1.6], [1.6, 1.3]], 150); Xp -= Xp.mean(0)
S = Xp.T @ Xp / len(Xp); lam, Q = np.linalg.eigh(S); order = np.argsort(lam)[::-1]; lam, Q = lam[order], Q[:, order]
fig, ax = plt.subplots(figsize=(5.2, 4))
proj = (Xp @ Q[:, :1]) @ Q[:, :1].T
for x0, x1 in zip(Xp, proj):
    ax.plot([x0[0], x1[0]], [x0[1], x1[1]], color=GREY, lw=0.5, alpha=0.6)
ax.scatter(*Xp.T, s=12, color="black", zorder=3); ax.scatter(*proj.T, s=8, color=BLUE, zorder=4)
for j, c in enumerate([BLUE, RED]):
    v = Q[:, j] * np.sqrt(lam[j]) * 2
    ax.annotate("", xy=v, xytext=(0, 0), arrowprops=dict(arrowstyle="->", lw=2.4, color=c))
    ax.text(*(v * 1.1), rf"$\mathbf{{q}}_{j+1}$, $\lambda_{j+1}={lam[j]:.2f}$", color=c, fontsize=10)
ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); fig.tight_layout(); fig.savefig(OUT / "fig_pca_2d.pdf"); plt.close(fig)

# ---------- fig_scree & fig_eigendigits: sklearn digits ----------
D = load_digits(); Xg = D.data.astype(float); Xg -= Xg.mean(0)
U, d, Vt = np.linalg.svd(Xg, full_matrices=False); lam_g = d**2 / len(Xg)
fig, ax = plt.subplots(figsize=(4.6, 3.4))
ax.plot(np.arange(1, 65), lam_g, "o-", ms=3, color=BLUE, label=r"$\lambda_j$")
ax2 = ax.twinx(); ax2.plot(np.arange(1, 65), np.cumsum(lam_g) / lam_g.sum(), color=RED, lw=1.6, label="cumulative")
ax2.axhline(0.9, color=GREY, ls=":", lw=0.8); ax2.set_ylim(0, 1.02); ax2.set_ylabel("proportion explained", color=RED)
ax.set_xlabel("component $j$"); ax.set_ylabel(r"$\lambda_j$", color=BLUE); ax.set_title("digits: scree plot", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "fig_scree.pdf"); plt.close(fig)

fig, axes = plt.subplots(2, 8, figsize=(9, 2.6))
axes[0, 0].imshow(D.data.mean(0).reshape(8, 8), cmap="gray"); axes[0, 0].set_title("mean", fontsize=8)
for j in range(7):
    axes[0, j + 1].imshow(Vt[j].reshape(8, 8), cmap="RdBu"); axes[0, j + 1].set_title(f"PC {j+1}", fontsize=8)
x0 = Xg[7]
axes[1, 0].imshow(D.data[7].reshape(8, 8), cmap="gray"); axes[1, 0].set_title("original", fontsize=8)
for a, k in zip(axes[1, 1:], [1, 2, 4, 8, 16, 32, 64]):
    rec = D.data.mean(0) + Vt[:k].T @ (Vt[:k] @ x0)
    a.imshow(rec.reshape(8, 8), cmap="gray"); a.set_title(f"$k={k}$", fontsize=8)
for a in axes.ravel():
    a.set_xticks([]); a.set_yticks([])
fig.tight_layout(); fig.savefig(OUT / "fig_eigendigits.pdf"); plt.close(fig)
print("figures written:", sorted(p.name for p in OUT.glob("*.pdf")))
