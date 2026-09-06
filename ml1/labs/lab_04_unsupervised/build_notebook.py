"""Build lab_04_student.ipynb and lab_04_solution.ipynb. Format as in lab_01_pipeline/build_notebook.py.

    python build_notebook.py [--check]
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
md = lambda s: ("markdown", s)
code = lambda sol, stu=None: ("code", sol, stu)

CELLS = [
md("""# Lab 4 — Clustering, PCA and the ML I Project
**Machine Learning I · PEU-CD 2026 · ENEI**

Companion to `tutorial.pdf`."""),
code("""import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, load_digits, load_diabetes
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
np.set_printoptions(precision=4, suppress=True)
rng = np.random.default_rng(155)"""),

md("""## 1. k-means
### Task 1 — Lloyd's algorithm, and the lecture example"""),
code("""def kmeans(X, mu0, iters=100):
    mu = np.array(mu0, float).copy(); Js = []
    for _ in range(iters):
        d2 = ((X[:, None, :] - mu[None, :, :]) ** 2).sum(-1)      # N x K squared distances
        c = d2.argmin(1)
        Js.append(d2[np.arange(len(X)), c].sum())                  # J after the assignment step
        new_mu = np.array([X[c == k].mean(0) if (c == k).any() else mu[k] for k in range(len(mu))])
        if np.allclose(new_mu, mu):
            break
        mu = new_mu
    return c, mu, np.array(Js)

P = np.array([[1, 1], [1.5, 2], [3, 3.5], [5, 7], [3.5, 5], [4.5, 5.0]])
c, mu, Js = kmeans(P, [P[0], P[3]])
print("assignments:", c, "\\ncentroids:\\n", mu, "\\nJ sequence:", Js)
assert np.isclose(Js[0], 22) and np.isclose(Js[-1], 9.1667, atol=1e-3)
assert np.allclose(mu, [[1.8333, 2.1667], [4.3333, 5.6667]], atol=1e-3)
assert np.all(np.diff(Js) <= 1e-12), "J must be non-increasing" """,
"""def kmeans(X, mu0, iters=100):
    mu = np.array(mu0, float).copy(); Js = []
    for _ in range(iters):
        # TODO: assignment step (nearest centroid), record J, update step (means), stop when centroids do not move
        ...
    return c, mu, np.array(Js)

P = np.array([[1, 1], [1.5, 2], [3, 3.5], [5, 7], [3.5, 5], [4.5, 5.0]])
c, mu, Js = kmeans(P, [P[0], P[3]])
print("assignments:", c, "\\ncentroids:\\n", mu, "\\nJ sequence:", Js)
assert np.isclose(Js[0], 22) and np.isclose(Js[-1], 9.1667, atol=1e-3)
assert np.all(np.diff(Js) <= 1e-12), "J must be non-increasing" """),

md("""### Task 2 — local minima, and k-means++"""),
code("""Xb, yb = make_blobs(300, centers=4, cluster_std=0.9, random_state=155)

def kmeanspp_seeds(X, K, r):
    seeds = [X[r.integers(len(X))]]
    for _ in range(K - 1):
        d2 = np.min(((X[:, None, :] - np.array(seeds)[None]) ** 2).sum(-1), axis=1)
        seeds.append(X[r.choice(len(X), p=d2 / d2.sum())])
    return np.array(seeds)

J_rand = [kmeans(Xb, Xb[rng.choice(300, 4, replace=False)])[2][-1] for _ in range(50)]
J_pp = [kmeans(Xb, kmeanspp_seeds(Xb, 4, rng))[2][-1] for _ in range(50)]
fig, ax = plt.subplots(1, 2, figsize=(9, 3), sharex=True)
ax[0].hist(J_rand, bins=20); ax[0].set_title(f"random init: best J={min(J_rand):.1f}, #distinct minima={len(np.unique(np.round(J_rand, 1)))}")
ax[1].hist(J_pp, bins=20); ax[1].set_title(f"k-means++: best J={min(J_pp):.1f}, #distinct={len(np.unique(np.round(J_pp, 1)))}")
plt.show()
sk = KMeans(4, n_init=10, random_state=0).fit(Xb)
print("sklearn J:", round(sk.inertia_, 1), " ours (best):", round(min(J_pp), 1))
assert np.isclose(sk.inertia_, min(J_pp), rtol=1e-3)""",
"""Xb, yb = make_blobs(300, centers=4, cluster_std=0.9, random_state=155)
# TODO: 50 random initializations -> histogram of final J; implement k-means++ seeding (D^2 sampling) and repeat
# TODO: compare best J with KMeans(4, n_init=10)
"""),

md("""### Task 3 — choosing K"""),
code("""def elbow_sil(X):
    Js, sil = [], []
    for K in range(1, 11):
        km = KMeans(K, n_init=5, random_state=0).fit(X); Js.append(km.inertia_)
        sil.append(silhouette_score(X, km.labels_) if K > 1 else np.nan)
    return Js, sil
Js, sil = elbow_sil(Xb)
Xo = np.vstack([Xb, [[40, 40]]]); Jo, silo = elbow_sil(Xo)
fig, ax = plt.subplots(1, 2, figsize=(9, 3))
ax[0].plot(range(1, 11), Js, "o-", label="clean"); ax[0].plot(range(1, 11), Jo, "s--", label="+1 outlier"); ax[0].set_xlabel("K"); ax[0].set_ylabel("J"); ax[0].legend()
ax[1].plot(range(1, 11), sil, "o-", label="clean"); ax[1].plot(range(1, 11), silo, "s--", label="+1 outlier"); ax[1].set_xlabel("K"); ax[1].set_ylabel("silhouette"); ax[1].legend()
plt.show()
print("best K by silhouette: clean", int(np.nanargmax(sil)) + 1, " with outlier", int(np.nanargmax(silo)) + 1)
# One far point costs a huge squared distance until it gets its own centroid, so the elbow moves to K=2:
# k-means has no notion of an outlier, only of squared distance.""",
"""# TODO: J and silhouette for K=1..10 on Xb; then add one far outlier (e.g. (40,40)) and repeat; explain
"""),

md("""## 2. Hierarchical clustering
### Task 4 — Ward's formula, verified"""),
code("""lab = KMeans(4, n_init=10, random_state=0).fit(Xb).labels_
A, B = Xb[lab == 0], Xb[lab == 1]
wss = lambda S: ((S - S.mean(0)) ** 2).sum()
dJ = wss(np.vstack([A, B])) - wss(A) - wss(B)
formula = len(A) * len(B) / (len(A) + len(B)) * np.sum((A.mean(0) - B.mean(0)) ** 2)
print(f"Delta J direct = {dJ:.6f}   formula = {formula:.6f}"); assert np.isclose(dJ, formula)

Z = linkage(Xb, "ward")
# scipy's Ward height is sqrt(2 * Delta J); check on the last merge
last = fcluster(Z, 2, "maxclust"); C1, C2 = Xb[last == 1], Xb[last == 2]
dJ_last = wss(Xb) - wss(C1) - wss(C2)
print(f"last merge: scipy height^2/2 = {Z[-1, 2]**2 / 2:.4f}   Delta J = {dJ_last:.4f}"); assert np.isclose(Z[-1, 2] ** 2 / 2, dJ_last)""",
"""lab = KMeans(4, n_init=10, random_state=0).fit(Xb).labels_
A, B = Xb[lab == 0], Xb[lab == 1]
# TODO: within-cluster SS separate vs merged; compare the increase with n_A n_B/(n_A+n_B) ||mu_A - mu_B||^2
# TODO: linkage(Xb, "ward"): reconcile the last merge height with Delta J (scipy reports sqrt(2*Delta J))
"""),

md("""### Task 5 — four linkages"""),
code("""Xh, yh = make_blobs(40, centers=3, cluster_std=1.0, random_state=155)
fig, axes = plt.subplots(1, 4, figsize=(13, 3))
for ax, m in zip(axes, ["single", "complete", "average", "ward"]):
    Zm = linkage(Xh, m); dendrogram(Zm, ax=ax, no_labels=True); ax.set_title(m)
    ari = adjusted_rand_score(yh, fcluster(Zm, 3, "maxclust")); ax.set_xlabel(f"ARI at 3 clusters = {ari:.2f}")
plt.tight_layout(); plt.show()
# Single linkage chains through the nearest points and tends to peel off one outlier as a 'cluster';
# it is the right choice for elongated / manifold-shaped groups where centroid-based methods fail.""",
"""Xh, yh = make_blobs(40, centers=3, cluster_std=1.0, random_state=155)
# TODO: dendrograms for single, complete, average, ward; cut at 3; adjusted Rand index vs yh
"""),

md("""## 3. PCA
### Task 6 — two computations, one answer"""),
code("""D = load_digits(); X = D.data.astype(float); X -= X.mean(0); N, p = X.shape
S = X.T @ X / N
lam, Q = np.linalg.eigh(S); order = np.argsort(lam)[::-1]; lam, Q = lam[order], Q[:, order]
U, d, Vt = np.linalg.svd(X, full_matrices=False); V = Vt.T
print("lambda_j = d_j^2/N :", np.allclose(lam, d**2 / N))
print("V matches eigenvectors up to sign:", np.allclose(np.abs(V[:, :20] * Q[:, :20]).sum(0), 1, atol=1e-6))
k = 5
top = np.trace(V[:, :k].T @ S @ V[:, :k])
print(f"tr(V_k' S V_k) = {top:.4f}   sum of top-{k} eigenvalues = {lam[:k].sum():.4f}"); assert np.isclose(top, lam[:k].sum())
rand = [np.trace(W.T @ S @ W) for W in (np.linalg.qr(rng.normal(size=(p, k)))[0] for _ in range(100))]
print(f"max over 100 random orthonormal W: {max(rand):.4f}  <  {top:.4f}"); assert max(rand) < top""",
"""D = load_digits(); X = D.data.astype(float); X -= X.mean(0); N, p = X.shape
# TODO: eigh of S = X'X/N (sorted descending); thin SVD of X; check lambda_j = d_j^2/N and V = eigenvectors up to sign
# TODO: Ky Fan check for k=5 against 100 random orthonormal 64x5 matrices (np.linalg.qr)
"""),

md("""### Task 7 — variance = total − reconstruction"""),
code("""ks = [1, 2, 4, 8, 16, 32, 64]; rec, viaeig = [], []
for k in ks:
    Vk = V[:, :k]; Xr = X @ Vk @ Vk.T
    rec.append(((X - Xr) ** 2).sum(1).mean()); viaeig.append(np.trace(S) - lam[:k].sum())
print(np.c_[ks, rec, viaeig]); assert np.allclose(rec, viaeig)
pve = np.cumsum(lam) / lam.sum(); k90 = int(np.argmax(pve >= 0.9)) + 1
plt.plot(range(1, 65), pve, "o-", ms=3); plt.axhline(0.9, ls=":", color="gray"); plt.xlabel("k"); plt.ylabel("proportion of variance explained"); plt.show()
print("smallest k with >= 90%:", k90)""",
"""# TODO: for k in [1,2,4,8,16,32,64]: mean squared reconstruction error vs tr(S) - sum of top-k eigenvalues; assert equal
# TODO: proportion of variance explained vs k; smallest k reaching 90%
"""),

md("""### Task 8 — eigendigits"""),
code("""fig, axes = plt.subplots(2, 8, figsize=(10, 2.8))
for j in range(8):
    axes[0, j].imshow(V[:, j].reshape(8, 8), cmap="RdBu"); axes[0, j].set_title(f"PC {j+1}", fontsize=8)
x0 = X[7]
for a, k in zip(axes[1], [1, 2, 4, 8, 16, 32, 64, 64]):
    a.imshow((D.data.mean(0) + V[:, :k] @ (V[:, :k].T @ x0)).reshape(8, 8), cmap="gray"); a.set_title(f"k={k}", fontsize=8)
for a in axes.ravel(): a.set_xticks([]); a.set_yticks([])
plt.tight_layout(); plt.show()""",
"""# TODO: first 8 PCs as 8x8 images; one digit reconstructed with k = 1,2,4,8,16,32,64
"""),

md("""### Task 9 — hard vs. soft: PCR against ridge"""),
code("""Xd, yd = load_diabetes(return_X_y=True); Xd = (Xd - Xd.mean(0)) / Xd.std(0); yd = yd - yd.mean()
kf = KFold(10, shuffle=True, random_state=155)
_, dd, _ = np.linalg.svd(Xd, full_matrices=False)
pcr = [(-cross_val_score(make_pipeline(PCA(k), LinearRegression()), Xd, yd, cv=kf, scoring="neg_mean_squared_error").mean(), k) for k in range(1, 11)]
lams = np.logspace(-1, 4, 40)
rid = [(-cross_val_score(Ridge(alpha=l, fit_intercept=False), Xd, yd, cv=kf, scoring="neg_mean_squared_error").mean(), (dd**2 / (dd**2 + l)).sum()) for l in lams]
plt.plot([k for _, k in pcr], [e for e, _ in pcr], "o-", label="PCR (hard)"); plt.plot([df for _, df in rid], [e for e, _ in rid], "s-", ms=3, label="ridge (soft)")
plt.xlabel("effective degrees of freedom"); plt.ylabel("10-fold CV MSE"); plt.legend(); plt.show()
print(f"best PCR: MSE={min(pcr)[0]:.1f} at k={min(pcr)[1]}   best ridge: MSE={min(rid)[0]:.1f} at df={min(rid)[1]:.2f}")""",
"""Xd, yd = load_diabetes(return_X_y=True); Xd = (Xd - Xd.mean(0)) / Xd.std(0); yd = yd - yd.mean()
# TODO: 10-fold CV MSE of PCA(k)+LinearRegression for k=1..10, and of Ridge over a log grid; plot both vs effective df
"""),
md("""## 4. The ML I project — see `tutorial.pdf`, Section 4. Work in a separate notebook."""),
]


def build(student):
    cells = []
    for c in CELLS:
        if c[0] == "markdown":
            cells.append({"cell_type": "markdown", "metadata": {}, "source": c[1]})
        else:
            src = c[2] if (student and len(c) > 2 and c[2] is not None) else c[1]
            cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src})
    return {"cells": cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
            "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}


def check():
    import matplotlib; matplotlib.use("Agg")
    ns = {}
    for i, c in enumerate(CELLS):
        if c[0] == "code":
            try:
                exec(compile(c[1], f"<cell {i}>", "exec"), ns)
            except Exception as e:  # noqa: BLE001
                print(f"cell {i} FAILED: {type(e).__name__}: {e}"); raise SystemExit(1)
    print("all solution cells executed")


if __name__ == "__main__":
    (HERE / "lab_04_student.ipynb").write_text(json.dumps(build(True), indent=1))
    (HERE / "lab_04_solution.ipynb").write_text(json.dumps(build(False), indent=1))
    print("wrote lab_04_student.ipynb, lab_04_solution.ipynb")
    if "--check" in sys.argv:
        check()
