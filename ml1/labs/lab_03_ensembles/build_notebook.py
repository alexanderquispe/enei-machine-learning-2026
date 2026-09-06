"""Build lab_03_student.ipynb and lab_03_solution.ipynb. Format as in lab_01_pipeline/build_notebook.py.

    python build_notebook.py [--check]
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
md = lambda s: ("markdown", s)
code = lambda sol, stu=None: ("code", sol, stu)

CELLS = [
md("""# Lab 3 — Trees, Random Forests and Gradient Boosting
**Machine Learning I · PEU-CD 2026 · ENEI**

Companion to `tutorial.pdf`."""),
code("""import numpy as np, time
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from sklearn.inspection import permutation_importance
np.set_printoptions(precision=4, suppress=True)
rng = np.random.default_rng(155)

Xa, ya = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(Xa, ya, test_size=0.2, random_state=155, stratify=ya)
N = len(ytr); print("train", Xtr.shape, "test", Xte.shape)"""),

md("""## 1. Impurity and information gain
### Task 1"""),
code("""def props(counts):
    counts = np.asarray(counts, float); return counts / counts.sum()
def entropy(counts):
    p = props(counts); p = p[p > 0]; return float(-(p * np.log2(p)).sum())
def gini(counts):
    p = props(counts); return float(1 - (p**2).sum())
def misclass(counts):
    p = props(counts); return float(1 - p.max())
def gain(parent, children, impurity):
    n = sum(sum(c) for c in children)
    return impurity(parent) - sum(sum(c) / n * impurity(c) for c in children)

for name, I in [("entropy", entropy), ("gini", gini), ("misclass", misclass)]:
    print(f"{name:9s} gain (8,4)->(6,1),(2,3): {gain([8, 4], [[6, 1], [2, 3]], I):.4f}")
assert np.isclose(gain([8, 4], [[6, 1], [2, 3]], entropy), 0.1685, atol=1e-4)
assert np.isclose(gain([8, 4], [[6, 1], [2, 3]], gini), 0.1016, atol=1e-4)
assert np.isclose(gain([8, 4], [[6, 1], [2, 3]], misclass), 0.0833, atol=1e-4)

after = lambda children, I: sum(sum(c) / 800 * I(c) for c in children)
A = [[300, 100], [100, 300]]; B = [[200, 400], [200, 0]]
for name, I in [("misclass", misclass), ("gini", gini), ("entropy", entropy)]:
    print(f"{name:9s} after split A: {after(A, I):.3f}   after split B: {after(B, I):.3f}")
assert np.isclose(after(A, misclass), after(B, misclass)) and after(B, gini) < after(A, gini)""",
"""def props(counts):
    counts = np.asarray(counts, float); return counts / counts.sum()
# TODO: entropy(counts), gini(counts), misclass(counts) on a class-count vector; gain(parent, children, impurity)
def entropy(counts): ...
def gini(counts): ...
def misclass(counts): ...
def gain(parent, children, impurity): ...

assert np.isclose(gain([8, 4], [[6, 1], [2, 3]], entropy), 0.1685, atol=1e-4)
assert np.isclose(gain([8, 4], [[6, 1], [2, 3]], gini), 0.1016, atol=1e-4)
assert np.isclose(gain([8, 4], [[6, 1], [2, 3]], misclass), 0.0833, atol=1e-4)
# TODO: the (400,400) example: impurity after split A vs split B under the three criteria
"""),

md("""### Task 2 — the root split, by hand"""),
code("""def best_split(X, y, impurity):
    best = (None, None, -np.inf)
    parent = np.bincount(y, minlength=2)
    for j in range(X.shape[1]):
        order = np.argsort(X[:, j]); xs, ys = X[order, j], y[order]
        left = np.zeros(2)
        for i in range(len(ys) - 1):
            left[ys[i]] += 1
            if xs[i] == xs[i + 1]:
                continue
            g = gain(parent, [left, parent - left], impurity)
            if g > best[2]:
                best = (j, (xs[i] + xs[i + 1]) / 2, g)
    return best

j, s, g = best_split(Xtr, ytr, gini)
stump = DecisionTreeClassifier(max_depth=1).fit(Xtr, ytr)
print(f"mine   : feature {j}, threshold {s:.4f}, gain {g:.4f}")
print(f"sklearn: feature {stump.tree_.feature[0]}, threshold {stump.tree_.threshold[0]:.4f}")
assert j == stump.tree_.feature[0] and np.isclose(s, stump.tree_.threshold[0], atol=1e-3)""",
"""def best_split(X, y, impurity):
    # TODO: scan every feature and every midpoint between consecutive sorted values; return (j, s, gain)
    ...

j, s, g = best_split(Xtr, ytr, gini)
stump = DecisionTreeClassifier(max_depth=1).fit(Xtr, ytr)
print(f"mine   : feature {j}, threshold {s:.4f}, gain {g:.4f}")
print(f"sklearn: feature {stump.tree_.feature[0]}, threshold {stump.tree_.threshold[0]:.4f}")
assert j == stump.tree_.feature[0] and np.isclose(s, stump.tree_.threshold[0], atol=1e-3)"""),

md("""## 2. One tree
### Task 3 — overfitting, in two directions"""),
code("""depths = range(1, 13); tr_e, te_e = [], []
for d in depths:
    t = DecisionTreeClassifier(max_depth=d, random_state=0).fit(Xtr, ytr)
    tr_e.append(1 - t.score(Xtr, ytr)); te_e.append(1 - t.score(Xte, yte))
leaves = [1, 2, 5, 10, 20, 50]; tr_l, te_l = [], []
for m in leaves:
    t = DecisionTreeClassifier(min_samples_leaf=m, random_state=0).fit(Xtr, ytr)
    tr_l.append(1 - t.score(Xtr, ytr)); te_l.append(1 - t.score(Xte, yte))
fig, ax = plt.subplots(1, 2, figsize=(9, 3))
ax[0].plot(depths, tr_e, "o-", label="train"); ax[0].plot(depths, te_e, "s-", label="test"); ax[0].set_xlabel("max_depth"); ax[0].legend()
ax[1].plot(leaves, tr_l, "o-", label="train"); ax[1].plot(leaves, te_l, "s-", label="test"); ax[1].set_xscale("log"); ax[1].set_xlabel("min_samples_leaf")
plt.show()
full = DecisionTreeClassifier(random_state=0).fit(Xtr, ytr)
single_tree_err = 1 - full.score(Xte, yte)
print(f"unrestricted tree: {full.get_n_leaves()} leaves, train err {1 - full.score(Xtr, ytr):.3f}, test err {single_tree_err:.3f}")""",
"""# TODO: train/test error vs max_depth in 1..12, and vs min_samples_leaf in [1,2,5,10,20,50]; two plots
# TODO: the unrestricted tree: leaves, train error, test error -> store test error as single_tree_err
full = DecisionTreeClassifier(random_state=0).fit(Xtr, ytr)
single_tree_err = 1 - full.score(Xte, yte)
"""),

md("""## 3. Bagging and the variance formula
### Task 4 — bagging from scratch"""),
code("""def bagged_trees(X, y, B, seed=155):
    r = np.random.default_rng(seed); trees, idxs = [], []
    for _ in range(B):
        idx = r.integers(0, len(y), len(y))
        trees.append(DecisionTreeClassifier(random_state=0).fit(X[idx], y[idx])); idxs.append(idx)
    return trees, idxs

def vote(trees, X):
    return (np.mean([t.predict(X) for t in trees], axis=0) >= 0.5).astype(int)

Bs = [1, 2, 5, 10, 25, 50, 100, 200]
trees200, idxs200 = bagged_trees(Xtr, ytr, 200)
bag_err = [1 - accuracy_score(yte, vote(trees200[:b], Xte)) for b in Bs]
plt.plot(Bs, bag_err, "o-", label="bagging"); plt.axhline(single_tree_err, ls="--", color="gray", label="single tree")
plt.xscale("log"); plt.xlabel("B"); plt.ylabel("test error"); plt.legend(); plt.show()

# out-of-bag
votes = np.zeros(N); counts = np.zeros(N)
for t, idx in zip(trees200, idxs200):
    oob = np.setdiff1d(np.arange(N), idx)
    votes[oob] += t.predict(Xtr[oob]); counts[oob] += 1
oob_pred = (votes / np.maximum(counts, 1) >= 0.5).astype(int)
oob_err = 1 - accuracy_score(ytr[counts > 0], oob_pred[counts > 0])
sk = BaggingClassifier(DecisionTreeClassifier(), n_estimators=200, oob_score=True, random_state=0).fit(Xtr, ytr)
print(f"OOB error mine={oob_err:.3f}  sklearn={1 - sk.oob_score_:.3f}  test error (B=200)={bag_err[-1]:.3f}")
print("mean fraction of distinct points per bootstrap:", np.mean([len(np.unique(i)) / N for i in idxs200]).round(3), " (1 - 1/e = 0.632)")""",
"""def bagged_trees(X, y, B, seed=155):
    # TODO: B bootstrap samples (rng.integers(0, N, N)), one unrestricted tree each; return trees and index arrays
    ...
def vote(trees, X):
    # TODO: majority vote
    ...
Bs = [1, 2, 5, 10, 25, 50, 100, 200]
trees200, idxs200 = bagged_trees(Xtr, ytr, 200)
# TODO: test error vs B, overlay single_tree_err
# TODO: out-of-bag error by hand; compare with BaggingClassifier(oob_score=True) and with the test error
# TODO: mean fraction of distinct training points per bootstrap sample (expect ~0.632)
"""),

md("""### Task 5 — measure rho and test the variance formula"""),
code("""def rho_and_var(trees, X):
    P = np.array([2 * t.predict(X) - 1 for t in trees], float)          # B x n_test, entries +-1
    B = len(trees)
    C = np.corrcoef(P)                                                  # B x B correlations across test points
    rho = (C.sum() - B) / (B * (B - 1))
    sigma2 = P.var(axis=1, ddof=0).mean()                               # mean per-tree variance across test points
    var_avg = P.mean(axis=0).var(ddof=0)                                # variance of the averaged vote across test points
    return rho, sigma2, var_avg

rho, s2, v = rho_and_var(trees200[:100], Xte)
print(f"rho={rho:.3f}  sigma^2={s2:.3f}  Var(average)={v:.4f}  formula rho*s2+(1-rho)*s2/B={rho * s2 + (1 - rho) * s2 / 100:.4f}")""",
"""def rho_and_var(trees, X):
    # TODO: predictions as +-1 (B x n_test); average pairwise correlation across trees; mean per-tree variance;
    # variance of the averaged vote. Return (rho, sigma2, var_avg).
    ...
rho, s2, v = rho_and_var(trees200[:100], Xte)
print(f"rho={rho:.3f}  Var(average)={v:.4f}  formula={rho * s2 + (1 - rho) * s2 / 100:.4f}")"""),

md("""### Task 6 — random forests lower rho"""),
code("""print(f"{'m':>3} {'rho':>7} {'tree err':>9} {'forest err':>11}")
best_rf, best_err = None, 1
for m in [1, 3, 5, 10, 30]:
    rf = RandomForestClassifier(n_estimators=100, max_features=m, random_state=0).fit(Xtr, ytr)
    rho_m, _, _ = rho_and_var(rf.estimators_, Xte)
    tree_err = np.mean([1 - t.score(Xte, yte) for t in rf.estimators_])
    f_err = 1 - rf.score(Xte, yte)
    print(f"{m:3d} {rho_m:7.3f} {tree_err:9.3f} {f_err:11.3f}")
    if f_err < best_err:
        best_rf, best_err = rf, f_err
imp = permutation_importance(best_rf, Xte, yte, n_repeats=10, random_state=0)
top = np.argsort(imp.importances_mean)[::-1][:5]
print("top-5 features:", [load_breast_cancer().feature_names[i] for i in top])""",
"""# TODO: for m in [1,3,5,10,30]: RandomForestClassifier(max_features=m); rho via rho_and_var(rf.estimators_, Xte),
# mean single-tree test error, forest test error. Then permutation_importance on the best forest: top-5 features.
"""),

md("""## 4. Boosting
### Task 7 — AdaBoost by hand (must match Lecture 5's table)"""),
code("""x = np.array([1, 2, 3, 4, 5.0]); y = np.array([1, 1, -1, -1, 1])
thresholds = [1.5, 2.5, 3.5, 4.5]
stumps = [(s, t) for t in thresholds for s in (+1, -1)]            # h(x) = s * sign(x - t)
H = lambda s, t, x: s * np.where(x > t, 1, -1)

w = np.ones(5) / 5; f = np.zeros(5); alphas, eps, Zs = [], [], []
for m in range(3):
    errs = [(w * (H(s, t, x) != y)).sum() for s, t in stumps]
    k = int(np.argmin(errs)); s, t = stumps[k]; e = errs[k]
    a = 0.5 * np.log((1 - e) / e)
    unnorm = w * np.exp(-a * y * H(s, t, x)); Z = unnorm.sum(); w = unnorm / Z
    f += a * H(s, t, x); alphas.append(a); eps.append(e); Zs.append(Z)
    print(f"round {m+1}: stump h=+1 iff x {'>' if s > 0 else '<='} {t},  eps={e:.3f}  alpha={a:.3f}  Z={Z:.3f}  new w={w}")
    assert np.isclose(Z, 2 * np.sqrt(e * (1 - e)))
print("train error after 3 rounds:", np.mean(np.sign(f) != y), "  prod Z =", np.prod(Zs).round(3))
print("mean exp loss =", np.mean(np.exp(-y * f)).round(4), " equals prod Z:", np.isclose(np.mean(np.exp(-y * f)), np.prod(Zs)))
assert np.allclose(eps, [0.2, 0.25, 1/3]) and np.allclose(alphas, [0.6931, 0.5493, 0.3466], atol=1e-4)
assert np.allclose(Zs, [0.8, 0.8660, 0.9428], atol=1e-4) and np.isclose(np.prod(Zs), 0.6532, atol=1e-4)""",
"""x = np.array([1, 2, 3, 4, 5.0]); y = np.array([1, 1, -1, -1, 1])
thresholds = [1.5, 2.5, 3.5, 4.5]
stumps = [(s, t) for t in thresholds for s in (+1, -1)]            # h(x) = s * sign(x - t)
H = lambda s, t, x: s * np.where(x > t, 1, -1)
# TODO: three rounds of AdaBoost: weighted error of each stump, pick the best, alpha, weight update with Z.
# Print eps, alpha, Z, new weights each round. Then check Z = 2 sqrt(eps(1-eps)) and mean exp-loss = prod Z.
w = np.ones(5) / 5; f = np.zeros(5)
"""),

md("""### Task 8 — early stopping on M"""),
code("""Xa_, Xv_, ya_, yv_ = train_test_split(Xtr, ytr, test_size=0.2, random_state=155, stratify=ytr)
best = {}
for nu in [1.0, 0.1, 0.01]:
    gb = GradientBoostingClassifier(n_estimators=2000, learning_rate=nu, max_depth=3, random_state=0).fit(Xa_, ya_)
    val = [log_loss(yv_, p[:, 1]) for p in gb.staged_predict_proba(Xv_)]
    M = int(np.argmin(val)) + 1
    gb_M = GradientBoostingClassifier(n_estimators=M, learning_rate=nu, max_depth=3, random_state=0).fit(Xtr, ytr)
    best[nu] = (M, 1 - gb_M.score(Xte, yte))
    plt.plot(np.arange(1, 2001), val, label=f"nu={nu}: best M={M}")
plt.xscale("log"); plt.xlabel("rounds M"); plt.ylabel("validation log loss"); plt.legend(); plt.show()
for nu, (M, err) in best.items():
    print(f"nu={nu:5}: best M={M:5d}  test error={err:.3f}")""",
"""Xa_, Xv_, ya_, yv_ = train_test_split(Xtr, ytr, test_size=0.2, random_state=155, stratify=ytr)
# TODO: for nu in [1, 0.1, 0.01]: GradientBoostingClassifier(n_estimators=2000, learning_rate=nu, max_depth=3);
# validation log loss after every round via staged_predict_proba; best M; refit on Xtr with that M; test error.
"""),

md("""### Task 9 — three ensembles, one table"""),
code("""models = {
  "tree": GridSearchCV(DecisionTreeClassifier(random_state=0), {"max_depth": [2, 4, 6, 8], "min_samples_leaf": [1, 5, 10]}, cv=5),
  "bagging": GridSearchCV(BaggingClassifier(DecisionTreeClassifier(), random_state=0), {"n_estimators": [50, 100, 200]}, cv=5),
  "random forest": GridSearchCV(RandomForestClassifier(random_state=0), {"n_estimators": [100, 200], "max_features": [3, 5, 10]}, cv=5),
  "gradient boosting": GridSearchCV(GradientBoostingClassifier(random_state=0),
                                    {"learning_rate": [0.1, 0.03], "n_estimators": [100, 300], "max_depth": [2, 3]}, cv=5),
}
print(f"{'model':18s} {'test acc':>9} {'AUC':>7} {'fit s':>7}   best params")
for name, gs in models.items():
    t0 = time.time(); gs.fit(Xtr, ytr); dt = time.time() - t0
    acc = gs.score(Xte, yte); auc = roc_auc_score(yte, gs.predict_proba(Xte)[:, 1])
    print(f"{name:18s} {acc:9.3f} {auc:7.3f} {dt:7.1f}   {gs.best_params_}")""",
"""# TODO: GridSearchCV (cv=5) for a tree, bagging, random forest and gradient boosting over small grids;
# table of test accuracy, AUC and wall-clock fit time; two sentences on which to deploy.
"""),
md("""## 5. Exercises — see `tutorial.pdf`, Section 5"""),
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
    (HERE / "lab_03_student.ipynb").write_text(json.dumps(build(True), indent=1))
    (HERE / "lab_03_solution.ipynb").write_text(json.dumps(build(False), indent=1))
    print("wrote lab_03_student.ipynb, lab_03_solution.ipynb")
    if "--check" in sys.argv:
        check()
