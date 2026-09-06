"""Build lab_02_student.ipynb and lab_02_solution.ipynb. See lab_01_pipeline/build_notebook.py for the format.

    python build_notebook.py          # write both notebooks
    python build_notebook.py --check  # also execute every solution code cell in order
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
md = lambda s: ("markdown", s)
code = lambda sol, stu=None: ("code", sol, stu)

CELLS = [
md("""# Lab 2 — Ridge, Lasso and Cross-Validation
**Machine Learning I · PEU-CD 2026 · ENEI**

Companion to `tutorial.pdf`. Conventions: Lecture 3's ridge is `Ridge(alpha=lam)`; Lecture 3's lasso
$\\tfrac12\\|y-X\\beta\\|^2+\\lambda\\|\\beta\\|_1$ is `Lasso(alpha=lam/N)`."""),
code("""import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.linear_model import Ridge, Lasso, LogisticRegression, lasso_path
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (mean_squared_error, accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, roc_curve, roc_auc_score)
np.set_printoptions(precision=4, suppress=True)

X_raw, y_raw = load_diabetes(return_X_y=True)
N, p = X_raw.shape
X = (X_raw - X_raw.mean(0)) / X_raw.std(0)      # standardized columns
y = y_raw - y_raw.mean()                         # centred response -> no intercept needed
print(X.shape, "column means ~0:", np.allclose(X.mean(0), 0), " column sds = 1:", np.allclose(X.std(0), 1))"""),

md("""## 1. Ridge from the formula
### Task 1 — closed form"""),
code("""def ridge_closed_form(X, y, lam):
    p = X.shape[1]
    return np.linalg.solve(X.T @ X + lam * np.eye(p), X.T @ y)

for lam in [0.1, 1, 10, 100]:
    mine = ridge_closed_form(X, y, lam)
    sk = Ridge(alpha=lam, fit_intercept=False).fit(X, y).coef_
    print(f"lam={lam:6}: max |diff| = {np.abs(mine - sk).max():.2e}")
    assert np.allclose(mine, sk, atol=1e-8)""",
"""def ridge_closed_form(X, y, lam):
    # TODO: (X^T X + lam I)^{-1} X^T y, via np.linalg.solve
    ...

for lam in [0.1, 1, 10, 100]:
    mine = ridge_closed_form(X, y, lam)
    sk = Ridge(alpha=lam, fit_intercept=False).fit(X, y).coef_
    print(f"lam={lam:6}: max |diff| = {np.abs(mine - sk).max():.2e}")
    assert np.allclose(mine, sk, atol=1e-8)"""),

md("""### Task 2 — the SVD picture"""),
code("""U, d, Vt = np.linalg.svd(X, full_matrices=False)
lam = 10
shrink = d**2 / (d**2 + lam)
yhat_svd = U @ (shrink * (U.T @ y))
yhat_ridge = X @ ridge_closed_form(X, y, lam)
print("SVD formula matches closed form:", np.allclose(yhat_svd, yhat_ridge))
print("singular values d_j :", d)
print("shrinkage factors   :", shrink)
j = np.argmin(shrink); print(f"most shrunk direction: j={j}, d_j={d[j]:.3f}, factor={shrink[j]:.3f}")

lams = np.logspace(-2, 4, 200)
df = np.array([(d**2 / (d**2 + l)).sum() for l in lams])
plt.semilogx(lams, df); plt.xlabel("lambda"); plt.ylabel("df(lambda)"); plt.show()
print("df(0) =", (d**2 / (d**2 + 0)).sum())""",
"""U, d, Vt = np.linalg.svd(X, full_matrices=False)
lam = 10
# TODO: shrinkage factors d^2/(d^2+lam); y_hat via the SVD formula; compare with X @ ridge_closed_form(X, y, lam)
shrink = ...
yhat_svd = ...
print("SVD formula matches closed form:", np.allclose(yhat_svd, X @ ridge_closed_form(X, y, lam)))
# TODO: which direction is shrunk most? plot df(lambda) on a log axis; confirm df(0) = 10
"""),

md("""### Task 3 — the ridge path"""),
code("""coefs = np.array([ridge_closed_form(X, y, l) for l in lams])
plt.semilogx(lams, coefs); plt.xlabel("lambda"); plt.ylabel("coefficients"); plt.title("ridge path"); plt.show()
# In the SVD basis beta_lam = V diag(d_j/(d_j^2+lam)) U^T y: each coordinate of V^T beta is the OLS value
# times a strictly positive factor, so V^T beta never crosses zero; beta itself can only be zero where
# a linear combination of shrunk OLS coordinates happens to cancel, which is a measure-zero event in lam.
print("min |beta_j| over the path (never exactly 0):", np.abs(coefs).min())""",
"""# TODO: coefficient paths for lams (log axis). Then explain, from the SVD formula, why no path crosses zero.
"""),

md("""## 2. Lasso from the formula
### Task 4 — soft thresholding"""),
code("""def soft(b, lam):
    b = np.asarray(b, dtype=float)
    return np.sign(b) * np.maximum(np.abs(b) - lam, 0.0)

print("lasso :", soft([3, 0.5, -2], 1))
print("ridge :", np.array([3, 0.5, -2]) / (1 + 1))
assert np.allclose(soft([3, 0.5, -2], 1), [2, 0, -1])
assert np.allclose(np.array([3, 0.5, -2]) / 2, [1.5, 0.25, -1])""",
"""def soft(b, lam):
    # TODO: sign(b) * max(|b| - lam, 0), vectorized
    ...

assert np.allclose(soft([3, 0.5, -2], 1), [2, 0, -1])
assert np.allclose(np.array([3, 0.5, -2]) / 2, [1.5, 0.25, -1])"""),

md("""### Task 5 — coordinate descent"""),
code("""def lasso_cd(X, y, lam, iters=20000, tol=1e-10):
    n, p = X.shape
    beta = np.zeros(p)
    r = y.copy()
    for it in range(iters):
        max_change = 0.0
        for j in range(p):
            r_j = r + X[:, j] * beta[j]
            b_new = soft(X[:, j] @ r_j, lam) / (X[:, j] @ X[:, j])
            r = r_j - X[:, j] * b_new
            max_change = max(max_change, abs(b_new - beta[j])); beta[j] = b_new
        if max_change < tol:          # correlated columns make plain CD slow; sweep until nothing moves
            break
    return beta

for lam in [50, 200, 800]:
    mine = lasso_cd(X, y, lam)
    sk = Lasso(alpha=lam / N, fit_intercept=False, max_iter=100000, tol=1e-10).fit(X, y).coef_
    r = y - X @ mine
    corr = X.T @ r
    active = np.abs(mine) > 1e-8
    print(f"lam={lam:4}: max|diff vs sklearn|={np.abs(mine - sk).max():.1e}  zeros={int((~active).sum())}/{p}"
          f"  KKT active max|x'r|-lam={np.abs(np.abs(corr[active]) - lam).max():.1e}"
          f"  inactive max|x'r|={np.abs(corr[~active]).max() if (~active).any() else 0:.1f} <= {lam}")
    assert np.allclose(mine, sk, atol=1e-4)""",
"""def lasso_cd(X, y, lam, iters=20000, tol=1e-10):
    n, p = X.shape
    beta = np.zeros(p)
    r = y.copy()                           # full residual y - X beta
    for _ in range(iters):                 # stop early once no coordinate moves more than tol
        for j in range(p):
            r_j = r + X[:, j] * beta[j]    # partial residual with coordinate j removed
            # TODO: coordinate update: soft-threshold x_j' r_j, divide by ||x_j||^2; then update r and beta[j]
            b_new = ...
            r = ...
            beta[j] = b_new
    return beta

for lam in [50, 200, 800]:
    mine = lasso_cd(X, y, lam)
    sk = Lasso(alpha=lam / N, fit_intercept=False, max_iter=100000, tol=1e-10).fit(X, y).coef_
    print(f"lam={lam}: max|diff|={np.abs(mine - sk).max():.1e}  zeros={int((np.abs(mine) < 1e-8).sum())}")
    assert np.allclose(mine, sk, atol=1e-4)
# TODO: verify the KKT conditions |x_j' r| = lam (active) and <= lam (inactive) on one solution
"""),

md("""### Task 6 — the lasso path and $\\lambda_{\\max}$"""),
code("""lam_max = np.abs(X.T @ y).max()
print(f"lam_max = {lam_max:.2f}")
print("all zero at 1.01*lam_max:", np.allclose(lasso_cd(X, y, 1.01 * lam_max), 0))
print("not all zero at 0.99*lam_max:", not np.allclose(lasso_cd(X, y, 0.99 * lam_max), 0))

alphas, coefs_l, _ = lasso_path(X, y, n_alphas=100)          # alphas are lam/N
names = load_diabetes().feature_names
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
for j in range(p):
    ax[0].plot(alphas * N, coefs_l[j], label=names[j])
ax[0].set_xscale("log"); ax[0].invert_xaxis(); ax[0].set_title("lasso path"); ax[0].legend(fontsize=7, ncol=2)
ax[1].semilogx(lams, coefs); ax[1].invert_xaxis(); ax[1].set_title("ridge path")
plt.show()
order = [names[j] for j in np.argsort([np.argmax(np.abs(coefs_l[j]) > 1e-8) for j in range(p)])]
print("order of entry (decreasing lambda):", order)""",
"""# TODO: lam_max = ||X^T y||_inf; show lasso_cd returns zeros above it and not below it
lam_max = ...
# TODO: lasso_path (note: its alphas are lam/N) next to the ridge path; list the order in which features enter
"""),

md("""## 3. Choosing lambda
### Task 7 — cross-validation with error bars"""),
code("""grid = np.logspace(-1, 3.2, 40)                     # lam values (Lecture 3 scale)
kf = KFold(10, shuffle=True, random_state=155)
cv_mean, cv_se, nnz = [], [], []
for lam in grid:
    pipe = make_pipeline(StandardScaler(), Lasso(alpha=lam / (0.9 * N), max_iter=50000))   # 9/10 of N rows per fold
    scores = -cross_val_score(pipe, X_raw, y_raw, cv=kf, scoring="neg_mean_squared_error")
    cv_mean.append(scores.mean()); cv_se.append(scores.std(ddof=1) / np.sqrt(10))
    nnz.append(int((np.abs(pipe.fit(X_raw, y_raw)[-1].coef_) > 1e-8).sum()))
cv_mean, cv_se = np.array(cv_mean), np.array(cv_se)
i_min = int(np.argmin(cv_mean))
i_1se = int(np.max(np.where(cv_mean <= cv_mean[i_min] + cv_se[i_min])[0]))   # largest lam within one SE
plt.errorbar(grid, cv_mean, yerr=cv_se, fmt="o-", ms=3); plt.xscale("log")
plt.axvline(grid[i_min], ls="--", label=f"lam_min={grid[i_min]:.1f}, nnz={nnz[i_min]}")
plt.axvline(grid[i_1se], ls=":", label=f"lam_1se={grid[i_1se]:.1f}, nnz={nnz[i_1se]}")
plt.xlabel("lambda"); plt.ylabel("10-fold CV MSE"); plt.legend(); plt.show()""",
"""grid = np.logspace(-1, 3.2, 40)
kf = KFold(10, shuffle=True, random_state=155)
# TODO: for each lam, a Pipeline(StandardScaler, Lasso) evaluated by cross_val_score; record mean, SE, and #nonzero
# TODO: plot with error bars; mark lam_min and the one-SE choice (largest lam with CV <= min + SE)
"""),

md("""### Task 8 — optimism, measured"""),
code("""beta_ols = np.linalg.solve(X.T @ X, X.T @ y)
sigma2 = ((y - X @ beta_ols) ** 2).sum() / (N - p - 1)
print(f"sigma^2 (OLS) = {sigma2:.1f}")
print(f"{'lam':>6} {'train':>9} {'10-CV':>9} {'CV-train':>9} {'2s2df/N':>9}")
for lam in [0.1, 1, 10, 100, 1000]:
    tr = mean_squared_error(y, X @ ridge_closed_form(X, y, lam))
    cv = -cross_val_score(Ridge(alpha=lam, fit_intercept=False), X, y, cv=kf, scoring="neg_mean_squared_error").mean()
    dfl = (d**2 / (d**2 + lam)).sum()
    print(f"{lam:6} {tr:9.1f} {cv:9.1f} {cv - tr:9.1f} {2 * sigma2 * dfl / N:9.1f}")""",
"""# TODO: sigma^2 from OLS residuals (divide by N-p-1). For lam in [0.1,1,10,100,1000]: train MSE, 10-fold CV MSE,
# their difference, and 2*sigma^2*df(lam)/N side by side.
"""),

md("""## 4. Regularized logistic regression
### Task 9 — separable data has no MLE"""),
code("""rng = np.random.default_rng(155)
A = rng.normal([-2, 0], 1, (30, 2)); B = rng.normal([2, 0], 1, (30, 2))
Xs = np.vstack([A, B]); ys = np.r_[np.zeros(30), np.ones(30)]
import warnings; warnings.filterwarnings("ignore")
unreg = LogisticRegression(penalty=None, max_iter=10000).fit(Xs, ys)
print("unpenalized: ||w|| =", np.linalg.norm(unreg.coef_).round(2), " min/max fitted prob:",
      unreg.predict_proba(Xs)[:, 1].min().round(4), unreg.predict_proba(Xs)[:, 1].max().round(4))
Cs = [1e3, 1e2, 10, 1, 0.1]
norms = [np.linalg.norm(LogisticRegression(C=C, max_iter=10000).fit(Xs, ys).coef_) for C in Cs]
plt.semilogx(Cs, norms, "o-"); plt.xlabel("C = 1/lambda"); plt.ylabel("||w||"); plt.show()
# As C grows the penalty vanishes and ||w|| grows without bound: the likelihood keeps improving along
# c * w_0 for any separating w_0 (Lecture 2). Any finite lambda pins it down.""",
"""rng = np.random.default_rng(155)
A = rng.normal([-2, 0], 1, (30, 2)); B = rng.normal([2, 0], 1, (30, 2))
Xs = np.vstack([A, B]); ys = np.r_[np.zeros(30), np.ones(30)]
# TODO: fit LogisticRegression(penalty=None, max_iter=10000); print ||w|| and the range of fitted probabilities
# TODO: for C in [1e3,1e2,10,1,0.1] plot ||w|| vs C (log axis) and explain the shape
"""),

md("""### Task 10 — a real classifier, evaluated properly"""),
code("""Xc, yc = load_breast_cancer(return_X_y=True)          # y=1 is benign in sklearn; make malignant the positive class
yc = 1 - yc
Xtr, Xte, ytr, yte = train_test_split(Xc, yc, test_size=0.2, random_state=155, stratify=yc)
pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
gs = GridSearchCV(pipe, {"logisticregression__C": np.logspace(-3, 2, 11)}, cv=5, scoring="neg_log_loss").fit(Xtr, ytr)
print("chosen C:", gs.best_params_["logisticregression__C"])
prob = gs.predict_proba(Xte)[:, 1]; pred = (prob >= 0.5).astype(int)
print(f"acc={accuracy_score(yte, pred):.3f} prec={precision_score(yte, pred):.3f} rec={recall_score(yte, pred):.3f} F1={f1_score(yte, pred):.3f}")
print("confusion (rows=true 0/1, cols=pred 0/1):\\n", confusion_matrix(yte, pred))

fpr, tpr, _ = roc_curve(yte, prob)
auc_sk = roc_auc_score(yte, prob)
auc_pairs = np.mean(prob[yte == 1][:, None] > prob[yte == 0][None, :])
plt.plot(fpr, tpr, label=f"AUC={auc_sk:.4f}"); plt.plot([0, 1], [0, 1], "k--"); plt.legend(); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.show()
print(f"AUC sklearn={auc_sk:.6f}  AUC pairwise={auc_pairs:.6f}")
assert np.isclose(auc_sk, auc_pairs)

c_fp, c_fn = 1, 10
tau = c_fp / (c_fp + c_fn)
for t in [0.5, tau]:
    pr = (prob >= t).astype(int); cm = confusion_matrix(yte, pr)
    cost = c_fp * cm[0, 1] + c_fn * cm[1, 0]
    print(f"threshold={t:.3f}: FP={cm[0,1]} FN={cm[1,0]} expected cost={cost}")""",
"""Xc, yc = load_breast_cancer(return_X_y=True); yc = 1 - yc      # malignant = 1
Xtr, Xte, ytr, yte = train_test_split(Xc, yc, test_size=0.2, random_state=155, stratify=yc)
# TODO: Pipeline(StandardScaler, LogisticRegression) with C chosen by 5-fold GridSearchCV on the training set
# TODO: accuracy, precision, recall, F1, confusion matrix at threshold 0.5
# TODO: ROC curve; AUC via roc_auc_score AND as the fraction of (pos, neg) pairs ranked correctly; assert equal
# TODO: with c_FP=1, c_FN=10: tau* = c_FP/(c_FP+c_FN); confusion matrix and expected cost at 0.5 and at tau*
"""),
md("""## 5. Exercises — see `tutorial.pdf`, Section 6"""),
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
    (HERE / "lab_02_student.ipynb").write_text(json.dumps(build(True), indent=1))
    (HERE / "lab_02_solution.ipynb").write_text(json.dumps(build(False), indent=1))
    print("wrote lab_02_student.ipynb, lab_02_solution.ipynb")
    if "--check" in sys.argv:
        check()
