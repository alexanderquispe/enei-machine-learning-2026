"""Build lab_01_student.ipynb and lab_01_solution.ipynb from one cell list.

    python build_notebook.py          # write both notebooks
    python build_notebook.py --check  # also execute every solution code cell in order

A cell is (kind, source) or (kind, solution_source, student_source). When a student
version is given, the solution notebook gets the first and the student notebook the second.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

md = lambda s: ("markdown", s)
code = lambda sol, stu=None: ("code", sol, stu)

CELLS = [
md("""# Lab 1 — Working Environment and a First End-to-End Pipeline
**Machine Learning I · PEU-CD 2026 · ENEI**

Companion to `tutorial.pdf`. Cells marked `# TODO` are yours. Run top to bottom.
"""),
code("""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score

rng = np.random.default_rng(155)
np.set_printoptions(precision=4, suppress=True)"""),

md("""## 1. numpy warm-up
Shapes, `@`, broadcasting, `solve`. Nothing to submit here — just run and read."""),
code("""A = rng.normal(size=(4, 3))
print("A.shape        =", A.shape)
print("(A.T @ A).shape=", (A.T @ A).shape)
print("column means   =", A.mean(axis=0))
print("centred means  =", (A - A.mean(axis=0)).mean(axis=0))   # broadcasting
b = rng.normal(size=3)
M = A.T @ A
print("solve vs inv agree:", np.allclose(np.linalg.solve(M, b), np.linalg.inv(M) @ b))"""),

md("""## 2. Lecture 1, recomputed
### Task 1 — normal equations by hand
Expected: $\\hat\\beta = (1.49, 0.75)$."""),
code("""x = np.array([1, 2, 3, 4, 5.0])
y = np.array([2.2, 2.8, 4.5, 3.7, 5.5])
X = np.column_stack([np.ones_like(x), x])          # design matrix, intercept first

XtX = X.T @ X
Xty = X.T @ y
beta_hat = np.linalg.solve(XtX, Xty)
print("X^T X =\\n", XtX)
print("X^T y =", Xty)
print("beta_hat =", beta_hat)
assert np.allclose(beta_hat, [1.49, 0.75])""",
"""x = np.array([1, 2, 3, 4, 5.0])
y = np.array([2.2, 2.8, 4.5, 3.7, 5.5])
X = np.column_stack([np.ones_like(x), x])          # design matrix, intercept first

# TODO: form X^T X and X^T y, solve the normal equations with np.linalg.solve
XtX = ...
Xty = ...
beta_hat = ...
print("beta_hat =", beta_hat)
assert np.allclose(beta_hat, [1.49, 0.75])"""),

md("""### Task 2 — four properties of the hat matrix"""),
code("""H = X @ np.linalg.solve(XtX, X.T)                    # X (X^T X)^{-1} X^T without forming the inverse
r = y - H @ y
print("symmetric  :", np.allclose(H, H.T))
print("idempotent :", np.allclose(H @ H, H))
print("trace      :", np.trace(H).round(6), "= p+1 =", X.shape[1])
print("X^T r = 0  :", np.allclose(X.T @ r, 0))
print("residuals  :", r)
print("eigenvalues:", np.sort(np.linalg.eigvalsh(H))[::-1].round(6))
# A projection has eigenvalues in {0,1}: H^2 = H forces lambda^2 = lambda. Exactly p+1 of them are 1.""",
"""# TODO: build H without calling np.linalg.inv, then check the four properties with np.allclose
H = ...
r = y - H @ y
print("symmetric  :", ...)
print("idempotent :", ...)
print("trace      :", ...)
print("X^T r = 0  :", ...)
print("eigenvalues:", np.sort(np.linalg.eigvalsh(H))[::-1].round(6))
# Why are the eigenvalues what they are? Write one sentence:
#"""),

md("""### Task 3 — gradient descent: implement, then break it"""),
code("""def gradient_descent(X, y, eta, iters, beta0=None):
    N, d = X.shape
    beta = np.zeros(d) if beta0 is None else beta0.copy()
    losses = []
    for _ in range(iters):
        resid = y - X @ beta
        grad = -(X.T @ resid) / N
        beta = beta - eta * grad
        losses.append(0.5 * np.mean((y - X @ beta) ** 2))
    return beta, np.array(losses)

beta_gd, L = gradient_descent(X, y, eta=0.05, iters=2000)
print("GD  :", beta_gd, "  closed form:", beta_hat)
assert np.allclose(beta_gd, beta_hat, atol=1e-3)

lam_max = np.linalg.eigvalsh(XtX / len(y)).max()
eta_crit = 2 / lam_max
print(f"lambda_max = {lam_max:.4f}   critical step 2/lambda_max = {eta_crit:.4f}")

fig, ax = plt.subplots(1, 2, figsize=(9, 3))
for a, eta in zip(ax, [0.9 * eta_crit, 1.05 * eta_crit]):
    _, Ls = gradient_descent(X, y, eta=eta, iters=40)
    a.plot(Ls); a.set_title(f"eta = {eta:.3f}"); a.set_xlabel("iteration"); a.set_yscale("log")
ax[0].set_ylabel("loss"); plt.tight_layout(); plt.show()

# standardize x and repeat
xs = (x - x.mean()) / x.std()
Xs = np.column_stack([np.ones_like(xs), xs])
lam_s = np.linalg.eigvalsh(Xs.T @ Xs / len(y)).max()
_, Ls = gradient_descent(Xs, y, eta=0.5, iters=200)
print(f"after standardizing: lambda_max = {lam_s:.4f};  iterations to |loss change|<1e-8:",
      int(np.argmax(np.abs(np.diff(Ls)) < 1e-8)) + 1)""",
"""def gradient_descent(X, y, eta, iters, beta0=None):
    N, d = X.shape
    beta = np.zeros(d) if beta0 is None else beta0.copy()
    losses = []
    for _ in range(iters):
        # TODO: gradient of L(beta) = (1/2N) ||y - X beta||^2, then the update
        grad = ...
        beta = ...
        losses.append(0.5 * np.mean((y - X @ beta) ** 2))
    return beta, np.array(losses)

beta_gd, L = gradient_descent(X, y, eta=0.05, iters=2000)
print("GD  :", beta_gd, "  closed form:", beta_hat)
assert np.allclose(beta_gd, beta_hat, atol=1e-3)

# TODO: lambda_max of X^T X / N, critical step 2/lambda_max, then plot the loss for eta just below and just above it
lam_max = ...
eta_crit = ...

# TODO: standardize x, rebuild the design matrix, and report the new lambda_max and iterations to converge
"""),

md("""## 3. A real pipeline
`load_diabetes`: 442 patients, 10 standardized covariates, disease progression as target."""),
code("""data = load_diabetes()
X_all, y_all = data.data, data.target
print(X_all.shape, y_all.shape)
X_tr, X_te, y_tr, y_te = train_test_split(X_all, y_all, test_size=0.2, random_state=155)
print("train:", X_tr.shape, " test (sealed):", X_te.shape)"""),

md("""### Task 4 — two numbers to beat"""),
code("""baseline_mse = mean_squared_error(y_tr, np.full_like(y_tr, y_tr.mean()))
model = LinearRegression().fit(X_tr, y_tr)
train_mse = mean_squared_error(y_tr, model.predict(X_tr))
print(f"baseline MSE = {baseline_mse:.1f}")
print(f"linear   MSE = {train_mse:.1f}   R^2 = {r2_score(y_tr, model.predict(X_tr)):.3f}")

# same thing with the normal equations
Xd = np.column_stack([np.ones(len(y_tr)), X_tr])
beta_ne = np.linalg.solve(Xd.T @ Xd, Xd.T @ y_tr)
print("coef agree with normal equations:", np.allclose(beta_ne[1:], model.coef_), np.isclose(beta_ne[0], model.intercept_))""",
"""# TODO: baseline MSE (predict the training mean), then LinearRegression train MSE and R^2
baseline_mse = ...
model = LinearRegression().fit(X_tr, y_tr)
train_mse = ...
print(f"baseline MSE = {baseline_mse:.1f}")
print(f"linear   MSE = {train_mse:.1f}")

# TODO: solve the normal equations on (X_tr with intercept column) and confirm model.coef_ matches
"""),

md("""### Task 5 — K-fold cross-validation, by hand"""),
code("""def kfold_mse(X, y, K, make_model, seed=155):
    kf = KFold(n_splits=K, shuffle=True, random_state=seed)
    scores = []
    for tr, va in kf.split(X):
        m = make_model().fit(X[tr], y[tr])
        scores.append(mean_squared_error(y[va], m.predict(X[va])))
    return np.array(scores)

mine = kfold_mse(X_tr, y_tr, 5, LinearRegression)
theirs = -cross_val_score(LinearRegression(), X_tr, y_tr, cv=KFold(5, shuffle=True, random_state=155),
                          scoring="neg_mean_squared_error")
print("mine  :", mine.round(1), " mean", mine.mean().round(1))
print("theirs:", theirs.round(1), " mean", theirs.mean().round(1))
assert np.allclose(mine, theirs)""",
"""def kfold_mse(X, y, K, make_model, seed=155):
    kf = KFold(n_splits=K, shuffle=True, random_state=seed)
    scores = []
    for tr, va in kf.split(X):
        # TODO: fit on the K-1 folds, evaluate MSE on the held-out fold
        ...
    return np.array(scores)

mine = kfold_mse(X_tr, y_tr, 5, LinearRegression)
theirs = -cross_val_score(LinearRegression(), X_tr, y_tr, cv=KFold(5, shuffle=True, random_state=155),
                          scoring="neg_mean_squared_error")
print("mine  :", mine.round(1)); print("theirs:", theirs.round(1))
assert np.allclose(mine, theirs)"""),

md("""### Task 6 — polynomial degree: the bias–variance picture, for real"""),
code("""degrees = [1, 2, 3]
tr_err, cv_err = [], []
for d in degrees:
    make = lambda d=d: make_pipeline(PolynomialFeatures(d, include_bias=False), StandardScaler(), LinearRegression())
    m = make().fit(X_tr, y_tr)
    tr_err.append(mean_squared_error(y_tr, m.predict(X_tr)))
    cv_err.append(kfold_mse(X_tr, y_tr, 5, make).mean())
for d, a, b in zip(degrees, tr_err, cv_err):
    print(f"degree {d}: train MSE {a:8.1f}   5-fold CV MSE {b:8.1f}")
plt.plot(degrees, tr_err, "o-", label="train"); plt.plot(degrees, cv_err, "s-", label="5-fold CV")
plt.xlabel("polynomial degree"); plt.ylabel("MSE"); plt.legend(); plt.show()
# Training error falls because bias falls (H grows). CV error rises because variance grows faster than
# bias falls: with N=354 and p=65 (degree 2) or 285 (degree 3), the fit follows the noise.
best_degree = int(degrees[int(np.argmin(cv_err))])
print("chosen degree:", best_degree)""",
"""degrees = [1, 2, 3]
tr_err, cv_err = [], []
for d in degrees:
    make = lambda d=d: make_pipeline(PolynomialFeatures(d, include_bias=False), StandardScaler(), LinearRegression())
    # TODO: training MSE of the fitted pipeline, and mean 5-fold CV MSE via kfold_mse
    ...
# TODO: plot both curves against degree, choose best_degree by CV, and explain each curve in one sentence
best_degree = ...
"""),

md("""### Task 7 — learning curves"""),
code("""X_a, X_v, y_a, y_v = train_test_split(X_tr, y_tr, test_size=0.2, random_state=155)
sizes = [20, 40, 80, 160, len(y_a)]
tr_c, va_c = [], []
for n in sizes:
    m = LinearRegression().fit(X_a[:n], y_a[:n])
    tr_c.append(mean_squared_error(y_a[:n], m.predict(X_a[:n])))
    va_c.append(mean_squared_error(y_v, m.predict(X_v)))
plt.plot(sizes, tr_c, "o-", label="train"); plt.plot(sizes, va_c, "s-", label="validation")
plt.xscale("log"); plt.xlabel("training-set size n"); plt.ylabel("MSE"); plt.legend(); plt.show()
print("gap at n=20:", round(va_c[0] - tr_c[0], 1), "   gap at n=%d:" % sizes[-1], round(va_c[-1] - tr_c[-1], 1))""",
"""X_a, X_v, y_a, y_v = train_test_split(X_tr, y_tr, test_size=0.2, random_state=155)
sizes = [20, 40, 80, 160, len(y_a)]
# TODO: for each n, fit on the first n rows of (X_a, y_a); record train MSE on those rows and MSE on (X_v, y_v)
# TODO: plot both against n (log x-axis)
"""),

md("""### The one number
Open the test set once."""),
code("""final = make_pipeline(PolynomialFeatures(best_degree, include_bias=False), StandardScaler(), LinearRegression()).fit(X_tr, y_tr)
print(f"TEST MSE (degree {best_degree}): {mean_squared_error(y_te, final.predict(X_te)):.1f}")""",
"""# TODO: fit the chosen pipeline on the full training set and report MSE on X_te. Then stop.
"""),

md("""## 4. Exercises — see `tutorial.pdf`, Section 6
Add your cells below."""),
]


def build(student: bool):
    cells = []
    for c in CELLS:
        kind = c[0]
        if kind == "markdown":
            cells.append({"cell_type": "markdown", "metadata": {}, "source": c[1]})
        else:
            src = c[2] if (student and len(c) > 2 and c[2] is not None) else c[1]
            cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src})
    return {"cells": cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
            "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}


def check():
    """Execute the solution code cells in order, in one namespace, with plots suppressed."""
    import matplotlib
    matplotlib.use("Agg")
    ns = {}
    for i, c in enumerate(CELLS):
        if c[0] != "code":
            continue
        try:
            exec(compile(c[1], f"<cell {i}>", "exec"), ns)
        except Exception as e:  # noqa: BLE001
            print(f"cell {i} FAILED: {type(e).__name__}: {e}")
            raise SystemExit(1)
    print("all solution cells executed")


if __name__ == "__main__":
    (HERE / "lab_01_student.ipynb").write_text(json.dumps(build(True), indent=1))
    (HERE / "lab_01_solution.ipynb").write_text(json.dumps(build(False), indent=1))
    print("wrote lab_01_student.ipynb, lab_01_solution.ipynb")
    if "--check" in sys.argv:
        check()
