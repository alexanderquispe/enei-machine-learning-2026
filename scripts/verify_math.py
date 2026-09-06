"""Recompute every numerical example printed in the ML I slides and compare with the slide.

Each check is tagged in the .tex source with a comment  `% verify: <tag>`  on the frame that
prints the numbers. Run from anywhere:

    python scripts/verify_math.py

Exit status is non-zero if any check fails. Only numpy is needed.
"""
import sys
import numpy as np

FAILURES = []


def check(tag, computed, slide, atol=1e-3):
    computed, slide = np.asarray(computed, float), np.asarray(slide, float)
    ok = np.allclose(computed, slide, atol=atol)
    print(f"  [{'ok' if ok else 'FAIL'}] {tag:38s} slide={np.round(slide, 4).tolist()}  computed={np.round(computed, 4).tolist()}")
    if not ok:
        FAILURES.append(tag)


# ---------------------------------------------------------------- Lecture 1
def L1_least_squares_example():
    x = np.array([1, 2, 3, 4, 5.0]); y = np.array([2.2, 2.8, 4.5, 3.7, 5.5])
    X = np.column_stack([np.ones(5), x])
    check("L1 X'X", X.T @ X, [[5, 15], [15, 55]])
    check("L1 X'y", X.T @ y, [18.7, 63.6])
    check("L1 det(X'X)", np.linalg.det(X.T @ X), 50)
    beta = np.linalg.solve(X.T @ X, X.T @ y)
    check("L1 beta_hat", beta, [1.49, 0.75])
    r = y - X @ beta
    check("L1 residuals", r, [-0.04, -0.19, 0.76, -0.79, 0.26], atol=5e-3)
    check("L1 residuals sum to 0", r.sum(), 0, atol=1e-10)
    H = X @ np.linalg.solve(X.T @ X, X.T)
    check("L1 tr(H) = p+1", np.trace(H), 2, atol=1e-10)
    check("L1 H idempotent", np.abs(H @ H - H).max(), 0, atol=1e-10)


# ---------------------------------------------------------------- Lecture 2
def L2_perceptron_trace():
    pts = np.array([[1, 1, 1], [3, 3, 1], [0, 2, 1], [4, 1, 1]], float); ys = np.array([-1, 1, -1, 1])
    w = np.zeros(3); mistakes = 0; passes = 0
    while True:
        changed = False; passes += 1
        for xt, yt in zip(pts, ys):
            if yt * (w @ xt) <= 0:
                w = w + yt * xt; mistakes += 1; changed = True
        if not changed or passes > 50:
            break
    check("L2 perceptron final w", w, [3, -1, -3])
    check("L2 perceptron #mistakes", mistakes, 7)
    check("L2 perceptron passes to converge", passes, 4)      # 3 passes with mistakes + 1 clean pass
    check("L2 R^2 = max ||x||^2", (pts**2).sum(1).max(), 19)
    check("L2 final scores", pts @ w, [-1, 3, -5, 8])


def L2_irls_step():
    X = np.array([[1, -1], [1, 0], [1, 1.0]]); y = np.array([0, 0, 1.0])
    beta = np.zeros(2); p = 1 / (1 + np.exp(-X @ beta))
    W = np.diag(p * (1 - p)); z = X @ beta + np.linalg.solve(W, y - p)
    check("L2 IRLS working response z", z, [-2, -2, 2])
    check("L2 IRLS X'WX", X.T @ W @ X, [[0.75, 0], [0, 0.5]])
    check("L2 IRLS X'Wz", X.T @ W @ z, [-0.5, 1])
    b1 = np.linalg.solve(X.T @ W @ X, X.T @ W @ z)
    check("L2 IRLS beta^(1)", b1, [-2 / 3, 2])
    grad = X.T @ (y - p); newton = beta + np.linalg.solve(-(-X.T @ W @ X), grad)
    check("L2 Newton form agrees", newton, b1, atol=1e-12)
    p1 = 1 / (1 + np.exp(-X @ b1))
    check("L2 probabilities after one step", p1, [0.065, 0.339, 0.791], atol=2e-3)


# ---------------------------------------------------------------- Lecture 3
def L3_soft_threshold_example():
    b = np.array([3, 0.5, -2.0]); lam = 1.0
    check("L3 ridge b/(1+lam)", b / (1 + lam), [1.5, 0.25, -1])
    check("L3 lasso soft threshold", np.sign(b) * np.maximum(np.abs(b) - lam, 0), [2, 0, -1])


# ---------------------------------------------------------------- Lecture 4
def _entropy(c):
    p = np.asarray(c, float); p = p / p.sum(); p = p[p > 0]; return float(-(p * np.log2(p)).sum())
def _gini(c):
    p = np.asarray(c, float); p = p / p.sum(); return float(1 - (p**2).sum())
def _mis(c):
    p = np.asarray(c, float); p = p / p.sum(); return float(1 - p.max())
def _after(children, I):
    n = sum(sum(c) for c in children); return sum(sum(c) / n * I(c) for c in children)


def L4_gini_vs_misclass():
    A = [[300, 100], [100, 300]]; B = [[200, 400], [200, 0]]
    check("L4 misclass after A / B", [_after(A, _mis), _after(B, _mis)], [0.25, 0.25])
    check("L4 gini after A / B", [_after(A, _gini), _after(B, _gini)], [0.375, 0.333])
    check("L4 entropy after A / B", [_after(A, _entropy), _after(B, _entropy)], [0.811, 0.689])


def L4_information_gain_example():
    S, L, R = [8, 4], [6, 1], [2, 3]
    check("L4 entropy S, L, R", [_entropy(S), _entropy(L), _entropy(R)], [0.9183, 0.5917, 0.9710], atol=1e-4)
    check("L4 entropy gain", _entropy(S) - _after([L, R], _entropy), 0.1685, atol=1e-4)
    check("L4 gini S, L, R", [_gini(S), _gini(L), _gini(R)], [0.4444, 0.2449, 0.48], atol=1e-4)
    check("L4 gini gain", _gini(S) - _after([L, R], _gini), 0.1016, atol=1e-4)
    check("L4 misclass gain", _mis(S) - _after([L, R], _mis), 0.0833, atol=1e-4)


# ---------------------------------------------------------------- Lecture 5
def L5_adaboost_trace():
    x = np.array([1, 2, 3, 4, 5.0]); y = np.array([1, 1, -1, -1, 1.0])
    stumps = [(s, t) for t in [1.5, 2.5, 3.5, 4.5] for s in (1, -1)]
    H = lambda s, t: s * np.where(x > t, 1, -1)
    w = np.ones(5) / 5; f = np.zeros(5); eps, al, Z, W = [], [], [], []
    for _ in range(3):
        errs = [(w * (H(s, t) != y)).sum() for s, t in stumps]
        k = int(np.argmin(errs)); s, t = stumps[k]; e = errs[k]; a = 0.5 * np.log((1 - e) / e)
        u = w * np.exp(-a * y * H(s, t)); z = u.sum(); w = u / z
        f += a * H(s, t); eps.append(e); al.append(a); Z.append(z); W.append(w.copy())
    check("L5 eps_m", eps, [0.2, 0.25, 1 / 3])
    check("L5 alpha_m", al, [0.693, 0.549, 0.347])
    check("L5 Z_m", Z, [0.800, 0.866, 0.943])
    check("L5 Z_m = 2 sqrt(eps(1-eps))", Z, [2 * np.sqrt(e * (1 - e)) for e in eps], atol=1e-12)
    check("L5 weights after round 1", W[0], [0.125, 0.125, 0.125, 0.125, 0.5])
    check("L5 weights after round 2", W[1], [0.25, 0.25, 0.083, 0.083, 0.333])
    check("L5 f_3 on x=1..5", f, [0.491, 0.491, -1.589, -1.589, -0.491])
    check("L5 training error", np.mean(np.sign(f) != y), 0.2)
    check("L5 prod Z", np.prod(Z), 0.653)
    check("L5 mean exp-loss = prod Z", np.mean(np.exp(-y * f)), np.prod(Z), atol=1e-12)


# ---------------------------------------------------------------- Lecture 6
def L6_kmeans_example():
    P = np.array([[1, 1], [1.5, 2], [3, 3.5], [5, 7], [3.5, 5], [4.5, 5.0]])
    mu = np.array([P[0], P[3]])
    d2 = ((P[:, None] - mu[None]) ** 2).sum(-1)
    check("L6 d^2 to mu1 (pass 1)", d2[:, 0], [0, 1.25, 10.25, 52, 22.25, 28.25])
    check("L6 d^2 to mu2 (pass 1)", d2[:, 1], [52, 37.25, 16.25, 0, 6.25, 4.25])
    c = d2.argmin(1); J1 = d2[np.arange(6), c].sum()
    check("L6 J after first assignment", J1, 22)
    mu = np.array([P[c == k].mean(0) for k in range(2)])
    check("L6 updated centroids", mu, [[1.833, 2.167], [4.333, 5.667]])
    d2 = ((P[:, None] - mu[None]) ** 2).sum(-1); c2 = d2.argmin(1)
    check("L6 assignments unchanged", c2, c)
    check("L6 J at convergence", d2[np.arange(6), c2].sum(), 9.167)


if __name__ == "__main__":
    checks = [L1_least_squares_example, L2_perceptron_trace, L2_irls_step, L3_soft_threshold_example,
              L4_gini_vs_misclass, L4_information_gain_example, L5_adaboost_trace, L6_kmeans_example]
    for fn in checks:
        print(fn.__name__)
        fn()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) FAILED: {FAILURES}"); sys.exit(1)
    print(f"all numerical examples verified ({sum(1 for _ in checks)} tagged frames)")
