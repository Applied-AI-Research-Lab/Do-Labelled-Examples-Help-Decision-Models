from __future__ import annotations

import numpy as np


def paired_bootstrap(stat_fn, idx_n: int, n_boot: int = 10000, seed: int = 0):
    rng = np.random.default_rng(seed)
    point = stat_fn(np.arange(idx_n))
    boots = np.empty(n_boot)
    for b in range(n_boot):
        boots[b] = stat_fn(rng.integers(0, idx_n, idx_n))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def mcnemar_exact(correct_a, correct_b) -> float:
    from scipy.stats import binomtest
    a, b = np.asarray(correct_a, bool), np.asarray(correct_b, bool)
    n01, n10 = int(np.sum(a & ~b)), int(np.sum(~a & b))
    if n01 + n10 == 0:
        return 1.0
    return float(binomtest(n01, n01 + n10, 0.5).pvalue)


def holm(pvals: list[float]) -> list[float]:
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj.tolist()


def tost_equivalent(lo90: float, hi90: float, margin: float) -> bool:
    return (lo90 > -margin) and (hi90 < margin)


def bootstrap_diff_ci(values_a, values_b, n_boot=10000, seed=0, level=0.95):
    a, b = np.asarray(values_a, float), np.asarray(values_b, float)
    rng = np.random.default_rng(seed)
    n = len(a)
    d = a - b
    boots = np.array([d[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    q = (1 - level) / 2 * 100
    lo, hi = np.percentile(boots, [q, 100 - q])
    return float(d.mean()), float(lo), float(hi)
