from __future__ import annotations

import numpy as np


def accuracy(P, y):
    return float(np.mean(P.argmax(1) == y))


def macro_f1(P, y, k=None):
    from sklearn.metrics import f1_score
    k = k or P.shape[1]
    return float(f1_score(y, P.argmax(1), average="macro", labels=list(range(k)), zero_division=0))


def ece_equal_mass(P, y, n_bins: int = 15) -> float:
    conf = P.max(1)
    correct = (P.argmax(1) == y).astype(float)
    order = np.argsort(conf)
    bins = np.array_split(order, n_bins)
    n = len(conf)
    return float(sum(len(b) / n * abs(correct[b].mean() - conf[b].mean()) for b in bins if len(b)))


def brier(P, y):
    Y = np.zeros_like(P)
    Y[np.arange(len(y)), y] = 1.0
    return float(np.mean(np.sum((P - Y) ** 2, axis=1)))


def nll(P, y, eps=1e-12):
    return float(-np.mean(np.log(np.clip(P[np.arange(len(y)), y], eps, 1.0))))


def risk_coverage(P, y):
    conf = P.max(1)
    err = (P.argmax(1) != y).astype(float)
    order = np.argsort(-conf, kind="stable")
    cum_err = np.cumsum(err[order])
    k = np.arange(1, len(y) + 1)
    return k / len(y), cum_err / k


def aurc(P, y):
    _, risk = risk_coverage(P, y)
    return float(risk.mean())


def coverage_at_risk(P, y, max_risk=0.02):
    cov, risk = risk_coverage(P, y)
    ok = np.where(risk <= max_risk)[0]
    return float(cov[ok].max()) if len(ok) else 0.0


def flip_rate(labels_a, labels_b):
    a, b = np.asarray(labels_a), np.asarray(labels_b)
    return float(np.mean(a != b))


def total_variation(P1, P2):
    return float(np.mean(0.5 * np.abs(P1 - P2).sum(1)))


def auroc(scores, positives):
    from sklearn.metrics import roc_auc_score
    positives = np.asarray(positives).astype(int)
    if positives.min() == positives.max():
        return float("nan")
    return float(roc_auc_score(positives, scores))


def summary(P, y) -> dict:
    return {
        "n": int(len(y)), "accuracy": accuracy(P, y), "macro_f1": macro_f1(P, y),
        "ece": ece_equal_mass(P, y), "brier": brier(P, y), "nll": nll(P, y),
        "aurc": aurc(P, y), "cov_at_2pct_risk": coverage_at_risk(P, y, 0.02),
    }
