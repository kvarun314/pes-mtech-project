"""Split conformal star ranges (score = 1 - p(true star)) for the Hand-off agent: a contiguous span of stars that holds the
true star with probability >= 1 - alpha on exchangeable data (Q1, LOG.md 2026-10-28)."""
import numpy as np


def qhat(P_cal, y_cal, alpha=0.1):
    s = 1 - np.asarray(P_cal)[np.arange(len(y_cal)), np.asarray(y_cal) - 1]; n = len(s)
    return float(np.quantile(s, min(1.0, np.ceil((n + 1) * (1 - alpha)) / n), method="higher"))


def star_range(p, q):
    keep = np.flatnonzero(1 - np.asarray(p) <= q)
    if keep.size == 0:
        keep = np.array([int(np.argmax(p))])
    return int(keep.min()) + 1, int(keep.max()) + 1


def evaluate(P, y, q):
    R = [star_range(p, q) for p in P]
    cover = np.array([a <= t <= b for (a, b), t in zip(R, y)]); size = np.array([b - a + 1 for a, b in R])
    return {"coverage": float(cover.mean()), "mean_size": float(size.mean()), "single_star": float((size == 1).mean())}
