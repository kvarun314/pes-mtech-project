"""Reflector (LR on agents' log-probabilities, as the main stack) with an Evidence-only fallback, and the Router threshold search."""
from dataclasses import dataclass
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

EPS = 1e-6
GRID = np.round(np.arange(0.30, 0.96, 0.05), 2)


def logp(p) -> np.ndarray:
    return np.log(np.clip(np.asarray(p, float), EPS, 1))


@dataclass
class Reflector:
    full: LogisticRegression       # features [log probe(5), log evidence(5)], labels 0-4
    fallback: LogisticRegression   # features [log evidence(5)]

    def combine(self, probe_p, evid_p):
        if probe_p is None:
            P = self.fallback.predict_proba(logp(evid_p)[None])[0]; shares = {"probe": 0.0, "evidence": 1.0}
        else:
            x = np.r_[logp(probe_p), logp(evid_p)][None]; P = self.full.predict_proba(x)[0]
            c = self.full.coef_[int(P.argmax())] * x[0]; a, b = abs(c[:5].sum()), abs(c[5:].sum()); s = (a + b) or 1.0
            shares = {"probe": a / s, "evidence": b / s}
        return int(P.argmax()) + 1, P, shares


def fit_reflector(P, E, stars) -> Reflector:
    y = np.asarray(stars) - 1
    return Reflector(LogisticRegression(C=1, max_iter=3000).fit(np.c_[logp(P), logp(E)], y),
                     LogisticRegression(C=1, max_iter=3000).fit(logp(E), y))


def choose_tau(P, E, stars, grid=None, tol=0.005, seed=0):
    """Smallest tau whose routed 5-fold CV accuracy is within tol of the full stack (skip the probe when max Evidence prob >= tau)."""
    grid = GRID if grid is None else grid; y = np.asarray(stars)
    full_ok = np.zeros(len(y), bool); routed = {t: np.zeros(len(y), bool) for t in grid}
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(E, y):
        r = fit_reflector(P[tr], E[tr], y[tr])
        fp = r.full.predict(np.c_[logp(P[te]), logp(E[te])]) + 1; bp = r.fallback.predict(logp(E[te])) + 1
        full_ok[te] = fp == y[te]; conf = E[te].max(1)
        for t in grid:
            routed[t][te] = np.where(conf >= t, bp, fp) == y[te]
    base = float(full_ok.mean())
    rows = [(float(t), float(routed[t].mean()), float((E.max(1) >= t).mean())) for t in grid]
    ok = [t for t, acc, _ in rows if acc >= base - tol]
    return (min(ok) if ok else None), base, rows
