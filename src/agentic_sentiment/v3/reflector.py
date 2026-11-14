"""Reflector (LR on agents' log-probabilities, as the main stack) with a fallback for reviews where the Router skips the probe,
the Router threshold search, and the shared loader of stored agent probabilities. Main stack SH3 = probe + Evidence + DeBERTa
(LOG.md 2026-11-10); with B=None everything reduces to the previous two-agent stack SH."""
import json
from dataclasses import dataclass
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

EPS = 1e-6
GRID = np.round(np.arange(0.30, 0.96, 0.05), 2)


def logp(p) -> np.ndarray:
    return np.log(np.clip(np.asarray(p, float), EPS, 1))


def feats(P, E, B=None) -> np.ndarray:
    """Full-stack features: [log probe(5), log evidence(5)(, log deberta(5))]."""
    return np.c_[logp(P), logp(E)] if B is None else np.c_[logp(P), logp(E), logp(B)]


def fb_feats(E, B=None) -> np.ndarray:
    """Fallback features (probe skipped): [log evidence(5)(, log deberta(5))]."""
    return logp(E) if B is None else np.c_[logp(E), logp(B)]


@dataclass
class Reflector:
    full: LogisticRegression       # features feats(P, E[, B]), labels 0-4
    fallback: LogisticRegression   # features fb_feats(E[, B])

    def combine(self, probe_p, evid_p, deb_p=None):
        three = self.full.n_features_in_ == 15
        assert three == (deb_p is not None), "DeBERTa probabilities must be given exactly when the Reflector was fit with them"
        B = None if deb_p is None else np.asarray(deb_p)[None]
        if probe_p is None:
            x = fb_feats(np.asarray(evid_p)[None], B); P = self.fallback.predict_proba(x)[0]
            parts = {"evidence": slice(0, 5)} | ({"deberta": slice(5, 10)} if three else {}); names = ["probe"]
        else:
            x = feats(np.asarray(probe_p)[None], np.asarray(evid_p)[None], B); P = self.full.predict_proba(x)[0]
            parts = {"probe": slice(0, 5), "evidence": slice(5, 10)} | ({"deberta": slice(10, 15)} if three else {}); names = []
        m = self.full if probe_p is not None else self.fallback; c = m.coef_[int(P.argmax())] * x[0]
        w = {k: abs(c[s].sum()) for k, s in parts.items()}; t = sum(w.values()) or 1.0
        shares = {k: 0.0 for k in names} | {k: v / t for k, v in w.items()}
        return int(P.argmax()) + 1, P, shares


def fit_reflector(P, E, stars, B=None) -> Reflector:
    y = np.asarray(stars) - 1
    return Reflector(LogisticRegression(C=1, max_iter=3000).fit(feats(P, E, B), y),
                     LogisticRegression(C=1, max_iter=3000).fit(fb_feats(E, B), y))


def choose_tau(P, E, stars, grid=None, tol=0.005, seed=0, B=None):
    """Smallest tau whose routed 5-fold CV accuracy is within tol of the full stack (skip the probe when max Evidence prob >= tau)."""
    grid = GRID if grid is None else grid; y = np.asarray(stars)
    full_ok = np.zeros(len(y), bool); routed = {t: np.zeros(len(y), bool) for t in grid}
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(E, y):
        b = (lambda i: None) if B is None else (lambda i: B[i])
        r = fit_reflector(P[tr], E[tr], y[tr], b(tr))
        fp = r.full.predict(feats(P[te], E[te], b(te))) + 1; bp = r.fallback.predict(fb_feats(E[te], b(te))) + 1
        full_ok[te] = fp == y[te]; conf = E[te].max(1)
        for t in grid:
            routed[t][te] = np.where(conf >= t, bp, fp) == y[te]
    base = float(full_ok.mean())
    rows = [(float(t), float(routed[t].mean()), float((E.max(1) >= t).mean())) for t in grid]
    ok = [t for t, acc, _ in rows if acc >= base - tol]
    return (min(ok) if ok else None), base, rows


def load_agents(split, three=True):
    """Stored agent probabilities for a split: ids, probe P, Evidence E, DeBERTa B (None if three=False), stars y. Row order = probe file."""
    from agentic_sentiment.v3 import paths as PT
    G = lambda p: {x["review_id"]: x for x in map(json.loads, open(f"{PT.D}{p}_{split}_probs.jsonl"))}
    a, e = G(PT.PROBE_PREFIX), G("evid6b"); b = G(PT.DEB_PREFIX) if three else None; ids = list(a)
    assert set(ids) == set(e) and (b is None or set(ids) == set(b)), f"{split}: review ids differ between agents"
    arr = lambda D: np.array([D[i]["probs"] for i in ids])
    return ids, arr(a), arr(e), (arr(b) if three else None), np.array([a[i]["gt_rating"] for i in ids])
