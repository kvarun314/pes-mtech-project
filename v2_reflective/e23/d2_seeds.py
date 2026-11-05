"""D2 (declared LOG.md 2026-11-05): DeBERTa seed variance, seeds 0, 1, 2. Reporting only; selects nothing.
Needs deb3_<split>_probs.jsonl (seed 0) and deb3s<seed>_<split>_probs.jsonl in v2_reflective/e23/.
usage: d2_seeds.py   (tee to d2_seeds.out; writes d2_seeds.json)"""
import json, sys, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
sys.path.insert(0, "src")
from agentic_sentiment.v3 import paths as PT
from agentic_sentiment.v3.reflector import logp
L = lambda f: {x["review_id"]: x for x in map(json.loads, open(PT.D + f))}
LR = lambda: LogisticRegression(C=1, max_iter=3000)
SEEDS = (0, 1, 2); TESTS = ("eval", "natural_test")


def load(s, seed):
    P, E, B = L(f"evid7_{s}_probs.jsonl"), L(f"evid6b_{s}_probs.jsonl"), L(f"deb3{'s' + str(seed) if seed else ''}_{s}_probs.jsonl"); ids = list(P)
    assert set(ids) == set(E) == set(B), f"{s} seed {seed}: review ids differ"
    g = lambda D: logp([D[i]["probs"] for i in ids]); pb = np.array([B[i]["probs"] for i in ids])
    return np.c_[g(P), g(E)], np.c_[g(P), g(E), g(B)], np.array([P[i]["gt_rating"] for i in ids]), pb.argmax(1) + 1


res = {}
for seed in SEEDS:
    X2, X3, y, deb = load("dev2", seed); ok2, ok3 = np.zeros(len(y), bool), np.zeros(len(y), bool)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X2, y):
        ok2[te] = LR().fit(X2[tr], y[tr]).predict(X2[te]) == y[te]; ok3[te] = LR().fit(X3[tr], y[tr]).predict(X3[te]) == y[te]
    r = {"dev2_cv": {"alone": float((deb == y).mean()), "SH": float(ok2.mean()), "with": float(ok3.mean())}}; m2, m3 = LR().fit(X2, y), LR().fit(X3, y)
    for s in ("natural_dev",) + TESTS:
        T2, T3, yt, dt = load(s, seed); a2, a3 = m2.predict(T2) == yt, m3.predict(T3) == yt
        r[s] = {"alone": float((dt == yt).mean()), "SH": float(a2.mean()), "with": float(a3.mean()), "fixed": int((a3 & ~a2).sum()), "broken": int((a2 & ~a3).sum())}
    res[seed] = r
out = {"per_seed": res, "summary": {}}
for s in ("dev2_cv", "natural_dev") + TESTS:
    d = np.array([res[k][s]["with"] - res[k][s]["SH"] for k in SEEDS]); A = np.array([res[k][s]["alone"] for k in SEEDS]); W = np.array([res[k][s]["with"] for k in SEEDS])
    out["summary"][s] = {"SH": res[0][s]["SH"], "alone_mean": float(A.mean()), "alone_sd": float(A.std(ddof=1)), "with_mean": float(W.mean()), "with_sd": float(W.std(ddof=1)),
                         "diff_per_seed": [float(x) for x in d], "diff_mean": float(d.mean()), "diff_sd": float(d.std(ddof=1)),
                         "stable": bool((np.all(d > 0) or np.all(d < 0)) and abs(d.mean()) >= 2 * d.std(ddof=1))}
    print(s, json.dumps(out["summary"][s]))
for k in SEEDS: print("seed", k, {s: (res[k][s]["fixed"], res[k][s]["broken"]) for s in TESTS})
json.dump(out, open(PT.D + "d2_seeds.json", "w"), indent=1)
