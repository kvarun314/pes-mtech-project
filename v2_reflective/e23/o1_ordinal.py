"""O1 (declared LOG.md 2026-11-02): main stack vs Phase 1 on ordinal metrics (quadratic weighted kappa, MAE), eval and natural_test. Frozen predictions; no selection.
usage: o1_ordinal.py   (tee to o1_ordinal.out; writes o1_ordinal.json)"""
import json, sys, numpy as np, joblib
from sklearn.metrics import cohen_kappa_score
sys.path.insert(0, "src")
from agentic_sentiment.v3 import paths as PT
from agentic_sentiment.v3.reflector import logp
full = joblib.load(PT.REFLECTOR_JOBLIB)["full"]
L = lambda f: {x["review_id"]: x for x in map(json.loads, open(PT.D + f))}
qwk = lambda y, p: cohen_kappa_score(y, p, weights="quadratic")
mae = lambda y, p: float(np.abs(y - p).mean())
rng = np.random.default_rng(0); res = {}
for s in ("eval", "natural_test"):
    P, E, R = L(f"evid7_{s}_probs.jsonl"), L(f"evid6b_{s}_probs.jsonl"), L(f"p1/probs_R_{s}.jsonl"); ids = list(P)
    y = np.array([P[i]["gt_rating"] for i in ids])
    sh = full.predict(np.c_[logp([P[i]["probs"] for i in ids]), logp([E[i]["probs"] for i in ids])]) + 1
    p1 = np.array([R[i]["probs"] for i in ids]).argmax(1) + 1
    B = [rng.integers(0, len(y), len(y)) for _ in range(1000)]
    ci = lambda f: [float(x) for x in np.percentile([f(y[b], sh[b]) - f(y[b], p1[b]) for b in B], [2.5, 97.5])]
    res[s] = {"n": len(y), "exact_SH": float((sh == y).mean()), "exact_P1": float((p1 == y).mean()),
              "qwk_SH": float(qwk(y, sh)), "qwk_P1": float(qwk(y, p1)), "qwk_diff_ci": ci(qwk),
              "mae_SH": mae(y, sh), "mae_P1": mae(y, p1), "mae_diff_ci": ci(mae)}
    print(s, json.dumps(res[s], indent=1), flush=True)
json.dump(res, open(PT.D + "o1_ordinal.json", "w"), indent=1)
