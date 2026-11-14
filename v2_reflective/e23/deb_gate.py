"""D1 gate and readout (declared LOG.md 2026-10-28): does a DeBERTa-v3 agent add to the main stack (probe + Evidence + Reflector)?
Needs deb3_<split>_probs.jsonl from colab_deberta.py copied into v2_reflective/e23/. Test sets are read only on PASS.
usage: deb_gate.py   (tee to deb_gate.out; writes deb_gate.json)"""
import json, sys, numpy as np
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
sys.path.insert(0, "src")
from agentic_sentiment.v3 import paths as PT
from agentic_sentiment.v3.reflector import logp
L = lambda f: {x["review_id"]: x for x in map(json.loads, open(PT.D + f))}
LR = lambda: LogisticRegression(C=1, max_iter=3000)


def load(s):
    P, E, B = L(f"evid7_{s}_probs.jsonl"), L(f"evid6b_{s}_probs.jsonl"), L(f"deb3_{s}_probs.jsonl"); ids = list(P)
    assert set(ids) == set(E) == set(B), f"{s}: review ids differ between agent files"
    g = lambda D: logp([D[i]["probs"] for i in ids])
    return np.c_[g(P), g(E)], np.c_[g(P), g(E), g(B)], np.array([P[i]["gt_rating"] for i in ids]), np.array([B[i]["probs"] for i in ids]).argmax(1) + 1


def boot(a, b, rng=np.random.default_rng(0)):
    d = a.astype(float) - b; s = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(1000)]
    return float(d.mean()), float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def mcnemar(new, old):
    f, b = int((new & ~old).sum()), int((old & ~new).sum()); return f, b, binomtest(f, f + b).pvalue if f + b else 1.0


res = {}
X2, X3, y, deb = load("dev2")
ok2, ok3 = np.zeros(len(y), bool), np.zeros(len(y), bool)
for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X2, y):
    ok2[te] = LR().fit(X2[tr], y[tr]).predict(X2[te]) == y[te]; ok3[te] = LR().fit(X3[tr], y[tr]).predict(X3[te]) == y[te]
d, lo, hi = boot(ok3, ok2)
m2, m3 = LR().fit(X2, y), LR().fit(X3, y)
N2, N3, yn, debn = load("natural_dev"); nd2, nd3 = m2.predict(N2) == yn, m3.predict(N3) == yn
g = {"deberta_alone_dev2": float((deb == y).mean()), "deberta_alone_natural_dev": float((debn == yn).mean()),
     "dev2_cv_SH": float(ok2.mean()), "dev2_cv_SH_plus_DeBERTa": float(ok3.mean()), "dev2_cv_diff": d, "dev2_cv_ci": [lo, hi],
     "natural_dev_SH": float(nd2.mean()), "natural_dev_SH_plus_DeBERTa": float(nd3.mean())}
g["pass"] = bool(d >= 0.010 and g["natural_dev_SH_plus_DeBERTa"] >= g["natural_dev_SH"])
res["gate"] = g; print("GATE", json.dumps(g, indent=1))
if g["pass"]:
    res["readout"] = {}
    for s in ("eval", "natural_test"):
        T2, T3, yt, debt = load(s); a2, a3 = m2.predict(T2) == yt, m3.predict(T3) == yt; f, b, p = mcnemar(a3, a2)
        r = {"SH": float(a2.mean()), "SH_plus_DeBERTa": float(a3.mean()), "DeBERTa_alone": float((debt == yt).mean()), "diff_ci": boot(a3, a2),
             "fixed": f, "broken": b, "mcnemar_p": p, "within1_SH_plus_DeBERTa": float((np.abs(m3.predict(T3) - yt) <= 1).mean())}
        res["readout"][s] = r; print(s, json.dumps(r, indent=1))
else:
    print("FAIL: no test read")
json.dump(res, open(PT.D + "deb_gate.json", "w"), indent=1)
