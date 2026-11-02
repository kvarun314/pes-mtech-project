"""G1 and N1 gate and readout (declared LOG.md 2026-11-02). Test sets are read only on PASS.
usage: gn_gate.py g1|n1   (tee to gn_gate_<which>.out; writes gn_gate_<which>.json)"""
import json, sys, numpy as np
from scipy.stats import binomtest
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
sys.path.insert(0, "src")
from agentic_sentiment.v3 import paths as PT
from agentic_sentiment.v3.reflector import logp
L = lambda f: {x["review_id"]: x for x in map(json.loads, open(PT.D + f))}
LR = lambda: LogisticRegression(C=1, max_iter=3000)
GB = lambda: HistGradientBoostingClassifier(max_iter=200, learning_rate=.05, max_depth=3, random_state=0)
which = sys.argv[1]; CANDS = {"g1": {"G1": (GB, ("evid7", "evid6b"))}, "n1": {"N1_replace": (LR, ("evid7", "nb1")), "N1_join": (LR, ("evid7", "evid6b", "nb1"))}}[which]
BASE = (LR, ("evid7", "evid6b"))


def load(s, agents):
    F = [L(f"{a}_{s}_probs.jsonl") for a in agents]; ids = list(F[0]); assert all(set(ids) == set(f) for f in F), s
    return np.hstack([logp([f[i]["probs"] for i in ids]) for f in F]), np.array([F[0][i]["gt_rating"] for i in ids])


def oof(cfg):
    X, y = load("dev2", cfg[1]); ok = np.zeros(len(y), bool)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y): ok[te] = cfg[0]().fit(X[tr], y[tr]).predict(X[te]) == y[te]
    return ok


def boot(a, b):
    rng = np.random.default_rng(0); d = a.astype(float) - b; s = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(1000)]
    return float(d.mean()), float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


fit = lambda cfg: cfg[0]().fit(*load("dev2", cfg[1]))
base_ok = oof(BASE); cv = {n: oof(c) for n, c in CANDS.items()}; name = max(cv, key=lambda n: cv[n].mean()); cand = CANDS[name]
d, lo, hi = boot(cv[name], base_ok); mb, mc = fit(BASE), fit(cand)
Xb, yn = load("natural_dev", BASE[1]); Xc, _ = load("natural_dev", cand[1])
g = {"dev2_cv_SH": float(base_ok.mean()), "dev2_cv": {n: float(a.mean()) for n, a in cv.items()}, "chosen": name, "dev2_cv_diff": d, "dev2_cv_ci": [lo, hi],
     "natural_dev_SH": float((mb.predict(Xb) == yn).mean()), "natural_dev_cand": float((mc.predict(Xc) == yn).mean())}
g["pass"] = bool(d >= 0.010 and lo > 0 and g["natural_dev_cand"] >= g["natural_dev_SH"])
res = {"gate": g}; print("GATE", json.dumps(g, indent=1))
if g["pass"]:
    res["readout"] = {}
    for s in ("eval", "natural_test"):
        Tb, yt = load(s, BASE[1]); Tc, _ = load(s, cand[1]); pb, pc = mb.predict(Tb), mc.predict(Tc); a, b = pc == yt, pb == yt
        f, br = int((a & ~b).sum()), int((b & ~a).sum())
        r = {"SH": float(b.mean()), "cand": float(a.mean()), "diff_ci": boot(a, b), "fixed": f, "broken": br, "mcnemar_p": binomtest(f, f + br).pvalue if f + br else 1.0, "within1_cand": float((np.abs(pc - yt) <= 1).mean())}
        res["readout"][s] = r; print(s, json.dumps(r, indent=1))
else:
    print("FAIL: no test read")
json.dump(res, open(PT.D + f"gn_gate_{which}.json", "w"), indent=1)
