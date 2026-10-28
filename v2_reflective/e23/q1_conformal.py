"""Q1 (declared LOG.md 2026-10-28): conformal star ranges. dev2 gate on out-of-fold main-stack probabilities; test readout only on PASS.
usage: q1_conformal.py   (tee to q1_conformal.out; writes q1_conformal.json)"""
import json, sys, numpy as np, joblib
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
sys.path.insert(0, "src")
from agentic_sentiment.v3 import paths as PT
from agentic_sentiment.v3.reflector import logp
from agentic_sentiment.v3.conformal import qhat, star_range, evaluate
full = joblib.load(PT.REFLECTOR_JOBLIB)["full"]
L = lambda f: {x["review_id"]: x for x in map(json.loads, open(PT.D + f))}
htau = [x for x in json.load(open(PT.D + "s1_selective.json"))["SH"]["eval"] if x["target"] == 0.4][0]["threshold"]


def load(s):
    P, E, R = L(f"evid7_{s}_probs.jsonl"), L(f"evid6b_{s}_probs.jsonl"), L(f"p1/probs_R_{s}.jsonl"); ids = list(P)
    X = np.c_[logp([P[i]["probs"] for i in ids]), logp([E[i]["probs"] for i in ids])]
    return X, np.array([R[i]["probs"] for i in ids]), np.array([P[i]["gt_rating"] for i in ids])


def cv(S, y):
    out = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=2).split(S, y):
        out.append(evaluate(S[te], y[te], qhat(S[tr], y[tr])))
    return {k: float(np.mean([o[k] for o in out])) for k in out[0]}


res = {}
X, P1, y = load("dev2"); S = np.zeros((len(y), 5))
for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
    S[te] = clone(full).fit(X[tr], y[tr] - 1).predict_proba(X[te])
g = {"SH": cv(S, y), "P1": cv(P1, y)}
g["pass"] = bool(0.88 <= g["SH"]["coverage"] <= 0.92 and g["P1"]["mean_size"] - g["SH"]["mean_size"] >= 0.2)
res["gate_dev2"] = g; print("GATE dev2", json.dumps(g, indent=1))
if g["pass"]:
    res["readout"] = {}
    for cal, test in (("dev2", "eval"), ("natural_dev", "natural_test")):
        Xc, P1c, yc = load(cal); Sc = S if cal == "dev2" else full.predict_proba(Xc)
        Xt, P1t, yt = load(test); St = full.predict_proba(Xt)
        qs, qp = qhat(Sc, yc), qhat(P1c, yc)
        r = {"q_SH": qs, "q_P1": qp, "SH": evaluate(St, yt, qs), "P1": evaluate(P1t, yt, qp)}
        R = [star_range(p, qs) for p in St]; size = np.array([b - a + 1 for a, b in R]); cover = np.array([a <= t <= b for (a, b), t in zip(R, yt)])
        auto = St.max(1) >= htau; rng_ = ~auto & (size <= 2); human = ~auto & ~rng_
        r["three_way"] = {"auto_share": float(auto.mean()), "auto_exact": float((St.argmax(1) + 1 == yt)[auto].mean()),
                          "range_share": float(rng_.mean()), "range_coverage": float(cover[rng_].mean()) if rng_.any() else None,
                          "human_share": float(human.mean())}
        res["readout"][test] = r; print(test, json.dumps(r, indent=1))
else:
    print("FAIL: no test read")
json.dump(res, open(PT.D + "q1_conformal.json", "w"), indent=1)
