"""Q6 (declared LOG.md 2026-11-23): disagreement guard. Below the bar and probe >= 2 stars from the Reflector -> a person, not a range.
Gate on dev2 (out-of-fold, as Q1) and natural_dev; test readout only on PASS.  usage: q6_disagree_guard.py (tee to .out; writes .json)"""
import json, sys, numpy as np, joblib
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
sys.path.insert(0, "src")
from agentic_sentiment.v3 import paths as PT
from agentic_sentiment.v3.reflector import feats, load_agents
from agentic_sentiment.v3.conformal import qhat, star_range
full = joblib.load(PT.REFLECTOR_JOBLIB)["full"]
htau = [x for x in json.load(open(PT.D + "s1_selective.json"))["SH"]["eval"] if x["target"] == 0.4][0]["threshold"]
GAP = 2


def load(s):
    ids, P, E, B, y = load_agents(s); return feats(P, E, B), np.asarray(P), y


def three_way(S, Pp, y, q):
    R = [star_range(p, q) for p in S]; size = np.array([b - a + 1 for a, b in R]); cover = np.array([a <= t <= b for (a, b), t in zip(R, y)])
    star = S.argmax(1) + 1; auto = S.max(1) >= htau; fire = ~auto & (np.abs(Pp.argmax(1) + 1 - star) >= GAP)
    rng0 = ~auto & (size <= 2); rng1 = rng0 & ~fire
    m = lambda a: float(a.mean())
    return {"n": len(y), "fires": int(fire.sum()), "fires_share": m(fire), "auto_share": m(auto), "auto_exact": m((star == y)[auto]),
            "before": {"range_share": m(rng0), "range_coverage": m(cover[rng0]), "human_share": m(~auto & ~rng0)},
            "after": {"range_share": m(rng1), "range_coverage": m(cover[rng1]), "human_share": m(~auto & ~rng1)},
            "fired_ranges": int((rng0 & fire).sum()), "fired_range_coverage": float(cover[rng0 & fire].mean()) if (rng0 & fire).any() else None,
            "other_range_coverage": m(cover[rng1]), "fired_reflector_exact": float((star == y)[fire].mean()) if fire.any() else None,
            "fired_probe_exact": float((Pp.argmax(1) + 1 == y)[fire].mean()) if fire.any() else None}


def ok(r):
    a = r["fired_range_coverage"] is not None and r["fired_range_coverage"] < r["other_range_coverage"]
    return bool(a and r["after"]["human_share"] - r["before"]["human_share"] <= 0.03)


res = {}
X, Pp, y = load("dev2"); S = np.zeros((len(y), 5))
for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):       # same folds as Q1
    S[te] = clone(full).fit(X[tr], y[tr] - 1).predict_proba(X[te])
g = {"dev2": three_way(S, Pp, y, qhat(S, y))}
Xn, Pn, yn = load("natural_dev"); Sn = full.predict_proba(Xn)
g["natural_dev"] = three_way(Sn, Pn, yn, qhat(Sn, yn))
g["pass"] = ok(g["dev2"]) and ok(g["natural_dev"])
res["gate"] = g; print("GATE", json.dumps(g, indent=1))
if g["pass"]:
    res["readout"] = {}
    for cal, test in (("dev2", "eval"), ("natural_dev", "natural_test")):
        Sc, yc = (S, y) if cal == "dev2" else (Sn, yn)
        Xt, Pt, yt = load(test); St = full.predict_proba(Xt)
        r = three_way(St, Pt, yt, qhat(Sc, yc)); res["readout"][test] = r; print(test, json.dumps(r, indent=1))
else:
    print("FAIL: no test read")
json.dump(res, open(PT.D + "q6_disagree_guard.json", "w"), indent=1)
