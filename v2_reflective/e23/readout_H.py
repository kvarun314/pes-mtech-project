"""Phase H single readout (declared LOG.md 2026-10-24). Stackers fit on dev2; read eval + natural_test once."""
import json, numpy as np
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
D = "v2_reflective/e23/"
G = lambda pref, s: {x["review_id"]: x for x in map(json.loads, open(f"{D}{pref}_{s}_probs.jsonl" if pref.startswith("evid") else f"{D}p1/probs_{pref}_{s}.jsonl"))}
def feats(s, ag):
    k = list(G("R", s)); return np.array([G("R", s)[i]["gt_rating"] - 1 for i in k]), np.hstack([np.log(np.clip([G(a, s)[i]["probs"] for i in k], 1e-6, 1)) for a in ag])
cfg = {"S0 A+E (dev2 fit)": (["R", "evid"], "dev2"), "S0 A+E (dev fit, old headline)": (["R", "evid"], "dev"), "SH P3+E6b (dev2 fit)": (["evid7", "evid6b"], "dev2")}
M = {n: LogisticRegression(C=1, max_iter=3000).fit(*feats(f, ag)[::-1][::-1] if False else (feats(f, ag)[1], feats(f, ag)[0])) for n, (ag, f) in cfg.items()}
pol = lambda a: np.where(a <= 1, 0, np.where(a == 2, 1, 2)); rng = np.random.default_rng(0)
def cmp(a, b, y):
    ca, cb = a == y, b == y; d = ca.astype(float) - cb; ci = np.percentile([d[rng.integers(0, len(d), len(d))].mean() for _ in range(3000)], [2.5, 97.5]) * 100
    fx, rg = int((ca & ~cb).sum()), int((~ca & cb).sum()); pa, pb = pol(a) == pol(y), pol(b) == pol(y); dp = pa.astype(float) - pb
    cip = np.percentile([dp[rng.integers(0, len(dp), len(dp))].mean() for _ in range(3000)], [2.5, 97.5]) * 100
    return f"exact {100*d.mean():+.1f} CI [{ci[0]:+.1f},{ci[1]:+.1f}] fixed {fx} regr {rg} McNemar p {binomtest(fx, fx+rg, .5).pvalue:.2g} | polarity {100*dp.mean():+.1f} CI [{cip[0]:+.1f},{cip[1]:+.1f}]"
for s in ("eval", "natural_test"):
    print(f"===== {s}")
    P = {n: M[n].predict(feats(s, ag)[1]) for n, (ag, f) in cfg.items()}; y = feats(s, ["R"])[0]
    for a in ("R", "evid", "evid6b", "evid7"): P[f"{a} alone"] = feats(s, [a])[1].argmax(1)
    for n, p in P.items():
        print(f"{n:32s} exact {100*(p==y).mean():.1f} macroF1 {f1_score(y,p,average='macro'):.3f} within1 {100*(abs(p-y)<=1).mean():.1f} polarity {100*(pol(p)==pol(y)).mean():.1f} recall {[round(float((p[y==k]==k).mean()),2) for k in range(5)]}")
    h = P["SH P3+E6b (dev2 fit)"]
    for o in ("S0 A+E (dev2 fit)", "S0 A+E (dev fit, old headline)", "evid6b alone", "evid alone", "evid7 alone", "R alone"): print(f"  SH vs {o}: {cmp(h, P[o], y)}")
