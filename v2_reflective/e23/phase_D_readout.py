"""Phase D single readout (declared LOG.md 2026-10-22). Fit on pooled dev+natural_dev, read eval + natural_test once."""
import json, numpy as np
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
D = "v2_reflective/e23/"
G = lambda pref, s: {x["review_id"]: x for x in map(json.loads, open(f"{D}{pref}_{s}_probs.jsonl" if pref.startswith("evid") else f"{D}p1/probs_{pref}_{s}.jsonl"))}
def feats(s):
    ids = list(G("R", s)); y = np.array([G("R", s)[i]["gt_rating"] - 1 for i in ids])
    return y, np.hstack([np.log(np.clip([G(a, s)[i]["probs"] for i in ids], 1e-6, 1)) for a in ("R", "evid")])
yd, Xd = feats("dev"); yn, Xn = feats("natural_dev")
m0 = LogisticRegression(C=1, max_iter=3000).fit(Xd, yd)
m1 = LogisticRegression(C=1, max_iter=3000).fit(np.vstack([Xd, Xn]), np.r_[yd, yn])
pol = lambda a: np.where(a <= 1, 0, np.where(a == 2, 1, 2))
rng = np.random.default_rng(0)
for s in ("eval", "natural_test"):
    y, X = feats(s); p0 = m0.predict(X); p1 = np.clip(np.rint(m1.predict_proba(X) @ np.arange(5)), 0, 4).astype(int)
    c0, c1 = p0 == y, p1 == y; q0, q1 = pol(p0) == pol(y), pol(p1) == pol(y)
    def ci(a, b):
        d = [(a[i] .mean() - b[i].mean()) for i in (rng.integers(0, len(a), len(a)) for _ in range(3000))]; return np.percentile(d, [2.5, 97.5]) * 100
    fx, rg = int((c1 & ~c0).sum()), int((~c1 & c0).sum()); p = binomtest(fx, fx + rg, .5).pvalue if fx + rg else 1
    print(f"{s}: S0 exact {c0.mean()*100:.1f} F1 {f1_score(y,p0,average='macro'):.3f} pol {q0.mean()*100:.1f} | D2 exact {c1.mean()*100:.1f} F1 {f1_score(y,p1,average='macro'):.3f} pol {q1.mean()*100:.1f}")
    print(f"   D2-S0 exact {100*(c1.mean()-c0.mean()):+.1f} CI {ci(c1,c0).round(1)} fixed {fx} regressed {rg} McNemar p {p:.2g}; polarity {100*(q1.mean()-q0.mean()):+.1f} CI {ci(q1,q0).round(1)}")
    print("   per-class recall S0", [round(float((p0[y==k]==k).mean()),2) for k in range(5)], "D2", [round(float((p1[y==k]==k).mean()),2) for k in range(5)])
