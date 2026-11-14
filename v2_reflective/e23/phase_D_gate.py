"""Phase D gate (declared LOG.md 2026-10-22). DEV-ONLY."""
import json, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
D = "v2_reflective/e23/"
G = lambda pref, s: {x["review_id"]: x for x in map(json.loads, open(f"{D}{pref}_{s}_probs.jsonl" if pref.startswith("evid") else f"{D}p1/probs_{pref}_{s}.jsonl"))}
def feats(s):
    ids = list(G("R", s)); y = np.array([G("R", s)[i]["gt_rating"] - 1 for i in ids])
    return y, np.hstack([np.log(np.clip([G(a, s)[i]["probs"] for i in ids], 1e-6, 1)) for a in ("R", "evid")])
yd, Xd = feats("dev"); yn, Xn = feats("natural_dev")
X = np.vstack([Xd, Xn]); y = np.r_[yd, yn]; sp = np.r_[np.zeros(len(yd)), np.ones(len(yn))].astype(int)
strat = sp * 5 + y
R = {k: [] for k in ("S0", "D1", "D2")}
def sc(t, p, m): return ((p[m] == t[m]).mean(), f1_score(t[m], p[m], average="macro"))
for rep in range(3):
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=rep).split(X, strat):
        tr0 = tr[sp[tr] == 0]
        m0 = LogisticRegression(C=1, max_iter=3000).fit(X[tr0], y[tr0]); m1 = LogisticRegression(C=1, max_iter=3000).fit(X[tr], y[tr])
        P1 = m1.predict_proba(X[te]); preds = {"S0": m0.predict(X[te]), "D1": P1.argmax(1), "D2": np.clip(np.rint(P1 @ np.arange(5)), 0, 4).astype(int)}
        for k, p in preds.items():
            full = np.full(len(y), -1); full[te] = p; t = y.copy()
            msk = np.zeros(len(y), bool); msk[te] = True
            R[k].append([sc(t, full, msk & (sp == s)) for s in (0, 1)])
out = {}
for k, v in R.items():
    a = np.array(v)  # reps*folds x split x (exact,f1)
    out[k] = a.mean(0); print(f"{k}: dev exact {out[k][0,0]:.3f} f1 {out[k][0,1]:.3f} | natural_dev exact {out[k][1,0]:.3f} f1 {out[k][1,1]:.3f} | mean exact {out[k][:,0].mean():.3f} f1 {out[k][:,1].mean():.3f}")
base = out["S0"]
ok = {k: out[k][:, 0].mean() >= base[:, 0].mean() + .005 and out[k][:, 1].mean() >= base[:, 1].mean() for k in ("D1", "D2")}
print("=== GATE", ok, "->", "PASS" if any(ok.values()) else "FAIL")
