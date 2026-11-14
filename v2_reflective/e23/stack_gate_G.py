"""Phase G2 gate (declared LOG.md 2026-10-24). DEV-ONLY: S0 A+E vs SG A+E6, 5-fold CV x3 on dev and natural_dev."""
import json, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
D = "v2_reflective/e23/"
G = lambda pref, s: {x["review_id"]: x for x in map(json.loads, open(f"{D}{pref}_{s}_probs.jsonl" if pref.startswith("evid") else f"{D}p1/probs_{pref}_{s}.jsonl"))}
def feats(s, ag):
    ids = list(G("R", s)); y = np.array([G("R", s)[i]["gt_rating"] - 1 for i in ids]); return y, np.hstack([np.log(np.clip([G(a, s)[i]["probs"] for i in ids], 1e-6, 1)) for a in ag])
sets = {"S0 A+E": ["R", "evid"], "SG A+E6": ["R", "evid6"], "E alone": ["evid"], "E6 alone": ["evid6"]}; res = {}
for s in ("dev", "natural_dev"):
    print(f"=== {s} 5-fold CV (3 repeats)")
    for n, ag in sets.items():
        y, X = feats(s, ag); ex, f1 = [], []
        for rep in range(3):
            p = cross_val_predict(LogisticRegression(C=1, max_iter=3000), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=rep)); ex.append((p == y).mean()); f1.append(f1_score(y, p, average="macro"))
        res[(s, n)] = (np.mean(ex), np.mean(f1)); print(f"{n:9s} exact {np.mean(ex):.3f} macroF1 {np.mean(f1):.3f}")
a, b, c, d = res[("dev", "S0 A+E")], res[("dev", "SG A+E6")], res[("natural_dev", "S0 A+E")], res[("natural_dev", "SG A+E6")]
ok = b[0] >= a[0] + .01 and d[0] >= c[0] and b[1] >= a[1] and d[1] >= c[1]
print("=== GATE (dev exact +.010, natural_dev exact >= S0, macroF1 >= S0 on both):", "PASS" if ok else "FAIL")
