"""B3 gate (declared LOG.md 2026-10-21). DEV-ONLY: 5-fold CV on dev and on natural_dev. No eval/natural_test touched."""
import json, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
D = "v2_reflective/e23/"
G = lambda pref, s: {x["review_id"]: x for x in map(json.loads, open(f"{D}{pref}_{s}_probs.jsonl" if pref.startswith("evid") else f"{D}p1/probs_{pref}_{s}.jsonl"))}
def feats(s, agents):
    ids = list(G("R", s)); y = np.array([G("R", s)[i]["gt_rating"] - 1 for i in ids]); return y, np.hstack([np.log(np.clip([G(a, s)[i]["probs"] for i in ids], 1e-6, 1)) for a in agents])
sets = {"S0 A+E": ["R", "evid"], "S1 A+E+E2(emb)": ["R", "evid", "evid2"], "S2 A+E+E3(char)": ["R", "evid", "evid3"], "S3 A+E+E2+E3": ["R", "evid", "evid2", "evid3"]}
res = {}
for s in ("dev", "natural_dev"):
    print(f"=== {s} 5-fold CV (3 repeats averaged)")
    for n, ag in sets.items():
        y, X = feats(s, ag); ex, f1 = [], []
        for rep in range(3):
            p = cross_val_predict(LogisticRegression(C=1, max_iter=3000), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=rep)); ex.append((p == y).mean()); f1.append(f1_score(y, p, average="macro"))
        res[(s, n)] = (np.mean(ex), np.mean(f1)); print(f"{n:18s} exact {np.mean(ex):.3f} macroF1 {np.mean(f1):.3f}")
for a in ("evid", "evid2", "evid3"):
    for s in ("dev", "natural_dev"):
        y, X = feats(s, [a]); print(f"single agent {a} on {s}: exact {((X.argmax(1) if False else np.exp(X).argmax(1))==y).mean():.3f}")
print("=== GATE (pass: exact >= S0+0.010 on dev AND >= S0 on natural_dev AND dev macroF1 >= S0)")
for n in list(sets)[1:]:
    ok = res[("dev", n)][0] >= res[("dev", "S0 A+E")][0] + .01 and res[("natural_dev", n)][0] >= res[("natural_dev", "S0 A+E")][0] and res[("dev", n)][1] >= res[("dev", "S0 A+E")][1]
    print(n, "PASS" if ok else "FAIL")
