"""Phase F2 gate (declared LOG.md 2026-10-22). DEV-ONLY. Pick (feature block, C) by probe-alone dev exact; stack gate once for the pick."""
import json, numpy as np, glob, warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
D = "v2_reflective/e23/"; ids = json.load(open(D + "hid/ids.json"))
H = {s: np.vstack([np.load(f) for f in sorted(glob.glob(f"{D}hid2/{s}_*.npy"))]).astype(np.float32) for s in ids}
Lj = lambda n: {r["review_id"]: r["gt_rating"] for r in map(json.loads, open(f"{D}{n}.jsonl"))}
lab = {"dev": Lj("dev"), "natural_dev": Lj("natural_dev"), "eval": Lj("eval"), "natural_test": Lj("natural_test")}; tr = Lj("evid_train")
y = {s: np.array([(tr if s == "fit" else lab[s])[i] - 1 for i in ids[s]]) for s in ids}
B = {"L16": slice(0, 4096), "L20": slice(4096, 8192), "L24": slice(8192, 12288), "mean20": slice(12288, 16384)}
best = None
for n, sl in B.items():
    sc = StandardScaler().fit(H["fit"][:, sl])
    for C in (0.003, 0.01, 0.03):
        m = LogisticRegression(C=C, max_iter=300).fit(sc.transform(H["fit"][:, sl]), y["fit"]); acc = (m.predict(sc.transform(H["dev"][:, sl])) == y["dev"]).mean(); print(f"{n} C={C}: dev exact {acc:.3f}", flush=True)
        if best is None or acc > best[0]: best = (acc, n, C, m, sc)
_, n, C, m, sc = best; sl = B[n]; print("chosen on dev:", n, C)
P = {s: m.predict_proba(sc.transform(H[s][:, sl])) for s in ("dev", "natural_dev", "eval", "natural_test")}
for s, p in P.items():
    with open(f"{D}evid5_{s}_probs.jsonl", "w") as o:
        for i, pp, yy in zip(ids[s], p, y[s]): o.write(json.dumps({"review_id": i, "gt_rating": int(yy) + 1, "probs": [float(x) for x in pp]}) + "\n")
G = lambda pref, s: {x["review_id"]: x for x in map(json.loads, open(f"{D}{pref}_{s}_probs.jsonl" if pref.startswith("evid") else f"{D}p1/probs_{pref}_{s}.jsonl"))}
feats = lambda s, ag: np.hstack([np.log(np.clip([G(a, s)[i]["probs"] for i in ids[s]], 1e-6, 1)) for a in ag])
sets = {"S0 A+E": ["R", "evid"], "SP A+E+P2": ["R", "evid", "evid5"], "P2 alone": ["evid5"], "E+P2": ["evid", "evid5"]}; res = {}
for s in ("dev", "natural_dev"):
    print(f"=== {s} 5-fold CV (3 repeats)")
    for k, ag in sets.items():
        X = feats(s, ag); ex, f1 = [], []
        for rep in range(3):
            p = cross_val_predict(LogisticRegression(C=1, max_iter=3000), X, y[s], cv=StratifiedKFold(5, shuffle=True, random_state=rep)); ex.append((p == y[s]).mean()); f1.append(f1_score(y[s], p, average="macro"))
        res[(s, k)] = (np.mean(ex), np.mean(f1)); print(f"{k:10s} exact {np.mean(ex):.3f} macroF1 {np.mean(f1):.3f}")
a, b = res[("dev", "S0 A+E")], res[("dev", "SP A+E+P2")]; c, d = res[("natural_dev", "S0 A+E")], res[("natural_dev", "SP A+E+P2")]
ok = b[0] >= a[0] + .01 and d[0] >= c[0] and b[1] >= a[1] and d[1] >= c[1]
print("=== GATE (dev exact +.010, natural_dev exact >= S0, macroF1 >= S0 on both):", "PASS" if ok else "FAIL")
