"""Phase F gate (declared LOG.md 2026-10-22). Fits LR probe on fit-set hidden states, C chosen on dev only, writes evid4_<split>_probs.jsonl, DEV-ONLY CV gate for stack [A+E+P]. eval/natural_test hidden states are used only to write probs, not scored here."""
import json, numpy as np, glob
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
D = "v2_reflective/e23/"; ids = json.load(open(D + "hid/ids.json"))
H = {s: np.vstack([np.load(f) for f in sorted(glob.glob(f"{D}hid/{s}_*.npy"))]) for s in ids}
Lj = lambda n: {r["review_id"]: r["gt_rating"] for r in map(json.loads, open(f"{D}{n}.jsonl"))}
lab = {"dev": Lj("dev"), "natural_dev": Lj("natural_dev"), "eval": Lj("eval"), "natural_test": Lj("natural_test")}; tr = Lj("evid_train")
y = {s: np.array([(tr if s == "fit" else lab[s])[i] - 1 for i in ids[s]]) for s in ids}
sc = StandardScaler().fit(H["fit"]); Z = {s: sc.transform(H[s]) for s in H}; best = None
for C in (0.01, 0.1, 1):
    m = LogisticRegression(C=C, max_iter=500).fit(Z["fit"], y["fit"]); acc = (m.predict(Z["dev"]) == y["dev"]).mean(); print(f"probe C={C}: dev exact {acc:.3f}", flush=True)
    if best is None or acc > best[0]: best = (acc, C, m)
_, C, m = best; print("chosen on dev: C =", C)
for s in ("dev", "natural_dev", "eval", "natural_test"):
    P = m.predict_proba(Z[s])
    with open(f"{D}evid4_{s}_probs.jsonl", "w") as o:
        for i, p, yy in zip(ids[s], P, y[s]): o.write(json.dumps({"review_id": i, "gt_rating": int(yy) + 1, "probs": [float(x) for x in p]}) + "\n")
G = lambda pref, s: {x["review_id"]: x for x in map(json.loads, open(f"{D}{pref}_{s}_probs.jsonl" if pref.startswith("evid") else f"{D}p1/probs_{pref}_{s}.jsonl"))}
def feats(s, ag):
    i_ = ids[s]; return np.hstack([np.log(np.clip([G(a, s)[i]["probs"] for i in i_], 1e-6, 1)) for a in ag])
sets = {"S0 A+E": ["R", "evid"], "SP A+E+P": ["R", "evid", "evid4"], "P alone": ["evid4"], "A alone": ["R"], "E alone": ["evid"], "A+P": ["R", "evid4"]}; res = {}
for s in ("dev", "natural_dev"):
    print(f"=== {s} 5-fold CV (3 repeats)")
    for n, ag in sets.items():
        X = feats(s, ag); ex, f1 = [], []
        for rep in range(3):
            p = cross_val_predict(LogisticRegression(C=1, max_iter=3000), X, y[s], cv=StratifiedKFold(5, shuffle=True, random_state=rep)); ex.append((p == y[s]).mean()); f1.append(f1_score(y[s], p, average="macro"))
        res[(s, n)] = (np.mean(ex), np.mean(f1)); print(f"{n:10s} exact {np.mean(ex):.3f} macroF1 {np.mean(f1):.3f}")
    print("direct argmax (no stacker): " + ", ".join(f"{a} {(np.exp(feats(s,[a])).argmax(1)==y[s]).mean():.3f}" for a in ("R", "evid", "evid4")))
ok = res[("dev", "SP A+E+P")][0] >= res[("dev", "S0 A+E")][0] + .01 and res[("natural_dev", "SP A+E+P")][0] >= res[("natural_dev", "S0 A+E")][0] and res[("dev", "SP A+E+P")][1] >= res[("dev", "S0 A+E")][1]
print("=== GATE (exact >= S0+0.010 on dev AND >= S0 on natural_dev AND dev macroF1 >= S0):", "PASS" if ok else "FAIL")
