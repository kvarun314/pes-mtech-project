"""Phase A1 gate (declared LOG.md 2026-10-24 night). dev2 + natural_dev only. P3b = L20 last-token probe on fit 10K + fit2 30K + fit3 (rows extracted so far).
Writes evid8_<split>_probs.jsonl; compares P3b+E6b vs SH (P3+E6b) by the declared rule. Written before extraction finished."""
import json, glob, warnings, numpy as np
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
D = "v2_reflective/e23/"; i2, i3 = json.load(open(D + "hid/ids.json")), json.load(open(D + "hid3/ids.json"))
ld = lambda d, s: np.vstack([np.load(f) for f in sorted(glob.glob(f"{D}{d}/{s}_*.npy"))]).astype(np.float32)
f3 = ld("hid4", "fit3"); id3 = [json.loads(l)["review_id"] for l in open(D + "fit3.jsonl")][:len(f3)]  # chunks are written in fit3.jsonl order
H = {"fit": np.vstack([ld("hid2", "fit")[:, 4096:8192], ld("hid3", "fit2")[:, :4096], f3]), "dev2": ld("hid3", "dev2")[:, :4096],
     **{s: ld("hid2", s)[:, 4096:8192] for s in ("dev", "natural_dev", "eval", "natural_test")}}
ids = {"fit": i2["fit"] + i3["fit2"] + id3, "dev2": i3["dev2"], **{s: i2[s] for s in ("dev", "natural_dev", "eval", "natural_test")}}
lab = {}
for n in ("evid_train", "fit3", "dev2", "dev", "natural_dev", "eval", "natural_test"):
    for l in open(f"{D}{n}.jsonl"): r = json.loads(l); lab[r["review_id"]] = r["gt_rating"] - 1
y = {s: np.array([lab[i] for i in ids[s]]) for s in ids}; assert all(len(H[s]) == len(y[s]) for s in H)
print("fit rows", len(y["fit"]), "(fit3 rows", len(id3), "of 80000)", flush=True)
sc = StandardScaler().fit(H["fit"]); Xf = sc.transform(H["fit"]); best = None
for C in (0.0003, 0.001, 0.003):
    m = LogisticRegression(C=C, max_iter=500).fit(Xf, y["fit"]); acc = (m.predict(sc.transform(H["dev2"])) == y["dev2"]).mean()
    print(f"P3b C={C}: dev2 exact {acc:.3f}", flush=True)
    if best is None or acc > best[0]: best = (acc, C, m)
_, C, m = best; print("chosen on dev2: C =", C)
for s in ("dev2", "dev", "natural_dev", "eval", "natural_test"):
    with open(f"{D}evid8_{s}_probs.jsonl", "w") as o:
        for i, p in zip(ids[s], m.predict_proba(sc.transform(H[s]))): o.write(json.dumps({"review_id": i, "gt_rating": int(lab[i]) + 1, "probs": [float(x) for x in p]}) + "\n")
G = lambda pref, s: {x["review_id"]: x for x in map(json.loads, open(f"{D}{pref}_{s}_probs.jsonl"))}
def feats(s, ag):
    k = list(G("evid7", s)); return np.array([lab[i] for i in k]), np.hstack([np.log(np.clip([G(a, s)[i]["probs"] for i in k], 1e-6, 1)) for a in ag])
sets = {"SH P3+E6b": ["evid7", "evid6b"], "P3b+E6b": ["evid8", "evid6b"], "P3 alone": ["evid7"], "P3b alone": ["evid8"]}; res, pred0 = {}, {}
for s in ("dev2", "natural_dev"):
    print(f"=== {s} 5-fold CV (3 repeats)")
    for n, ag in sets.items():
        yy, X = feats(s, ag); ex, f1 = [], []
        for rep in range(3):
            p = cross_val_predict(LogisticRegression(C=1, max_iter=3000), X, yy, cv=StratifiedKFold(5, shuffle=True, random_state=rep)); ex.append((p == yy).mean()); f1.append(f1_score(yy, p, average="macro"))
            if rep == 0: pred0[(s, n)] = p == yy
        res[(s, n)] = (np.mean(ex), np.mean(f1)); print(f"{n:10s} exact {np.mean(ex):.3f} macroF1 {np.mean(f1):.3f}")
b, s0 = "P3b+E6b", "SH P3+E6b"; d = pred0[("dev2", b)].astype(float) - pred0[("dev2", s0)]; rng = np.random.default_rng(0)
lo, hi = np.percentile([d[rng.integers(0, len(d), len(d))].mean() for _ in range(3000)], [2.5, 97.5])
print(f"gain on dev2 {res[('dev2', b)][0] - res[('dev2', s0)][0]:+.3f}; paired CI (rep 0) [{lo:+.3f}, {hi:+.3f}]")
ok = res[("dev2", b)][0] >= res[("dev2", s0)][0] + .01 and lo > 0 and all(res[(s, b)][1] >= res[(s, s0)][1] for s in ("dev2", "natural_dev")) and res[("natural_dev", b)][0] >= res[("natural_dev", s0)][0]
print("=== GATE A1:", "PASS" if ok else "FAIL")
