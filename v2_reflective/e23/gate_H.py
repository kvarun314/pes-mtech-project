"""Phase H gate (declared LOG.md 2026-10-24). Uses dev2 + natural_dev only. Fits P3 probe (fit 10K + fit2 30K), writes evid7_<split>_probs.jsonl, runs the declared gate."""
import json, glob, warnings, numpy as np
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
D = "v2_reflective/e23/"; i2, i3 = json.load(open(D + "hid/ids.json")), json.load(open(D + "hid3/ids.json"))
ld = lambda d, s: np.vstack([np.load(f) for f in sorted(glob.glob(f"{D}{d}/{s}_*.npy"))]).astype(np.float32)
h2 = lambda s: (lambda a: np.hstack([a[:, 4096:8192], a[:, 12288:16384]]))(ld("hid2", s))
H = {"fit": np.vstack([h2("fit"), ld("hid3", "fit2")]), "dev2": ld("hid3", "dev2"), **{s: h2(s) for s in ("dev", "natural_dev", "eval", "natural_test")}}
ids = {"fit": i2["fit"] + i3["fit2"], "dev2": i3["dev2"], **{s: i2[s] for s in ("dev", "natural_dev", "eval", "natural_test")}}
lab = {}
for n in ("evid_train", "dev2", "dev", "natural_dev", "eval", "natural_test"):
    for l in open(f"{D}{n}.jsonl"): r = json.loads(l); lab[r["review_id"]] = r["gt_rating"] - 1
y = {s: np.array([lab[i] for i in ids[s]]) for s in ids}; assert all(len(H[s]) == len(y[s]) for s in H), {s: (len(H[s]), len(y[s])) for s in H}
print("fit rows", len(y["fit"]))
F = {"L20": slice(0, 4096), "mean20": slice(4096, 8192), "L20+mean20": slice(0, 8192)}; best = None
for fn, sl in F.items():
    sc = StandardScaler().fit(H["fit"][:, sl])
    for C in (0.001, 0.003, 0.01):
        m = LogisticRegression(C=C, max_iter=500).fit(sc.transform(H["fit"][:, sl]), y["fit"]); acc = (m.predict(sc.transform(H["dev2"][:, sl])) == y["dev2"]).mean()
        print(f"P3 {fn} C={C}: dev2 exact {acc:.3f}", flush=True)
        if best is None or acc > best[0]: best = (acc, fn, C, m, sc)
_, fn, C, m, sc = best; sl = F[fn]; print("chosen on dev2:", fn, C)
for s in ("dev2", "dev", "natural_dev", "eval", "natural_test"):
    P = m.predict_proba(sc.transform(H[s][:, sl]))
    with open(f"{D}evid7_{s}_probs.jsonl", "w") as o:
        for i, p in zip(ids[s], P): o.write(json.dumps({"review_id": i, "gt_rating": int(lab[i]) + 1, "probs": [float(x) for x in p]}) + "\n")
G = lambda pref, s: {x["review_id"]: x for x in map(json.loads, open(f"{D}{pref}_{s}_probs.jsonl" if pref.startswith("evid") else f"{D}p1/probs_{pref}_{s}.jsonl"))}
def feats(s, ag):
    k = list(G("R", s)); return np.array([lab[i] for i in k]), np.hstack([np.log(np.clip([G(a, s)[i]["probs"] for i in k], 1e-6, 1)) for a in ag])
sets = {"S0 A+E": ["R", "evid"], "A+E6b": ["R", "evid6b"], "A+E+P3": ["R", "evid", "evid7"], "A+E6b+P3": ["R", "evid6b", "evid7"], "P3+E": ["evid7", "evid"], "P3+E6b": ["evid7", "evid6b"], "P3 alone": ["evid7"], "E6b alone": ["evid6b"]}
res, pred0 = {}, {}
for s in ("dev2", "natural_dev"):
    print(f"=== {s} 5-fold CV (3 repeats)")
    for n, ag in sets.items():
        yy, X = feats(s, ag); ex, f1 = [], []
        for rep in range(3):
            p = cross_val_predict(LogisticRegression(C=1, max_iter=3000), X, yy, cv=StratifiedKFold(5, shuffle=True, random_state=rep)); ex.append((p == yy).mean()); f1.append(f1_score(yy, p, average="macro"))
            if rep == 0: pred0[(s, n)] = (p == yy)
        res[(s, n)] = (np.mean(ex), np.mean(f1)); print(f"{n:10s} exact {np.mean(ex):.3f} macroF1 {np.mean(f1):.3f}")
cands = ["A+E6b", "A+E+P3", "A+E6b+P3", "P3+E", "P3+E6b"]; b = max(cands, key=lambda n: res[("dev2", n)][0]); s0 = "S0 A+E"
d = pred0[("dev2", b)].astype(float) - pred0[("dev2", s0)]; rng = np.random.default_rng(0)
lo, hi = np.percentile([d[rng.integers(0, len(d), len(d))].mean() for _ in range(3000)], [2.5, 97.5])
print(f"best candidate on dev2: {b}; gain {res[('dev2', b)][0] - res[('dev2', s0)][0]:+.3f}; paired CI (rep 0) [{lo:+.3f}, {hi:+.3f}]")
ok = res[("dev2", b)][0] >= res[("dev2", s0)][0] + .01 and lo > 0 and res[("dev2", b)][1] >= res[("dev2", s0)][1] and res[("natural_dev", b)][1] >= res[("natural_dev", s0)][1] and res[("natural_dev", b)][0] >= res[("natural_dev", s0)][0]
print("=== GATE:", b, "PASS" if ok else "FAIL")
