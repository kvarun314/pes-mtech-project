"""Phase H variance (declared LOG.md 2026-10-24). Probe-fit bootstrap + dev2 bootstrap for the Reflector, 5 seeds. E6b fixed. Resumable per seed."""
import json, glob, os, warnings, numpy as np
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
D = "v2_reflective/e23/"; i2, i3 = json.load(open(D + "hid/ids.json")), json.load(open(D + "hid3/ids.json"))
ld = lambda d, s: np.vstack([np.load(f) for f in sorted(glob.glob(f"{D}{d}/{s}_*.npy"))]).astype(np.float32)
H = {"fit": np.vstack([ld("hid2", "fit")[:, 4096:8192], ld("hid3", "fit2")[:, :4096]]), "dev2": ld("hid3", "dev2")[:, :4096], **{s: ld("hid2", s)[:, 4096:8192] for s in ("eval", "natural_test")}}
ids = {"fit": i2["fit"] + i3["fit2"], "dev2": i3["dev2"], "eval": i2["eval"], "natural_test": i2["natural_test"]}
lab = {}
for n in ("evid_train", "dev2", "eval", "natural_test"):
    for l in open(f"{D}{n}.jsonl"): r = json.loads(l); lab[r["review_id"]] = r["gt_rating"] - 1
y = {s: np.array([lab[i] for i in ids[s]]) for s in ids}
G = lambda pref, s: {x["review_id"]: x for x in map(json.loads, open(f"{D}{pref}_{s}_probs.jsonl" if pref.startswith("evid") else f"{D}p1/probs_{pref}_{s}.jsonl"))}
lg = lambda pref, s: np.log(np.clip([G(pref, s)[i]["probs"] for i in ids[s]], 1e-6, 1))
E6, A, E = ({s: lg(p, s) for s in ("dev2", "eval", "natural_test")} for p in ("evid6b", "R", "evid"))
for seed in range(5):
    out = f"{D}variance_H_seed{seed}.json"
    if os.path.exists(out): continue
    rng = np.random.default_rng(seed); b = rng.integers(0, len(y["fit"]), len(y["fit"])); sc = StandardScaler().fit(H["fit"][b])
    pm = LogisticRegression(C=.001, max_iter=500).fit(sc.transform(H["fit"][b]), y["fit"][b])
    P3 = {s: np.log(np.clip(pm.predict_proba(sc.transform(H[s])), 1e-6, 1)) for s in ("dev2", "eval", "natural_test")}
    d = rng.integers(0, 2000, 2000)
    sh = LogisticRegression(C=1, max_iter=3000).fit(np.hstack([P3["dev2"], E6["dev2"]])[d], y["dev2"][d])
    s0 = LogisticRegression(C=1, max_iter=3000).fit(np.hstack([A["dev2"], E["dev2"]])[d], y["dev2"][d]); r = {}
    for s in ("eval", "natural_test"):
        p = {"SH": sh.predict(np.hstack([P3[s], E6[s]])), "S0": s0.predict(np.hstack([A[s], E[s]])), "E6b": E6[s].argmax(1), "P3": P3[s].argmax(1)}
        r[s] = {k: [float((v == y[s]).mean()), float(f1_score(y[s], v, average="macro"))] for k, v in p.items()}
    json.dump(r, open(out, "w")); print("seed", seed, json.dumps(r), flush=True)
R = [json.load(open(f"{D}variance_H_seed{k}.json")) for k in range(5)]
for s in ("eval", "natural_test"):
    for k in ("SH", "S0", "E6b", "P3"):
        a = np.array([x[s][k] for x in R]); print(f"{s} {k}: exact {100*a[:,0].mean():.1f} ± {100*a[:,0].std(ddof=1):.1f}  macroF1 {a[:,1].mean():.3f} ± {a[:,1].std(ddof=1):.3f}")
    for o in ("E6b", "P3", "S0"):
        d = np.array([100 * (x[s]["SH"][0] - x[s][o][0]) for x in R]); print(f"{s} SH-minus-{o}: {d.mean():+.2f} ± {d.std(ddof=1):.2f} pts, per seed {d.round(1).tolist()}")
