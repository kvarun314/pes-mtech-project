"""Workstream C2-C5 (Phase 2 v3 plan): Holm-corrected main comparisons, SH error analysis, calibration, per-agent cost. Reads stored probs only;
no selection. SH = LR(C=1) on [log P3, log E6b] fit on dev2 (as readout_H.py). Review text is never printed (ids only)."""
import json, time, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix
D = "v2_reflective/e23/"; FIG = "paper/figures/e23/"
G = lambda p, s: {x["review_id"]: x for x in map(json.loads, open(f"{D}{p}_{s}_probs.jsonl" if p.startswith("evid") else f"{D}p1/probs_{p}_{s}.jsonl"))}
ids = {s: list(G("R", s)) for s in ("dev", "dev2", "eval", "natural_test")}
lg = lambda p, s: np.log(np.clip([G(p, s)[i]["probs"] for i in ids[s]], 1e-6, 1))
y = {s: np.array([G("R", s)[i]["gt_rating"] - 1 for i in ids[s]]) for s in ids}
fit = lambda ag, f: LogisticRegression(C=1, max_iter=3000).fit(np.hstack([lg(a, f) for a in ag]), y[f])
M = {"SH": (["evid7", "evid6b"], "dev2"), "S0 dev2": (["R", "evid"], "dev2"), "S0 dev (old headline)": (["R", "evid"], "dev")}; M = {n: (ag, fit(ag, f)) for n, (ag, f) in M.items()}
words = {s: np.array([len(json.loads(l)["review_text"].split()) for l in open(f"{D}{s}.jsonl")]) for s in ("eval", "natural_test")}
order = {s: [json.loads(l)["review_id"] for l in open(f"{D}{s}.jsonl")] for s in ("eval", "natural_test")}
pvals = []
for s in ("eval", "natural_test"):
    print(f"\n===== {s} (n={len(y[s])})")
    P = {n: m.predict_proba(np.hstack([lg(a, s) for a in ag])) for n, (ag, m) in M.items()}; pred = {n: p.argmax(1) for n, p in P.items()}
    pred["E6b alone"], pred["P3 alone"] = lg("evid6b", s).argmax(1), lg("evid7", s).argmax(1); h = pred["SH"]; yy = y[s]
    # C2: Holm over the 8 declared SH comparisons (4 per split)
    for o in ("S0 dev (old headline)", "S0 dev2", "E6b alone", "P3 alone"):
        a, b = h == yy, pred[o] == yy; fx, rg = int((a & ~b).sum()), int((~a & b).sum()); pvals.append((f"{s}: SH vs {o}", 100 * (a.mean() - b.mean()), binomtest(fx, fx + rg, .5).pvalue))
    # C3: error analysis
    cm = confusion_matrix(yy, h, labels=range(5)); err = h != yy
    print("SH confusion (rows true 1-5, cols pred 1-5):\n" + "\n".join(" ".join(f"{v:5d}" for v in r) for r in cm))
    print(f"SH errors {err.sum()} | off-by-one {np.mean(abs(h[err] - yy[err]) == 1):.3f} of errors | 2+ stars off {int((abs(h - yy) >= 2).sum())}")
    print("per-class recall", [round(float(cm[k, k] / cm[k].sum()), 3) for k in range(5)], "| macro-F1", round(f1_score(yy, h, average="macro"), 3))
    w = words[s][[order[s].index(i) for i in ids[s]]]; q = np.quantile(w, [.25, .5, .75])
    for lo, hi, name in ((0, q[0], "Q1 shortest"), (q[0], q[1], "Q2"), (q[1], q[2], "Q3"), (q[2], 1e9, "Q4 longest")):
        k = (w > lo) & (w <= hi) if lo else w <= hi; print(f"length {name:12s} ({int(lo)}-{int(min(hi, w.max()))} words, n={k.sum()}): SH {100*(h[k]==yy[k]).mean():.1f} | E6b {100*(pred['E6b alone'][k]==yy[k]).mean():.1f} | P3 {100*(pred['P3 alone'][k]==yy[k]).mean():.1f}")
    ag = pred["E6b alone"] == pred["P3 alone"]
    print(f"agents agree on {ag.mean():.3f}: SH acc {100*(h[ag]==yy[ag]).mean():.1f} | disagree: SH {100*(h[~ag]==yy[~ag]).mean():.1f}, E6b {100*(pred['E6b alone'][~ag]==yy[~ag]).mean():.1f}, P3 {100*(pred['P3 alone'][~ag]==yy[~ag]).mean():.1f}")
    rng = np.random.default_rng(1); ex = {"SH right, both agents wrong": np.where((h == yy) & (pred["E6b alone"] != yy) & (pred["P3 alone"] != yy))[0],
          "SH wrong, an agent right": np.where((h != yy) & ((pred["E6b alone"] == yy) | (pred["P3 alone"] == yy)))[0], "SH 2+ stars off": np.where(abs(h - yy) >= 2)[0]}
    for n, ix in ex.items(): print(f"examples [{n}] (n={len(ix)}):", [(ids[s][i], int(yy[i]) + 1, int(h[i]) + 1) for i in rng.choice(ix, min(2, len(ix)), replace=False)])
    # C4: calibration of SH (15-bin ECE on top-class probability)
    c = P["SH"].max(1); ok = h == yy; bins = np.linspace(0, 1, 16); idx = np.digitize(c, bins[1:-1]); ece = sum(abs(c[idx == k].mean() - ok[idx == k].mean()) * (idx == k).mean() for k in range(15) if (idx == k).any())
    print(f"SH ECE {ece:.3f} | mean confidence {c.mean():.3f} vs accuracy {ok.mean():.3f}")
    plt.figure(figsize=(3.6, 3.4)); xs = [c[idx == k].mean() for k in range(15) if (idx == k).sum() >= 10]; ys = [ok[idx == k].mean() for k in range(15) if (idx == k).sum() >= 10]
    plt.plot([0, 1], [0, 1], "--", color="grey"); plt.plot(xs, ys, "o-", color="#C44E52"); plt.xlabel("SH confidence"); plt.ylabel("accuracy"); plt.title(f"{s}, ECE {ece:.3f}"); plt.tight_layout(); plt.savefig(f"{FIG}reliability_SH_{s}.png", dpi=150); plt.close()
print("\n===== Holm correction over 8 SH comparisons (exact McNemar p)")
pvals.sort(key=lambda t: t[2]); m = len(pvals); run = 0
for k, (n, d, p) in enumerate(pvals):
    run = max(run, min(1, (m - k) * p)); print(f"{n:42s} {d:+.1f} pts  p {p:.2g}  Holm-adj {run:.2g}  {'sig' if run < .05 else 'ns'}")
# C5: per-agent cost on this Mac (probe from A1 extraction timing; E6b and Reflector measured here)
rows = [json.loads(l) for l in open(D + "eval.jsonl")][:500]
from sklearn.feature_extraction.text import TfidfVectorizer
vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1).fit(r["review_text"] for r in rows); t0 = time.time(); vec.transform(r["review_text"] for r in rows); tv = (time.time() - t0) / len(rows)
X = np.hstack([lg("evid7", "eval"), lg("evid6b", "eval")]); t0 = time.time(); [M["SH"][1].predict(X[i:i + 1]) for i in range(500)]; tr = (time.time() - t0) / 500
print(f"\n===== C5 cost per review (M1, 16 GB): P3 probe = LLaMA-3-8B Q4 forward to layer 20 ~0.67 s (A1 log: 500 rows in 335 s) | TF-IDF transform ~{1000*tv:.2f} ms (+ LR dot product) | Reflector ~{1000*tr:.2f} ms")
