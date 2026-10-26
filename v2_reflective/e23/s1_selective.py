"""S1 hand-off agent (declared LOG.md 2026-10-26): accuracy vs coverage of the frozen SH, thresholds fixed on dev2 quantiles.
Phase 1 LoRA (arm R) curve for comparison, its thresholds from its own dev2 quantiles. Writes s1_selective.out/.json and the figure."""
import json, sys, joblib, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
sys.path.insert(0, "src")
from agentic_sentiment.v3 import paths as PT
from agentic_sentiment.v3.reflector import Reflector
r = joblib.load(PT.REFLECTOR_JOBLIB); rf = Reflector(r["full"], r["fallback"])
L = lambda f: {x["review_id"]: x for x in map(json.loads, open(PT.D + f))}
def sh(s):
    P, E = L(f"evid7_{s}_probs.jsonl"), L(f"evid6b_{s}_probs.jsonl")
    out = [(rf.combine(np.array(P[i]["probs"]), np.array(E[i]["probs"])), P[i]["gt_rating"]) for i in P]
    return np.array([o[1].max() for o, g in out]), np.array([o[0] for o, g in out]), np.array([g for o, g in out])
def p1(s):
    R = list(L(f"p1/probs_R_{s}.jsonl").values()); P = np.array([x["probs"] for x in R])
    return P.max(1), P.argmax(1) + 1, np.array([x["gt_rating"] for x in R])
def ci(ok, rng):
    b = [ok[rng.integers(0, len(ok), len(ok))].mean() for _ in range(2000)]; return [float(x) for x in np.percentile(b, [2.5, 97.5])]
TARGETS = [1.0, .9, .8, .7, .6, .5, .4]; res = {}; fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
for name, fn in (("SH", sh), ("Phase1_LoRA", p1)):
    c2 = fn("dev2")[0]; res[name] = {}
    for ax, s in zip(axes, ("eval", "natural_test")):
        c, pred, y = fn(s); ok = pred == y; rng = np.random.default_rng(0); rows = []
        for t in TARGETS:
            th = -1.0 if t == 1.0 else float(np.quantile(c2, 1 - t)); m = c >= th
            rows.append({"target": t, "threshold": th, "coverage": float(m.mean()), "acc_auto": float(ok[m].mean()), "ci": ci(ok[m], rng),
                         "macroF1_auto": float(f1_score(y[m], pred[m], average="macro")), "acc_handed_off": float(ok[~m].mean()) if (~m).any() else None})
        res[name][s] = rows
        grid = np.linspace(0.2, 1, 41); cv = [ok[c >= np.quantile(c, 1 - g)].mean() for g in grid]
        ax.plot(grid * 100, np.array(cv) * 100, color="#c0392b" if name == "SH" else "#7f8c8d", lw=2, label="Main stack" if name == "SH" else "Phase 1 LoRA")
        ax.set_title({"eval": "Balanced eval", "natural_test": "Natural test"}[s]); ax.set_xlabel("Reviews answered automatically (%)"); ax.grid(alpha=.3)
axes[0].set_ylabel("Accuracy on answered reviews (%)"); axes[1].legend(); fig.tight_layout()
fig.savefig("paper/figures/e23/selective_SH.png", dpi=200)
with open(PT.D + "s1_selective.out", "w") as f:
    for name in res:
        for s, rows in res[name].items():
            print(f"=== {name} {s}", file=f)
            for x in rows:
                ho = "—" if x["acc_handed_off"] is None else f"{x['acc_handed_off']:.3f}"
                print(f"target {x['target']:.0%}  thr {x['threshold']:.2f}  coverage {x['coverage']:.3f}  acc_auto {x['acc_auto']:.3f} [{x['ci'][0]:.3f}, {x['ci'][1]:.3f}]  "
                      f"macroF1_auto {x['macroF1_auto']:.3f}  acc_handed_off {ho}", file=f)
json.dump(res, open(PT.D + "s1_selective.json", "w"), indent=1)
print(open(PT.D + "s1_selective.out").read())
