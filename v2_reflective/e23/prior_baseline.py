"""Fair baseline for the natural_test gains in stack_week1: Evidence alone and Analyst alone with the SAME natural-prior correction (prior from natural_dev only):
p_adj ∝ p * prior_natural_dev (training was balanced, so this is the Bayes prior shift). Declared after seeing stack_week1.out (LOG 2026-10-18): this is a missing baseline, not a new stacker."""
import json, numpy as np
from sklearn.metrics import f1_score
D = "v2_reflective/e23/"; Lj = lambda p: {r["review_id"]: r for r in map(json.loads, open(p))}
def get(split):
    a, e = Lj(f"{D}p1/probs_R_{split}.jsonl"), Lj(f"{D}evid_{split}_probs.jsonl"); ids = list(a)
    return np.array([a[i]["gt_rating"] - 1 for i in ids]), np.array([a[i]["probs"] for i in ids]), np.array([e[i]["probs"] for i in ids])
yn, _, _ = get("natural_dev"); prior = np.bincount(yn, minlength=5) / len(yn); print("natural_dev prior", prior.round(3))
m = lambda y, p: f"exact {(p==y).mean():.1%} macroF1 {f1_score(y,p,average='macro'):.3f} MAE {abs(p-y).mean():.3f} recall {[round(float(((p==c)&(y==c)).sum()/max((y==c).sum(),1)),2) for c in range(5)]}"
for split in ("eval", "natural_test"):
    y, A, E = get(split); print(f"\n== {split}")
    for nm, P in (("Analyst", A), ("Evidence", E)):
        print(f"{nm:9s} raw          ", m(y, P.argmax(1))); print(f"{nm:9s} prior-adjusted", m(y, (P * prior).argmax(1)))
