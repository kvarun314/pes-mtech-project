"""Per-star breakdown for the demo's Results tab (descriptive, no selection; LOG.md 2026-11-22).
Phase 1 (arm R) vs main stack (SH3: probe evid7 + Evidence evid6b + DeBERTa deb3, Reflector LR C=1 fit on dev2), exactly as in freeze_tables.py.
usage: .venv/bin/python v2_reflective/e23/demo_breakdown.py | tee v2_reflective/e23/demo_breakdown.out   -> demo_breakdown.json"""
import json, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
D = "v2_reflective/e23/"
L = lambda p, s: {x["review_id"]: x for x in map(json.loads, open(f"{D}{p}_{s}_probs.jsonl" if p.startswith(("evid", "deb")) else f"{D}p1/probs_{p}_{s}.jsonl"))}
S = ("dev2", "eval", "natural_test"); R = {s: L("R", s) for s in S}; ids = {s: list(R[s]) for s in S}
C = {}
arr = lambda p, s: np.clip([C.setdefault((p, s), L(p, s))[i]["probs"] for i in ids[s]], 1e-6, 1)
y = {s: np.array([R[s][i]["gt_rating"] - 1 for i in ids[s]]) for s in S}
AG = ("evid7", "evid6b", "deb3"); XH = lambda s: np.hstack([np.log(arr(a, s)) for a in AG])
rf = LogisticRegression(C=1, max_iter=3000).fit(XH("dev2"), y["dev2"])
out = {}
for s in ("eval", "natural_test"):
    out[s] = {}
    for name, p in (("p1", arr("R", s).argmax(1)), ("p2", rf.predict(XH(s)))):
        cm = confusion_matrix(y[s], p, labels=range(5))
        out[s][name] = {"exact": float((p == y[s]).mean() * 100), "confusion": cm.tolist(),
                        "recall": (cm.diagonal() / cm.sum(1) * 100).round(1).tolist(), "f1": f1_score(y[s], p, average=None, labels=range(5)).round(3).tolist(),
                        "off_by_one": float((abs(p - y[s]) == 1).mean() * 100), "off_by_more": float((abs(p - y[s]) > 1).mean() * 100)}
        o = out[s][name]; print(f"{s:12s} {name} exact {o['exact']:.1f} recall {o['recall']} f1 {o['f1']} off1 {o['off_by_one']:.1f} off2+ {o['off_by_more']:.1f}")
json.dump(out, open(D + "demo_breakdown.json", "w"), indent=1)
