"""B1 Router gate + SC1 escalation self-correction (declared LOG.md 2026-10-25). dev2 CV picks tau; natural_dev check; if PASS a single readout on eval + natural_test."""
import json, sys, numpy as np
sys.path.insert(0, "src")
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from agentic_sentiment.v3 import paths as PT
from agentic_sentiment.v3.reflector import choose_tau, fit_reflector, logp
import os
D = PT.D; LR = lambda: LogisticRegression(C=1, max_iter=3000)
TOL = float(os.environ.get("TOL", "0.005")); OUTN = os.environ.get("OUT", "gate_B1")   # B1: 0.005; B1-b (second attempt): 0
def load(s):
    G = lambda p: {x["review_id"]: x for x in map(json.loads, open(f"{D}{p}_{s}_probs.jsonl"))}
    a, b = G(PT.PROBE_PREFIX), G("evid6b"); ids = list(a)
    return np.array([a[i]["probs"] for i in ids]), np.array([b[i]["probs"] for i in ids]), np.array([a[i]["gt_rating"] for i in ids])
def esc(draft, revised, y, escalated):
    f, b = int(((draft != y) & (revised == y) & escalated).sum()), int(((draft == y) & (revised != y) & escalated).sum())
    return {"escalated": int(escalated.sum()), "changed": int(((draft != revised) & escalated).sum()), "fixed": f, "broken": b, "net": f - b,
            "p": float(binomtest(f, f + b).pvalue) if f + b else 1.0}
def routed(r, P, E, tau):
    fp = r.full.predict(np.c_[logp(P), logp(E)]) + 1; bp = r.fallback.predict(logp(E)) + 1; skip = E.max(1) >= tau
    return fp, np.where(skip, bp, fp), skip, bp
P2, E2, y2 = load("dev2"); tau, base, rows = choose_tau(P2, E2, y2, tol=TOL)
for t, acc, sk in rows: print(f"tau {t:.2f}: routed {acc:.3f} (full {base:.3f}) skip {sk:.2f}")
out = {"tol": TOL, "tau": tau, "dev2_full": base, "dev2_routed": {f"{t:.2f}": a for t, a, _ in rows}}; ok = tau is not None
if ok:
    cv = StratifiedKFold(5, shuffle=True, random_state=0)                      # same folds as choose_tau
    fp2 = cross_val_predict(LR(), np.c_[logp(P2), logp(E2)], y2 - 1, cv=cv) + 1; bp2 = cross_val_predict(LR(), logp(E2), y2 - 1, cv=cv) + 1
    out["sc1_dev2"] = e2 = esc(bp2, fp2, y2, E2.max(1) < tau); print("SC1 dev2 (CV):", e2)
    r = fit_reflector(P2, E2, y2); Pn, En, yn = load("natural_dev"); fp, rp, sk, _ = routed(r, Pn, En, tau)
    print(f"natural_dev: full {(fp == yn).mean():.3f} routed {(rp == yn).mean():.3f} skip {sk.mean():.2f}")
    out["natural_dev"] = {"full": float((fp == yn).mean()), "routed": float((rp == yn).mean()), "skip_share": float(sk.mean())}
    ok = (rp == yn).mean() >= (fp == yn).mean() - 0.005 and e2["fixed"] > e2["broken"] and e2["p"] < 0.05
print(f"=== GATE {OUTN} + SC1 (tol {TOL}):", "PASS" if ok else "FAIL", "tau", tau)
if ok:
    out["readout"] = {}
    for s in ("eval", "natural_test"):
        P, E, y = load(s); fp, rp, sk, bp = routed(r, P, E, tau); a, b = int(((rp == y) & (fp != y)).sum()), int(((rp != y) & (fp == y)).sum())
        p = binomtest(a, a + b).pvalue if a + b else 1.0; sc1 = esc(bp, fp, y, ~sk)
        out["readout"][s] = {"full": float((fp == y).mean()), "routed": float((rp == y).mean()), "skip_share": float(sk.mean()), "mcnemar_p": float(p), "sc1": sc1}
        print(f"{s}: full {(fp == y).mean():.3f} routed {(rp == y).mean():.3f} skip {sk.mean():.2f} McNemar p {p:.2g} | SC1 {sc1}")
json.dump(out, open(D + OUTN + ".json", "w"), indent=1)
