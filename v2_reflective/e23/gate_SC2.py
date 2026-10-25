"""SC2 gate (declared LOG.md 2026-10-25): LLM reflective re-rating on uncertain reviews. dev2 + natural_dev; readout only if PASS."""
import json, sys, time, joblib, numpy as np
sys.path.insert(0, "src")
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from agentic_sentiment.v3 import paths as PT
from agentic_sentiment.v3.evidence import EvidenceAgent
from agentic_sentiment.v3.reflector import Reflector, logp
from agentic_sentiment.v3.explainer import load_instruct, make_chat_fn
from agentic_sentiment.v3.sc import reflect_pick
D, M = PT.D, 0.20
e = joblib.load(PT.EVIDENCE_JOBLIB); ev = EvidenceAgent(e["vectorizer"], e["model"]); rj = joblib.load(PT.REFLECTOR_JOBLIB); rf = Reflector(rj["full"], rj["fallback"])
model, tok = load_instruct(); chat = make_chat_fn(model, tok, max_tokens=4)
def load(s):
    G = lambda p: {x["review_id"]: x["probs"] for x in map(json.loads, open(f"{D}{p}_{s}_probs.jsonl"))}
    rows = [json.loads(l) for l in open(f"{D}{s}.jsonl")]; a, b = G(PT.PROBE_PREFIX), G("evid6b")
    return rows, np.c_[logp([a[r["review_id"]] for r in rows]), logp([b[r["review_id"]] for r in rows])], np.array([r["gt_rating"] for r in rows])
def run(s, cv):
    rows, X, y = load(s); t0 = time.time()
    proba = (cross_val_predict(LogisticRegression(C=1, max_iter=3000), X, y - 1, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")
             if cv else rf.full.predict_proba(X))
    srt = np.sort(proba, 1); top2 = np.argsort(-proba, 1)[:, :2] + 1; stack = top2[:, 0].copy(); final = stack.copy(); unsure = (srt[:, -1] - srt[:, -2]) < M; abst = 0
    for n, i in enumerate(np.where(unsure)[0]):
        r, cand = rows[i], (int(top2[i, 0]), int(top2[i, 1]))
        phr = {c: ev.phrases(r["title"], r["review_text"], c)[0] for c in cand}
        pick = reflect_pick(chat, r["title"], r["review_text"], cand, phr)
        if pick is None: abst += 1
        else: final[i] = pick
        if n % 100 == 0: print(f"  {s} {n}/{unsure.sum()} {time.time() - t0:.0f}s", flush=True)
    f, b = int(((stack != y) & (final == y)).sum()), int(((stack == y) & (final != y)).sum())
    res = {"uncertain": int(unsure.sum()), "abstained": abst, "changed": int((final != stack).sum()), "fixed": f, "broken": b, "net": f - b,
           "p": float(binomtest(f, f + b).pvalue) if f + b else 1.0, "stack_exact": float((stack == y).mean()), "sc_exact": float((final == y).mean()), "seconds": round(time.time() - t0)}
    print(s, res, flush=True); return res
out = {"margin": M, "dev2": run("dev2", cv=True)}; out["natural_dev"] = run("natural_dev", cv=False)
ok = out["dev2"]["fixed"] > out["dev2"]["broken"] and out["dev2"]["p"] < 0.05 and out["natural_dev"]["net"] >= 0
print("=== GATE SC2:", "PASS" if ok else "FAIL", flush=True)
if ok:
    out["readout"] = {s: run(s, cv=False) for s in ("eval", "natural_test")}
json.dump(out, open(D + "gate_SC2.json", "w"), indent=1)
