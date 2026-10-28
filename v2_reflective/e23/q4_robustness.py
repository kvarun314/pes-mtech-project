"""Q4 (declared LOG.md 2026-10-28): flip rate and accuracy under typos / flat text / no title, main stack vs Phase 1, 300 eval reviews.
usage: q4_robustness.py   resumable (q4/robustness.jsonl); tee to q4_robustness.out; writes q4_robustness.json"""
import json, os, sys, time, numpy as np, joblib
from scipy.stats import binomtest
sys.path.insert(0, "src")
from agentic_sentiment.v3 import paths as PT
from agentic_sentiment.v3.evidence import EvidenceAgent
from agentic_sentiment.v3.reflector import Reflector
from agentic_sentiment.v3.perturb import typos, flat
V = {"original": lambda t, x: (t, x), "typos": lambda t, x: (t, typos(x)), "flat": lambda t, x: (flat(t), flat(x)), "no_title": lambda t, x: ("", x)}
rows = [json.loads(l) for l in open(PT.D + "eval.jsonl")]
sample = [r for k in range(1, 6) for r in [r for r in rows if r["gt_rating"] == k][:60]]
os.makedirs(PT.D + "q4", exist_ok=True); out = PT.D + "q4/robustness.jsonl"
done = {(d["review_id"], d["variant"]) for d in map(json.loads, open(out))} if os.path.exists(out) else set()
if len(done) < len(sample) * len(V):
    from agentic_sentiment.v3.llama import SharedLlama, load_probe, phase1_probs
    e, r = joblib.load(PT.EVIDENCE_JOBLIB), joblib.load(PT.REFLECTOR_JOBLIB)
    ev, rf = EvidenceAgent(e["vectorizer"], e["model"]), Reflector(r["full"], r["fallback"])
    llama = SharedLlama(); probe = load_probe(llama); t0 = time.time()
    with open(out, "a") as f:
        for k, row in enumerate(sample):
            for v, fn in V.items():
                if (row["review_id"], v) in done: continue
                title, text = fn(row["title"], row["review_text"])
                sh, _, _ = rf.combine(probe.probs(text), ev.probs(title, text))
                p1 = int(np.argmax(phase1_probs(llama, text))) + 1
                f.write(json.dumps({"review_id": row["review_id"], "variant": v, "gt": row["gt_rating"], "SH": int(sh), "P1": p1}) + "\n"); f.flush()
            if (k + 1) % 25 == 0: print(f"{k + 1}/{len(sample)} {time.time() - t0:.0f}s", flush=True)
D = {(d["review_id"], d["variant"]): d for d in map(json.loads, open(out))}; ids = [r["review_id"] for r in sample]; res = {}
for v in V:
    a = {m: np.array([D[i, v][m] == D[i, v]["gt"] for i in ids]) for m in ("SH", "P1")}
    fl = {m: np.array([D[i, v][m] != D[i, "original"][m] for i in ids]) for m in ("SH", "P1")}
    b, c = int((fl["SH"] & ~fl["P1"]).sum()), int((fl["P1"] & ~fl["SH"]).sum())
    res[v] = {"SH_exact": float(a["SH"].mean()), "P1_exact": float(a["P1"].mean()), "SH_flip": float(fl["SH"].mean()), "P1_flip": float(fl["P1"].mean()),
              "only_SH_flips": b, "only_P1_flips": c, "p": binomtest(b, b + c).pvalue if b + c else 1.0}
    print(f"{v:9s} exact SH {res[v]['SH_exact']:.3f} P1 {res[v]['P1_exact']:.3f} | flips SH {res[v]['SH_flip']:.3f} P1 {res[v]['P1_flip']:.3f} | only-SH {b} only-P1 {c} p {res[v]['p']:.2g}")
json.dump(res, open(PT.D + "q4_robustness.json", "w"), indent=1)
