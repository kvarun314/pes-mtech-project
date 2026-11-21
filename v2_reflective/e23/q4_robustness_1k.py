"""Q4b (declared LOG.md 2026-11-21): Q4 robustness extended from 300 to all 1,000 eval reviews (200 per star). Same variants, same systems
(SH3 main stack vs Phase 1 arm R), same metrics. Reported analysis, no selection. The 300 rows of q4/robustness_sh3.jsonl are reused unchanged.
usage: caffeinate -i .venv/bin/python v2_reflective/e23/q4_robustness_1k.py | tee -a v2_reflective/e23/q4_robustness_1k.out   (resumable)
writes q4_robustness_1k.json"""
import json, os, sys, time, numpy as np, joblib
from scipy.stats import binomtest
sys.path.insert(0, "src")
from agentic_sentiment.v3 import paths as PT
from agentic_sentiment.v3.evidence import EvidenceAgent
from agentic_sentiment.v3.reflector import Reflector
from agentic_sentiment.v3.perturb import typos, flat
V = {"original": lambda t, x: (t, x), "typos": lambda t, x: (t, typos(x)), "flat": lambda t, x: (flat(t), flat(x)), "no_title": lambda t, x: ("", x)}
sample = [json.loads(l) for l in open(PT.D + "eval.jsonl")]
assert len(sample) == 1000, len(sample)
out = PT.D + "q4/robustness_1k.jsonl"
if not os.path.exists(out):   # seed with the 300 reviews already scored by q4_robustness.py (SH3 rerun)
    with open(out, "w") as f: f.writelines(open(PT.D + "q4/robustness_sh3.jsonl").readlines())
done = {(d["review_id"], d["variant"]) for d in map(json.loads, open(out))}
if len(done) < len(sample) * len(V):
    from agentic_sentiment.v3.llama import SharedLlama, load_probe, phase1_probs
    from agentic_sentiment.v3.deberta import DebertaAgent
    e, r = joblib.load(PT.EVIDENCE_JOBLIB), joblib.load(PT.REFLECTOR_JOBLIB)
    ev, rf = EvidenceAgent(e["vectorizer"], e["model"]), Reflector(r["full"], r["fallback"])
    llama = SharedLlama(); probe = load_probe(llama); deb = DebertaAgent(); t0 = time.time(); k0 = len(done) // len(V)
    with open(out, "a") as f:
        for k, row in enumerate(sample):
            for v, fn in V.items():
                if (row["review_id"], v) in done: continue
                title, text = fn(row["title"], row["review_text"])
                sh, _, _ = rf.combine(probe.probs(text), ev.probs(title, text), deb.probs(title, text))
                p1 = int(np.argmax(phase1_probs(llama, text))) + 1
                f.write(json.dumps({"review_id": row["review_id"], "variant": v, "gt": row["gt_rating"], "SH": int(sh), "P1": p1}) + "\n"); f.flush()
            if (k + 1) % 50 == 0: print(f"{k + 1}/{len(sample)} {time.time() - t0:.0f}s", flush=True)
D = {(d["review_id"], d["variant"]): d for d in map(json.loads, open(out))}; ids = [r["review_id"] for r in sample]; res = {"_n": len(ids)}
for v in V:
    a = {m: np.array([D[i, v][m] == D[i, v]["gt"] for i in ids]) for m in ("SH", "P1")}
    fl = {m: np.array([D[i, v][m] != D[i, "original"][m] for i in ids]) for m in ("SH", "P1")}
    b, c = int((fl["SH"] & ~fl["P1"]).sum()), int((fl["P1"] & ~fl["SH"]).sum())
    res[v] = {"SH_exact": float(a["SH"].mean()), "P1_exact": float(a["P1"].mean()), "SH_flip": float(fl["SH"].mean()), "P1_flip": float(fl["P1"].mean()),
              "only_SH_flips": b, "only_P1_flips": c, "p": binomtest(b, b + c).pvalue if b + c else 1.0}
    print(f"{v:9s} exact SH {res[v]['SH_exact']:.3f} P1 {res[v]['P1_exact']:.3f} | flips SH {res[v]['SH_flip']:.3f} P1 {res[v]['P1_flip']:.3f} | only-SH {b} only-P1 {c} p {res[v]['p']:.2g}")
json.dump(res, open(PT.D + "q4_robustness_1k.json", "w"), indent=1)
