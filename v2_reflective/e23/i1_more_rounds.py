"""I1 (declared LOG.md 2026-10-28): rerun the B2 graph with max_rewrites=4 for explanations still unfaithful after 2 rounds.
Same models and stored probe probabilities as b2_run.py; greedy, so rounds 0-2 should repeat the stored drafts.
usage: i1_more_rounds.py SPLIT   resumable: appends to b2/i1_SPLIT.jsonl; prints the summary at the end (tee to i1_more_rounds_SPLIT.out)"""
import json, os, sys, time, joblib
import numpy as np
from scipy.stats import binomtest
sys.path.insert(0, "src")
from agentic_sentiment.v3 import paths as PT
from agentic_sentiment.v3.evidence import EvidenceAgent
from agentic_sentiment.v3.reflector import Reflector
from agentic_sentiment.v3.explainer import load_instruct, make_chat_fn
from agentic_sentiment.v3.graph import build_graph_v3
split = sys.argv[1]; out = f"{PT.B2}i1_{split}.jsonl"
B = [json.loads(l) for l in open(f"{PT.B2}{split}.jsonl")]; todo = [b for b in B if not b["faithful"]]
done = {json.loads(l)["review_id"] for l in open(out)} if os.path.exists(out) else set()
if len(done) < len(todo):
    e, r = joblib.load(PT.EVIDENCE_JOBLIB), joblib.load(PT.REFLECTOR_JOBLIB)
    probe = {x["review_id"]: x["probs"] for x in map(json.loads, open(f"{PT.D}{PT.PROBE_PREFIX}_{split}_probs.jsonl"))}
    T = {x["review_id"]: x for x in map(json.loads, open(f"{PT.D}{split}.jsonl"))}
    model, tok = load_instruct(); chat = make_chat_fn(model, tok)
    g = build_graph_v3(EvidenceAgent(e["vectorizer"], e["model"]), Reflector(r["full"], r["fallback"]), probe_fn=None, explain_fn=chat, judge_fn=chat, max_rewrites=4)
    t0 = time.time(); print(f"resume: {len(done)} done of {len(todo)}", flush=True)
    with open(out, "a") as f:
        for k, b in enumerate(todo):
            if b["review_id"] in done: continue
            row = T[b["review_id"]]
            s = g.invoke({"review_id": row["review_id"], "title": row["title"], "review_text": row["review_text"], "probe_probs": probe[row["review_id"]]})
            keep = ("drafts", "feedback", "failed_checks", "rounds", "faithful", "judge_unparsed", "llm_calls", "timings")
            f.write(json.dumps({"review_id": row["review_id"], "same_first3": s["drafts"][:3] == b["drafts"][:3], **{x: s[x] for x in keep}}) + "\n"); f.flush()
            if (k + 1) % 10 == 0: print(f"{k + 1}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
I = {x["review_id"]: x for x in map(json.loads, open(out))}
fixed = sum(I[b["review_id"]]["faithful"] for b in todo); n = len(B); f2 = sum(b["faithful"] for b in B); f0 = sum(b["faithful_first"] for b in B)
f4 = f2 + fixed; extra = np.mean([I[b["review_id"]]["llm_calls"] - b["llm_calls"] for b in todo])
print(f"{split}: still unfaithful after 2 rounds {len(todo)}; fixed by rounds 3-4 {fixed} ({fixed / len(todo):.1%}); rounds 0-2 reproduced exactly {sum(x['same_first3'] for x in I.values())}/{len(I)}")
print(f"faithful: first {f0 / n:.3f}  max-2 {f2 / n:.3f}  max-4 {f4 / n:.3f}  | max-4 vs max-2 McNemar fixed {fixed} broken 0 p {binomtest(fixed, fixed).pvalue if fixed else 1.0:.2g}")
print(f"extra LLM calls per re-run review {extra:.2f}; per review over the split {extra * len(todo) / n:.2f}")
print("still failing after round 4, by check:", {c: sum(c in I[b['review_id']]['failed_checks'][-1] for b in todo if not I[b['review_id']]['faithful']) for c in range(1, 6)})
if split == "dev2": print("GATE (>= 15 of the 62 fixed):", "PASS" if fixed >= 15 else "FAIL")
