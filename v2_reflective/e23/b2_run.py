"""B2 (declared LOG.md): Explainer + Critic loop on a split, SH star fixed beforehand (stored probe probs, full SH, no Router).
usage: b2_run.py SPLIT [LIMIT]   resumable: appends to v2_reflective/e23/b2/SPLIT.jsonl and skips done ids."""
import json, os, sys, time, joblib
sys.path.insert(0, "src")
from agentic_sentiment.v3 import paths as PT
from agentic_sentiment.v3.evidence import EvidenceAgent
from agentic_sentiment.v3.reflector import Reflector
from agentic_sentiment.v3.explainer import load_instruct, make_chat_fn
from agentic_sentiment.v3.graph import build_graph_v3
split = sys.argv[1]; limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
os.makedirs(PT.B2, exist_ok=True); out = f"{PT.B2}{split}.jsonl"
done = {json.loads(l)["review_id"] for l in open(out)} if os.path.exists(out) else set()
e, r = joblib.load(PT.EVIDENCE_JOBLIB), joblib.load(PT.REFLECTOR_JOBLIB)
probe = {x["review_id"]: x["probs"] for x in map(json.loads, open(f"{PT.D}{PT.PROBE_PREFIX}_{split}_probs.jsonl"))}
model, tok = load_instruct(); chat = make_chat_fn(model, tok)
g = build_graph_v3(EvidenceAgent(e["vectorizer"], e["model"]), Reflector(r["full"], r["fallback"]), probe_fn=None, explain_fn=chat, judge_fn=chat)
rows = [json.loads(l) for l in open(f"{PT.D}{split}.jsonl")][:limit]; t0 = time.time(); n = 0
print(f"resume: {len(done)} done of {len(rows)}", flush=True)
with open(out, "a") as f:
    for row in rows:
        if row["review_id"] in done: continue
        s = g.invoke({"review_id": row["review_id"], "title": row["title"], "review_text": row["review_text"], "probe_probs": probe[row["review_id"]]})
        keep = ("star", "pack", "drafts", "feedback", "failed_checks", "rounds", "faithful_first", "faithful", "judge_unparsed", "llm_calls", "timings")
        f.write(json.dumps({"review_id": row["review_id"], "gt": row["gt_rating"], **{k: s[k] for k in keep}}) + "\n"); f.flush(); n += 1
        if n % 50 == 0: print(f"{len(done) + n} rows, {time.time() - t0:.0f}s", flush=True)
print("done", out)
