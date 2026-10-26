"""B2 deletion test (declared LOG.md). Per review: SH prob of the explained star on review text only, after removing the cited
phrases, after removing the same number of random words (3 draws), and with only cited / only random words kept. Resumable."""
import json, os, random, sys, time, zlib, joblib, numpy as np
sys.path.insert(0, "src")
from agentic_sentiment.v3 import paths as PT
from agentic_sentiment.v3.evidence import EvidenceAgent
from agentic_sentiment.v3.reflector import Reflector
from agentic_sentiment.v3.llama import SharedLlama, load_probe
from agentic_sentiment.v3.text import quoted, phrase_token_idx, drop_tokens, token_spans, tokens, contains
split = sys.argv[1]; out = f"{PT.B2}deletion_{split}.jsonl"
done = {json.loads(l)["review_id"] for l in open(out)} if os.path.exists(out) else set()
e, r = joblib.load(PT.EVIDENCE_JOBLIB), joblib.load(PT.REFLECTOR_JOBLIB); ev, rf = EvidenceAgent(e["vectorizer"], e["model"]), Reflector(r["full"], r["fallback"])
probe = load_probe(SharedLlama(adapter=None)); text = {x["review_id"]: x["review_text"] for x in map(json.loads, open(f"{PT.D}{split}.jsonl"))}
def p_star(t, star): return float(rf.combine(probe.probs(t), ev.probs("", t))[1][star - 1])
t0 = time.time(); n = 0
with open(out, "a") as f:
    for b in map(json.loads, open(f"{PT.B2}{split}.jsonl")):
        if b["review_id"] in done: continue
        t, star = text[b["review_id"]], b["star"]
        cited = [q for q in quoted(b["drafts"][-1]) if contains(tokens(t), q)] or [p["display"] for p in b["pack"]["for"][:3]]
        idx = set().union(*[set(phrase_token_idx(t, q) or []) for q in cited]); k = len(idx); nt = len(token_spans(t))
        rng = random.Random(zlib.crc32(b["review_id"].encode())); draws = [set(rng.sample(range(nt), min(k, nt))) for _ in range(3)]
        keep = lambda s: " ".join(t[a:z] for i, (a, z) in enumerate(token_spans(t)) if i in s)
        rec = {"review_id": b["review_id"], "star": star, "n_tokens": k, "p_ref": p_star(t, star),
               "p_cited_removed": p_star(drop_tokens(t, idx), star), "p_random_removed": [p_star(drop_tokens(t, d), star) for d in draws],
               "p_cited_only": p_star(keep(idx), star), "p_random_only": [p_star(keep(d), star) for d in draws]}
        f.write(json.dumps(rec) + "\n"); f.flush(); n += 1
        if n % 100 == 0: print(f"{len(done) + n} rows, {time.time() - t0:.0f}s", flush=True)
print("done", out)
