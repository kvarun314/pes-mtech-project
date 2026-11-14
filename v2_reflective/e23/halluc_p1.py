"""Hallucination study on the Electronics-trained Phase 1 adapter (arm R). First 150 rows of e23/eval.jsonl (balanced).
Full greedy generation (120 tokens) under the training-format zero-shot prompt (same as eval_p1.py). Resumable: appends, skips done ids.
usage: halluc_p1.py gen | score    (score output -> e23/halluc_p1.out via tee)"""
import json, os, re, sys, time
sys.path.insert(0, "src")
from agentic_sentiment.agents.parsing import parse_rating
OUT = "v2_reflective/e23/p1/halluc_R_generations.jsonl"; N = 150
rows = [json.loads(l) for l in open("v2_reflective/e23/eval.jsonl")][:N]
if sys.argv[1] == "gen":
    from mlx_lm import load, generate
    from agentic_sentiment.phase1.config import DataConfig
    from agentic_sentiment.phase1.data.preprocessing import build_prompt
    model, tok = load("v2_reflective/mlx_model", adapter_path="v2_reflective/e23/p1/adapter_R"); cfg = DataConfig(use_one_shot=False)
    done = {json.loads(l)["review_id"] for l in open(OUT)} if os.path.exists(OUT) else set(); t0 = time.time()
    print(f"resume: {len(done)}/{N} done", flush=True)
    with open(OUT, "a") as f:
        for i, r in enumerate(rows):
            if r["review_id"] in done: continue
            out = generate(model, tok, prompt=build_prompt(r["review_text"][:1200], cfg), max_tokens=120, verbose=False)
            f.write(json.dumps({"review_id": r["review_id"], "gt_rating": r["gt_rating"], "review_text": r["review_text"], "output": out}) + "\n"); f.flush()
            if (i + 1) % 10 == 0: print(f"{i+1}/{N} {time.time()-t0:.0f}s", flush=True)
    print("done", OUT)
else:
    G = [json.loads(l) for l in open(OUT)]; n = len(G); P = [parse_rating(g["output"]) for g in G]
    desc = {1: "dissatisfaction", 2: "softened", 3: "neutral", 4: "positive experience", 5: "satisfaction"}
    wrong = sum(p is not None and p != g["gt_rating"] for g, p in zip(G, P)); sev = [(g, p) for g, p in zip(G, P) if p is not None and abs(p - g["gt_rating"]) >= 2]
    mism = sum(p is not None and p in desc and desc[p] not in g["output"].lower().split("(rating")[0] and "rating " + str(p) not in g["output"].lower() for g, p in zip(G, P))
    invent = sum(bool(re.search(r"\n\s*(Review|Example)\s*:", g["output"])) for g in G)
    print(f"n={n} unparseable {sum(p is None for p in P)}  wrong rating {wrong} ({wrong/n:.0%})  description doesn't match own rating {mism}  runs on into invented 'Review:/Example:' {invent}")
    print("severe misses (>=2 stars):", len(sev))
    for g, p in sev[:5]: print(f" gt={g['gt_rating']} pred={p} | {g['review_text'][:160]!r}\n   -> {g['output'][:200]!r}")
    print("\nsample raw outputs:"); [print(" ", repr(g["output"][:220])) for g in G[:4]]
