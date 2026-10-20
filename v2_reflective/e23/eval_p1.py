"""First-digit probabilities of an Electronics-trained Phase 1 adapter. usage: eval_p1.py ARM split   (split = dev|eval|natural_test)
Same zero-shot training-format prompt as training (build_prompt without one-shot), review truncated to 1,200 chars."""
import json, math, os, sys, time
sys.path.insert(0, "src")
import mlx.core as mx
from mlx_lm import load
from agentic_sentiment.phase1.config import DataConfig
from agentic_sentiment.phase1.data.preprocessing import build_prompt
arm, split = sys.argv[1], sys.argv[2]
model, tok = load("v2_reflective/mlx_model", adapter_path=None if arm == "base" else f"v2_reflective/e23/p1/adapter_{arm}")  # arm "base" = LLaMA-3-8B Q4 without LoRA
dig = {d: tok.encode(str(d), add_special_tokens=False)[-1] for d in range(1, 6)}; sp = tok.encode(" ", add_special_tokens=False)
cfg = DataConfig(use_one_shot=False); out = f"v2_reflective/e23/p1/probs_{arm}_{split}.jsonl"
# resume: keep rows already written (append mode), skip their ids; log progress/timing so the run is documented
done = {json.loads(l)["review_id"] for l in open(out)} if os.path.exists(out) else set(); t0 = time.time(); n = 0
print(f"resume: {len(done)} rows already in {out}", flush=True)
with open(out, "a") as f:
    for l in open(f"v2_reflective/e23/{split}.jsonl"):
        r = json.loads(l)
        if r["review_id"] in done: continue
        n += 1
        if n % 100 == 0: print(f"{len(done)+n} rows, {time.time()-t0:.0f}s since start/resume", flush=True)
        ids = tok.encode(build_prompt(r["review_text"][:1200], cfg), add_special_tokens=False) + sp
        lg = model(mx.array(ids)[None])[0, -1]; v = [float(lg[dig[k]]) for k in range(1, 6)]; m = max(v); e = [math.exp(x - m) for x in v]; s = sum(e)
        f.write(json.dumps({"review_id": r["review_id"], "gt_rating": r["gt_rating"], "probs": [x / s for x in e]}) + "\n"); f.flush()
print("done", out)
