"""Phase F (declared LOG.md 2026-10-22): base-LLaMA last-token hidden-state probe. Chunked/resumable."""
import json, os, random, sys, time, numpy as np
sys.path.insert(0, "src")
import mlx.core as mx
from mlx_lm import load
from agentic_sentiment.phase1.config import DataConfig
from agentic_sentiment.phase1.data.preprocessing import build_prompt
from agentic_sentiment.phase1.data.sampling import stratified_sample
D = "v2_reflective/e23/"; os.makedirs(D + "hid", exist_ok=True); CH = 500
model, tok = load("v2_reflective/mlx_model"); cfg = DataConfig(use_one_shot=False); sp = tok.encode(" ", add_special_tokens=False)
L = lambda n: [json.loads(l) for l in open(f"{D}{n}.jsonl")]
tr = [{"review_text": r["review_text"], "gt_rating": r["gt_rating"], "rating": r["gt_rating"], "review_id": r["review_id"]} for r in L("evid_train")]
fit = list(stratified_sample(tr, samples_per_rating=2000, rng=random.Random(11)))
assert len(fit) == 10000, len(fit)
rows = {"fit": fit, "dev": L("dev"), "natural_dev": L("natural_dev"), "eval": L("eval"), "natural_test": L("natural_test")}
t0 = time.time()
for s, rs in rows.items():
    for c in range(0, len(rs), CH):
        f = f"{D}hid/{s}_{c//CH:03d}.npy"
        if os.path.exists(f): continue
        out = []
        for r in rs[c:c + CH]:
            ids = tok.encode(f"Review: {r['review_text'][:1200]}\nRating (1-5):", add_special_tokens=False) + sp
            h = model.model(mx.array(ids)[None])[0, -1]; mx.eval(h); out.append(np.array(h.astype(mx.float32)))
        np.save(f, np.stack(out)); print(s, c, f"{time.time()-t0:.0f}s", flush=True)
json.dump({s: [r["review_id"] for r in rs] for s, rs in rows.items()}, open(D + "hid/ids.json", "w"))
print("extraction done")
