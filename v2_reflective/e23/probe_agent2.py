"""Phase F2 (declared LOG.md 2026-10-22): multi-layer hidden-state extraction. Chunked/resumable."""
import json, os, random, sys, time, numpy as np
sys.path.insert(0, "src")
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.base import create_attention_mask
from agentic_sentiment.phase1.data.sampling import stratified_sample
D = "v2_reflective/e23/"; os.makedirs(D + "hid2", exist_ok=True); CH = 500; LAY = (16, 20, 24)
model, tok = load("v2_reflective/mlx_model"); sp = tok.encode(" ", add_special_tokens=False)
L = lambda n: [json.loads(l) for l in open(f"{D}{n}.jsonl")]
tr = [{"review_text": r["review_text"], "gt_rating": r["gt_rating"], "rating": r["gt_rating"], "review_id": r["review_id"]} for r in L("evid_train")]
fit = list(stratified_sample(tr, samples_per_rating=2000, rng=random.Random(11)))
assert len(fit) == 10000 and [r["review_id"] for r in fit] == json.load(open(D + "hid/ids.json"))["fit"]
rows = {"dev": L("dev"), "natural_dev": L("natural_dev"), "eval": L("eval"), "natural_test": L("natural_test"), "fit": fit}
def fwd(ids):
    h = model.model.embed_tokens(mx.array(ids)[None]); mask = create_attention_mask(h, None); out = []
    for i, l in enumerate(model.model.layers[:max(LAY)], 1):
        h = l(h, mask, None)
        if i in LAY: out.append(h[0, -1])
        if i == 20: out.append(h[0].mean(0))
    return mx.concatenate(out)
if os.environ.get("SELFTEST"):
    ids = tok.encode("Review: great\nRating (1-5):", add_special_tokens=False) + sp
    a = model.model(mx.array(ids)[None])[0, -1]; h = model.model.embed_tokens(mx.array(ids)[None]); m = create_attention_mask(h, None)
    for l in model.model.layers: h = l(h, m, None)
    b = model.model.norm(h)[0, -1]; print("selftest maxdiff", float(mx.abs(a - b).max()), "vec", fwd(ids).shape); sys.exit()
t0 = time.time()
for s, rs in rows.items():
    for c in range(0, len(rs), CH):
        f = f"{D}hid2/{s}_{c//CH:03d}.npy"
        if os.path.exists(f): continue
        out = []
        for r in rs[c:c + CH]:
            ids = tok.encode(f"Review: {r['review_text'][:1200]}\nRating (1-5):", add_special_tokens=False) + sp
            v = fwd(ids); mx.eval(v); out.append(np.array(v.astype(mx.float16)))
        np.save(f, np.stack(out)); print(s, c, f"{time.time()-t0:.0f}s", flush=True)
print("extraction done")
