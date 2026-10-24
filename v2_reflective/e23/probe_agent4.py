"""Phase A1 (declared LOG.md 2026-10-24 night): L20 last-token state only (the P3 feature), same prompt and model as probe_agent3, for fit3. Resumable chunks in hid4/."""
import json, os, sys, time, numpy as np
sys.path.insert(0, "src")
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.base import create_attention_mask
D = "v2_reflective/e23/"; os.makedirs(D + "hid4", exist_ok=True); CH = 500
model, tok = load("v2_reflective/mlx_model"); sp = tok.encode(" ", add_special_tokens=False)
L = lambda n: [json.loads(l) for l in open(f"{D}{n}.jsonl")]
def fwd(ids):
    h = model.model.embed_tokens(mx.array(ids)[None]); mask = create_attention_mask(h, None)
    for l in model.model.layers[:20]: h = l(h, mask, None)
    return h[0, -1]
rows = {"fit3": L("fit3")}; t0 = time.time()
for s, rs in rows.items():
    for c in range(0, len(rs), CH):
        f = f"{D}hid4/{s}_{c//CH:03d}.npy"
        if os.path.exists(f): continue
        out = []
        for r in rs[c:c + CH]:
            ids = tok.encode(f"Review: {r['review_text'][:1200]}\nRating (1-5):", add_special_tokens=False) + sp
            v = fwd(ids); mx.eval(v); out.append(np.array(v.astype(mx.float16)))
        np.save(f, np.stack(out)); print(s, c, f"{time.time()-t0:.0f}s", flush=True)
json.dump({s: [r["review_id"] for r in rs] for s, rs in rows.items()}, open(D + "hid4/ids.json", "w")); print("extraction done")
