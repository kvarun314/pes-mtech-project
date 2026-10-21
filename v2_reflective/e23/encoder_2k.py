"""DistilBERT baseline on the same 2,000 arm-R rows. Pre-declared in LOG.md 2026-10-21. Resumable: skips training if saved model exists; scoring skips existing probs files."""
import json, os, random, sys, time, numpy as np, torch
sys.path.insert(0, "src")
from agentic_sentiment.phase1.data.sampling import stratified_sample
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
D = "v2_reflective/e23/"; CK = D + "p1/enc2k_model"; dev_ = "mps"; torch.manual_seed(0); random.seed(0); np.random.seed(0)
tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
enc = lambda texts: tok(texts, truncation=True, max_length=256, padding=True, return_tensors="pt")
if not os.path.exists(CK):
    rows = [{"text": r["review_text"][:1200], "rating": r["gt_rating"]} for r in map(json.loads, open(D + "evid_train.jsonl"))]
    rs = list(stratified_sample(rows, samples_per_rating=400, rng=random.Random(42))); random.Random(42).shuffle(rs); va, tr = rs[:400], rs[400:]
    print(len(tr), "train", len(va), "valid", flush=True)
    m = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=5).to(dev_)
    opt = torch.optim.AdamW(m.parameters(), lr=5e-5, weight_decay=0.01); E, B = 3, 16; steps = E * ((len(tr) + B - 1) // B); sch = get_linear_schedule_with_warmup(opt, int(.06 * steps), steps)
    t0 = time.time()
    for ep in range(E):
        m.train(); random.Random(ep).shuffle(tr); tl = 0
        for i in range(0, len(tr), B):
            b = tr[i:i + B]; x = enc([r["text"] for r in b]).to(dev_); y = torch.tensor([r["rating"] - 1 for r in b]).to(dev_)
            loss = m(**x, labels=y).loss; loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sch.step(); opt.zero_grad(); tl += loss.item()
        m.eval(); c = 0
        with torch.no_grad():
            for i in range(0, len(va), 32):
                b = va[i:i + 32]; c += (m(**enc([r["text"] for r in b]).to(dev_)).logits.argmax(1).cpu() == torch.tensor([r["rating"] - 1 for r in b])).sum().item()
        print(f"epoch {ep+1} train loss {tl/((len(tr)+B-1)//B):.4f} valid acc {c/len(va):.3f} ({time.time()-t0:.0f}s)", flush=True)
    m.save_pretrained(CK)
m = AutoModelForSequenceClassification.from_pretrained(CK).to(dev_).eval()
for sp in ("dev", "eval", "natural_test"):
    out = f"{D}p1/probs_enc2k_{sp}.jsonl"
    if os.path.exists(out): continue
    rows = [json.loads(l) for l in open(f"{D}{sp}.jsonl")]; res = []
    with torch.no_grad():
        for i in range(0, len(rows), 32):
            b = rows[i:i + 32]; p = torch.softmax(m(**enc([r["review_text"][:1200] for r in b]).to(dev_)).logits, -1).cpu().numpy()
            res += [{"review_id": r["review_id"], "gt_rating": r["gt_rating"], "probs": [float(x) for x in q]} for r, q in zip(b, p)]
    with open(out, "w") as f:
        for r in res: f.write(json.dumps(r) + "\n")
    print("scored", sp, len(res), flush=True)
