"""SH3 prep (declared LOG.md 2026-11-10): the local DeBERTa agent (D1 checkpoint, run on this Mac) must reproduce the Colab probabilities
before it is used live (robustness check, deletion test, demo). Pass: top star agrees on >= 99% of dev rows and mean |diff| <= 0.01.
usage: deb_local_check.py   (tee to deb_local_check.out)"""
import json, sys, time, numpy as np
sys.path.insert(0, "src")
from agentic_sentiment.v3 import paths as PT
from agentic_sentiment.v3.deberta import DebertaAgent
rows = [json.loads(l) for l in open(PT.D + "dev.jsonl")]; st = {x["review_id"]: x["probs"] for x in map(json.loads, open(f"{PT.D}{PT.DEB_PREFIX}_dev_probs.jsonl"))}
a = DebertaAgent(); print("device", a.dev, flush=True); t0 = time.time()
P = a.probs_many([(r["title"], r["review_text"]) for r in rows]); dt = (time.time() - t0) / len(rows)
S = np.array([st[r["review_id"]] for r in rows]); d = np.abs(P - S); agree = float((P.argmax(1) == S.argmax(1)).mean())
y = np.array([r["gt_rating"] for r in rows]); ok = agree >= 0.99 and d.mean() <= 0.01
print(f"rows {len(rows)} | top-star agreement {agree:.4f} | mean |diff| {d.mean():.5f} | max |diff| {d.max():.4f} | exact local {(P.argmax(1) + 1 == y).mean():.3f} vs stored {(S.argmax(1) + 1 == y).mean():.3f}")
print(f"cost: {1000 * dt:.1f} ms per review (batch 32, {a.dev})")
print("=== CHECK:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
