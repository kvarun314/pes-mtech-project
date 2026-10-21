"""Arm R10 data: 10K raw balanced rows from evid_train, same prompt builder as build_p1_data.py. Declared LOG.md 2026-10-21."""
import json, random, sys
sys.path.insert(0, "src")
from agentic_sentiment.phase1.config import DataConfig
from agentic_sentiment.phase1.data.preprocessing import prepare_conversation_format
from agentic_sentiment.phase1.data.sampling import stratified_sample
rows = [{"text": r["review_text"][:1200], "rating": r["gt_rating"]} for r in map(json.loads, open("v2_reflective/e23/evid_train.jsonl"))]
cfg = DataConfig(use_one_shot=False); rs = list(stratified_sample(rows, samples_per_rating=2000, rng=random.Random(7))); random.Random(7).shuffle(rs)
parts = {"valid": rs[:200], "train": rs[200:]}; import os; os.makedirs("v2_reflective/e23/p1/data_R10", exist_ok=True)
for name, part in parts.items():
    with open(f"v2_reflective/e23/p1/data_R10/{name}.jsonl", "w") as f:
        for r in part:
            ex = prepare_conversation_format(r["text"], r["rating"], cfg); f.write(json.dumps({"prompt": ex["prompt"], "completion": " " + ex["answer"]}) + "\n")
    print(name, len(part))
