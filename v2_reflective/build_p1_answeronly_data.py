"""Phase 1 retrain data, paper-format prompt, completion-only loss. 400 per star (2,000 total, the paper's Table 1 size)."""
import json, sys, os
import pandas as pd
sys.path.insert(0, "src")
from agentic_sentiment.phase1.config import DataConfig
from agentic_sentiment.phase1.data.preprocessing import prepare_conversation_format
held = set()
for f in ("kaggle_eval_rows", "kaggle_dev_rows"):
    held |= {int(json.loads(l)["review_id"].rsplit("_", 1)[1]) for l in open(f".scratch/{f}.jsonl")}
held |= {int(json.loads(l)["review_id"].rsplit("_", 1)[1]) for l in open("v2_reflective/paper_protocol_rows.jsonl")}
df = pd.read_csv(".scratch/kaggle_data/Reviews.csv", usecols=["Score", "Text"]).dropna()
df["Text"] = df["Text"].astype(str).str.strip()
df = df[df["Text"].str.len().between(20, 800) & ~df.index.isin(held)].drop_duplicates("Text")
cfg = DataConfig(); os.makedirs("v2_reflective/p1_data", exist_ok=True)
picked = pd.concat([g.sample(420, random_state=33) for _, g in df.groupby("Score")])
train = picked.groupby("Score").head(400).sample(frac=1, random_state=33); valid = picked.drop(train.index)
for name, part in (("train", train), ("valid", valid)):
    with open(f"v2_reflective/p1_data/{name}.jsonl", "w") as f:
        for t, r in zip(part["Text"], part["Score"]):
            ex = prepare_conversation_format(t, int(r), cfg)
            f.write(json.dumps({"prompt": ex["prompt"], "completion": " " + ex["answer"]}) + "\n")
print(len(train), len(valid))
