"""Paper-protocol test set: 4,000 random held-out Kaggle rows, NATURAL rating distribution (paper §3:
'4,000 reviews randomly selected from the remaining data'). Disjoint from the balanced eval + dev sets.
Caveat: the committed adapter's own training rows were not recorded, so overlap is excluded only statistically
(4K of ~568K rows)."""
import json, random
import pandas as pd
ex = {int(json.loads(l)["review_id"].rsplit("_", 1)[1]) for f in ("kaggle_eval_rows", "kaggle_dev_rows") for l in open(f".scratch/{f}.jsonl")}
df = pd.read_csv(".scratch/kaggle_data/Reviews.csv", usecols=["ProductId", "Score", "Text"]).dropna()
idx = [i for i in df.index if i not in ex and len(str(df.at[i, "Text"])) >= 20]
random.Random(2024).shuffle(idx)
rows = [{"review_id": f"kaggle_{int(df.at[i,'Score'])}_{i}", "asin": str(df.at[i, "ProductId"]), "gt_rating": int(df.at[i, "Score"]),
         "review_text": str(df.at[i, "Text"])[:800], "meta_title": "", "meta_description": ""} for i in idx[:4000]]
assert not {int(r["review_id"].rsplit("_", 1)[1]) for r in rows} & ex
with open("v2_reflective/paper_protocol_rows.jsonl", "w") as f:
    for r in rows: f.write(json.dumps(r) + "\n")
print(len(rows), pd.Series([r["gt_rating"] for r in rows]).value_counts(normalize=True).sort_index().round(3).to_dict())
