"""Eval the answer-only Phase 1 retrain on the first 300 balanced eval rows (same prompt/parser as .scratch/mlx_eval_scale.py)."""
import json, sys
sys.path.insert(0, "src")
from mlx_lm import generate, load
from agentic_sentiment.agents.parsing import parse_rating
from agentic_sentiment.phase1.config import DataConfig
from agentic_sentiment.phase1.data.preprocessing import build_prompt
N = 300
rows = [json.loads(l) for l in open(".scratch/kaggle_eval_rows.jsonl")][:N]
base = {r["review_id"]: r["analyst_rating"] for r in map(json.loads, open(".scratch/kaggle_text_only.jsonl"))}
print(f"[committed adapter, llama.cpp] exact {sum(base[r['review_id']] == r['gt_rating'] for r in rows)/N:.2%}")
model, tok = load("v2_reflective/mlx_model", adapter_path="v2_reflective/p1_answeronly_adapter")
cfg = DataConfig()
preds = [parse_rating(generate(model, tok, build_prompt(r["review_text"], cfg), max_tokens=20, verbose=False)) or 3 for r in rows]
gt = [r["gt_rating"] for r in rows]
print(f"[answer-only, 2,000 rows] exact {sum(p==g for p,g in zip(preds,gt))/N:.2%}  within1 {sum(abs(p-g)<=1 for p,g in zip(preds,gt))/N:.2%}")
for s in range(1, 6):
    k = [i for i in range(N) if gt[i] == s]; print(f"  {s}* recall {sum(preds[i]==s for i in k)/len(k):.2f} (n={len(k)})")
json.dump([{"id": r["review_id"], "gt": g, "pred": p} for r, g, p in zip(rows, gt, preds)], open("v2_reflective/p1_answeronly_eval.json", "w"))
