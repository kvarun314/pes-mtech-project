"""Phase 1 close-out experiments on the committed adapter via llama-server (port 8090). First 300 balanced eval rows.
A  = ANALYST_PROMPT + one-shot chosen with a different rating than truth (as used for every reported number; from kaggle_eval_probs.jsonl)
B  = ANALYST_PROMPT + one-shot chosen at random, independent of truth   -> A vs B = effect of the label leak
C  = training-format build_prompt + random one-shot                       -> B vs C = effect of prompt-format mismatch
D  = ANALYST_PROMPT, no example (zero-shot)
H  = hallucination study: full generation (120 tokens) on first 150 rows, raw text kept."""
import json, math, random, sys, requests
sys.path.insert(0, "src")
from agentic_sentiment.agents.graph import ANALYST_PROMPT, SENTIMENT_INSTRUCTION
from agentic_sentiment.eval.one_shot import COT_PHRASE, build_one_shot_pool, format_one_shot_example
from agentic_sentiment.phase1.config import DataConfig
from agentic_sentiment.phase1.data.preprocessing import build_prompt
URL = "http://localhost:8090/completion"
rows = [json.loads(l) for l in open(".scratch/kaggle_eval_rows.jsonl")]
pool = build_one_shot_pool(rows); cfg = DataConfig()

def probs(prompt):
    r = requests.post(URL, timeout=300, json={"prompt": prompt, "n_predict": 2, "temperature": 0, "n_probs": 50}); r.raise_for_status()
    s = r.json()["completion_probabilities"]; top = s[1 if s[0]["token"] == " " and len(s) > 1 else 0]["top_logprobs"]
    lp = {t["token"].strip(): t["logprob"] for t in top}
    return [math.exp(lp[str(k)]) if str(k) in lp else 0.0 for k in range(1, 6)]

def rand_shot(row): 
    p = random.Random("rand" + row["review_id"]).choice(pool); return format_one_shot_example(p["review_text"], p["gt_rating"])
def analyst(row, shot):
    return ANALYST_PROMPT.format(instruction=SENTIMENT_INSTRUCTION, cot_block=COT_PHRASE, one_shot_block=f"Example:\n{shot}\n\n" if shot else "",
                                 metadata_block="", review_text=row["review_text"], critique_block="")
variants = {"B_random_shot": lambda r: analyst(r, rand_shot(r)),
            "C_train_format": lambda r: build_prompt(r["review_text"], cfg, one_shot_example=rand_shot(r)),
            "D_zero_shot": lambda r: analyst(r, None)}
R300 = rows[:300]
for name, mk in variants.items():
    with open(f"v2_reflective/p1final_{name}.jsonl", "w") as f:
        for r in R300:
            f.write(json.dumps({"review_id": r["review_id"], "gt_rating": r["gt_rating"], "probs": probs(mk(r))}) + "\n"); f.flush()
    print("done", name, flush=True)
# hallucination study: full generations under the standard (A) prompt
import agentic_sentiment.eval.one_shot as os_
with open("v2_reflective/p1final_H_generations.jsonl", "w") as f:
    for r in rows[:150]:
        rng = random.Random(r["review_id"]); cands = [p for p in pool if p["gt_rating"] != r["gt_rating"]] or pool
        c = rng.choice(cands); shot = format_one_shot_example(c["review_text"], c["gt_rating"])
        out = requests.post(URL, timeout=600, json={"prompt": analyst(r, shot), "n_predict": 120, "temperature": 0}).json()["content"]
        f.write(json.dumps({"review_id": r["review_id"], "gt_rating": r["gt_rating"], "review_text": r["review_text"], "output": out}) + "\n"); f.flush()
print("done H", flush=True)
