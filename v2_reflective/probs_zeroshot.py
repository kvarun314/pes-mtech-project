"""First-digit probabilities, committed adapter, ANALYST_PROMPT with NO one-shot (leak-free). usage: src out"""
import json, math, sys, requests
sys.path.insert(0, "src")
from agentic_sentiment.agents.graph import ANALYST_PROMPT, SENTIMENT_INSTRUCTION
from agentic_sentiment.eval.one_shot import COT_PHRASE
src, out = sys.argv[1], sys.argv[2]
done = set()
try: done = {json.loads(l)["review_id"] for l in open(out)}
except FileNotFoundError: pass
with open(out, "a") as f:
    for l in open(src):
        r = json.loads(l)
        if r["review_id"] in done: continue
        p = ANALYST_PROMPT.format(instruction=SENTIMENT_INSTRUCTION, cot_block=COT_PHRASE, one_shot_block="", metadata_block="", review_text=r["review_text"][:2000], critique_block="")
        j = requests.post("http://localhost:8090/completion", timeout=300, json={"prompt": p, "n_predict": 2, "temperature": 0, "n_probs": 50}).json()
        s = j["completion_probabilities"]; top = s[1 if s[0]["token"] == " " and len(s) > 1 else 0]["top_logprobs"]
        lp = {t["token"].strip(): t["logprob"] for t in top}
        f.write(json.dumps({"review_id": r["review_id"], "gt_rating": r["gt_rating"], "probs": [math.exp(lp[str(k)]) if str(k) in lp else 0.0 for k in range(1, 6)]}) + "\n"); f.flush()
