"""Phase 1 training data from Amazon 2023 Electronics (evid_train partition; disjoint from dev/eval/natural_test).
P = paper pipeline (TextBlob DQC -> stratified -> VGST -> neutral x2 -> cap 2,000) | R = raw balanced 2,000 (400/star).
Prompt = paper instruction + CoT phrase, zero-shot (documented deviation: no one-shot example), review truncated to 1,200 chars.
Answer = ' {rating}. {description}' (paper-style). 80/20 train/valid."""
import json, random, sys
sys.path.insert(0, "src")
from agentic_sentiment.phase1.config import DataConfig
from agentic_sentiment.phase1.data.preprocessing import prepare_conversation_format
from agentic_sentiment.phase1.data.sampling import apply_paper_preprocessing, stratified_sample

def main():
    rows = [{"text": r["review_text"][:1200], "rating": r["gt_rating"]} for r in map(json.loads, open("v2_reflective/e23/evid_train.jsonl"))]
    cfg = DataConfig(use_one_shot=False)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
    arms = {"P": apply_paper_preprocessing(rows, cfg, tokenizer=tok, max_samples=2000, seed=42),
            "R": stratified_sample(rows, samples_per_rating=400, rng=random.Random(42))}
    for arm, rs in arms.items():
        rs = list(rs); random.Random(42).shuffle(rs); n_val = int(len(rs) * 0.2); parts = {"valid": rs[:n_val], "train": rs[n_val:]}
        from collections import Counter
        print(arm, len(rs), sorted(Counter(r["rating"] for r in rs).items()), "avg words", sum(len(r["text"].split()) for r in rs) / len(rs))
        for name, part in parts.items():
            with open(f"v2_reflective/e23/p1/{arm}_{name}.jsonl", "w") as f:
                for r in part:
                    ex = prepare_conversation_format(r["text"], r["rating"], cfg)
                    f.write(json.dumps({"prompt": ex["prompt"], "completion": " " + ex["answer"]}) + "\n")
if __name__ == "__main__": main()
