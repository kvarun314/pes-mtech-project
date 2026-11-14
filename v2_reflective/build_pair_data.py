"""Pair-adjudicator training data: 'is this review rating A or B?' on neighbouring (and one-apart) stars.
Rows exclude the balanced eval, dev and paper-protocol sets. Both answer orders appear. prompt/completion format so
mlx_lm can mask the prompt (loss only on the 1-token answer)."""
import json, random
import pandas as pd
RUB = {1: "very dissatisfied: serious flaws, strongly negative", 2: "negative but softened: some positives or unmet expectations",
       3: "neutral or mixed: both good and bad, mediocre", 4: "positive overall: praise with a few shortcomings",
       5: "highly satisfied: praise, little or no complaint, would recommend"}
def prompt(text, a, b):
    return (f"A customer review is below. Its star rating is either {a} or {b}.\n"
            f"{a} stars = {RUB[a]}.\n{b} stars = {RUB[b]}.\n\nReview:\n{text}\n\nThe rating is {a} or {b}. Answer with one digit.\nAnswer:")
if __name__ == "__main__":
    held = set()
    for f in ("kaggle_eval_rows", "kaggle_dev_rows"):
        held |= {int(json.loads(l)["review_id"].rsplit("_", 1)[1]) for l in open(f".scratch/{f}.jsonl")}
    held |= {int(json.loads(l)["review_id"].rsplit("_", 1)[1]) for l in open("v2_reflective/paper_protocol_rows.jsonl")}
    df = pd.read_csv(".scratch/kaggle_data/Reviews.csv", usecols=["Score", "Text"]).dropna()
    df["Text"] = df["Text"].astype(str).str.replace("<br />", " ").str.strip()
    df = df[df["Text"].str.len().between(20, 600) & ~df.index.isin(held)].drop_duplicates("Text")
    pools = {r: df[df["Score"] == r].sample(frac=1, random_state=11).Text.tolist() for r in range(1, 6)}
    take = {r: 0 for r in pools}
    rng = random.Random(5)
    out, val = [], []
    spec = [((1, 2), 300), ((2, 3), 300), ((3, 4), 300), ((4, 5), 300), ((1, 3), 100), ((2, 4), 100), ((3, 5), 100)]
    for (lo, hi), n in spec:
        for i in range(n):  # n examples per pair, half each true rating, random display order
            true = lo if i % 2 == 0 else hi
            text = pools[true][take[true]]; take[true] += 1
            a, b = (lo, hi) if rng.random() < 0.5 else (hi, lo)
            rec = {"prompt": prompt(text, a, b), "completion": " " + str(true)}
            (val if i >= n - 8 else out).append(rec)
    rng.shuffle(out)
    import os; os.makedirs("v2_reflective/pair_data", exist_ok=True)
    for name, part in (("train", out), ("valid", val)):
        with open(f"v2_reflective/pair_data/{name}.jsonl", "w") as f:
            for r in part: f.write(json.dumps(r) + "\n")
    print(len(out), len(val))
