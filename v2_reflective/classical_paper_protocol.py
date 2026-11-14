"""Paper Table 3 classical baselines (same configs) on the SAME 4,000 paper-protocol test rows used for the LLaMA run.
Tests whether the paper's 78.6/87.2/87.6 are consistent with 5-class or with polarity labels."""
import json, sys, time
import pandas as pd
sys.path.insert(0, "src")
from agentic_sentiment.baselines.classical import train_decision_tree, train_svm, train_naive_bayes
test = [json.loads(l) for l in open("v2_reflective/paper_protocol_rows.jsonl")]
held = {int(r["review_id"].rsplit("_", 1)[1]) for r in test}
for f in ("kaggle_eval_rows", "kaggle_dev_rows"):
    held |= {int(json.loads(l)["review_id"].rsplit("_", 1)[1]) for l in open(f".scratch/{f}.jsonl")}
df = pd.read_csv(".scratch/kaggle_data/Reviews.csv", usecols=["Score", "Text"]).dropna()
df = df[~df.index.isin(held)].sample(16000, random_state=42)   # natural distribution, as in the paper's random split
pol = lambda s: 0 if s <= 2 else (1 if s == 3 else 2)
Xtr, ytr = df["Text"].astype(str).tolist(), df["Score"].astype(int).tolist()
Xte, yte = [r["review_text"] for r in test], [r["gt_rating"] for r in test]
print("train", len(Xtr), "test", len(Xte), "always-5", round(sum(y == 5 for y in yte) / len(yte), 4), "always-positive", round(sum(pol(y) == 2 for y in yte) / len(yte), 4))
for name, fit in (("DecisionTree", train_decision_tree), ("SVM(sigmoid,g=1)", train_svm), ("MultinomialNB(a=.2)", train_naive_bayes)):
    t = time.time()
    m5 = fit(Xtr, ytr); p5 = list(m5.predict(Xte))
    mp = fit(Xtr, [pol(y) for y in ytr]); pp = list(mp.predict(Xte))
    a = lambda p, y: round(sum(i == j for i, j in zip(p, y)) / len(y), 4)
    print(f"{name:20s} 5-class {a(p5, yte)}  5-class->polarity {a([pol(x) for x in p5], [pol(y) for y in yte])}  trained-on-polarity {a(pp, [pol(y) for y in yte])}   ({time.time()-t:.0f}s)")
