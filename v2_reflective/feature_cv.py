"""Feature check for the Reflector calibrator. 5-fold CV on the 500 dev rows ONLY (eval stays untouched)."""
import json, re, numpy as np
from textblob import TextBlob
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
rows = {json.loads(l)["review_id"]: json.loads(l) for l in open(".scratch/kaggle_dev_rows.jsonl")}
P, y, T = [], [], []
for l in open(".scratch/kaggle_dev_probs.jsonl"):
    r = json.loads(l); P.append(r["probs"]); y.append(r["gt_rating"] - 1); T.append(rows[r["review_id"]]["review_text"])
X0 = np.log(np.clip(np.array(P), 1e-6, 1)); y = np.array(y)
def feats(t):
    b = TextBlob(t)
    return [np.log1p(len(t)), b.sentiment.polarity, b.sentiment.subjectivity, t.count("!"), len(re.findall(r"\b(but|however|although|though)\b", t.lower()))]
F = np.array([feats(t) for t in T])
names = ["len", "polarity", "subjectivity", "excl", "contrast"]
def cv(X, C):
    accs = []
    for seed in range(5):
        p = cross_val_predict(LogisticRegression(C=C, max_iter=3000), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=seed))
        accs.append((p == y).mean())
    return np.mean(accs)
print("base", (X0.argmax(1) == y).mean())
for C in (1, 10):
    print(f"C={C} logprobs only {cv(X0, C):.3f}")
    for i, n in enumerate(names): print(f"  +{n} {cv(np.c_[X0, F[:, i]], C):.3f}")
    print(f"  +all {cv(np.c_[X0, F], C):.3f}")
