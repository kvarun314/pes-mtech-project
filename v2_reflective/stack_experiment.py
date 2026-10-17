"""Does a second, independent signal help? Evidence model = TF-IDF(1-2gram) + LR trained on 100K class-balanced Kaggle reviews
(held-out ids excluded). Stack with leak-free LLaMA zero-shot probs. Fit stacker on 500 dev rows; eval (1000) read once at the end."""
import json, sys, numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
ids = lambda p: {int(json.loads(l)["review_id"].rsplit("_", 1)[1]) for l in open(p)}
held = ids("v2_reflective/paper_protocol_rows.jsonl") | ids(".scratch/kaggle_eval_rows.jsonl") | ids(".scratch/kaggle_dev_rows.jsonl")
df = pd.read_csv(".scratch/kaggle_data/Reviews.csv", usecols=["Score", "Text"]).dropna()
df = df[~df.index.isin(held)]
tr = pd.concat([g.sample(min(20000, len(g)), random_state=42) for _, g in df.groupby("Score")])
print("evidence-model train rows", len(tr), tr.Score.value_counts().sort_index().tolist())
vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=300000, sublinear_tf=True)
Xt = vec.fit_transform(tr.Text.astype(str)); ev = LogisticRegression(C=3, max_iter=300).fit(Xt, tr.Score.astype(int))
def load(pf, rf):
    rows = {json.loads(l)["review_id"]: json.loads(l) for l in open(rf)}
    P, y, T = [], [], []
    for l in open(pf):
        r = json.loads(l); P.append(r["probs"]); y.append(r["gt_rating"] - 1); T.append(rows[r["review_id"]]["review_text"].replace("<br />", " "))
    P = np.clip(np.array(P), 1e-6, 1); E = np.clip(ev.predict_proba(vec.transform(T)), 1e-6, 1)
    return np.log(P), np.log(E), np.array(y), np.log1p([len(t) for t in T])
Ld, Ed, yd, Nd = load("v2_reflective/zs_dev_probs.jsonl", ".scratch/kaggle_dev_rows.jsonl")
Le, Ee, ye, Ne = load("v2_reflective/zs_eval_probs.jsonl", ".scratch/kaggle_eval_rows.jsonl")
cv = StratifiedKFold(5, shuffle=True, random_state=0)
print("\nDEV 5-fold CV exact: llama", (Ld.argmax(1) == yd).mean().round(3), " evidence alone", (Ed.argmax(1) == yd).mean().round(3))
cfg = {"llama+evidence": (np.c_[Ld, Ed], np.c_[Le, Ee]), "llama+evidence+len": (np.c_[Ld, Ed, Nd], np.c_[Le, Ee, Ne]), "llama+len (no evidence)": (np.c_[Ld, Nd], np.c_[Le, Ne])}
for n, (a, b) in cfg.items():
    for C in (0.1, 1):
        print(f"  {n:24s} C={C}: {np.mean([(cross_val_predict(LogisticRegression(C=C, max_iter=3000), a, yd, cv=StratifiedKFold(5, shuffle=True, random_state=s)) == yd).mean() for s in range(3)]):.3f}")
# eval, pre-declared: llama+evidence+len, C=1
base = Le.argmax(1) == ye; rng = np.random.default_rng(0)
print("\nEVAL (1000 held-out, once): llama alone", base.mean().round(3), " evidence alone", (Ee.argmax(1) == ye).mean().round(3))
for n in ("llama+evidence", "llama+evidence+len"):
    a, b = cfg[n]; p = LogisticRegression(C=1, max_iter=3000).fit(a, yd).predict(b); ok = p == ye; d = ok.astype(float) - base
    ci = np.percentile([d[rng.integers(0, len(d), len(d))].mean() for _ in range(3000)], [2.5, 97.5])
    rec = [round(float(((p == k) & (ye == k)).sum() / (ye == k).sum()), 2) for k in range(5)]
    print(f"  {n}: exact {ok.mean():.1%} within1 {(abs(p-ye)<=1).mean():.1%} MAE {abs(p-ye).mean():.3f} delta {d.mean()*100:+.1f} CI [{ci[0]*100:+.1f},{ci[1]*100:+.1f}] fixed {int((ok&~base).sum())} regressed {int((base&~ok).sum())} recall {rec}")
a, b = cfg["llama+evidence"]; p = LogisticRegression(C=1, max_iter=3000).fit(a, yd).predict(b); ok = p == ye; eb = Ee.argmax(1) == ye; d = ok.astype(float) - eb
ci = np.percentile([d[rng.integers(0, len(d), len(d))].mean() for _ in range(3000)], [2.5, 97.5])
print(f"stack vs EVIDENCE ALONE: {d.mean()*100:+.1f} CI [{ci[0]*100:+.1f},{ci[1]*100:+.1f}]  evidence-alone vs LLaMA: {(eb.astype(float)-base).mean()*100:+.1f}")
