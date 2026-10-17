"""Does the Phase 1 preprocessing help the EVIDENCE model? Electronics 2023. Validation = 10K held-out rows from evid_train (raw, unfiltered, balanced).
A raw 90K | B TextBlob-DQC 90K | C random 10K (control) | D TextBlob+VGST 10K | E TextBlob + random 10K (control for D)."""
import json, random, sys, time
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, "src")
from agentic_sentiment.phase1.data.sampling import filter_by_textblob_polarity, vgst_sample
def main():
    global val
    rows = [json.loads(l) for l in open("v2_reflective/e23/evid_train.jsonl")]
    rng = random.Random(7); rng.shuffle(rows)
    val, tr = rows[:10000], rows[10000:]
    mk = lambda rs: [{"text": r["review_text"], "rating": r["gt_rating"], "title": r["title"]} for r in rs]
    tr, val = mk(tr), mk(val)
    t = time.time(); dqc = filter_by_textblob_polarity(tr); print(f"TextBlob DQC kept {len(dqc)}/{len(tr)} ({len(dqc)/len(tr):.1%}) in {time.time()-t:.0f}s", flush=True)
    from collections import Counter
    print(" kept per star", sorted(Counter(r["rating"] for r in dqc).items()), flush=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
    class T:  # cache token sets so VGST's inner loop is cheap
        def __init__(s): s.c = {}
        def encode(s, text, add_special_tokens=False):
            if text not in s.c: s.c[text] = tok.encode(text, add_special_tokens=False)
            return s.c[text]
    t = time.time(); vg = vgst_sample(dqc, target_size=10000, batch_size=32, wishlist_len=10, tokenizer=T(), rng=random.Random(7)); print(f"VGST 10K in {time.time()-t:.0f}s", flush=True)
    lw = lambda rs: np.mean([len(r["text"].split()) for r in rs])
    def fit_eval(name, rs):
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=300000, sublinear_tf=True)
        X = vec.fit_transform([r["text"] for r in rs]); m = LogisticRegression(C=3, max_iter=300).fit(X, [r["rating"] for r in rs])
        p = m.predict(vec.transform([r["text"] for r in val])); y = np.array([r["rating"] for r in val]); p = np.array(p)
        se = np.sqrt((p == y).mean() * (1 - (p == y).mean()) / len(y))
        print(f"{name:34s} n={len(rs):6d} avg words {lw(rs):5.1f}  val exact {(p==y).mean():.3f} (+-{1.96*se:.3f})  within1 {(abs(p-y)<=1).mean():.3f}  star mix {sorted(Counter(r['rating'] for r in rs).items())}", flush=True)
    r2 = random.Random(11)
    fit_eval("A raw balanced", tr)
    fit_eval("B TextBlob-DQC", dqc)
    fit_eval("C random 10K (control)", r2.sample(tr, 10000))
    fit_eval("E TextBlob + random 10K (control)", r2.sample(dqc, 10000))
    fit_eval("D TextBlob + VGST 10K", vg)
    from agentic_sentiment.phase1.data.sampling import stratified_sample
    k = min(sum(r["rating"] == s for r in dqc) for s in range(1, 6))
    fit_eval(f"F TextBlob then re-balance ({k}/star)", stratified_sample(dqc, samples_per_rating=k, rng=random.Random(5)))
    fit_eval(f"G raw balanced same size ({k}/star)", stratified_sample(tr, samples_per_rating=k, rng=random.Random(5)))

if __name__ == '__main__':
    main()
