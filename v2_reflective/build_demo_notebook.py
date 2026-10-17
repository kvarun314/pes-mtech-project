"""Generates Presentation_Demo.ipynb (small-sample live demo of the pipeline). Run: .venv/bin/python v2_reflective/build_demo_notebook.py"""
import nbformat as nbf
nb = nbf.v4.new_notebook(); C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip("\n")))

md("""
# Presentation Demo — Phase 1 reproduction and Phase 2 agentic pipeline
Small-sample, CPU-only demo of the Phase 1 reproduction and the Phase 2 pipeline. Everything runs in about 1–2 minutes from saved files; the one optional **live model** cell needs the local llama.cpp server and falls back to saved results if it is not running.

| Section | What it shows |
|---|---|
| 1 | The Phase 1 data pipeline: TextBlob filter, stratified sampling, VGST, neutral oversampling, the prompt |
| 2 | Phase 1 results: metrics, per-star table, confusion matrix, optional live prediction |
| 3 | Reproducing the paper: classical baselines against LLaMA, and a data-scale curve |
| 4 | Phase 2: the critic, a real LangGraph run with a scripted model, why the override never fires |
| 5 | Where reflection can help: the uncertainty gate and the adjudicator |

Change `SAMPLE_N` below to use a smaller or larger sample.
""")
code("""
import sys, json, random, zlib, re, warnings
from pathlib import Path
from collections import Counter
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib.pyplot as plt

ROOT = Path.cwd()
while not (ROOT / "src").exists() and ROOT != ROOT.parent: ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "v2_reflective"))
SAMPLE_N, SEED = 300, 7
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})
BLUE, GREY, ORANGE, RED = "#2C5AA0", "#8792A0", "#C25B1E", "#A93636"
rows_all = [json.loads(l) for l in open(ROOT / "v2_reflective/paper_protocol_rows.jsonl")]
probs_all = {r["review_id"]: r for r in map(json.loads, open(ROOT / "v2_reflective/paper_protocol_probs.jsonl"))}
print(f"project root: {ROOT}\\n{len(rows_all)} saved test rows, {len(probs_all)} saved model outputs")
""")

md("""
## 1. The Phase 1 data pipeline
The paper builds its training set in five steps. Each step below is the project's real function from `src/agentic_sentiment/phase1/data/sampling.py`, run on a small example.

### 1a. TextBlob quality filter (DQC)
Drop a review when the words disagree with the star rating: negative text with a rating above 3, or positive text with a rating below 3. Zero polarity or a TextBlob failure keeps the row.
""")
code("""
from textblob import TextBlob
from agentic_sentiment.phase1.data.sampling import filter_by_textblob_polarity
toy = [
 {"text": "Absolutely love it, works perfectly and arrived early.", "rating": 5},
 {"text": "Terrible. Broke after two days, total waste of money.", "rating": 5},     # contradicts: negative text, 5 stars
 {"text": "Awful quality, I hate it.", "rating": 1},
 {"text": "Great product, so happy with this purchase!", "rating": 1},               # contradicts: positive text, 1 star
 {"text": "It is okay, does the job but nothing special.", "rating": 3},
 {"text": "Pretty good overall but the battery could be better.", "rating": 4},
 {"text": "Not what I expected, a bit disappointing.", "rating": 2},
 {"text": "The box arrived on Tuesday.", "rating": 3},                                  # polarity 0 -> kept
]
kept = filter_by_textblob_polarity(toy)
kept_texts = {r["text"] for r in kept}
df = pd.DataFrame([{"rating": r["rating"], "polarity": round(TextBlob(r["text"]).sentiment.polarity, 2),
                    "decision": "keep" if r["text"] in kept_texts else "DROP", "text": r["text"][:60]} for r in toy])
df.style.apply(lambda s: ["background-color:#f6d5d5" if v == "DROP" else "" for v in s], subset=["decision"])
""")
md("### 1b. Stratified sampling — remove the five-star skew\nThe raw reviews are about 64% five-star. A model trained on that would learn to say \"5\" and still look accurate.")
code("""
from agentic_sentiment.phase1.data.sampling import stratified_sample, vgst_sample, oversample_neutral
raw = [{"text": r["review_text"], "rating": r["gt_rating"]} for r in rows_all]          # natural distribution
bal = stratified_sample(raw, samples_per_rating=150, rng=random.Random(SEED))
cnt = lambda rs: [Counter(r["rating"] for r in rs)[k] for k in range(1, 6)]
fig, ax = plt.subplots(1, 2, figsize=(8, 2.8), sharey=False)
for a, (title, rs, col) in zip(ax, [("Raw (natural)", raw, GREY), ("After stratified sampling", bal, BLUE)]):
    a.bar(range(1, 6), cnt(rs), color=col); a.set_title(title, fontsize=10); a.set_xlabel("stars")
    for i, v in enumerate(cnt(rs), 1): a.text(i, v, v, ha="center", va="bottom", fontsize=8)
plt.tight_layout(); plt.show()
""")
md("""
### 1c. VGST — pick the reviews that add the most new vocabulary
Draw a random batch of 32 candidates, keep the one that adds the most new token ids to what is already selected, repeat 10 times to fill a wishlist, take the best of the wishlist, and repeat until the target size is reached. Below, a simple stand-in tokenizer maps each word to an id (the real run uses the LLaMA tokenizer). We compare token coverage against a random sample of the same size.
""")
code("""
class WordTok:                                    # stand-in for the LLaMA tokenizer: one id per distinct word
    def encode(self, text, add_special_tokens=False): return [zlib.crc32(w.encode()) for w in re.findall(r"[a-z']+", text.lower())]
tok = WordTok(); pool = [{"text": r["review_text"], "rating": r["gt_rating"]} for r in rows_all[:2000]]
target = 100
v = vgst_sample(pool, target_size=target, batch_size=32, wishlist_len=10, tokenizer=tok, rng=random.Random(SEED))
rnd = random.Random(SEED).sample(pool, target)
cover = lambda rs: len({t for r in rs for t in tok.encode(r["text"])})
cv, cr = cover(v), cover(rnd)
print(f"unique word ids covered by {target} reviews:  VGST {cv}   random {cr}   ({cv/cr - 1:+.0%})")
print(f"average review length (words):                VGST {np.mean([len(tok.encode(r['text'])) for r in v]):.0f}   random {np.mean([len(tok.encode(r['text'])) for r in rnd]):.0f}")
plt.figure(figsize=(3.6, 2.6)); plt.bar(["Random", "VGST"], [cr, cv], color=[GREY, BLUE]); plt.title("Vocabulary covered", fontsize=10); plt.tight_layout(); plt.show()
""")
md("**Read it honestly:** VGST covers far more vocabulary, but largely because it prefers much longer reviews (compare the average lengths above): a long review adds more new tokens than a short one. The paper claims diversity helps; we implemented it as described but never tested accuracy with and without it.")
md("### 1d. Oversample neutral (3-star) reviews, then the prompt the model sees")
code("""
b2 = oversample_neutral(bal, neutral_rating=3, multiplier=2.0, rng=random.Random(SEED))
print("star counts before:", cnt(bal), "  after oversampling neutral:", cnt(b2))
from agentic_sentiment.phase1.config import DataConfig
from agentic_sentiment.phase1.data.preprocessing import build_prompt, format_one_shot_example, prepare_conversation_format
cfg = DataConfig(); ex = rows_all[3]
shot = format_one_shot_example("Does what it says, nothing more, nothing less.", 3, cfg)
print("="*30, "PROMPT", "="*30); print(build_prompt(ex["review_text"][:300], cfg, one_shot_example=shot))
print("="*30, "TRAINING ANSWER", "="*30); print(prepare_conversation_format(ex["review_text"], ex["gt_rating"], cfg)["answer"], "  <- loss is computed on this part only")
""")

md("""
## 2. Phase 1 results
The model's output for every test review is saved as the probability of each rating 1–5 (read at the answer token). The prediction is the most likely rating. Scores below use a random sample of `SAMPLE_N` reviews, with the full 4,000-row numbers for comparison.
""")
code("""
pol = lambda s: 0 if s <= 2 else (1 if s == 3 else 2)
def evaluate(ids):
    gt = np.array([probs_all[i]["gt_rating"] for i in ids]); pr = np.array([int(np.argmax(probs_all[i]["probs"])) + 1 for i in ids])
    return gt, pr, {"5-class exact": (gt == pr).mean(), "within ±1": (abs(gt - pr) <= 1).mean(), "MAE (stars)": abs(gt - pr).mean(),
                    "polarity": (np.vectorize(pol)(gt) == np.vectorize(pol)(pr)).mean(),
                    "always-5 baseline": (gt == 5).mean(), "always-positive (polarity)": (np.vectorize(pol)(gt) == 2).mean()}
ids_s = random.Random(SEED).sample(list(probs_all), SAMPLE_N); ids_f = list(probs_all)
gt_s, pr_s, m_s = evaluate(ids_s); gt_f, pr_f, m_f = evaluate(ids_f)
pd.DataFrame({f"sample (n={SAMPLE_N})": m_s, "all 4,000 rows": m_f}).T.round(3)
""")
code("""
from sklearn.metrics import precision_recall_fscore_support as prf, confusion_matrix
P, R, F, S = prf(gt_f, pr_f, labels=[1, 2, 3, 4, 5], zero_division=0)
print("Per-star results, all 4,000 rows (natural distribution)")
display(pd.DataFrame({"n": S, "precision": P, "recall": R, "F1": F}, index=[f"{k}★" for k in range(1, 6)]).round(2))
cm = confusion_matrix(gt_f, pr_f, labels=range(1, 6)); cmn = cm / cm.sum(1, keepdims=True)
fig, ax = plt.subplots(figsize=(4.2, 3.6)); im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
for i in range(5):
    for j in range(5): ax.text(j, i, f"{cmn[i, j]:.0%}", ha="center", va="center", color="white" if cmn[i, j] > .5 else "black", fontsize=8)
ax.set_xticks(range(5)); ax.set_xticklabels(range(1, 6)); ax.set_yticks(range(5)); ax.set_yticklabels(range(1, 6))
ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title("Rows sum to 100%: neighbours get the errors", fontsize=9); plt.tight_layout(); plt.show()
""")
md("### 2b. Optional: one live prediction\nIf the local model server is running (`llama-server ... --port 8090`) this calls the real fine-tuned model; otherwise it shows the saved probabilities for the same review.")
code("""
import math
ex = next(r for r in rows_all if r["gt_rating"] == 2 and len(r["review_text"]) < 350)
live = None
try:
    import requests
    from agentic_sentiment.agents.graph import ANALYST_PROMPT, SENTIMENT_INSTRUCTION
    from agentic_sentiment.eval.one_shot import COT_PHRASE
    pr = ANALYST_PROMPT.format(instruction=SENTIMENT_INSTRUCTION, cot_block=COT_PHRASE, one_shot_block="", metadata_block="", review_text=ex["review_text"], critique_block="")
    r = requests.post("http://localhost:8090/completion", timeout=15, json={"prompt": pr, "n_predict": 2, "temperature": 0, "n_probs": 50}).json()
    steps = r["completion_probabilities"]; top = steps[1 if steps[0]["token"] == " " and len(steps) > 1 else 0]["top_logprobs"]
    lp = {t["token"].strip(): t["logprob"] for t in top}; live = [math.exp(lp.get(str(k), -99)) for k in range(1, 6)]
except Exception as e:
    print("live server not available -> using saved probabilities  (", type(e).__name__, ")")
p = live or probs_all[ex["review_id"]]["probs"]
print("REVIEW:", ex["review_text"][:350], "\\nTRUE RATING:", ex["gt_rating"], "  source:", "LIVE model" if live else "saved output")
plt.figure(figsize=(4.2, 2.4)); plt.bar(range(1, 6), p, color=[ORANGE if k == ex["gt_rating"] else BLUE for k in range(1, 6)])
plt.xlabel("rating"); plt.ylabel("probability"); plt.title("Model's probability per rating (orange = true)", fontsize=9); plt.tight_layout(); plt.show()
""")

md("""
## 3. Reproducing the paper
The paper's classical baselines (Decision Tree, SVM, Naive Bayes) are trained here with the paper's configurations on a small training sample and scored on the **same** held-out rows as LLaMA. "Polarity" groups stars as 1–2 / 3 / 4–5.
""")
code("""
from agentic_sentiment.baselines.classical import train_decision_tree, train_svm, train_naive_bayes
csv = ROOT / ".scratch/kaggle_data/Reviews.csv"
held = {int(r["review_id"].rsplit("_", 1)[1]) for r in rows_all}
full = pd.read_csv(csv, usecols=["Score", "Text"]).dropna(); full = full[~full.index.isin(held)]
TRAIN_N = 4000
tr = full.sample(TRAIN_N, random_state=1); Xtr, ytr = tr.Text.astype(str).tolist(), tr.Score.astype(int).tolist()
Xte = [r["review_text"] for r in rows_all]; yte = np.array([r["gt_rating"] for r in rows_all])
res = {}
for name, fit in [("Decision Tree", train_decision_tree), ("SVM (sigmoid)", train_svm), ("Multinomial NB", train_naive_bayes)]:
    p = np.array(fit(Xtr, ytr).predict(Xte)); res[name] = ((p == yte).mean(), (np.vectorize(pol)(p) == np.vectorize(pol)(yte)).mean())
res["LLaMA-3-8B + LoRA (ours)"] = (m_f["5-class exact"], m_f["polarity"])
paper = {"Decision Tree": .786, "SVM (sigmoid)": .872, "Multinomial NB": .876, "LLaMA-3-8B + LoRA (ours)": .926}
t = pd.DataFrame(res, index=["5-class exact", "polarity"]).T; t["paper reports"] = pd.Series(paper)
t.loc["always guess majority"] = [(yte == 5).mean(), (np.vectorize(pol)(yte) == 2).mean(), np.nan]
print(f"classical models trained on {TRAIN_N} rows (earlier runs used 16,000); LLaMA row = saved result on the same 4,000 test rows")
(t * 100).round(1).fillna("—")
""")
md("### 3b. Data-scale mini curve\nMore training data keeps helping the classical models. Here a TF-IDF logistic regression and Naive Bayes are trained on 1K, 4K and 16K reviews.")
code("""
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
sizes, out = [1000, 4000, 16000], []
big = full.sample(max(sizes), random_state=2)
for n in sizes:
    d = big.iloc[:n]; X, y = d.Text.astype(str).tolist(), d.Score.astype(int).tolist()
    t1 = TfidfVectorizer(min_df=2, ngram_range=(1, 2), max_features=200000); lr = LogisticRegression(C=5, max_iter=300).fit(t1.fit_transform(X), y); a = lr.predict(t1.transform(Xte))
    c1 = CountVectorizer(); nbm = MultinomialNB(alpha=.2).fit(c1.fit_transform(X), y); b = nbm.predict(c1.transform(Xte))
    out.append({"n": n, "LogReg exact": (a == yte).mean(), "LogReg polarity": (np.vectorize(pol)(a) == np.vectorize(pol)(yte)).mean(),
                "NB exact": (b == yte).mean(), "NB polarity": (np.vectorize(pol)(b) == np.vectorize(pol)(yte)).mean()})
cur = pd.DataFrame(out).set_index("n"); display((cur * 100).round(1))
fig, ax = plt.subplots(1, 2, figsize=(8, 2.8))
for a_, cols, title in [(ax[0], ["LogReg exact", "NB exact"], "5-class exact"), (ax[1], ["LogReg polarity", "NB polarity"], "polarity")]:
    for c, col in zip(cols, [BLUE, GREY]): a_.plot(cur.index, cur[c] * 100, "o-", color=col, label=c)
    a_.axhline(m_f["5-class exact" if "exact" in title else "polarity"] * 100, color=ORANGE, ls="--", label="LLaMA (2–4K rows)")
    a_.set_xscale("log"); a_.set_title(title, fontsize=10); a_.set_xlabel("training rows"); a_.legend(fontsize=7)
plt.tight_layout(); plt.show()
""")

md("""
## 4. Phase 2: the agentic pipeline
### 4a. The critic's disagreement score
`dissonance_norm(h, ev, gf) = |h − (ev + gf)/2|`, with each rating mapped to 0–1 as (rating − 1)/4. A correction round triggers when it reaches **0.4**.
""")
code("""
from agentic_sentiment.agents.critic import dissonance_norm, dissonance_stdev, normalize_rating, DISSONANCE_THRESHOLD
cases = [("all agree: 4,4,4", 4, 4, 4), ("mild: analyst 4, rag 3", 4, 4, 3), ("analyst 5, rag 3", 5, 5, 3), ("analyst 5, rag 1", 5, 5, 1), ("analyst 1, rag 5", 1, 1, 5), ("analyst 3, rag 1", 3, 3, 1)]
tb = pd.DataFrame([{"votes (analyst, visual, rag)": f"{a}, {v}, {r}", "case": n, "D (norm)": round(dissonance_norm(*[normalize_rating(x) for x in (a, v, r)]), 3),
                    "D (Run 1: stdev/2)": round(dissonance_stdev([a, v, r]), 3)} for n, a, v, r in cases])
tb["triggers correction (norm)"] = tb["D (norm)"] >= DISSONANCE_THRESHOLD
tb
""")
md("""
### 4b. A real LangGraph run with a scripted model
This uses the project's actual `build_graph` with a stand-in for the language model that returns canned answers, so the flow is visible and repeatable without a GPU. Watch the loop: the analyst says 5★, retrieval says 1★, the critic sees high disagreement, writes feedback, and the analyst reruns.
""")
code("""
from agentic_sentiment.agents.graph import build_graph
calls = []
def scripted_llm(prompt):
    if prompt.lstrip().startswith(("Evaluate", "Evaluate")) and "Critique:" in prompt:
        calls.append("CRITIC  -> feedback written"); return "The review complains about a defect; reconsider a 5-star rating."
    if "RAG Prover" in prompt:
        calls.append("RAG     -> rates 1"); return "Sentiment (1-5): 1"
    rerun = "Critic feedback:" in prompt
    calls.append(f"ANALYST -> rates {2 if rerun else 5}  ({'after critic feedback' if rerun else 'first pass'})"); return f"Sentiment (1-5): {2 if rerun else 5}"
g = build_graph(llm_fn=scripted_llm, spec_store=None, max_correction_iters=2)
state = {"review_id": "demo", "review_text": "Works, but it stopped charging after a week and support never replied.", "gt_rating": 2,
         "use_metadata": False, "use_image": False, "use_rag": True, "use_cot": True, "one_shot_example": None,
         "dissonance_method": "norm", "correction_iters": 0, "max_correction_iters": 2, "meta_title": "", "meta_description": "", "asin": ""}
out = g.invoke(state)
print("\\n".join(calls)); print("-" * 40)
print(f"analyst={out['analyst_rating']}  rag={out['rag_rating']}  dissonance={out['dissonance']:.3f}  correction rounds={out['correction_iters']}  FINAL={out['final_rating']}  (true={out['gt_rating']})")
""")
md("""
### 4c. Why the first Phase 2 almost never helped
The override rule in the critic only replaces the analyst's rating if **two or more other votes unanimously disagree**. With the visual vote excluded, only one other vote (retrieval) exists, so that can never happen. Test it: whatever retrieval says, the final rating stays the analyst's.
""")
code("""
def run(analyst_rating, rag_rating):
    def llm(p):
        if "Critique:" in p: return "reconsider"
        if "RAG Prover" in p: return f"Sentiment (1-5): {rag_rating}"
        return f"Sentiment (1-5): {analyst_rating}"
    s = dict(state, correction_iters=0, max_correction_iters=0)    # no correction rounds: look at the final decision rule alone
    return build_graph(llm_fn=llm, spec_store=None, max_correction_iters=0).invoke(s)["final_rating"]
grid = pd.DataFrame([[run(a, r) for r in range(1, 6)] for a in range(1, 6)], index=[f"analyst={a}" for a in range(1, 6)], columns=[f"rag={r}" for r in range(1, 6)])
display(grid); print("Final rating equals the analyst's in every cell:", all(grid.iloc[a - 1, r - 1] == a for a in range(1, 6) for r in range(1, 6)))
""")

md("""
## 5. Where reflection can help
### 5a. The uncertainty gate
The model's two most likely ratings contain the true rating on almost 9 of 10 reviews. The gate sends a review to reflection only when the **margin** between its top two probabilities is small. Saved probabilities from the 1,000-row balanced eval set are used here.
""")
code("""
ev = [json.loads(l) for l in open(ROOT / ".scratch/kaggle_eval_probs.jsonl")]
rows_g = []
for r in ev:
    p = r["probs"]; o = sorted(range(5), key=lambda k: -p[k]); g = r["gt_rating"] - 1
    rows_g.append((p[o[0]] - p[o[1]], o[0] == g, g in o[:2]))
M = np.array(rows_g, dtype=float); print(f"top-1 correct {M[:,1].mean():.1%}   true rating inside the model's top two {M[:,2].mean():.1%}")
ths = np.linspace(0.05, 1.0, 20)
cover = [(M[:, 0] < t).mean() for t in ths]; acc_in = [M[M[:, 0] < t, 1].mean() if (M[:, 0] < t).any() else np.nan for t in ths]
acc_out = [M[M[:, 0] >= t, 1].mean() if (M[:, 0] >= t).any() else np.nan for t in ths]
fig, ax = plt.subplots(figsize=(5.2, 3)); ax.plot(ths, cover, color=GREY, label="share of rows sent to reflection"); ax.plot(ths, acc_in, color=RED, label="accuracy inside the gate")
ax.plot(ths, acc_out, color=BLUE, label="accuracy outside the gate"); ax.axvline(0.3, ls=":", color="k"); ax.set_xlabel("margin threshold"); ax.legend(fontsize=7); plt.tight_layout(); plt.show()
th = 0.3; g_ = M[:, 0] < th
print(f"gate at margin<{th}: {g_.mean():.0%} of rows; top-1 right on {M[g_,1].mean():.0%} inside vs {M[~g_,1].mean():.0%} outside; true rating in top two for {M[g_,2].mean():.0%} of gated rows")
""")
md("""
### 5b. What the adjudicator must beat
Inside the gate the model already picks the right one of its two candidates about 60% of the time. An adjudicator only helps if it picks better than that. This chart is the ceiling: expected exact accuracy for different adjudicator skill levels.
""")
code("""
inside = (M[:, 0] < 0.3) & (M[:, 2] == 1)
base_pick = M[inside, 1].mean(); print(f"model's own pick among its top two, inside the gate: {base_pick:.1%}")
skills = np.linspace(0.5, 0.9, 9)
exact = [(M[~(M[:, 0] < 0.3), 1].sum() + s * ((M[:, 0] < 0.3) & (M[:, 2] == 1)).sum()) / len(M) for s in skills]
plt.figure(figsize=(4.8, 2.8)); plt.plot(skills * 100, np.array(exact) * 100, "o-", color=BLUE); plt.axvline(base_pick * 100, color=RED, ls="--", label="Phase 1's own pick rate")
plt.axhline(M[:, 1].mean() * 100, color=GREY, ls=":", label="Phase 1 exact"); plt.xlabel("adjudicator picks the right neighbour (%)"); plt.ylabel("expected exact accuracy (%)"); plt.legend(fontsize=7); plt.tight_layout(); plt.show()
""")
md("### 5c. The adjudicator's training question, and its result if the check has been run")
code("""
from build_pair_data import prompt as pair_prompt
print(pair_prompt("Great screen but the battery barely lasts four hours.", 3, 4)); print("\\ncompletion the adjudicator is trained to write:  ' 3' or ' 4'")
g0 = ROOT / "v2_reflective/gate0_dev.json"
if g0.exists():
    d = json.load(open(g0)); n = len(d); rows_ = []
    for th in (0.2, 0.3, 0.4, 0.5):
        gp = [x for x in d if x["margin"] < th and x["gt"] in (x["a"], x["b"])]
        if gp:
            ph1 = np.mean([x["gt"] == x["a"] for x in gp]); adj = np.mean([(x["gt"] == x["a"]) == (x["adj"] > 0) for x in gp])
            sim = np.mean([(x["a"] if (x["margin"] >= th or x["adj"] > 0) else x["b"]) == x["gt"] for x in d])
            rows_.append({"gate margin<": th, "pairs": len(gp), "Phase 1 picks right": round(ph1, 3), "adjudicator picks right": round(adj, 3), "simulated Phase 2 exact": round(sim, 3)})
    print(f"\\nPhase 1 exact on the {n} dev rows: {np.mean([x['gt']==x['a'] for x in d]):.1%}"); display(pd.DataFrame(rows_))
else:
    print("\\nGate 0 has not been run yet (needs the finished adjudicator): no adjudicator numbers to show.")
""")
md("""
## Summary and caveats
- The pipeline functions above are the project's real code; the model outputs come from saved files (or the live server if running).
- Phase 1 reproduces the paper's **method**, not its headline number; the paper's baselines match polarity-style scoring, and its training scale is far larger.
- Phase 2's first design almost never changed the analyst's answer (section 4c); the redesign aims reflection at the model's top two candidates (section 5).
- The adjudicator result only appears if `gate0_dev.json` exists; no Phase 2 accuracy gain is claimed until that check is run and held-out data confirm it.
- Small samples are noisy: with 300 rows, ±3 points is normal.
""")
nb["cells"] = C
nb["metadata"] = {"kernelspec": {"name": "pes-mtech", "display_name": "PES MTech (.venv)", "language": "python"}}
nbf.write(nb, "Presentation_Demo.ipynb"); print("wrote", len(C), "cells")
