"""Phase 2 v3: refit and save the main-stack agents (Evidence model E6b, LLaMA probe P3, Reflector) with the exact frozen configs, then
verify they reproduce the stored probabilities / main-stack predictions. Writes v2_reflective/e23/agents/*.joblib (gitignored). Run once; ~30-40 min CPU."""
import json, glob, os, sys, time, joblib, numpy as np
sys.path.insert(0, "src")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from agentic_sentiment.v3 import paths as PT
from agentic_sentiment.v3.evidence import evidence_text
from agentic_sentiment.v3.reflector import fit_reflector, logp
D = PT.D; os.makedirs(PT.AGENTS, exist_ok=True); t0 = time.time()
L = lambda n: [json.loads(l) for l in open(f"{D}{n}.jsonl")]
G = lambda p, s: {x["review_id"]: x["probs"] for x in map(json.loads, open(f"{D}{p}_{s}_probs.jsonl"))}
# Evidence model (E6b): evid_H.py config
if not os.path.exists(PT.EVIDENCE_JOBLIB):
    tr = L("evid_train") + L("evid_big_extra_nodev2")
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=400000, sublinear_tf=True); X = vec.fit_transform(evidence_text(r["title"], r["review_text"]) for r in tr)
    m = LogisticRegression(C=3, max_iter=1000).fit(X, [r["gt_rating"] for r in tr]); joblib.dump({"vectorizer": vec, "model": m}, PT.EVIDENCE_JOBLIB); print("E6b fit", f"{time.time()-t0:.0f}s", flush=True)
    del tr, X
e = joblib.load(PT.EVIDENCE_JOBLIB); ev = L("eval"); st = G("evid6b", "eval")
diff = np.abs(e["model"].predict_proba(e["vectorizer"].transform(evidence_text(r["title"], r["review_text"]) for r in ev)) - np.array([st[r["review_id"]] for r in ev])).max()
print("E6b max |diff| vs evid6b_eval_probs:", float(diff), flush=True); assert diff < 1e-6
# LLaMA probe (P3): gate_H.py rows and config, feature L20 last token, C=0.001
ld = lambda d, s: np.vstack([np.load(f) for f in sorted(glob.glob(f"{D}{d}/{s}_*.npy"))]).astype(np.float32)
i2, i3 = json.load(open(D + "hid/ids.json")), json.load(open(D + "hid3/ids.json"))
lab = {r["review_id"]: r["gt_rating"] - 1 for n in ("evid_train", "dev2", "dev", "natural_dev", "eval", "natural_test") for r in L(n)}
Hf = np.vstack([ld("hid2", "fit")[:, 4096:8192], ld("hid3", "fit2")[:, :4096]]); yf = np.array([lab[i] for i in i2["fit"] + i3["fit2"]])
sc = StandardScaler().fit(Hf); clf = LogisticRegression(C=0.001, max_iter=500).fit(sc.transform(Hf), yf); joblib.dump({"scaler": sc, "clf": clf}, PT.PROBE_JOBLIB)
st7 = G("evid7", "eval"); He = ld("hid2", "eval")[:, 4096:8192]
diff = np.abs(clf.predict_proba(sc.transform(He)) - np.array([st7[i] for i in i2["eval"]])).max(); print("P3 max |diff| vs evid7_eval_probs:", float(diff), flush=True); assert diff < 1e-6
# Reflector on dev2 (row order as freeze_tables.py: p1/probs_R_dev2.jsonl)
ids = [json.loads(l)["review_id"] for l in open(D + "p1/probs_R_dev2.jsonl")]; P7, E6 = G(PT.PROBE_PREFIX, "dev2"), G("evid6b", "dev2")
r = fit_reflector(np.array([P7[i] for i in ids]), np.array([E6[i] for i in ids]), np.array([lab[i] + 1 for i in ids])); joblib.dump({"full": r.full, "fallback": r.fallback}, PT.REFLECTOR_JOBLIB)
ie = [json.loads(l)["review_id"] for l in open(D + "p1/probs_R_eval.jsonl")]; P7e, E6e = G(PT.PROBE_PREFIX, "eval"), G("evid6b", "eval")
pred = r.full.predict(np.c_[logp([P7e[i] for i in ie]), logp([E6e[i] for i in ie])]) + 1; gt = np.array([lab[i] + 1 for i in ie])
acc = (pred == gt).mean(); print("Reflector eval exact (must equal main stack 0.675):", round(float(acc), 3)); assert round(float(acc), 3) == 0.675 or PT.PROBE_PREFIX != "evid7"
print("done", f"{time.time()-t0:.0f}s")
