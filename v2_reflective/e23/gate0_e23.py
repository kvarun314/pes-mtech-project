"""Gate 0 (P2-2) on Electronics, pre-declared in LOG.md 2026-10-20. usage: gate0_e23.py build|run|score. Resumable; dev + natural_dev disagreement rows only."""
import json, os, sys, math, time, numpy as np
D = "v2_reflective/e23/"; PR, AN = D + "gate0_prompts.jsonl", D + "gate0_answers.jsonl"
Lj = lambda p: {r["review_id"]: r for r in map(json.loads, open(p))}
SPL = [("dev", "dev"), ("natural_dev", "natural_dev")]
def build():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    tr = [json.loads(l) for l in open(D + "evid_train.jsonl")]; txt = lambda r: r["title"] + " . " + r["review_text"]
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=400000, sublinear_tf=True)
    m = LogisticRegression(C=3, max_iter=400).fit(vec.fit_transform([txt(r) for r in tr]), [r["gt_rating"] for r in tr])
    vocab = np.array(vec.get_feature_names_out()); n = 0
    with open(PR, "w") as o:
        for name, sp in SPL:
            rows = Lj(f"{D}{sp}.jsonl"); a = Lj(f"{D}p1/probs_R_{name}.jsonl"); e = Lj(f"{D}evid_{name}_probs.jsonl")
            for i, r in rows.items():
                pa, pe = int(np.argmax(a[i]["probs"])), int(np.argmax(e[i]["probs"]))
                if pa == pe: continue
                x = vec.transform([txt(r)]); idx = x.indices; val = x.data
                def top(c, d):
                    s = val * (m.coef_[c, idx] - m.coef_[d, idx]); o_ = np.argsort(-s)[:5]; return [str(vocab[idx[k]]) for k in o_ if s[k] > 0]
                o.write(json.dumps({"review_id": i, "split": name, "truth": r["gt_rating"], "a": pa + 1, "e": pe + 1, "review": r["review_text"][:1200],
                                    "ev_a": top(pa, pe), "ev_e": top(pe, pa)}) + "\n"); n += 1
    print("prompts", n)
def prompt(r, first, second, ev1, ev2):
    f = lambda x: ", ".join(f'"{t}"' for t in x) or "(none)"
    return (f"Review: {r['review']}\n\nThe review's star rating (1-5) is either {first} or {second}.\n"
            f"A text classifier found phrases pointing to {first}: {f(ev1)}\nPhrases pointing to {second}: {f(ev2)}\n"
            f"Read the review and decide which rating the reviewer gave.\nRating:")
def run():
    import mlx.core as mx
    from mlx_lm import load
    model, tok = load("v2_reflective/mlx_model"); dig = {d: tok.encode(str(d), add_special_tokens=False)[-1] for d in range(1, 6)}; sp = tok.encode(" ", add_special_tokens=False)
    done = {json.loads(l)["review_id"] for l in open(AN)} if os.path.exists(AN) else set(); t0 = time.time(); n = 0
    with open(AN, "a") as f:
        for l in open(PR):
            r = json.loads(l)
            if r["review_id"] in done: continue
            n += 1
            if n % 50 == 0: print(f"{len(done)+n} rows, {time.time()-t0:.0f}s", flush=True)
            out = {}
            for tag, (x, y, ex, ey) in {"ae": (r["a"], r["e"], r["ev_a"], r["ev_e"]), "ea": (r["e"], r["a"], r["ev_e"], r["ev_a"])}.items():
                ids = tok.encode(prompt(r, x, y, ex, ey), add_special_tokens=False) + sp
                lg = model(mx.array(ids)[None])[0, -1]; vx, vy = float(lg[dig[x]]), float(lg[dig[y]])
                out[tag] = {x: 1 / (1 + math.exp(vy - vx)), y: 1 / (1 + math.exp(vx - vy))}
            pa = (out["ae"][r["a"]] + out["ea"][r["a"]]) / 2; f.write(json.dumps({"review_id": r["review_id"], "p_a": pa}) + "\n"); f.flush()
    print("done", AN)
def score():
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    rng = np.random.default_rng(0); P = [json.loads(l) for l in open(PR)]; A = {json.loads(l)["review_id"]: json.loads(l)["p_a"] for l in open(AN)}
    F = lambda name, ids: np.c_[np.log(np.clip([Lj(f"{D}p1/probs_R_{name}.jsonl")[i]["probs"] for i in ids], 1e-6, 1)), np.log(np.clip([Lj(f"{D}evid_{name}_probs.jsonl")[i]["probs"] for i in ids], 1e-6, 1))]
    # stack pick per row, restricted to the two candidates
    stk = {}
    ids_d = list(Lj(f"{D}dev.jsonl")); Xd = F("dev", ids_d); yd = np.array([Lj(f"{D}dev.jsonl")[i]["gt_rating"] - 1 for i in ids_d])
    pcv = cross_val_predict(LogisticRegression(C=1, max_iter=3000), Xd, yd, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")
    for i, p in zip(ids_d, pcv): stk[i] = p
    m = LogisticRegression(C=1, max_iter=3000).fit(Xd, yd); ids_n = list(Lj(f"{D}natural_dev.jsonl"))
    for i, p in zip(ids_n, m.predict_proba(F("natural_dev", ids_n))): stk[i] = p
    rows = [r for r in P if r["truth"] in (r["a"], r["e"]) and r["review_id"] in A]
    print(f"disagreement rows {len(P)}, scored {len([r for r in P if r['review_id'] in A])}, truth in pair {len(rows)} (dev {sum(r['split']=='dev' for r in rows)}, natural_dev {sum(r['split']=='natural_dev' for r in rows)})")
    llm = np.array([(r["a"] if A[r["review_id"]] >= .5 else r["e"]) == r["truth"] for r in rows])
    st = np.array([(r["a"] if stk[r["review_id"]][r["a"] - 1] >= stk[r["review_id"]][r["e"] - 1] else r["e"]) == r["truth"] for r in rows])
    ev = np.array([r["e"] == r["truth"] for r in rows]); an = np.array([r["a"] == r["truth"] for r in rows])
    print(f"pair accuracy: LLM adjudicator {llm.mean():.1%} | stack {st.mean():.1%} | Evidence always {ev.mean():.1%} | Analyst always {an.mean():.1%}")
    d = llm.astype(float) - st; bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(3000)]; lo, hi = np.percentile(bs, [2.5, 97.5])
    print(f"LLM - stack: {d.mean()*100:+.1f} pts, 95% CI [{lo*100:+.1f},{hi*100:+.1f}], LLM-only right {int(((llm==1)&(st==0)).sum())}, stack-only right {int(((llm==0)&(st==1)).sum())}")
    for s in ("dev", "natural_dev"):
        k = np.array([r["split"] == s for r in rows]); print(f"  {s}: n={k.sum()} LLM {llm[k].mean():.1%} stack {st[k].mean():.1%}")
    print("VERDICT:", "PASS" if d.mean() >= .03 and lo > 0 else "FAIL", "(criterion: LLM >= stack +3 pts and CI excludes 0)")
{"build": build, "run": run, "score": score}[sys.argv[1]]()
