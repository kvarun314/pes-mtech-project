"""Does the LoRA adapter add anything? (S2a) + missing paired CI (C2 vs prior-adjusted Evidence) + figures (S4). PRE-DECLARED 2026-10-19 (LOG.md). Read-out on eval + natural_test once.
1) base (no LoRA) vs arm R vs Evidence, exact/macroF1/recall, paired bootstrap R-vs-base.
2) Stacks fit on balanced dev (same recipe as C0): [base+E] vs [R+E] vs [base+R+E]; paired vs Evidence and vs [R+E].
3) C2 (A+E balanced dev reweighted to natural prior) vs Evidence prior-adjusted, natural_test, paired CI.
4) Figures -> paper/figures/e23/: confusion matrices (R+E stack, eval and natural_test) and per-class recall bars."""
import json, os, numpy as np
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix
D = "v2_reflective/e23/"; rng = np.random.default_rng(0); Lj = lambda p: {r["review_id"]: r for r in map(json.loads, open(p))}
def get(split):
    R, B, E = Lj(f"{D}p1/probs_R_{split}.jsonl"), Lj(f"{D}p1/probs_base_{split}.jsonl"), Lj(f"{D}evid_{split}_probs.jsonl"); ids = [i for i in R if i in B and i in E]
    g = lambda d: np.clip([d[i]["probs"] for i in ids], 1e-6, 1)
    return np.array([R[i]["gt_rating"] - 1 for i in ids]), g(R), g(B), g(E)
def met(y, p): return dict(exact=(p == y).mean(), macroF1=f1_score(y, p, average="macro"), within1=(abs(p - y) <= 1).mean(), recall=[round(float(((p == c) & (y == c)).sum() / max((y == c).sum(), 1)), 2) for c in range(5)])
fmt = lambda m: f"exact {m['exact']:.1%} macroF1 {m['macroF1']:.3f} within1 {m['within1']:.1%} recall {m['recall']}"
def cmp(ok, b):
    d = ok.astype(float) - b; ci = np.percentile([d[rng.integers(0, len(d), len(d))].mean() for _ in range(3000)], [2.5, 97.5]); fx, rg = int((ok & ~b).sum()), int((b & ~ok).sum())
    return f"{d.mean()*100:+.1f} [{ci[0]*100:+.1f},{ci[1]*100:+.1f}] fixed {fx}/regressed {rg} p={(binomtest(fx, fx+rg, .5).pvalue if fx+rg else 1):.2g}"
lg = lambda *P: np.log(np.hstack(P))
yd, Rd, Bd, Ed = get("dev"); yn = np.array([json.loads(l)["gt_rating"] - 1 for l in open(f"{D}natural_dev.jsonl")]); prior = np.bincount(yn, minlength=5) / len(yn)
print(f"dev n={len(yd)}  base {fmt(met(yd, Bd.argmax(1)))}\n       arm R {fmt(met(yd, Rd.argmax(1)))}")
fit = lambda X, y, w=None: LogisticRegression(C=1, max_iter=3000).fit(X, y, sample_weight=w)
S = {"[base+E]": (lambda R, B, E: lg(B, E), fit(lg(Bd, Ed), yd)), "[R+E] (=C0)": (lambda R, B, E: lg(R, E), fit(lg(Rd, Ed), yd)), "[base+R+E]": (lambda R, B, E: lg(B, R, E), fit(lg(Bd, Rd, Ed), yd))}
w = (prior / (np.bincount(yd, minlength=5) / len(yd)))[yd]; C2 = fit(lg(Rd, Ed), yd, w)
res = {}
for split in ("eval", "natural_test"):
    y, R, B, E = get(split); print(f"\n===== {split} n={len(y)}")
    pb, pr, pe = B.argmax(1), R.argmax(1), E.argmax(1); print("base (no LoRA):", fmt(met(y, pb))); print("arm R (LoRA)  :", fmt(met(y, pr))); print("Evidence      :", fmt(met(y, pe)))
    print("  R vs base:", cmp(pr == y, pb == y))
    st = {}
    for n, (f, m) in S.items():
        p = m.predict(f(R, B, E)); st[n] = p; print(f"{n:12s}", fmt(met(y, p))); print(f"{'':12s}vs Evidence {cmp(p == y, pe == y)}")
        if n != "[R+E] (=C0)": print(f"{'':12s}vs [R+E]    {cmp(p == y, st['[R+E] (=C0)'] == y) if '[R+E] (=C0)' in st else 'n/a (order)'}")
    if split == "natural_test":
        pc2 = C2.predict(lg(R, E)); pea = (E * prior).argmax(1); print("\nC2 (R+E bal-dev reweighted):", fmt(met(y, pc2))); print("Evidence prior-adjusted    :", fmt(met(y, pea))); print("  C2 vs Evidence prior-adjusted:", cmp(pc2 == y, pea == y))
    res[split] = {"base": met(y, pb), "R": met(y, pr), "E": met(y, pe), **{k: met(y, v) for k, v in st.items()}}
    # figures
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    os.makedirs("paper/figures/e23", exist_ok=True); p = st["[R+E] (=C0)"]
    cm = confusion_matrix(y, p, labels=range(5)); cm = cm / cm.sum(1, keepdims=True); fig, ax = plt.subplots(figsize=(4.2, 3.8)); ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    for i in range(5):
        for j in range(5): ax.text(j, i, f"{cm[i,j]:.2f}", ha="center", va="center", fontsize=8, color="white" if cm[i, j] > .5 else "black")
    ax.set_xticks(range(5), range(1, 6)); ax.set_yticks(range(5), range(1, 6)); ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(f"Stack (R+E) {split}, row-normalised"); fig.tight_layout(); fig.savefig(f"paper/figures/e23/cm_stack_{split}.png", dpi=200); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 3.2)); wd = .2
    for k, (nm, pp) in enumerate((("base", pb), ("arm R", pr), ("Evidence", pe), ("stack R+E", p))):
        ax.bar(np.arange(5) + (k - 1.5) * wd, met(y, pp)["recall"], wd, label=nm)
    ax.set_xticks(range(5), [f"{i}★" for i in range(1, 6)]); ax.set_ylabel("recall"); ax.set_title(f"Per-class recall, {split}"); ax.legend(fontsize=7, ncol=4); fig.tight_layout(); fig.savefig(f"paper/figures/e23/recall_{split}.png", dpi=200); plt.close(fig)
json.dump(res, open(f"{D}base_vs_lora_results.json", "w"), indent=1, default=float); print("\nfigures in paper/figures/e23/")
