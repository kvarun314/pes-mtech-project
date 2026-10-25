"""Deck figures from stored probabilities (no refits beyond the main-stack Reflector on dev2, as readout_H.py / analysis_C.py; no test-set selection).
Writes paper/figures/e23/{cm_main_<split>, recall_phase1_vs_main, agents_complement}.png and prints the numbers they show."""
import json, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
D = "v2_reflective/e23/"; FIG = "paper/figures/e23/"; S = ("eval", "natural_test"); NAME = {"eval": "Balanced eval (1,000)", "natural_test": "Natural test (2,000)"}
G = lambda p, s: {x["review_id"]: x for x in map(json.loads, open(f"{D}{p}_{s}_probs.jsonl" if p.startswith("evid") else f"{D}p1/probs_{p}_{s}.jsonl"))}
ids = {s: list(G("R", s)) for s in ("dev2",) + S}
lg = lambda p, s: np.log(np.clip([G(p, s)[i]["probs"] for i in ids[s]], 1e-6, 1))
y = {s: np.array([G("R", s)[i]["gt_rating"] - 1 for i in ids[s]]) for s in ids}
m = LogisticRegression(C=1, max_iter=3000).fit(np.hstack([lg("evid7", "dev2"), lg("evid6b", "dev2")]), y["dev2"])
pred = {s: {"main": m.predict(np.hstack([lg("evid7", s), lg("evid6b", s)])), "probe": lg("evid7", s).argmax(1), "evid": lg("evid6b", s).argmax(1), "p1": lg("R", s).argmax(1)} for s in S}
stars = ["1★", "2★", "3★", "4★", "5★"]
# 1. confusion matrices of the main stack (row-normalised, counts annotated)
for s in S:
    cm = confusion_matrix(y[s], pred[s]["main"], labels=range(5)); r = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(4.2, 3.8)); ax.imshow(r, cmap="Blues", vmin=0, vmax=1)
    for i in range(5):
        for j in range(5): ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=9, color="white" if r[i, j] > .5 else "#222")
    ax.set_xticks(range(5), stars); ax.set_yticks(range(5), stars); ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(f"{NAME[s]}: exact {100 * (pred[s]['main'] == y[s]).mean():.1f}%", fontsize=10); fig.tight_layout(); fig.savefig(f"{FIG}cm_main_{s}.png", dpi=200); plt.close(fig)
    print(s, "main-stack confusion:\n", cm)
# 2. per-star recall, Phase 1 LoRA vs main stack
fig, axs = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
for a, s in zip(axs, S):
    rec = {k: [float(((pred[s][k] == c) & (y[s] == c)).sum() / (y[s] == c).sum()) for c in range(5)] for k in ("p1", "main")}
    x = np.arange(5); a.bar(x - .2, rec["p1"], .4, label="Phase 1 LoRA", color="#7f8c8d"); a.bar(x + .2, rec["main"], .4, label="Phase 2 main stack", color="#C44E52")
    a.set_xticks(x, stars); a.set_ylim(0, 1); a.set_title(NAME[s], fontsize=10); print(s, "recall P1", np.round(rec["p1"], 3), "main", np.round(rec["main"], 3))
axs[0].set_ylabel("recall (share of each star found)"); axs[0].legend(fontsize=8, loc="upper left"); fig.tight_layout(); fig.savefig(f"{FIG}recall_phase1_vs_main.png", dpi=200); plt.close(fig)
# 3. agents complement: accuracy by review length (balanced eval) and agree / disagree (both splits)
words = np.array([len(json.loads(l)["review_text"].split()) for l in open(f"{D}eval.jsonl")]); order = [json.loads(l)["review_id"] for l in open(f"{D}eval.jsonl")]
w = words[[order.index(i) for i in ids["eval"]]]; q = np.quantile(w, [.25, .5, .75]); s = "eval"; yy = y[s]
bins = [w <= q[0], (w > q[0]) & (w <= q[1]), (w > q[1]) & (w <= q[2]), w > q[2]]; labs = ["shortest 25%", "short-mid", "mid-long", "longest 25%"]
acc = {k: [100 * (pred[s][k][b] == yy[b]).mean() for b in bins] for k in ("evid", "probe", "main")}
fig, axs = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [1.6, 1]})
x = np.arange(4)
for off, k, lab, col in ((-.27, "evid", "Evidence model", "#4C72B0"), (0, "probe", "LLaMA probe", "#55A868"), (.27, "main", "Main stack", "#C44E52")):
    axs[0].bar(x + off, acc[k], .27, label=lab, color=col)
axs[0].set_xticks(x, labs); axs[0].set_ylim(40, 80); axs[0].set_ylabel("exact %"); axs[0].set_title("Balanced eval: accuracy by review length", fontsize=10); axs[0].legend(fontsize=8)
print("length quartile acc", {k: np.round(v, 1) for k, v in acc.items()})
grp = []
for s in S:
    ag = pred[s]["evid"] == pred[s]["probe"]; yy = y[s]
    grp.append((s, 100 * ag.mean(), 100 * (pred[s]["main"][ag] == y[s][ag]).mean(), 100 * (pred[s]["main"][~ag] == y[s][~ag]).mean(),
                100 * (pred[s]["evid"][~ag] == y[s][~ag]).mean(), 100 * (pred[s]["probe"][~ag] == y[s][~ag]).mean()))
x = np.arange(2)
for off, idx, lab, col in ((-.27, 4, "Evidence model", "#4C72B0"), (0, 5, "LLaMA probe", "#55A868"), (.27, 3, "Main stack", "#C44E52")):
    axs[1].bar(x + off, [g[idx] for g in grp], .27, label=lab, color=col)
axs[1].set_xticks(x, ["eval", "natural test"]); axs[1].set_ylim(0, 80); axs[1].set_title("When the two agents disagree", fontsize=10)
fig.tight_layout(); fig.savefig(f"{FIG}agents_complement.png", dpi=200); plt.close(fig)
print("agree share / stack acc agree / disagree: stack, evid, probe", [tuple(round(v, 1) if isinstance(v, float) else v for v in g) for g in grp])
