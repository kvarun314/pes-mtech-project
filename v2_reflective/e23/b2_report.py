"""B2 report: faithfulness (first vs final, McNemar), rewrite stats, check failures, deletion test, latency. Reads b2/ outputs."""
import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import binomtest, wilcoxon
D = "v2_reflective/e23/"; F = "paper/figures/e23/"; res = {}
L = lambda p: [json.loads(l) for l in open(p)]
def boot(d, n=3000, seed=0):
    rng = np.random.default_rng(seed); return [float(x) for x in np.percentile([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)], [2.5, 97.5])]
for s in ("eval", "natural_test"):
    R = L(f"{D}b2/{s}.jsonl"); X = L(f"{D}b2/deletion_{s}.jsonl"); f0 = np.array([r["faithful_first"] for r in R]); f1 = np.array([r["faithful"] for r in R])
    a, b = int((f1 & ~f0).sum()), int((~f1 & f0).sum()); chk = np.zeros(6, int)
    for r in R:
        for c in r["failed_checks"][0]: chk[c] += 1
    comp_c = np.array([x["p_ref"] - x["p_cited_removed"] for x in X]); comp_r = np.array([x["p_ref"] - np.mean(x["p_random_removed"]) for x in X])
    suf_c = np.array([x["p_cited_only"] for x in X]); suf_r = np.array([np.mean(x["p_random_only"]) for x in X])
    tm = lambda k: float(np.mean([r["timings"].get(k, 0) for r in R]))
    res[s] = {"n": len(R), "faithful_first": float(f0.mean()), "faithful_final": float(f1.mean()), "mcnemar_p": float(binomtest(a, a + b).pvalue) if a + b else 1.0,
              "rewrite_rate": float(np.mean([r["rounds"] > 0 for r in R])), "mean_rounds": float(np.mean([r["rounds"] for r in R])),
              "first_draft_failures_by_check": {str(c): int(chk[c]) for c in range(1, 6)}, "judge_unparsed": int(sum(r["judge_unparsed"] for r in R)),
              "llm_calls": float(np.mean([r["llm_calls"] for r in R])), "explain_s": tm("explain_s"), "critic_s": tm("critic_s"),
              "comp_cited": float(comp_c.mean()), "comp_random": float(comp_r.mean()), "comp_diff_ci": boot(comp_c - comp_r), "comp_wilcoxon_p": float(wilcoxon(comp_c, comp_r).pvalue),
              "suff_cited": float(suf_c.mean()), "suff_random": float(suf_r.mean()), "suff_diff_ci": boot(suf_c - suf_r), "suff_wilcoxon_p": float(wilcoxon(suf_c, suf_r).pvalue)}
    print(s, json.dumps(res[s], indent=1))
res["router"] = json.load(open(D + "gate_B1.json")); json.dump(res, open(D + "b2_report.json", "w"), indent=1)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
for i, (k1, k2, t) in enumerate([("comp_cited", "comp_random", "Comprehensiveness (drop in p, higher = cited words matter)"), ("suff_cited", "suff_random", "Sufficiency (p with only these words)")]):
    x = np.arange(2); ax[i].bar(x - .18, [res[s][k1] for s in ("eval", "natural_test")], .36, label="cited phrases", color="#c0392b")
    ax[i].bar(x + .18, [res[s][k2] for s in ("eval", "natural_test")], .36, label="random words", color="#7f8c8d"); ax[i].set_xticks(x, ["eval", "natural_test"]); ax[i].set_title(t, fontsize=9)
ax[0].legend(fontsize=8); fig.tight_layout(); fig.savefig(F + "deletion_test.png", dpi=200)
fig, ax = plt.subplots(figsize=(5, 3.4)); x = np.arange(2)
ax.bar(x - .18, [res[s]["faithful_first"] for s in ("eval", "natural_test")], .36, label="first draft", color="#7f8c8d")
ax.bar(x + .18, [res[s]["faithful_final"] for s in ("eval", "natural_test")], .36, label="after Critic loop", color="#c0392b")
ax.set_xticks(x, ["eval", "natural_test"]); ax.set_ylim(0, 1); ax.set_ylabel("share faithful"); ax.legend(fontsize=8, loc="lower right"); fig.tight_layout(); fig.savefig(F + "faithfulness.png", dpi=200)
