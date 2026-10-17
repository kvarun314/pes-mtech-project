"""Phase 2 vs Phase 1 analysis on the 1,000 held-out balanced eval rows (touched once, after gate chosen on dev)."""
import os, sys, json, numpy as np, collections
sys.path.insert(0, "v2_reflective")
import reflective as R
dev = R.load(os.environ.get("DEV_PROBS", ".scratch/kaggle_dev_probs.jsonl"), ".scratch/kaggle_dev_rows.jsonl")
ev = R.load(os.environ.get("EVAL_PROBS", ".scratch/kaggle_eval_probs.jsonl"), ".scratch/kaggle_eval_rows.jsonl")
cv = R.choose_gate(dev); print("dev-CV exact by gate margin:", {k: round(v, 3) for k, v in cv.items()})
gate = max(cv, key=cv.get); print("chosen gate", gate)
tr = R.run(ev, R.Reflector().fit(dev), gate)
y = np.array([t["gt"] for t in tr]); a = np.array([t["first"] for t in tr]); b = np.array([t["final"] for t in tr])
ca, cb = a == y, b == y
rng = np.random.default_rng(0); d = cb.astype(float) - ca
bs = np.percentile([d[rng.integers(0, len(d), len(d))].mean() for _ in range(5000)], [2.5, 97.5])
fixed, regr = int((cb & ~ca).sum()), int((ca & ~cb).sum())
from scipy.stats import binomtest
p = binomtest(fixed, fixed + regr, 0.5).pvalue
print(f"\nPhase 1: exact {ca.mean():.1%} within1 {(abs(a-y)<=1).mean():.1%} MAE {abs(a-y).mean():.3f}")
print(f"Phase 2: exact {cb.mean():.1%} within1 {(abs(b-y)<=1).mean():.1%} MAE {abs(b-y).mean():.3f}")
print(f"delta {d.mean()*100:+.1f} pts, 95% CI [{bs[0]*100:+.1f}, {bs[1]*100:+.1f}], fixed {fixed}, regressed {regr}, McNemar p={p:.2g}")
print(f"reflected on {np.mean([t['reflected'] for t in tr]):.0%} rows, changed {np.mean([t['changed'] for t in tr]):.0%}")
print("\nper star (n, P1 recall -> P2 recall, P2 precision):")
for k in range(1, 6):
    m = y == k
    print(f" {k}*  n={m.sum()}  {ca[m].mean():.2f} -> {cb[m].mean():.2f}  prec {((b==k)&m).sum()/max((b==k).sum(),1):.2f} (P1 {((a==k)&m).sum()/max((a==k).sum(),1):.2f})")
pol = lambda x: np.where(x <= 2, 0, np.where(x == 3, 1, 2))
print(f"\n3-class polarity: P1 {(pol(a)==pol(y)).mean():.1%} -> P2 {(pol(b)==pol(y)).mean():.1%}")
bn = (y != 3); pb = lambda x: x >= 4
print(f"binary (3* dropped): P1 {(pb(a)[bn]==pb(y)[bn]).mean():.1%} -> P2 {(pb(b)[bn]==pb(y)[bn]).mean():.1%}")
print("\nadjacent-star confusion (|err|==1 share of errors): P1 %.0f%% P2 %.0f%%" % (100*(abs(a-y)==1)[~ca].mean(), 100*(abs(b-y)==1)[~cb].mean()))
print("\nP2 confusion (rows=true, cols=pred):")
for k in range(1, 6): print(" ", k, [int(((y == k) & (b == j)).sum()) for j in range(1, 6)])
# margin-bucket view: where does reflection help?
print("\nby Phase-1 margin bucket (n, P1 exact, P2 exact):")
mg = np.array([t["margin"] for t in tr])
for lo, hi in ((0, .2), (.2, .5), (.5, .8), (.8, 1.01)):
    m = (mg >= lo) & (mg < hi); print(f" margin {lo}-{hi}: n={m.sum()} {ca[m].mean():.2f} -> {cb[m].mean():.2f}")
txt = {r["id"]: r["text"] for r in ev}
bad = [t for t in tr if t["first"] == t["gt"] and t["final"] != t["gt"]][:4]
good = [t for t in tr if t["first"] != t["gt"] and t["final"] == t["gt"]][:4]
json.dump({"gate": gate, "trace": tr}, open(os.environ.get("TRACE_OUT", "v2_reflective/phase2_trace.json"), "w"))
for nm, L in (("REGRESSED examples", bad), ("FIXED examples", good)):
    print("\n" + nm)
    for t in L: print(f" gt={t['gt']} P1={t['first']} P2={t['final']} margin={t['margin']:.2f} | {txt[t['id']][:200]!r}")
