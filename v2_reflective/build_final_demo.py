"""Generates presentation/Final_Demo.ipynb: Phase 1 + Phase 2 live on 5 reviews, then the paper's frozen results. ~3 min on the M1.
Run: .venv/bin/python v2_reflective/build_final_demo.py   (then check: .venv/bin/python v2_reflective/check_demo.py)"""
import nbformat as nbf
nb = nbf.v4.new_notebook(); C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip("\n"))); code = lambda s: C.append(nbf.v4.new_code_cell(s.strip("\n")))
md("# Agentic sentiment on Amazon Reviews 2023 Electronics — final demo\nPhase 1 (baseline paper) and Phase 2 (agents) run live on five reviews; every paper number below is read from the frozen result files.")
code("""
import os, sys, json, time; os.chdir(os.path.abspath(os.path.join(os.getcwd(), "..")) if os.path.basename(os.getcwd()) == "presentation" else os.getcwd()); sys.path.insert(0, "src")
import numpy as np, joblib
from IPython.display import Markdown, Image, display
from agentic_sentiment.v3 import paths as PT
from agentic_sentiment.v3.demo import cached, paper_line, pick_demo_ids
T0 = time.time(); rows = {json.loads(l)["review_id"]: json.loads(l) for l in open(PT.D + "eval.jsonl")}
ids = cached("demo_ids", pick_demo_ids); R = [rows[i] for i in ids]
for r in R: print(f"{r['gt_rating']}★  {r['review_text'][:140]}")
""")
md("## 1. Phase 1 — the baseline paper's data pipeline (20-row toy table)")
code("""
from agentic_sentiment.phase1.data.sampling import filter_by_textblob_polarity, stratified_sample, oversample_neutral
import random
toy = [{"text": r["review_text"], "rating": r["gt_rating"]} for r in list(rows.values())[:20]]
dqc = filter_by_textblob_polarity(toy); print("TextBlob DQC kept", len(dqc), "of", len(toy), "| dropped ratings:", sorted(r["rating"] for r in toy if r not in dqc))
st = stratified_sample(toy, samples_per_rating=2, rng=random.Random(0)); print("stratified:", sorted(r["rating"] for r in st))
print("neutral oversampled:", sorted(r["rating"] for r in oversample_neutral(st, rng=random.Random(0))))
print("Why DQC was dropped on Electronics: it deletes 52% of 1★ and 67% of 2★ reviews (negative reviews use positive words).")
""")
md("## 2. Phase 1 — LLaMA-3-8B + LoRA rates the five reviews")
code("""
from agentic_sentiment.v3.llama import SharedLlama, phase1_probs, load_probe
try:
    llama = SharedLlama()                       # one model for both phases (~20 s)
except Exception as ex:
    llama = None; print(f"[model not loaded ({type(ex).__name__}); live cells will show cached results]")
p1 = cached("phase1_probs", lambda: [list(map(float, phase1_probs(llama, r["review_text"]))) for r in R])
for r, p in zip(R, p1): print(f"true {r['gt_rating']}★  Phase 1 → {int(np.argmax(p)) + 1}★  probs {np.round(p, 2)}")
paper_line("Phase 1 LoRA eval exact", "54.0"); paper_line("Phase 1 LoRA natural_test exact", "53.4")
paper_line("leak audit one-shot A vs random B", "-7.7")
""")
md("## 3. Phase 2 — agents decide the star (Evidence → Router → probe → Reflector → Hand-off)")
code("""
from agentic_sentiment.v3.evidence import EvidenceAgent
from agentic_sentiment.v3.reflector import Reflector
from agentic_sentiment.v3.graph import build_graph_v3
e, rj = joblib.load(PT.EVIDENCE_JOBLIB), joblib.load(PT.REFLECTOR_JOBLIB); ev, rf = EvidenceAgent(e["vectorizer"], e["model"]), Reflector(rj["full"], rj["fallback"])
tau = json.load(open(PT.D + "gate_B1b.json"))["tau"]                                    # Router attempt 2 (passed)
htau = [x for x in json.load(open(PT.D + "s1_selective.json"))["SH"]["eval"] if x["target"] == 0.4][0]["threshold"]   # Hand-off, dev2 quantile
probe = load_probe(llama) if llama else None; g = build_graph_v3(ev, rf, probe.probs if probe else None, tau=tau, handoff_tau=htau)
p2 = cached("phase2_states", lambda: [{k: v for k, v in g.invoke({"review_id": r["review_id"], "title": r["title"], "review_text": r["review_text"]}).items() if k in ("route", "star", "probs", "shares", "agreed", "handoff")} for r in R])
for r, s in zip(R, p2): print(f"true {r['gt_rating']}★  Phase 2 → {s['star']}★  route={s['route']}  shares={ {k: round(float(v), 2) for k, v in s['shares'].items()} }  agents agree={s['agreed']}  hand-off={s['handoff']}")
llama = probe = None; import gc, mlx.core as mx; gc.collect(); mx.clear_cache()   # free ~5 GB before loading the Instruct model
""")
md("## 4. Phase 2 — explanation, checked by the Critic")
code("""
from agentic_sentiment.v3.explainer import load_instruct, make_chat_fn
def explain():
    m, t = load_instruct(); chat = make_chat_fn(m, t); gx = build_graph_v3(ev, rf, None, explain_fn=chat, judge_fn=chat)
    out = []
    for r, s in list(zip(R, p2))[:2]:
        o = gx.invoke({"review_id": r["review_id"], "title": r["title"], "review_text": r["review_text"], "probe_probs": s["probs"]})
        out.append({k: o[k] for k in ("pack", "drafts", "feedback", "faithful")})
    return out
for o in cached("explanations", explain):
    print("evidence for:", [p["display"] for p in o["pack"]["for"]]); [print(f"draft {i}: {d}\\n  critic: {f or 'faithful'}") for i, (d, f) in enumerate(zip(o["drafts"], o["feedback"]))]
if os.path.exists(PT.D + "b2_report.json"):
    B = json.load(open(PT.D + "b2_report.json"))
    for s in ("eval", "natural_test"): paper_line(f"faithful first {s}", f"{B[s]['faithful_first']:.3f}"); paper_line(f"faithful final {s}", f"{B[s]['faithful_final']:.3f}")
""")
md("## 5. The paper's results (frozen)")
code("""
display(Markdown(open(PT.D + "results_frozen.md").read()))
for f in ("main_bars.png", "reliability_SH_eval.png", "deletion_test.png", "faithfulness.png"):
    p = "paper/figures/e23/" + f
    if os.path.exists(p): display(Image(p, width=800))
display(Markdown(open(PT.D + "agent_audit.md").read()))
""")
md("## 6. Phase 1 vs Phase 2 on the same reviews")
code("""
display(Markdown("| true | Phase 1 | Phase 2 |\\n|---|---|---|\\n" + "\\n".join(f"| {r['gt_rating']}★ | {int(np.argmax(a)) + 1}★ | {b['star']}★ |" for r, a, b in zip(R, p1, p2))))
paper_line("hand-off natural_test exact at 55.7% coverage", "93.1"); paper_line("SH eval exact", "67.5"); paper_line("SH natural_test exact", "75.7")
print(f"demo runtime {time.time() - T0:.0f}s")
""")
nb["metadata"] = {"kernelspec": {"name": "pes-mtech", "display_name": "PES MTech (.venv)", "language": "python"}}
nb["cells"] = C; nbf.write(nb, "presentation/Final_Demo.ipynb"); print("wrote", len(C), "cells")
