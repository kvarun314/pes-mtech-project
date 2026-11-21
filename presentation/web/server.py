"""Backend for the React demo (presentation/web). Same models and logic as presentation/dashboard_app.py (the Streamlit version, kept),
served as a small JSON API on 127.0.0.1 only, so review text never leaves this laptop.
  GET  /api/demos     demo reviews            POST /api/rate     {title, text} -> Phase 1 + Phase 2, step by step
  POST /api/explain   streams Explainer / Critic events (text/event-stream)     POST /api/stress   four altered versions
  GET  /api/results   frozen test results, read from v2_reflective/e23/ result files
Run: sh presentation/run_web.sh"""
import json, os, re, sys, threading, time
from functools import lru_cache
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); os.chdir(ROOT); sys.path.insert(0, os.path.join(ROOT, "src"))
import joblib
import numpy as np
from starlette.applications import Starlette
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles
from agentic_sentiment.v3 import paths as PT

LOCK = threading.Lock()                       # MLX models are not thread-safe; one model call at a time
CACHE = json.load(open("presentation/demo_cache.json")) if os.path.exists("presentation/demo_cache.json") else {}
SEL = json.load(open(PT.D + "s1_selective.json"))
fl = lambda v: [float(x) for x in v]


# ----------------------------------------------------------------------------------------------- models (loaded once)
@lru_cache(None)
def core():
    from agentic_sentiment.v3.evidence import EvidenceAgent
    from agentic_sentiment.v3.reflector import Reflector
    from agentic_sentiment.v3.deberta import DebertaAgent
    e, rj = joblib.load(PT.EVIDENCE_JOBLIB), joblib.load(PT.REFLECTOR_JOBLIB)
    htau = [x for x in SEL["SH"]["eval"] if x["target"] == 0.4][0]["threshold"]          # Hand-off threshold fixed on dev2
    q = json.load(open(PT.D + "q1_conformal.json"))["readout"]["eval"]["q_SH"]            # conformal quantile for star ranges
    return EvidenceAgent(e["vectorizer"], e["model"]), Reflector(rj["full"], rj["fallback"]), DebertaAgent(), htau, q


@lru_cache(None)
def llama():
    from agentic_sentiment.v3.llama import SharedLlama, load_probe
    L = SharedLlama(); return L, load_probe(L)


@lru_cache(None)
def instruct():
    from agentic_sentiment.v3.explainer import load_instruct, make_chat_fn
    m, t = load_instruct(); return make_chat_fn(m, t)


def rate(title, text):
    from agentic_sentiment.v3.llama import phase1_probs
    from agentic_sentiment.v3.conformal import star_range
    from agentic_sentiment.v3.reflector import feats
    ev, rf, deb, htau, q = core(); T = {}
    with LOCK:
        L, probe = llama()
        t0 = time.time(); p1 = phase1_probs(L, text); T["phase1"] = time.time() - t0
        t0 = time.time(); h = probe.hidden(text); pp = probe.clf.predict_proba(probe.scaler.transform(h[None]))[0]; T["probe"] = time.time() - t0   # same as probe.probs
    t0 = time.time(); ep = ev.probs(title, text); T["evidence"] = time.time() - t0
    t0 = time.time(); dp = deb.probs(title, text); T["deberta"] = time.time() - t0
    star, P, shares = rf.combine(pp, np.array(ep), np.array(dp))
    C = rf.full.coef_ * feats(np.array(pp)[None], np.array(ep)[None], np.array(dp)[None])[0]      # 5 stars x 15 features: exact logit terms
    push = {k: C[:, sl].sum(1) for k, sl in (("probe", slice(0, 5)), ("evidence", slice(5, 10)), ("deberta", slice(10, 15)))}
    push = {k: fl(v - v.mean()) for k, v in push.items()}                                       # centred per agent: softmax unchanged
    base = rf.full.intercept_ - rf.full.intercept_.mean()
    lo, hi = star_range(P, q)
    decision = "auto" if max(P) >= htau else ("range" if hi - lo <= 1 else "person")
    pos, neg = ev.phrases(title, text, star)
    agents = {"probe": fl(pp), "evidence": fl(ep), "deberta": fl(dp)}
    votes = {k: int(np.argmax(v)) + 1 for k, v in agents.items()}
    return {"p1": fl(p1), "agents": agents, "votes": votes, "agreed": len(set(votes.values())) == 1, "star": int(star), "P": fl(P),
            "shares": {k: float(v) for k, v in shares.items()}, "push": push, "base": fl(base), "decision": decision, "range": [int(lo), int(hi)],
            "htau": float(htau), "pos": pos, "neg": neg, "times": {k: round(v * 1000) for k, v in T.items()},
            "hidden": {"n": int(h.shape[0]), "sample": fl(h[:64])}}


def cached_rate(rid):
    i = CACHE["demo_ids"].index(rid); s = CACHE["phase2_states"][i]; P = s["probs"]
    agents = {"probe": s.get("probe_probs") or P, "evidence": s["evidence_probs"], "deberta": s["deberta_probs"]}
    votes = {k: int(np.argmax(v)) + 1 for k, v in agents.items()}
    return {"p1": CACHE["phase1_probs"][i], "agents": agents, "votes": votes, "agreed": s["agreed"], "star": s["star"], "P": P, "shares": s["shares"],
            "push": None, "base": None, "decision": "auto" if s["handoff"] == "auto" else "person", "range": [s["star"], s["star"]], "htau": None,
            "pos": [], "neg": [], "times": {}, "cached": True}


# ----------------------------------------------------------------------------------------------- routes
async def body(req):
    b = await req.json(); return (b.get("title") or "").strip(), (b.get("text") or "").strip(), b.get("id")


async def demos(req):
    if not CACHE or not os.path.exists(PT.D + "eval.jsonl"): return JSONResponse([])
    rows = {}
    for l in open(PT.D + "eval.jsonl"):
        r = json.loads(l)
        if r["review_id"] in CACHE["demo_ids"]: rows[r["review_id"]] = r
    return JSONResponse([{"id": i, "title": rows[i]["title"], "text": rows[i]["review_text"], "truth": rows[i]["gt_rating"]} for i in CACHE["demo_ids"] if i in rows])


async def api_rate(req):
    from starlette.concurrency import run_in_threadpool
    title, text, rid = await body(req)
    if not text: return JSONResponse({"error": "Empty review"}, 400)
    try:
        return JSONResponse(await run_in_threadpool(rate, title, text))
    except Exception as ex:
        if rid in CACHE.get("demo_ids", []): return JSONResponse({**cached_rate(rid), "warning": f"Live run failed ({type(ex).__name__}); recorded result."})
        return JSONResponse({"error": f"{type(ex).__name__}: {ex}"}, 500)


async def api_stress(req):
    from starlette.concurrency import run_in_threadpool
    from agentic_sentiment.v3.perturb import flat, typos
    title, text, _ = await body(req)
    V = [("Original", "as written", title, text), ("Typos", "one word in ten misspelled", title, typos(text, seed=0)),
         ("No punctuation or capitals", "all lower case, punctuation removed", flat(title), flat(text)), ("No title", "title deleted, text unchanged", "", text)]
    try:
        out = []
        for name, what, t, x in V:
            r = await run_in_threadpool(rate, t, x)
            out.append({"name": name, "what": what, "title": t, "text": x, "p1": int(np.argmax(r["p1"])) + 1, "p2": r["star"], "conf": max(r["P"])})
        return JSONResponse(out)
    except Exception as ex:
        return JSONResponse({"error": f"{type(ex).__name__}: {ex}"}, 500)


async def api_explain(req):
    b = await req.json(); title, text, rid, res = b.get("title", ""), b.get("text", ""), b.get("id"), b["res"]

    def events():
        ev_ = lambda d: "data: " + json.dumps(d) + "\n\n"
        try:
            from agentic_sentiment.v3.graph import build_graph_v3
            ev, rf, deb, htau, q = core()
            with LOCK:
                chat = instruct(); t0 = time.time()
                g = build_graph_v3(ev, rf, None, explain_fn=chat, judge_fn=chat, deberta=deb)
                state = {"review_id": rid or "live", "title": title, "review_text": text, "probe_probs": res["agents"]["probe"], "deberta_probs": res["agents"]["deberta"]}
                yield ev_({"type": "start"})
                for upd in g.stream(state, stream_mode="updates"):
                    node, out = next(iter(upd.items()))
                    if node == "pack": yield ev_({"type": "pack", "pack": out["pack"]})
                    if node == "explainer": yield ev_({"type": "draft", "n": len(out["drafts"]), "text": out["drafts"][-1]})
                    if node == "critic": yield ev_({"type": "critic", "n": len(out["feedback"]), "feedback": out["feedback"][-1], "failed": out["failed_checks"][-1]})
                yield ev_({"type": "done", "seconds": round(time.time() - t0)})
        except Exception as ex:
            i = CACHE.get("demo_ids", []).index(rid) if rid in CACHE.get("demo_ids", []) else None
            rec = CACHE.get("explanations", [None] * 99)[i] if i is not None else None
            if rec:
                yield ev_({"type": "warning", "text": f"Live run failed ({type(ex).__name__}); recorded run."})
                for n, (d, f) in enumerate(zip(rec["drafts"], rec["feedback"]), 1):
                    yield ev_({"type": "draft", "n": n, "text": d}); yield ev_({"type": "critic", "n": n, "feedback": f, "failed": None})
                yield ev_({"type": "done", "seconds": None})
            else:
                yield ev_({"type": "error", "text": f"{type(ex).__name__}: {ex}"})
    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


def frozen_table(split):
    md = open(PT.D + "results_frozen.md").read(); sec = md.split(f"## {split}")[1].split("## ")[0]; rows = {}
    for l in sec.splitlines():
        m = re.match(r"\| ([^|]+) \| ([\d.]+) \[[^\]]*\] \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \|", l)
        if m: rows[m.group(1).strip()] = tuple(float(v) for v in m.group(2, 3, 4, 5))
    return rows


async def api_results(req):
    B = json.load(open(PT.D + "b2_report.json")); Q = json.load(open(PT.D + "q4_robustness.json")); HC = json.load(open(PT.D + "hard_cases.json"))
    hc = open(PT.D + "human_check.out").read(); hum = float(re.search(r"human faithful ([\d.]+)", hc).group(1))
    split = {m.group(1): {"n": int(m.group(2)), "human": float(m.group(3)) * 100, "critic": float(m.group(4)) * 100}
             for m in re.finditer(r"agents (agreed|disagreed): n (\d+)\s+human faithful ([\d.]+)\s+Critic faithful ([\d.]+)", hc)}
    E, N = frozen_table("eval"), frozen_table("natural_test")
    main = next(k for k in E if "(main)" in k); p1 = next(k for k in E if k.startswith("Phase 1"))
    auto = [x for x in SEL["SH"]["natural_test"] if x["target"] == 0.4][0]
    curve = lambda key: [[p["coverage"] * 100, p["acc_auto"] * 100] for p in SEL[key]["natural_test"]]
    hcs = {s: HC[s]["surface_negative_positives"] for s in ("eval", "natural_test")}
    return JSONResponse({
        "exact": {"bal": [E[p1][0], E[main][0]], "nat": [N[p1][0], N[main][0]]},
        "f1": {"bal": [E[p1][1], E[main][1]], "nat": [N[p1][1], N[main][1]]},
        "auto": {"coverage": auto["coverage"] * 100, "acc": auto["acc_auto"] * 100}, "curve": {"p1": curve("Phase1_LoRA"), "p2": curve("SH")},
        "robust": {k: [Q[v]["P1_flip"] * 100, Q[v]["SH_flip"] * 100] for k, v in (("Typos", "typos"), ("No punctuation", "flat"), ("No title", "no_title"))},
        "critic": {"bal": [B["eval"]["faithful_first"] * 100, B["eval"]["faithful_final"] * 100], "nat": [B["natural_test"]["faithful_first"] * 100, B["natural_test"]["faithful_final"] * 100]},
        "hard": {"bal": [hcs["eval"]["p1_exact"] * 100, hcs["eval"]["sh_exact"] * 100], "nat": [hcs["natural_test"]["p1_exact"] * 100, hcs["natural_test"]["sh_exact"] * 100],
                 "n": [hcs["eval"]["n"], hcs["natural_test"]["n"]]},
        "human": hum * 100, "human_split": split, "main": main,
        "models": [{"name": n, "be": E[n][0], "bf": E[n][1], "ne": N[n][0], "nf": N[n][1]} for n in E if n in N]})


DIST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dist")
app = Starlette(routes=[Route("/api/demos", demos), Route("/api/rate", api_rate, methods=["POST"]), Route("/api/stress", api_stress, methods=["POST"]),
                        Route("/api/explain", api_explain, methods=["POST"]), Route("/api/results", api_results),
                        Mount("/", StaticFiles(directory=DIST, html=True, check_dir=False))])
