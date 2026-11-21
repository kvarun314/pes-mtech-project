"""Live demo dashboard: Phase 1 (reference method) vs Phase 2 (agents) on any review, the Explainer-Critic loop round by round,
live stress tests, and the frozen results as interactive charts. Runs fully on this laptop; review text never leaves it.
Run:  sh presentation/run_dashboard.sh      (or: .venv/bin/streamlit run presentation/dashboard_app.py)
Every result number on the Results tab is read from v2_reflective/e23/ result files, not typed in."""
import json, os, re, sys, threading, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(ROOT); sys.path.insert(0, os.path.join(ROOT, "src"))
import joblib
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from agentic_sentiment.v3 import paths as PT

st.set_page_config(page_title="Agentic review rating · live demo", page_icon="★", layout="wide")
BLUE, DEEP, SKY, GREY, INK, MUTED, RED, TINT = "#2C5AA0", "#1B3C6E", "#7FA8E8", "#C7C7CC", "#1D1D1F", "#6E6E73", "#A93636", "#F5F5F7"
LOCK = threading.Lock()                       # MLX models are not thread-safe; one call at a time
CACHE = json.load(open("presentation/demo_cache.json")) if os.path.exists("presentation/demo_cache.json") else {}

st.markdown(f"""<style>
html, body, [class*="css"] {{ font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Helvetica Neue", Arial, sans-serif; }}
.block-container {{ padding-top: 1.6rem; max-width: 1280px; }}
.hero {{ background: linear-gradient(135deg, #0B1A33 0%, {DEEP} 100%); color: white; border-radius: 22px; padding: 26px 32px; margin-bottom: 18px; }}
.hero h1 {{ font-size: 2.1rem; margin: 0; font-weight: 700; letter-spacing: -0.02em; color: white; }}
.hero p {{ color: #C9D7EA; margin: 6px 0 0 0; font-size: 1.05rem; }}
.card {{ background: {TINT}; border-radius: 18px; padding: 18px 20px; height: 100%; }}
.card .lab {{ color: {MUTED}; font-size: .78rem; font-weight: 600; letter-spacing: .06em; text-transform: uppercase; }}
.card .big {{ font-size: 2.6rem; font-weight: 700; letter-spacing: -0.03em; line-height: 1.1; color: {INK}; }}
.card .big.blue {{ color: {BLUE}; }} .card .sub {{ color: {MUTED}; font-size: .9rem; }}
.stars {{ color: {BLUE}; font-size: 1.5rem; letter-spacing: 2px; }} .stars.grey {{ color: #A1A1A6; }}
.pill {{ display: inline-block; padding: 4px 12px; border-radius: 999px; font-weight: 600; font-size: .85rem; }}
.pill.ok {{ background: #E7F1FB; color: {BLUE}; }} .pill.warn {{ background: #F9ECE7; color: #8A3B12; }} .pill.bad {{ background: #FBEBEB; color: {RED}; }}
.review {{ background: white; border: 1px solid #E5E5EA; border-radius: 16px; padding: 16px 18px; line-height: 1.7; font-size: 1.02rem; }}
mark.for {{ background: #DCE8F7; color: {DEEP}; border-radius: 6px; padding: 1px 4px; }}
mark.against {{ background: #F7E1E1; color: {RED}; border-radius: 6px; padding: 1px 4px; }}
.draft {{ background: white; border: 1px solid #E5E5EA; border-radius: 16px; padding: 14px 16px; margin-bottom: 10px; }}
.check {{ display: inline-block; margin: 3px 6px 3px 0; padding: 3px 10px; border-radius: 999px; font-size: .82rem; font-weight: 600; }}
.check.pass {{ background: #E7F1FB; color: {BLUE}; }} .check.fail {{ background: #FBEBEB; color: {RED}; }}
.note {{ color: {MUTED}; font-size: .85rem; }}
div[data-testid="stTabs"] button p {{ font-size: 1.02rem; font-weight: 600; }}
</style>""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------------------------- models (loaded once)
@st.cache_resource(show_spinner="Loading the Evidence model, DeBERTa-v3 and the Reflector…")
def core():
    from agentic_sentiment.v3.evidence import EvidenceAgent
    from agentic_sentiment.v3.reflector import Reflector
    from agentic_sentiment.v3.deberta import DebertaAgent
    e, rj = joblib.load(PT.EVIDENCE_JOBLIB), joblib.load(PT.REFLECTOR_JOBLIB)
    sel = json.load(open(PT.D + "s1_selective.json"))
    htau = [x for x in sel["SH"]["eval"] if x["target"] == 0.4][0]["threshold"]          # Hand-off threshold fixed on dev2
    q = json.load(open(PT.D + "q1_conformal.json"))["readout"]["eval"]["q_SH"]            # conformal quantile for star ranges
    return EvidenceAgent(e["vectorizer"], e["model"]), Reflector(rj["full"], rj["fallback"]), DebertaAgent(), htau, q


@st.cache_resource(show_spinner="Loading LLaMA-3-8B (Phase 1 adapter + Phase 2 probe), about 20 s…")
def llama():
    from agentic_sentiment.v3.llama import SharedLlama, load_probe
    L = SharedLlama(); return L, load_probe(L)


@st.cache_resource(show_spinner="Loading Llama-3-8B-Instruct (Explainer and Critic), about 20 s…")
def instruct():
    from agentic_sentiment.v3.explainer import load_instruct, make_chat_fn
    m, t = load_instruct(); return make_chat_fn(m, t)


def rate(title, text):
    """Phase 1 and Phase 2 on one review. Returns a dict; raises if a model cannot run."""
    from agentic_sentiment.v3.llama import phase1_probs
    from agentic_sentiment.v3.conformal import star_range
    ev, rf, deb, htau, q = core(); L, probe = llama(); T = {}
    with LOCK:
        t0 = time.time(); p1 = phase1_probs(L, text); T["Phase 1 (LLaMA + LoRA)"] = time.time() - t0
        t0 = time.time(); pp = probe.probs(text); T["LLaMA probe"] = time.time() - t0
    t0 = time.time(); ep = ev.probs(title, text); T["Evidence model"] = time.time() - t0
    t0 = time.time(); dp = deb.probs(title, text); T["DeBERTa-v3"] = time.time() - t0
    t0 = time.time(); star, P, shares = rf.combine(pp, np.array(ep), np.array(dp)); T["Reflector"] = time.time() - t0
    lo, hi = star_range(P, q)
    decision = "auto" if max(P) >= htau else ("range" if hi - lo <= 1 else "person")
    pos, neg = ev.phrases(title, text, star)
    votes = {"LLaMA probe": int(np.argmax(pp)) + 1, "Evidence model": int(np.argmax(ep)) + 1, "DeBERTa-v3": int(np.argmax(dp)) + 1}
    return {"p1": list(map(float, p1)), "probe": list(map(float, pp)), "evidence": list(map(float, ep)), "deberta": list(map(float, dp)),
            "star": int(star), "P": list(map(float, P)), "shares": {k: float(v) for k, v in shares.items()}, "votes": votes,
            "agreed": len(set(votes.values())) == 1, "decision": decision, "range": (lo, hi), "htau": float(htau), "pos": pos, "neg": neg, "times": T}


def cached_rate(rid):
    """Recorded result for a demo review (used only if the live run fails)."""
    i = CACHE["demo_ids"].index(rid); s = CACHE["phase2_states"][i]; P = s["probs"]
    return {"p1": CACHE["phase1_probs"][i], "probe": s.get("probe_probs") or P, "evidence": s["evidence_probs"], "deberta": s["deberta_probs"],
            "star": s["star"], "P": P, "shares": s["shares"], "votes": {}, "agreed": s["agreed"], "decision": "auto" if s["handoff"] == "auto" else "person",
            "range": (s["star"], s["star"]), "htau": None, "pos": [], "neg": [], "times": {}, "cached": True}


# ----------------------------------------------------------------------------------------------- helpers
def stars(n, grey=False):
    return f'<span class="stars{" grey" if grey else ""}">{"★" * n}{"☆" * (5 - n)}</span>'


def card(label, big, sub="", blue=False, html_big=False):
    st.markdown(f'<div class="card"><div class="lab">{label}</div><div class="big{" blue" if blue else ""}">{big}</div><div class="sub">{sub}</div></div>', unsafe_allow_html=True)


def bars(series, title="", ymax=1.0, pct=True, h=300, colors=None, cats=None):
    cats = cats or ["1★", "2★", "3★", "4★", "5★"]; fig = go.Figure()
    for (name, vals), c in zip(series, colors or [GREY, SKY, BLUE, DEEP]):
        fig.add_bar(name=name, x=cats, y=vals, marker_color=c, text=[f"{v:.0%}" if pct else f"{v:.1f}" for v in vals], textposition="outside")
    fig.update_layout(title=title, height=h, barmode="group", margin=dict(l=10, r=10, t=40 if title else 10, b=10), plot_bgcolor="white",
                      paper_bgcolor="white", font=dict(family="-apple-system, Helvetica Neue, Arial", size=13, color=INK),
                      legend=dict(orientation="h", y=1.12, x=0), yaxis=dict(visible=False, range=[0, ymax * 1.18]), bargap=0.25)
    return fig


def highlight(title, text, pos, neg):
    from html import escape
    out = escape(text); t = escape(title or "")
    for cls, items in (("for", pos), ("against", neg)):
        for p in sorted({x["display"] for x in items}, key=len, reverse=True):
            pat = re.compile(re.escape(escape(p)), re.I)
            out = pat.sub(lambda m: f'<mark class="{cls}">{m.group(0)}</mark>', out, count=1)
            t = pat.sub(lambda m: f'<mark class="{cls}">{m.group(0)}</mark>', t, count=1)
    return f'<div class="review"><b>{t}</b><br>{out}</div>'


@st.cache_data
def demo_reviews():
    if not CACHE: return []
    rows = {}
    for l in open(PT.D + "eval.jsonl"):
        r = json.loads(l)
        if r["review_id"] in CACHE["demo_ids"]: rows[r["review_id"]] = r
    return [rows[i] for i in CACHE["demo_ids"] if i in rows]


# ----------------------------------------------------------------------------------------------- header + input
st.markdown('<div class="hero"><h1>Agentic Self-Correction in LLaMA-3 · live demo</h1>'
            '<p>Three agents rate the star. LLaMA-3 explains it. A Critic checks it. Everything runs live on this laptop.</p></div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Review")
    D = demo_reviews()
    opts = ["Write my own"] + [f"Demo {i + 1}: true {r['gt_rating']}★ · {r['title'][:40]}" for i, r in enumerate(D)]
    pick = st.selectbox("Start from", opts, index=1 if D else 0)
    base = D[opts.index(pick) - 1] if pick != "Write my own" else {"title": "", "review_text": "", "gt_rating": None, "review_id": None}
    title = st.text_input("Title", base["title"], key=f"t_{pick}")
    text = st.text_area("Review text", base["review_text"], height=220, key=f"x_{pick}")
    go_btn = st.button("Rate this review", type="primary", width="stretch")
    st.markdown('<p class="note">Models load on first use (about 40 s). Close other apps before the talk; the two LLaMA models need about 10 GB.</p>', unsafe_allow_html=True)

if go_btn and text.strip():
    try:
        with st.spinner("Phase 1 and the three agents are reading the review…"):
            st.session_state.res = rate(title, text); st.session_state.res["cached"] = False
    except Exception as ex:
        if base.get("review_id") in CACHE.get("demo_ids", []) and title == base["title"] and text == base["review_text"]:
            st.session_state.res = cached_rate(base["review_id"]); st.warning(f"Live run failed ({type(ex).__name__}); showing the recorded result for this demo review.")
        else:
            st.error(f"Live run failed: {type(ex).__name__}: {ex}"); st.stop()
    st.session_state.inp = {"title": title, "text": text, "truth": base.get("gt_rating") if text == base["review_text"] else None, "rid": base.get("review_id")}
    st.session_state.pop("expl", None); st.session_state.pop("stress", None)

tabs = st.tabs(["Live tester", "Explanation and Critic", "Stress tests", "Results"])
res, inp = st.session_state.get("res"), st.session_state.get("inp")

# ----------------------------------------------------------------------------------------------- tab 1: live tester
with tabs[0]:
    if not res:
        st.info("Pick a demo review or write your own in the sidebar, then press **Rate this review**.")
    else:
        p1s = int(np.argmax(res["p1"])) + 1; truth = inp["truth"]
        c = st.columns([1, 1, 1, 1])
        with c[0]: card("Phase 1 · reference method", f'{p1s}★', stars(p1s, grey=True) + (f" · {max(res['p1']):.0%} sure" ))
        with c[1]: card("Phase 2 · three agents", f'{res["star"]}★', stars(res["star"]) + f" · {max(res['P']):.0%} sure", blue=True)
        with c[2]: card("True star", f"{truth}★" if truth else "—", "from the test set" if truth else "your own review")
        dec = {"auto": ('<span class="pill ok">Answers automatically</span>', "confident enough"),
               "range": (f'<span class="pill warn">Star range {res["range"][0]}–{res["range"][1]}★</span>', "holds the true star ~90% of the time"),
               "person": ('<span class="pill bad">Hands to a person</span>', "too unsure to answer")}[res["decision"]]
        with c[3]: card("Hand-off agent", dec[0], dec[1] + (f" (threshold {res['htau']:.2f})" if res.get("htau") else ""))
        if res.get("cached"): st.caption("Recorded result (live run unavailable).")
        st.write("")
        a, b = st.columns([1.15, 1])
        with a:
            st.markdown("##### What each agent thinks (probability per star)")
            st.plotly_chart(bars([("LLaMA probe", res["probe"]), ("Evidence model", res["evidence"]), ("DeBERTa-v3", res["deberta"]), ("Reflector (final)", res["P"])], h=330), width="stretch")
            if res["votes"]:
                st.markdown("Votes: " + " · ".join(f"**{k}** {v}★" for k, v in res["votes"].items()) +
                            ("  ·  <span class='pill ok'>all agree</span>" if res["agreed"] else "  ·  <span class='pill warn'>agents disagree</span>"), unsafe_allow_html=True)
        with b:
            st.markdown("##### Phase 1 vs Phase 2")
            st.plotly_chart(bars([("Phase 1", res["p1"]), ("Phase 2", res["P"])], h=230, colors=[GREY, BLUE]), width="stretch")
            st.markdown("##### Who the Reflector listened to")
            sh = res["shares"]; fig = go.Figure(go.Bar(x=list(sh.values()), y=list(sh.keys()), orientation="h", marker_color=[SKY, BLUE, DEEP][:len(sh)],
                                                     text=[f"{v:.0%}" for v in sh.values()], textposition="outside"))
            fig.update_layout(height=150, margin=dict(l=10, r=40, t=5, b=5), xaxis=dict(visible=False, range=[0, 1.15]), plot_bgcolor="white", paper_bgcolor="white")
            st.plotly_chart(fig, width="stretch")
        st.markdown("##### The review, with the evidence the decision used")
        st.markdown(highlight(inp["title"], inp["text"], res["pos"], res["neg"]), unsafe_allow_html=True)
        st.markdown('<p class="note"><mark class="for">blue</mark> supports the star · <mark class="against">red</mark> argues against it (Evidence model weights)</p>', unsafe_allow_html=True)
        if res["times"]:
            st.caption("Time: " + " · ".join(f"{k} {v * 1000:.0f} ms" for k, v in res["times"].items()))

# ----------------------------------------------------------------------------------------------- tab 2: explanation + critic
CHECKS = {1: "Star matches", 2: "Quotes are real", 3: "Uses key phrases", 4: "Mentions disagreement", 5: "No unsupported claim (LLM judge)"}

with tabs[1]:
    if not res:
        st.info("Rate a review first.")
    else:
        st.markdown("The Explainer writes from the evidence; the Critic runs five checks and sends it back until it passes (at most four rewrites). "
                    "Live this takes about 15–40 s.")
        run = st.button("Explain and check live", type="primary")
        box = st.container()
        if run:
            try:
                from agentic_sentiment.v3.graph import build_graph_v3
                ev, rf, deb, htau, q = core(); chat = instruct()
                g = build_graph_v3(ev, rf, None, explain_fn=chat, judge_fn=chat, deberta=deb)
                state = {"review_id": inp["rid"] or "live", "title": inp["title"], "review_text": inp["text"], "probe_probs": res["probe"], "deberta_probs": res["deberta"]}
                drafts, fbs, fails, t0 = [], [], [], time.time()
                with box, LOCK:
                    status = st.status("Explainer writing draft 1…", expanded=True)
                    for upd in g.stream(state, stream_mode="updates"):
                        node, out = next(iter(upd.items()))
                        if node == "pack":
                            status.write("Evidence pack: " + ", ".join(f"“{p['display']}”" for p in out["pack"]["for"]))
                        if node == "explainer":
                            drafts = out["drafts"]; status.write(f"**Draft {len(drafts)}:** {drafts[-1]}"); status.update(label=f"Critic checking draft {len(drafts)}…")
                        if node == "critic":
                            fbs, fails = out["feedback"], out["failed_checks"]
                            status.write(("✓ passes all five checks" if not fbs[-1] else "✗ Critic: " + " ".join(fbs[-1])))
                            status.update(label="Passed" if not fbs[-1] else f"Explainer rewriting (round {len(drafts)})…")
                    status.update(label=f"Done in {time.time() - t0:.0f} s", state="complete", expanded=False)
                st.session_state.expl = {"drafts": drafts, "feedback": fbs, "fails": fails, "cached": False}
            except Exception as ex:
                i = CACHE.get("demo_ids", []).index(inp["rid"]) if inp["rid"] in CACHE.get("demo_ids", []) else None
                rec = next((o for j, o in enumerate(CACHE.get("explanations", [])) if j == i), None)
                if rec:
                    st.warning(f"Live run failed ({type(ex).__name__}); showing the recorded run for this demo review.")
                    st.session_state.expl = {"drafts": rec["drafts"], "feedback": rec["feedback"], "fails": [], "cached": True}
                else:
                    st.error(f"Live explanation failed: {type(ex).__name__}: {ex}")
        e = st.session_state.get("expl")
        if e:
            for i, (d, f) in enumerate(zip(e["drafts"], e["feedback"])):
                failed = set(e["fails"][i]) if i < len(e["fails"]) else (set() if not f else None)
                chips = "".join(f'<span class="check {"fail" if failed and k in failed else "pass"}">{"✗" if failed and k in failed else "✓"} {v}</span>'
                                for k, v in CHECKS.items()) if failed is not None else ""
                verdict = '<span class="pill ok">Passes all checks</span>' if not f else f'<span class="pill bad">Critic</span> {" ".join(f)}'
                st.markdown(f'<div class="draft"><div class="lab note">DRAFT {i + 1}</div><p>{d}</p>{chips}<p>{verdict}</p></div>', unsafe_allow_html=True)
            st.caption("On all 3,000 test reviews the loop raises the Critic pass rate from 62–64% to 90.5–91.6% with none broken; a blind human check gives 80%.")

# ----------------------------------------------------------------------------------------------- tab 3: stress tests
with tabs[2]:
    if not res:
        st.info("Rate a review first.")
    else:
        from agentic_sentiment.v3.perturb import flat, typos
        st.markdown("Change the review the way real text varies, and watch whether the star moves. On 300 test reviews, flat text changed 9.3% of "
                    "Phase 2 stars vs 29.7% for Phase 1; removing the title changed 15.7% (Phase 2 reads the title, Phase 1 does not).")
        V = {"Original": (inp["title"], inp["text"]), "Title removed": ("", inp["text"]), "No punctuation or capitals": (flat(inp["title"]), flat(inp["text"])),
             "Typos (10% of words)": (inp["title"], typos(inp["text"], seed=0))}
        if st.button("Run all four versions live", type="primary"):
            out = {}
            try:
                with st.spinner("Rating four versions…"):
                    for k, (t, x) in V.items(): r = rate(t, x); out[k] = (int(np.argmax(r["p1"])) + 1, r["star"], max(r["P"]))
                st.session_state.stress = out
            except Exception as ex:
                st.error(f"Live stress test failed: {type(ex).__name__}: {ex}")
        s = st.session_state.get("stress")
        if s:
            o1, o2, _ = s["Original"]; cols = st.columns(4)
            for c, (k, (a, b, conf)) in zip(cols, s.items()):
                ch = (a != o1, b != o2)
                with c:
                    card(k, f"{b}★", f"Phase 2 {'changed' if ch[1] else 'unchanged'} · {conf:.0%} sure<br>Phase 1: {a}★ {'(changed)' if ch[0] else ''}", blue=not ch[1])
            with st.expander("See the altered text"):
                for k, (t, x) in V.items(): st.markdown(f"**{k}** — *{t or '(no title)'}* · {x}")

# ----------------------------------------------------------------------------------------------- tab 4: results (frozen)
def frozen_table(split):
    md = open(PT.D + "results_frozen.md").read(); sec = md.split(f"## {split}")[1].split("## ")[0]; rows = {}
    for l in sec.splitlines():
        m = re.match(r"\| ([^|]+) \| ([\d.]+) \[[^\]]*\] \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \|", l)
        if m: rows[m.group(1).strip()] = tuple(float(v) for v in m.group(2, 3, 4, 5))
    return rows


with tabs[3]:
    B = json.load(open(PT.D + "b2_report.json")); S = json.load(open(PT.D + "s1_selective.json")); Q = json.load(open(PT.D + "q4_robustness.json"))
    H = json.load(open(PT.D + "hard_cases.json")); hc = open(PT.D + "human_check.out").read()
    E, N = frozen_table("eval"), frozen_table("natural_test")
    main = next(k for k in E if "(main)" in k); p1 = next(k for k in E if k.startswith("Phase 1"))
    auto = [x for x in S["SH"]["natural_test"] if x["target"] == 0.4][0]
    hum = float(re.search(r"human faithful ([\d.]+)", hc).group(1))
    k = st.columns(5)
    with k[0]: card("Exact star, balanced", f"{E[main][0]:.1f}%", f"Phase 1 {E[p1][0]:.1f}% · +{E[main][0] - E[p1][0]:.1f}", blue=True)
    with k[1]: card("Macro-F1, natural", f"{N[main][1]:.3f}".lstrip("0"), f"Phase 1 {N[p1][1]:.3f}".replace("0.", "."), blue=True)
    with k[2]: card("Hand-off, natural", f"{auto['acc_auto']:.1%}", f"right on the {auto['coverage']:.0%} it answers alone", blue=True)
    with k[3]: card("Critic pass rate", f"{B['natural_test']['faithful_final']:.1%}", f"first draft {B['natural_test']['faithful_first']:.1%}; 0 broken", blue=True)
    with k[4]: card("Blind human check", f"{hum:.0%}", "author, 50 explanations")
    st.caption("Natural test: 65% of reviews are 5-star, so raw exact accuracy flatters Phase 2 (53.4% → 78.0%); against Phase 1 adjusted to that mix the gain is +2.5. "
               "The two-agent system fixed before any test read scores 67.5% / 75.7%.")
    l, r = st.columns(2)
    with l:
        split = st.radio("Test set", ["Balanced (1,000)", "Natural (2,000)"], horizontal=True)
        T = E if split.startswith("Bal") else N; metric = st.radio("Metric", ["Exact %", "Macro-F1"], horizontal=True)
        names = list(T.keys()); vals = [T[n][0] if metric == "Exact %" else T[n][1] for n in names]
        order = np.argsort(vals); fig = go.Figure(go.Bar(x=[vals[i] for i in order], y=[names[i] for i in order], orientation="h",
                                                        marker_color=[BLUE if "(main)" in names[i] else (GREY if names[i].startswith("Phase 1") else SKY) for i in order],
                                                        text=[f"{vals[i]:.1f}" if metric == "Exact %" else f"{vals[i]:.3f}" for i in order], textposition="outside"))
        fig.update_layout(title="Every model on the frozen test set", height=520, margin=dict(l=10, r=40, t=40, b=10), plot_bgcolor="white", paper_bgcolor="white",
                          xaxis=dict(visible=False), font=dict(size=12))
        st.plotly_chart(fig, width="stretch")
    with r:
        fig = go.Figure()
        for name, key, c in (("Phase 2", "SH", BLUE), ("Phase 1", "Phase1_LoRA", GREY)):
            pts = S[key]["natural_test"]; fig.add_scatter(x=[p["coverage"] * 100 for p in pts], y=[p["acc_auto"] * 100 for p in pts], name=name, mode="lines+markers",
                                                          line=dict(color=c, width=4), marker=dict(size=8))
        fig.add_annotation(x=auto["coverage"] * 100, y=auto["acc_auto"] * 100, text=f"dev2 threshold: {auto['acc_auto']:.1%} on {auto['coverage']:.0%}", showarrow=True, ay=-40)
        fig.update_layout(title="Hand-off: answer only when sure (natural test)", height=300, xaxis_title="Reviews answered automatically (%)", yaxis_title="Right on those (%)",
                          plot_bgcolor="white", paper_bgcolor="white", margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=1.12))
        st.plotly_chart(fig, width="stretch")
        rob = {"Typos": "typos", "No punctuation": "flat", "No title": "no_title"}
        fig = bars([("Phase 1", [Q[v]["P1_flip"] * 100 for v in rob.values()]), ("Phase 2", [Q[v]["SH_flip"] * 100 for v in rob.values()])],
                   title="Stars changed by altered text (%, lower is better)", ymax=35, pct=False, h=260, colors=[GREY, BLUE], cats=list(rob))
        st.plotly_chart(fig, width="stretch")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(bars([("First draft", [B[s]["faithful_first"] * 100 for s in ("eval", "natural_test")]), ("After the Critic loop", [B[s]["faithful_final"] * 100 for s in ("eval", "natural_test")])],
                             title="Critic pass rate (%)", ymax=100, pct=False, h=280, colors=[GREY, BLUE], cats=["Balanced", "Natural"]), width="stretch")
    with c2:
        hcs = {s: H[s]["surface_negative_positives"] for s in ("eval", "natural_test")}
        st.plotly_chart(bars([("Phase 1", [hcs[s]["p1_exact"] * 100 for s in hcs]), ("Phase 2", [hcs[s]["sh_exact"] * 100 for s in hcs])],
                             title=f"Negative words, happy reviewer (n = {hcs['eval']['n']} / {hcs['natural_test']['n']})", ymax=100, pct=False, h=280, colors=[GREY, BLUE],
                             cats=["Balanced", "Natural"]), width="stretch")
    st.caption("Sources: results_frozen.md, s1_selective.json, b2_report.json, q4_robustness.json, hard_cases.json, human_check.out (v2_reflective/e23/).")
