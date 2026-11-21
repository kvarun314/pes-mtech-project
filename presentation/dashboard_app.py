"""Live demo dashboard. Each tab shows the answer first, then how the system reached it, step by step:
How it rates (Phase 1 vs the three agents, the Reflector and the Hand-off agent), How it explains (Explainer and Critic, round by round),
Stress test (the same review, altered), Results (the frozen test results). Runs fully on this laptop; review text never leaves it.
Run:  sh presentation/run_dashboard.sh      (or: .venv/bin/streamlit run presentation/dashboard_app.py)
Every number on the Results tab is read from v2_reflective/e23/ result files, not typed in. Previous version: docs/archive/dashboard_app_2026-11-21.py"""
import json, os, re, sys, threading, time
from html import escape
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(ROOT); sys.path.insert(0, os.path.join(ROOT, "src"))
import joblib
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from agentic_sentiment.v3 import paths as PT

st.set_page_config(page_title="Agentic review rating", page_icon="★", layout="wide", initial_sidebar_state="collapsed")
BLUE, DEEP, SKY, GREY, INK, BODY, MUTED, RED, TINT, RULE = "#2C5AA0", "#1B3C6E", "#7FA8E8", "#C7C7CC", "#1D1D1F", "#424245", "#6E6E73", "#A93636", "#F5F5F7", "#E5E5EA"
LOCK = threading.Lock()                       # MLX models are not thread-safe; one call at a time
SEL = json.load(open(PT.D + "s1_selective.json")); ROB = json.load(open(PT.D + "q4_robustness.json"))
AUTO = [x for x in SEL["SH"]["natural_test"] if x["target"] == 0.4][0]                          # Hand-off at the dev2-fixed threshold, natural test
CACHE = json.load(open("presentation/demo_cache.json")) if os.path.exists("presentation/demo_cache.json") else {}
AGENTS = {"probe": ("LLaMA probe", "Reads the review with LLaMA-3 and takes the star from what the model understood (its hidden state)."),
          "evidence": ("Evidence model", "Weighs the words and short phrases in the title and text, as learned from labelled reviews."),
          "deberta": ("DeBERTa-v3", "A language model fine-tuned to rate reviews. Reads title and text together.")}
CHECKS = {1: ("Right star", "The explanation states the same star the system chose."),
          2: ("Real quotes", "Every quoted phrase really appears in the review."),
          3: ("Key evidence", "It uses at least two of the strongest evidence phrases."),
          4: ("Honest about doubt", "If the agents disagreed, it says so."),
          5: ("Nothing made up", "A second LLaMA-3 reads it and finds no claim the review does not support.")}

st.markdown(f"""<style>
html, body, [class*="css"] {{ font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Helvetica Neue", Arial, sans-serif; color: {INK}; }}
.block-container {{ padding-top: 2.2rem; max-width: 1180px; }}
header[data-testid="stHeader"] {{ background: transparent; }}
h1.app {{ font-size: 2.2rem; font-weight: 700; letter-spacing: -0.025em; margin: 0; }}
p.lede {{ color: {MUTED}; font-size: 1.08rem; margin: 4px 0 18px 0; }}
div[data-testid="stTabs"] button p {{ font-size: 1rem; font-weight: 600; }}
.tile {{ background: {TINT}; border-radius: 18px; padding: 18px 20px; height: 100%; }}
.tile.main {{ background: white; border: 2px solid {BLUE}; }}
.lab {{ color: {MUTED}; font-size: .74rem; font-weight: 600; letter-spacing: .07em; text-transform: uppercase; }}
.big {{ font-size: 2.5rem; font-weight: 700; letter-spacing: -0.03em; line-height: 1.15; }}
.big.blue {{ color: {BLUE}; }} .sub {{ color: {MUTED}; font-size: .92rem; line-height: 1.45; }}
.stars {{ color: {BLUE}; letter-spacing: 2px; font-size: 1.15rem; }} .stars.grey {{ color: #A1A1A6; }}
.step {{ display: flex; gap: 14px; align-items: flex-start; margin: 30px 0 12px 0; }}
.num {{ flex: none; width: 30px; height: 30px; border-radius: 50%; background: {INK}; color: white; font-weight: 700; display: flex; align-items: center; justify-content: center; font-size: .95rem; }}
.step h3 {{ margin: 0; font-size: 1.25rem; font-weight: 700; letter-spacing: -0.01em; }}
.step p {{ margin: 2px 0 0 0; color: {MUTED}; font-size: .97rem; }}
.agent {{ background: {TINT}; border-radius: 18px; padding: 16px 18px; height: 100%; }}
.agent .name {{ font-weight: 700; font-size: 1.05rem; }} .agent .what {{ color: {MUTED}; font-size: .86rem; min-height: 2.6em; margin: 2px 0 10px 0; }}
.vote {{ font-size: 1.9rem; font-weight: 700; letter-spacing: -0.02em; }} .vote small {{ font-size: .9rem; color: {MUTED}; font-weight: 500; }}
.hist {{ display: flex; gap: 6px; align-items: flex-end; height: 92px; margin-top: 8px; }}
.hc {{ flex: 1; text-align: center; display: flex; flex-direction: column; justify-content: flex-end; height: 100%; }}
.hb {{ border-radius: 5px 5px 2px 2px; }} .hv {{ font-size: .7rem; color: {MUTED}; margin-bottom: 2px; }} .hl {{ font-size: .75rem; color: {MUTED}; margin-top: 3px; }}
.stack {{ display: flex; height: 34px; border-radius: 10px; overflow: hidden; margin: 6px 0 6px 0; }}
.stack div {{ color: white; font-size: .82rem; font-weight: 600; display: flex; align-items: center; justify-content: center; white-space: nowrap; overflow: hidden; }}
.gauge {{ position: relative; height: 14px; background: {RULE}; border-radius: 7px; margin: 30px 0 8px 0; }}
.gauge .fill {{ position: absolute; left: 0; top: 0; bottom: 0; border-radius: 7px; }}
.gauge .tick {{ position: absolute; top: -8px; bottom: -8px; width: 2px; background: {INK}; }}
.gauge .tlab {{ position: absolute; top: -28px; transform: translateX(-50%); font-size: .78rem; color: {INK}; white-space: nowrap; font-weight: 600; }}
.review {{ background: white; border: 1px solid {RULE}; border-radius: 16px; padding: 16px 20px; line-height: 1.75; font-size: 1.02rem; }}
mark.for {{ background: #DCE8F7; color: {DEEP}; border-radius: 5px; padding: 1px 4px; }}
mark.against {{ background: #F7E1E1; color: {RED}; border-radius: 5px; padding: 1px 4px; }}
.chip {{ display: inline-block; margin: 3px 6px 3px 0; padding: 4px 11px; border-radius: 999px; font-size: .84rem; font-weight: 600; }}
.chip.for {{ background: #DCE8F7; color: {DEEP}; }} .chip.against {{ background: #F7E1E1; color: {RED}; }} .chip.ok {{ background: #E7F1FB; color: {BLUE}; }}
.chip.warn {{ background: #F9ECE7; color: #8A3B12; }} .chip.bad {{ background: #FBEBEB; color: {RED}; }}
.final {{ background: white; border: 2px solid {BLUE}; border-radius: 18px; padding: 20px 24px; font-size: 1.15rem; line-height: 1.6; }}
.draft {{ border-left: 3px solid {RULE}; padding: 4px 0 4px 18px; margin: 0 0 22px 6px; }}
.draft.pass {{ border-left-color: {BLUE}; }}
.draft .txt {{ font-size: 1rem; line-height: 1.6; margin: 4px 0 10px 0; }}
.checks {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 6px; }}
.ck {{ border-radius: 10px; padding: 7px 9px; font-size: .78rem; line-height: 1.3; }}
.ck.pass {{ background: #E7F1FB; color: {DEEP}; }} .ck.fail {{ background: #FBEBEB; color: {RED}; }} .ck.skip {{ background: {TINT}; color: {MUTED}; }}
.ask {{ color: {RED}; font-size: .9rem; margin-top: 8px; }}
.flow {{ display: flex; gap: 10px; align-items: stretch; margin: 6px 0 16px 0; }}
.flow .box {{ flex: 1; background: {TINT}; border-radius: 14px; padding: 12px 14px; font-size: .9rem; color: {BODY}; }}
.flow .box b {{ color: {INK}; display: block; font-size: .98rem; margin-bottom: 2px; }}
.flow .arr {{ align-self: center; color: #8E8E93; font-size: 1.3rem; }}
table.st {{ width: 100%; border-collapse: collapse; font-size: .98rem; }}
table.st th {{ text-align: left; color: {MUTED}; font-size: .74rem; font-weight: 600; letter-spacing: .07em; text-transform: uppercase; padding: 8px 10px; border-bottom: 1px solid {RULE}; }}
table.st td {{ padding: 12px 10px; border-bottom: 1px solid {RULE}; vertical-align: top; }}
.note {{ color: {MUTED}; font-size: .86rem; }}
.fl {{ display: flex; align-items: center; gap: 8px; margin: 8px 0; }}
.fl.p1 {{ opacity: .6; }}
.fcol {{ display: flex; flex-direction: column; gap: 6px; flex: 1; }}
.fn {{ background: {TINT}; border-radius: 12px; padding: 9px 12px; font-size: .86rem; color: {MUTED}; text-align: center; }}
.fn b {{ display: block; color: {INK}; font-size: .92rem; }}
.fn.ag {{ padding: 6px 10px; }} .fn.hi {{ background: {BLUE}; color: #DCE8F7; }} .fn.hi b {{ color: white; }}
.fa {{ color: #8E8E93; font-size: 1.2rem; flex: none; }}
.card-h {{ font-weight: 700; font-size: 1.08rem; margin: 0; }} .card-s {{ color: {MUTED}; font-size: .9rem; margin: 2px 0 0 0; }}
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
    ev, rf, deb, htau, q = core(); L, probe = llama()
    with LOCK:
        p1 = phase1_probs(L, text); pp = probe.probs(text)
    ep, dp = ev.probs(title, text), deb.probs(title, text)
    star, P, shares = rf.combine(pp, np.array(ep), np.array(dp))
    from agentic_sentiment.v3.reflector import feats
    C = rf.full.coef_ * feats(np.array(pp)[None], np.array(ep)[None], np.array(dp)[None])[0]      # 5 stars x 15 features, exact logit terms
    push = {k: C[:, sl].sum(1) for k, sl in (("probe", slice(0, 5)), ("evidence", slice(5, 10)), ("deberta", slice(10, 15)))}
    push = {k: list(map(float, v - v.mean())) for k, v in push.items()}                          # centred per agent: softmax is unchanged
    base = rf.full.intercept_ - rf.full.intercept_.mean()
    lo, hi = star_range(P, q)
    decision = "auto" if max(P) >= htau else ("range" if hi - lo <= 1 else "person")
    pos, neg = ev.phrases(title, text, star)
    agents = {"probe": list(map(float, pp)), "evidence": list(map(float, ep)), "deberta": list(map(float, dp))}
    votes = {k: int(np.argmax(v)) + 1 for k, v in agents.items()}
    return {"p1": list(map(float, p1)), "agents": agents, "star": int(star), "P": list(map(float, P)), "shares": {k: float(v) for k, v in shares.items()},
            "votes": votes, "agreed": len(set(votes.values())) == 1, "decision": decision, "range": (lo, hi), "htau": float(htau), "pos": pos, "neg": neg,
            "push": push, "base": list(map(float, base))}


def cached_rate(rid):
    """Recorded result for a demo review (used only if the live run fails)."""
    i = CACHE["demo_ids"].index(rid); s = CACHE["phase2_states"][i]; P = s["probs"]
    agents = {"probe": s.get("probe_probs") or P, "evidence": s["evidence_probs"], "deberta": s["deberta_probs"]}
    votes = {k: int(np.argmax(v)) + 1 for k, v in agents.items()}
    return {"p1": CACHE["phase1_probs"][i], "agents": agents, "star": s["star"], "P": P, "shares": s["shares"], "votes": votes, "agreed": s["agreed"],
            "decision": "auto" if s["handoff"] == "auto" else "person", "range": (s["star"], s["star"]), "htau": None, "pos": [], "neg": [], "cached": True}


@st.cache_data
def demo_reviews():
    if not CACHE or not os.path.exists(PT.D + "eval.jsonl"): return []
    rows = {}
    for l in open(PT.D + "eval.jsonl"):
        r = json.loads(l)
        if r["review_id"] in CACHE["demo_ids"]: rows[r["review_id"]] = r
    return [rows[i] for i in CACHE["demo_ids"] if i in rows]


# ----------------------------------------------------------------------------------------------- small html pieces
H = lambda s: st.markdown(s, unsafe_allow_html=True)
stars = lambda n, grey=False: f'<span class="stars{" grey" if grey else ""}">{"★" * n}{"☆" * (5 - n)}</span>'
pct = lambda v: f"{v:.0%}"


def step(n, title, sub):
    H(f'<div class="step"><div class="num">{n}</div><div><h3>{title}</h3><p>{sub}</p></div></div>')


def tile(label, big, sub="", main=False, blue=False):
    return f'<div class="tile{" main" if main else ""}"><div class="lab">{label}</div><div class="big{" blue" if blue else ""}">{big}</div><div class="sub">{sub}</div></div>'


def hist(p, color=BLUE):
    top = int(np.argmax(p))
    return '<div class="hist">' + "".join(
        f'<div class="hc"><div class="hv">{pct(v)}</div><div class="hb" style="height:{max(3, v * 62):.0f}px;background:{color if i == top else GREY}"></div>'
        f'<div class="hl">{i + 1}★</div></div>' for i, v in enumerate(p)) + "</div>"


def highlight(title, text, pos, neg):
    out, t = escape(text), escape(title or "")
    for cls, items in (("for", pos), ("against", neg)):
        for p in sorted({x["display"] for x in items}, key=len, reverse=True):
            pat = re.compile(re.escape(escape(p)), re.I)
            out = pat.sub(lambda m: f'<mark class="{cls}">{m.group(0)}</mark>', out, count=1)
            t = pat.sub(lambda m: f'<mark class="{cls}">{m.group(0)}</mark>', t, count=1)
    return f'<div class="review">{f"<b>{t}</b><br>" if t else ""}{out}</div>'


def style(fig, h=300, title=None):
    fig.update_layout(height=h, margin=dict(l=8, r=8, t=8, b=8), plot_bgcolor="white", paper_bgcolor="white", bargap=0.35,
                      font=dict(family="-apple-system, Helvetica Neue, Arial", size=13, color=INK), showlegend=False)
    fig.update_xaxes(showgrid=False, linecolor=RULE); fig.update_yaxes(showgrid=False, visible=False)
    return fig


def chart(fig):
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


# ----------------------------------------------------------------------------------------------- header + input
H('<h1 class="app">Agentic review rating</h1><p class="lede">Three agents rate the star. LLaMA-3 explains it. A Critic checks it. '
  'Everything runs live on this laptop.</p>')

D = demo_reviews()
with st.container(border=True):
    a, b = st.columns([3, 1.1], gap="large")
    with b:
        opts = ["Write my own"] + [f"Demo {i + 1} · true {r['gt_rating']}★" for i, r in enumerate(D)]
        pick = st.selectbox("Start from", opts, index=1 if D else 0)
        base = D[opts.index(pick) - 1] if pick != "Write my own" else {"title": "", "review_text": "", "gt_rating": None, "review_id": None}
        go_btn = st.button("Rate this review", type="primary", width="stretch")
        H('<p class="note">First run loads the models (about 40 s).</p>')
    with a:
        title = st.text_input("Title", base["title"], key=f"t_{pick}", placeholder="e.g. Stopped working after a week")
        text = st.text_area("Review", base["review_text"], height=110, key=f"x_{pick}", placeholder="Paste or type a product review…")

if go_btn and text.strip():
    try:
        with st.spinner("Phase 1 and the three agents are reading the review…"):
            st.session_state.res = rate(title, text)
    except Exception as ex:
        if base.get("review_id") in CACHE.get("demo_ids", []) and title == base["title"] and text == base["review_text"]:
            st.session_state.res = cached_rate(base["review_id"]); st.warning(f"Live run failed ({type(ex).__name__}); showing the recorded result for this demo review.")
        else:
            st.error(f"Live run failed: {type(ex).__name__}: {ex}"); st.stop()
    st.session_state.inp = {"title": title, "text": text, "truth": base.get("gt_rating") if text == base["review_text"] else None, "rid": base.get("review_id")}
    st.session_state.pop("expl", None); st.session_state.pop("stress", None)

tabs = st.tabs(["How it rates", "How it explains", "Stress test", "Results"])
res, inp = st.session_state.get("res"), st.session_state.get("inp")

# ----------------------------------------------------------------------------------------------- tab 1: how it rates
with tabs[0]:
    if not res:
        st.info("Pick a demo review or write your own above, then press **Rate this review**.")
    else:
        p1s, truth, P, star = int(np.argmax(res["p1"])) + 1, inp["truth"], res["P"], res["star"]
        verdict = lambda s: ("" if not truth else ("✓ matches the true star" if s == truth else f"off by {abs(s - truth)} star{'s' if abs(s - truth) > 1 else ''}"))
        c = st.columns(3, gap="medium")
        with c[0]: H(tile("Phase 2 · our system", f"{star}★", f"{stars(star)}<br>{pct(max(P))} confident · {verdict(star)}", main=True, blue=True))
        with c[1]: H(tile("Phase 1 · reference method", f"{p1s}★", f"{stars(p1s, True)}<br>{pct(max(res['p1']))} confident · {verdict(p1s)}"))
        with c[2]: H(tile("True star", f"{truth}★" if truth else "—", "what the reviewer gave" if truth else "not known for your own review"))
        if res.get("cached"): st.caption("Recorded result (live run unavailable).")

        dname = {"auto": "answers alone", "range": f"range {res['range'][0]}–{res['range'][1]}★", "person": "to a person"}[res["decision"]]
        ag = "".join(f'<div class="fn ag"><b>{AGENTS[k][0]}</b>{res["votes"][k]}★ · {pct(max(p))}</div>' for k, p in res["agents"].items())
        H('<div class="lab" style="margin-top:22px">How the data flows</div><div class="fl">'
          f'<div class="fcol"><div class="fn"><b>Review</b>title + text</div></div><div class="fa">→</div>'
          f'<div class="fcol">{ag}</div><div class="fa">→</div>'
          f'<div class="fcol"><div class="fn hi"><b>Reflector</b>{star}★ · {pct(max(P))}</div></div><div class="fa">→</div>'
          f'<div class="fcol"><div class="fn"><b>Hand-off</b>{dname}</div></div><div class="fa">→</div>'
          f'<div class="fcol"><div class="fn hi"><b>Answer</b>{star}★</div></div></div>'
          '<div class="fl p1"><div class="fcol"><div class="fn"><b>Review</b>text only</div></div><div class="fa">→</div>'
          f'<div class="fcol"><div class="fn"><b>Phase 1</b>LLaMA-3 + LoRA</div></div><div class="fa">→</div>'
          f'<div class="fcol"><div class="fn"><b>Answer</b>{p1s}★</div></div></div>'
          '<p class="note">Top: Phase 2, our system. Bottom: Phase 1, the reference method. The steps below open up each box.</p>')

        step(1, "Find the evidence", "The Evidence model marks the words that push toward the final star and the words that push against it.")
        H(highlight(inp["title"], inp["text"], res["pos"], res["neg"]))
        H('<p class="note" style="margin-top:6px"><mark class="for">blue</mark> supports the star &nbsp; <mark class="against">red</mark> argues against it</p>')

        step(2, "Three agents vote, independently", "Each agent reads the review its own way and gives a probability for every star. None sees the others.")
        cols = st.columns(3, gap="medium")
        for col, (k, p) in zip(cols, res["agents"].items()):
            name, what = AGENTS[k]; v = res["votes"][k]
            with col: H(f'<div class="agent"><div class="name">{name}</div><div class="what">{what}</div>'
                        f'<div class="vote">{v}★ <small>{pct(max(p))} sure</small></div>{hist(p, SKY)}</div>')
        vs = " · ".join(f"{AGENTS[k][0]} {v}★" for k, v in res["votes"].items())
        H(f'<p style="margin-top:12px">{"<span class=\'chip ok\'>All three agree</span>" if res["agreed"] else "<span class=\'chip warn\'>The agents disagree</span>"} '
          f'<span class="note">{vs}</span></p>')

        step(3, "The Reflector combines the votes", "A logistic-regression model, trained on agent votes for reviews none of them trained on, learned how far to trust each agent "
             "for each star. Each agent adds a push toward or away from every star; the pushes add up to a score, and the scores become the final probabilities.")
        sh = {k: v for k, v in res["shares"].items() if k in AGENTS}; lead = max(sh, key=sh.get)
        if res.get("push"):
            fig = go.Figure(); cats = [f"{i}★" for i in range(1, 6)]
            for (k, v), c in zip(res["push"].items(), (SKY, BLUE, DEEP)):
                fig.add_bar(x=cats, y=v, name=AGENTS[k][0], marker_color=c, hovertemplate="%{x}: %{y:+.2f}<extra>" + AGENTS[k][0] + "</extra>")
            tot = np.sum([res["push"][k] for k in res["push"]], 0) + np.array(res["base"])
            fig.add_scatter(x=cats, y=tot, mode="markers+text", name="Total score", marker=dict(symbol="line-ew-open", size=34, line=dict(width=3, color=INK)),
                            text=[f"{v:+.1f}" for v in tot], textposition="top center", hoverinfo="skip")
            style(fig, 300).update_layout(barmode="relative", showlegend=True, legend=dict(orientation="h", y=1.14, x=0))
            fig.update_yaxes(visible=True, zeroline=True, zerolinecolor=MUTED, title="push (log-odds)", gridcolor=TINT)
            a, b = st.columns([1.5, 1], gap="large")
            with a:
                H('<div class="lab">How each agent pushes each star</div>'); chart(fig)
            with b:
                H(f'<div class="lab">Final probability per star</div>{hist(P, BLUE)}')
                ranked = sorted(res["push"], key=lambda k: -res["push"][k][star - 1])
                H(f'<p class="sub" style="margin-top:14px"><b>Why {star}★:</b> ' + "; ".join(
                    f'{AGENTS[k][0]} {res["push"][k][star - 1]:+.1f}' for k in ranked) + f' toward {star}★. '
                  f'Overall it leaned most on <b>{AGENTS[lead][0]}</b> ({pct(sh[lead])} of the weight). '
                  f'{"All three agreed, so combining mostly raises confidence." if res["agreed"] else "They disagreed, so these pushes settle the star."}</p>')
            H('<p class="note">Bars above zero push toward a star, bars below push away. The black line is the total score; the highest score wins. '
              'An agent pushes hard when it is sure and the Reflector learned it is reliable for that star.</p>')
        else:
            H(f'<div class="lab">Final probability per star</div>{hist(P, BLUE)}')
        H(f'<p style="margin-top:6px">Result: <b>{star}★</b> with {pct(max(P))} confidence.</p>')

        step(4, "The Hand-off agent decides who answers", "If the system is confident enough it answers on its own. If not, it gives a narrow star range or passes the review to a person.")
        conf, tau = max(P), res["htau"]
        dec = {"auto": ("ok", "Answers automatically", f"Confidence {pct(conf)} is above the bar of {pct(tau) if tau else '—'}."),
               "range": ("warn", f"Gives a range: {res['range'][0]}–{res['range'][1]}★", f"Confidence {pct(conf)} is below the bar, but the likely stars are close together. Ranges like this hold the true star about 90% of the time."),
               "person": ("bad", "Hands it to a person", f"Confidence {pct(conf)} is below the bar and the likely stars are far apart.")}[res["decision"]]
        if tau:
            H(f'<div class="gauge"><div class="fill" style="width:{conf * 100:.1f}%;background:{BLUE if res["decision"] == "auto" else GREY}"></div>'
              f'<div class="tick" style="left:{tau * 100:.1f}%"></div><div class="tlab" style="left:{tau * 100:.1f}%">bar {pct(tau)}</div></div>')
        H(f'<p><span class="chip {dec[0]}">{dec[1]}</span> <span class="sub">{dec[2]}</span></p>')
        H(f'<p class="note">The bar was fixed on development data before testing. On the natural test the system answers {AUTO["coverage"]:.0%} of reviews alone and is right on {AUTO["acc_auto"]:.1%} of those.</p>')

        step(5, "Compare with the reference method", "Phase 1 is one LLaMA-3 fine-tuned with LoRA on 2,000 reviews. It reads only the text and answers in one step.")
        fig = go.Figure()
        cats = [f"{i}★" for i in range(1, 6)]
        fig.add_bar(x=cats, y=res["p1"], name="Phase 1", marker_color=GREY, text=[pct(v) for v in res["p1"]], textposition="outside")
        fig.add_bar(x=cats, y=P, name="Phase 2", marker_color=BLUE, text=[pct(v) for v in P], textposition="outside")
        style(fig, 250).update_layout(showlegend=True, legend=dict(orientation="h", y=1.15, x=0), yaxis=dict(range=[0, 1.2]))
        chart(fig)

# ----------------------------------------------------------------------------------------------- tab 2: how it explains
with tabs[1]:
    if not res:
        st.info("Rate a review first.")
    else:
        H('<div class="flow"><div class="box"><b>1 · Evidence</b>The star, its confidence and the strongest phrases for and against it.</div><div class="arr">→</div>'
          '<div class="box"><b>2 · Explainer</b>LLaMA-3 writes one or two plain sentences from that evidence only.</div><div class="arr">→</div>'
          '<div class="box"><b>3 · Critic</b>Runs five checks. If one fails, it says exactly what to fix and sends the draft back (up to four rewrites).</div></div>')
        run = st.button("Write and check the explanation", type="primary")
        if run:
            try:
                from agentic_sentiment.v3.graph import build_graph_v3
                ev, rf, deb, htau, q = core(); chat = instruct()
                g = build_graph_v3(ev, rf, None, explain_fn=chat, judge_fn=chat, deberta=deb)
                state = {"review_id": inp["rid"] or "live", "title": inp["title"], "review_text": inp["text"],
                         "probe_probs": res["agents"]["probe"], "deberta_probs": res["agents"]["deberta"]}
                out_s, t0 = {"pack": None, "drafts": [], "feedback": [], "fails": []}, time.time()
                with LOCK, st.status("Collecting the evidence…", expanded=False) as status:
                    for upd in g.stream(state, stream_mode="updates"):
                        node, out = next(iter(upd.items()))
                        if node == "pack": out_s["pack"] = out["pack"]; status.update(label="Explainer writing draft 1…")
                        if node == "explainer": out_s["drafts"] = out["drafts"]; status.update(label=f"Critic checking draft {len(out['drafts'])}…")
                        if node == "critic":
                            out_s["feedback"], out_s["fails"] = out["feedback"], out["failed_checks"]
                            if out["feedback"][-1]: status.update(label=f"Critic sent it back. Explainer writing draft {len(out['feedback']) + 1}…")
                    status.update(label=f"Done in {time.time() - t0:.0f} s", state="complete")
                st.session_state.expl = {**out_s, "cached": False}
            except Exception as ex:
                i = CACHE.get("demo_ids", []).index(inp["rid"]) if inp["rid"] in CACHE.get("demo_ids", []) else None
                rec = CACHE.get("explanations", [None] * 99)[i] if i is not None else None
                if rec:
                    st.warning(f"Live run failed ({type(ex).__name__}); showing the recorded run for this demo review.")
                    st.session_state.expl = {"pack": None, "drafts": rec["drafts"], "feedback": rec["feedback"], "fails": [], "cached": True}
                else:
                    st.error(f"Live explanation failed: {type(ex).__name__}: {ex}")
        e = st.session_state.get("expl")
        if e and e["drafts"]:
            pk = e["pack"] or {"for": res["pos"], "against": res["neg"]}
            ok = not e["feedback"][-1]; n = len(e["drafts"])
            mark = lambda s: highlight("", s, pk["for"], pk["against"]).replace('<div class="review">', "").replace("</div>", "")
            H(f'<div class="lab" style="margin-top:14px">Final explanation · {"passed all five checks" if ok else "best draft after four rewrites"} · '
              f'{n} draft{"s" if n > 1 else ""}</div><div class="final">{mark(e["drafts"][-1])}</div>')

            step(1, "What the Explainer was given", "It may only use this. The Critic checks the draft against it.")
            H(f'<p><b>{res["star"]}★</b> <span class="sub">at {pct(max(res["P"]))} confidence · '
              f'{"agents agreed" if res["agreed"] else "agents disagreed, so the explanation must say so"}</span></p>'
              + "".join(f'<span class="chip for">{escape(p["display"])}</span>' for p in pk["for"])
              + "".join(f'<span class="chip against">{escape(p["display"])}</span>' for p in pk["against"]))

            step(2, "Draft by draft", "Each draft and how it scored on the five checks. A red check comes with the Critic's instruction for the next draft.")
            for i, (d, f) in enumerate(zip(e["drafts"], e["feedback"])):
                failed = set(e["fails"][i]) if i < len(e["fails"]) else None
                cks = ""
                if failed is not None:
                    judged = not (failed - {5})          # check 5 only runs when the four rule checks pass
                    cks = '<div class="checks">' + "".join(
                        f'<div class="ck {"fail" if k in failed else ("skip" if k == 5 and not judged else "pass")}" title="{escape(CHECKS[k][1])}">'
                        f'{"✗" if k in failed else ("–" if k == 5 and not judged else "✓")} <b>{CHECKS[k][0]}</b><br>{CHECKS[k][1]}</div>' for k in CHECKS) + "</div>"
                ask = "".join(f'<div class="ask">Critic: {escape(x)}</div>' for x in f)
                H(f'<div class="draft{" pass" if not f else ""}"><div class="lab">Draft {i + 1}{" · passed" if not f else ""}</div>'
                  f'<div class="txt">{mark(d)}</div>{cks}{ask}</div>')
            H('<p class="note">Check 5 (grey dash) runs only after checks 1–4 pass. On all 3,000 test reviews this loop raises the Critic pass rate from '
              '62–64% to 90.5–91.6%; a blind human check of 50 explanations rated 80% faithful.</p>')

# ----------------------------------------------------------------------------------------------- tab 3: stress test
with tabs[2]:
    if not res:
        st.info("Rate a review first.")
    else:
        from agentic_sentiment.v3.perturb import flat, typos
        H('<p class="sub" style="font-size:1rem">Real reviews are messy. Here the same review is changed in three ways and rated again. '
          'A good system keeps the same star.</p>')
        V = {"Original": ("as written", inp["title"], inp["text"]),
             "Typos": ("one word in ten misspelled", inp["title"], typos(inp["text"], seed=0)),
             "No punctuation or capitals": ("all lower case, punctuation removed", flat(inp["title"]), flat(inp["text"])),
             "No title": ("title deleted, text unchanged", "", inp["text"])}
        if st.button("Rate all four versions", type="primary"):
            try:
                with st.spinner("Rating four versions…"):
                    st.session_state.stress = {k: (int(np.argmax(r["p1"])) + 1, r["star"], max(r["P"])) for k, (_, t, x) in V.items() for r in [rate(t, x)]}
            except Exception as ex:
                st.error(f"Live stress test failed: {type(ex).__name__}: {ex}")
        s = st.session_state.get("stress")
        if s:
            o1, o2, _ = s["Original"]
            badge = lambda new, old, k: "" if k == "Original" else ('<span class="chip ok">same</span>' if new == old else '<span class="chip bad">changed</span>')
            rows = "".join(f'<tr><td><b>{k}</b><br><span class="note">{V[k][0]}</span></td><td>{a}★ {badge(a, o1, k)}</td>'
                           f'<td><b>{b}★</b> {badge(b, o2, k)} <span class="note">{pct(conf)} sure</span></td></tr>' for k, (a, b, conf) in s.items())
            H(f'<table class="st"><tr><th>Version</th><th>Phase 1</th><th>Phase 2</th></tr>{rows}</table>')
            with st.expander("Show the altered text"):
                for k, (_, t, x) in V.items(): H(f'<p><b>{k}</b><br><span class="note">{escape(t) or "(no title)"} · {escape(x)}</span></p>')
        H(f'<p class="note" style="margin-top:12px">On 300 test reviews: typos changed {ROB["typos"]["SH_flip"]:.1%} of Phase 2 stars vs {ROB["typos"]["P1_flip"]:.1%} for Phase 1; '
          f'flat text {ROB["flat"]["SH_flip"]:.1%} vs {ROB["flat"]["P1_flip"]:.1%}. Removing the title changed {ROB["no_title"]["SH_flip"]:.1%} of Phase 2 stars, '
          'because Phase 2 reads the title and Phase 1 does not.</p>')

# ----------------------------------------------------------------------------------------------- tab 4: results (frozen)
def frozen_table(split):
    md = open(PT.D + "results_frozen.md").read(); sec = md.split(f"## {split}")[1].split("## ")[0]; rows = {}
    for l in sec.splitlines():
        m = re.match(r"\| ([^|]+) \| ([\d.]+) \[[^\]]*\] \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \|", l)
        if m: rows[m.group(1).strip()] = tuple(float(v) for v in m.group(2, 3, 4, 5))
    return rows


def pair(title, sub, labels, p1, p2, fmt, ymax):
    """One card: Phase 1 vs Phase 2 bars, side by side per group."""
    H(f'<p class="card-h">{title}</p><p class="card-s">{sub}</p>')
    fig = go.Figure()
    fig.add_bar(x=labels, y=p1, name="Phase 1", marker_color=GREY, text=[fmt(v) for v in p1], textposition="outside")
    fig.add_bar(x=labels, y=p2, name="Phase 2", marker_color=BLUE, text=[fmt(v) for v in p2], textposition="outside")
    chart(style(fig, 240).update_layout(showlegend=True, legend=dict(orientation="h", y=1.18, x=0), yaxis=dict(range=[0, ymax])))


with tabs[3]:
    B, S, Q = json.load(open(PT.D + "b2_report.json")), SEL, ROB
    HC = json.load(open(PT.D + "hard_cases.json")); hc = open(PT.D + "human_check.out").read()
    E, N = frozen_table("eval"), frozen_table("natural_test")
    main = next(k for k in E if "(main)" in k); p1 = next(k for k in E if k.startswith("Phase 1"))
    auto = AUTO
    hum = float(re.search(r"human faithful ([\d.]+)", hc).group(1))

    H('<p class="sub" style="font-size:1rem">Frozen results on two test sets the system never trained on: <b>balanced</b> (1,000 reviews, 200 per star) '
      'and <b>natural</b> (2,000 reviews in their real mix, 65% five-star).</p>')
    k = st.columns(4, gap="medium")
    with k[0]: H(tile("Exact star · balanced", f"{E[main][0]:.1f}%", f"vs {E[p1][0]:.1f}% for Phase 1 (+{E[main][0] - E[p1][0]:.1f} points)", blue=True))
    with k[1]: H(tile("Macro-F1 · natural", f"{N[main][1]:.3f}".lstrip("0"), f"vs {N[p1][1]:.3f} for Phase 1; fair to every star".replace("0.", "."), blue=True))
    with k[2]: H(tile("Right when it answers alone", f"{auto['acc_auto']:.0%}", f"on the {auto['coverage']:.0%} of natural reviews it is sure about", blue=True))
    with k[3]: H(tile("Explanations that pass", f"{B['natural_test']['faithful_final']:.0%}", f"Critic pass rate, from {B['natural_test']['faithful_first']:.0%} at first draft; human check {hum:.0%}", blue=True))

    st.write("")
    a, b = st.columns(2, gap="large")
    with a, st.container(border=True):
        pair("More stars exactly right", "Exact-star accuracy (%).", ["Balanced test", "Natural test"], [E[p1][0], N[p1][0]], [E[main][0], N[main][0]], lambda v: f"{v:.1f}", 100)
    with b, st.container(border=True):
        pair("Fairer to every star", "Macro-F1 averages all five stars equally, so the common 5★ cannot hide mistakes.", ["Balanced test", "Natural test"],
             [E[p1][1], N[p1][1]], [E[main][1], N[main][1]], lambda v: f"{v:.2f}", 1)
    a, b = st.columns(2, gap="large")
    with a, st.container(border=True):
        H('<p class="card-h">Knows when it is unsure</p><p class="card-s">Answer only the reviews you are most sure of: accuracy on those (natural test).</p>')
        fig = go.Figure()
        for name, key, c in (("Phase 1", "Phase1_LoRA", GREY), ("Phase 2", "SH", BLUE)):
            pts = S[key]["natural_test"]
            fig.add_scatter(x=[p["coverage"] * 100 for p in pts], y=[p["acc_auto"] * 100 for p in pts], name=name, mode="lines", line=dict(color=c, width=4))
        fig.add_scatter(x=[auto["coverage"] * 100], y=[auto["acc_auto"] * 100], mode="markers+text", marker=dict(size=11, color=BLUE),
                        text=[f"  {auto['acc_auto']:.0%} right on {auto['coverage']:.0%}"], textposition="middle right", showlegend=False)
        style(fig, 240).update_layout(showlegend=True, legend=dict(orientation="h", y=1.18, x=0))
        fig.update_xaxes(title="Reviews answered automatically (%)", range=[0, 105]); fig.update_yaxes(visible=True, title="Right on those (%)", gridcolor=TINT)
        chart(fig)
    with b, st.container(border=True):
        rob = {"Typos": "typos", "No punctuation": "flat", "No title": "no_title"}
        pair("Steadier on messy text", "Stars that changed when the review was altered (%, lower is better). Phase 1 never reads the title.", list(rob),
             [Q[v]["P1_flip"] * 100 for v in rob.values()], [Q[v]["SH_flip"] * 100 for v in rob.values()], lambda v: f"{v:.1f}", 36)
    a, b = st.columns(2, gap="large")
    with a, st.container(border=True):
        H('<p class="card-h">The Critic loop repairs explanations</p><p class="card-s">Share of explanations that pass all five checks (%).</p>')
        fig = go.Figure()
        L = ["Balanced test", "Natural test"]
        fig.add_bar(x=L, y=[B[s]["faithful_first"] * 100 for s in ("eval", "natural_test")], marker_color=GREY, name="First draft",
                    text=[f"{B[s]['faithful_first'] * 100:.1f}" for s in ("eval", "natural_test")], textposition="outside")
        fig.add_bar(x=L, y=[B[s]["faithful_final"] * 100 for s in ("eval", "natural_test")], marker_color=BLUE, name="After the loop",
                    text=[f"{B[s]['faithful_final'] * 100:.1f}" for s in ("eval", "natural_test")], textposition="outside")
        chart(style(fig, 240).update_layout(showlegend=True, legend=dict(orientation="h", y=1.18, x=0), yaxis=dict(range=[0, 108])))
    with b, st.container(border=True):
        hcs = {s: HC[s]["surface_negative_positives"] for s in ("eval", "natural_test")}
        pair("Not fooled by negative words", f"Happy reviews that use negative words: exact star (%, n = {hcs['eval']['n']} / {hcs['natural_test']['n']}).",
             ["Balanced test", "Natural test"], [hcs[s]["p1_exact"] * 100 for s in hcs], [hcs[s]["sh_exact"] * 100 for s in hcs], lambda v: f"{v:.1f}", 100)

    H('<p class="note">Read with care: on the natural test 65% of reviews are 5★, so raw exact accuracy flatters Phase 2. Against Phase 1 adjusted to that mix the '
      'gain is +2.5 points, which is why macro-F1 is the headline there. The two-agent system fixed before any test read scores 67.5% / 75.7%.</p>')
    with st.expander("Every model on the test sets"):
        names = list(E.keys())
        rows = "".join(f'<tr><td>{"<b>" + escape(n) + "</b>" if n == main else escape(n)}</td><td>{E[n][0]:.1f}</td><td>{E[n][1]:.3f}</td>'
                       f'<td>{N[n][0]:.1f}</td><td>{N[n][1]:.3f}</td></tr>' for n in names if n in N)
        H(f'<table class="st"><tr><th>Model</th><th>Balanced exact %</th><th>Balanced macro-F1</th><th>Natural exact %</th><th>Natural macro-F1</th></tr>{rows}</table>')
    st.caption("Sources: results_frozen.md, s1_selective.json, b2_report.json, q4_robustness.json, hard_cases.json, human_check.out (v2_reflective/e23/).")
