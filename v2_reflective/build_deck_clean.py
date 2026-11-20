"""The one current deck (clean version, 2026-11-20): one message per slide, big numbers, native charts, same blue palette.
Replaces build_deck_e23.py (archived in docs/archive/). Every number is copied from v2_reflective/LOG.md / e23/*.out;
the source of each slide is in its speaker note. Writes presentation/deck_slide_map.md and syncs the Drive copy.
Run: /opt/homebrew/bin/python3 v2_reflective/build_deck_clean.py   (system python has python-pptx; .venv does not)"""
import os, json, shutil
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION, XL_LABEL_POSITION

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUTER = os.path.dirname(ROOT)
DST = os.path.join(OUTER, "Agentic_LLaMA3_Presentation.pptx")
FIG = os.path.join(ROOT, "paper", "figures", "e23")
FONT = "Arial"
C = lambda h: RGBColor.from_string(h)
INK, BODY, MUTED = C("1A2332"), C("3D4C5C"), C("8792A0")
BLUE, DEEP, TINT, SURF, RULE = C("2C5AA0"), C("1B3C6E"), C("ECF2F9"), C("F7F9FB"), C("D8E0E8")
GREY, POS, NEG, WHITE = C("AAB4C0"), C("1E7A46"), C("A93636"), C("FFFFFF")
W, H, M = 13.333, 7.5, 0.7

prs = Presentation(); prs.slide_width, prs.slide_height = Inches(W), Inches(H)
LAY = prs.slide_layouts[6]
SLIDES, NOTES = [], {}


def text(sl, x, y, w, h, s, size=16, color=BODY, bold=False, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, line=1.1):
    tb = sl.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)); tf = tb.text_frame
    tf.word_wrap = True; tf.vertical_anchor = anchor
    for a in ("margin_left", "margin_right", "margin_top", "margin_bottom"): setattr(tf, a, 0)
    for i, part in enumerate(s if isinstance(s, list) else [s]):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        runs = part if isinstance(part, tuple) else ((part, {}),) if isinstance(part, str) else part
        if isinstance(part, tuple) and isinstance(part[0], str): runs = (part,)
        p.alignment = align; p.line_spacing = line
        if i: p.space_before = Pt(size * 0.5)
        for t, o in runs:
            r = p.add_run(); r.text = t; f = r.font; f.name = FONT
            f.size = Pt(o.get("size", size)); f.bold = o.get("bold", bold); f.color.rgb = o.get("color", color)
    return tb


def rect(sl, x, y, w, h, fill, shape=MSO_SHAPE.RECTANGLE, line=None):
    s = sl.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid(); s.fill.fore_color.rgb = fill; s.shadow.inherit = False
    if line: s.line.color.rgb = line; s.line.width = Pt(1)
    else: s.line.fill.background()
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE: s.adjustments[0] = 0.08
    return s


def slide(key, kicker, title, note, sub=None):
    sl = prs.slides.add_slide(LAY); SLIDES.append((key, title, sl)); NOTES[key] = note
    rect(sl, 0, 0, W, 0.08, BLUE)
    text(sl, M, 0.45, W - 2 * M, 0.3, kicker.upper(), size=11, color=BLUE, bold=True)
    text(sl, M, 0.75, W - 2 * M, 0.8, title, size=30, color=INK, bold=True)
    if sub: text(sl, M, 1.5, W - 2 * M, 0.4, sub, size=15, color=MUTED)
    return sl


def divider(key, num, title, line, note):
    sl = prs.slides.add_slide(LAY); SLIDES.append((key, title, sl)); NOTES[key] = note
    rect(sl, 0, 0, W, H, DEEP); rect(sl, M, 3.05, 0.9, 0.06, C("6F9BD8"))
    text(sl, M, 1.9, 6, 1.0, num, size=60, color=C("6F9BD8"), bold=True)
    text(sl, M, 3.35, W - 2 * M, 1.0, title, size=40, color=WHITE, bold=True)
    text(sl, M, 4.35, W - 2 * M, 0.8, line, size=18, color=C("C9D7EA"))
    return sl


def stat(sl, x, y, w, big, label, color=BLUE, size=54, lsize=15):
    text(sl, x, y, w, 0.95, big, size=size, color=color, bold=True)
    text(sl, x, y + size / 72 * 1.25, w, 1.2, label, size=lsize, color=BODY)


def card(sl, x, y, w, h, big, label, color=BLUE, size=44):
    rect(sl, x, y, w, h, SURF, MSO_SHAPE.ROUNDED_RECTANGLE, RULE); rect(sl, x, y + 0.25, 0.07, h - 0.5, color)
    stat(sl, x + 0.35, y + 0.3, w - 0.6, big, label, color, size)


def takeaway(sl, s, y=6.3):
    rect(sl, M, y, W - 2 * M, 0.6, TINT); rect(sl, M, y, 0.07, 0.6, BLUE)
    text(sl, M + 0.3, y, W - 2 * M - 0.5, 0.6, s, size=15, color=DEEP, bold=True, anchor=MSO_ANCHOR.MIDDLE)


def bars(sl, x, y, w, h, cats, series, colors, pct=True, maxv=100, legend=True, fmt='0.0"%"', horizontal=False):
    cd = CategoryChartData(); cd.categories = cats
    for n, v in series: cd.add_series(n, v)
    kind = XL_CHART_TYPE.BAR_CLUSTERED if horizontal else XL_CHART_TYPE.COLUMN_CLUSTERED
    ch = sl.shapes.add_chart(kind, Inches(x), Inches(y), Inches(w), Inches(h), cd).chart
    ch.font.name = FONT; ch.font.size = Pt(13); ch.font.color.rgb = BODY
    va, ca = ch.value_axis, ch.category_axis
    va.maximum_scale, va.minimum_scale = maxv, 0; va.has_major_gridlines = False; va.visible = False
    ca.format.line.color.rgb = RULE; ca.tick_labels.font.size = Pt(14); ca.has_major_gridlines = False
    if horizontal: ca.reverse_order = True
    pl = ch.plots[0]; pl.gap_width = 60; pl.overlap = -10
    pl.has_data_labels = True; dl = pl.data_labels; dl.number_format = fmt; dl.number_format_is_linked = False
    dl.font.size = Pt(15); dl.font.bold = True; dl.position = XL_LABEL_POSITION.OUTSIDE_END
    for s, c in zip(pl.series, colors): s.format.fill.solid(); s.format.fill.fore_color.rgb = c
    ch.has_legend = legend and len(series) > 1
    if ch.has_legend:
        ch.legend.position = XL_LEGEND_POSITION.TOP; ch.legend.include_in_layout = False; ch.legend.font.size = Pt(13)
    return ch


def pic(sl, f, x, y, w=None, h=None):
    return sl.shapes.add_picture(os.path.join(FIG, f), Inches(x), Inches(y), Inches(w) if w else None, Inches(h) if h else None)


def box(sl, x, y, w, h, title, sub, fill=WHITE, tc=INK, line=BLUE):
    s = rect(sl, x, y, w, h, fill, MSO_SHAPE.ROUNDED_RECTANGLE, line)
    text(sl, x + 0.1, y + 0.12, w - 0.2, 0.4, title, size=15, color=tc, bold=True, align=PP_ALIGN.CENTER)
    if sub: text(sl, x + 0.1, y + 0.5, w - 0.2, h - 0.55, sub, size=11.5, color=tc if fill != WHITE else BODY, align=PP_ALIGN.CENTER)
    return s


def arrow(sl, x1, y1, x2, y2, color=GREY):
    c = sl.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = color; c.line.width = Pt(2)
    ln = c.line._get_or_add_ln(); from pptx.oxml.ns import qn
    ln.append(ln.makeelement(qn("a:tailEnd"), {"type": "triangle", "w": "med", "len": "med"}))


# 1 ---------------------------------------------------------------- title
sl = prs.slides.add_slide(LAY); SLIDES.append(("title", "Agentic Self-Correction in LLaMA-3", sl))
rect(sl, 0, 0, 4.6, H, DEEP); rect(sl, 4.6, 0, 0.08, H, BLUE)
text(sl, 5.3, 2.2, 7.4, 1.8, "Agentic Self-Correction in LLaMA-3", size=40, color=INK, bold=True)
text(sl, 5.3, 3.75, 7.4, 1.0, "Several agents rate, explain and check e-commerce product reviews", size=20, color=BODY)
rect(sl, 5.3, 4.95, 0.9, 0.05, BLUE)
text(sl, 5.3, 5.2, 7.4, 0.9, ["Varun K  ·  PES University, M.Tech Project", "Amazon Reviews 2023, Electronics  ·  November 2026"], size=14, color=MUTED)
NOTES["title"] = ("Title. Say: we take one LLM review-rating method from a published paper, rebuild it, find where it falls short, and replace it "
                  "with several agents that rate, explain and check.")

# 2 ---------------------------------------------------------------- problem
sl = slide("problem", "The problem", "One model, one pass, nothing checks it",
           "Problem. Today a single LLM reads a review and gives a star and a justification. Three things go wrong: no second opinion, the explanation cannot be checked, and hard language (mixed or negative-sounding but positive reviews) trips it up. Keep it to one minute.")
for i, (n, t, s) in enumerate([("1", "No second opinion", "One model decides. A mistake has nothing to catch it."),
                                ("2", "Unchecked explanations", "The model writes a justification. Nothing tests whether it is true."),
                                ("3", "Hard language", "Mixed reviews, and reviews that sound negative but are happy, confuse it.")]):
    x = M + i * 4.05
    rect(sl, x, 2.4, 0.8, 0.8, BLUE, MSO_SHAPE.OVAL)
    text(sl, x, 2.4, 0.8, 0.8, n, size=26, color=WHITE, bold=True, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    text(sl, x, 3.5, 3.6, 0.5, t, size=22, color=INK, bold=True)
    text(sl, x, 4.15, 3.6, 1.5, s, size=16)
takeaway(sl, "Goal: a rating system whose answer is more accurate and whose explanation can be checked.")

# 3 ---------------------------------------------------------------- Phase 1 divider
divider("d1", "01", "Phase 1 · The reference paper", "We chose one published method, rebuilt it on a clean benchmark, and explored where it breaks.",
        "Section 1. Say: we chose Wang et al. (2025) as our reference paper. Phase 1 is our reproduction of it and our exploration of it.")

# 4 ---------------------------------------------------------------- the paper
sl = slide("paper", "Phase 1 · The reference paper", "One fine-tuned LLaMA rates each review",
           "The paper we chose. Walk the pipeline left to right: clean the data with TextBlob, balance the stars, fine-tune LLaMA-3-8B with a small LoRA add-on, then one prompt gives a star and a justification. The paper reports 92.6% on its own data and protocol. Source: Wang et al. (2025), ITM Web of Conferences 70, 04021.",
           sub="Wang et al. (2025), “Sentiment Analysis of Product Reviews Using Fine-Tuned LLaMA-3”")
steps = [("Clean", "TextBlob filter"), ("Balance", "equal reviews per star"), ("Fine-tune", "LLaMA-3-8B + LoRA"), ("Prompt", "one example, step by step"), ("Output", "star + justification")]
for i, (t, s) in enumerate(steps):
    x = M + i * 2.42
    box(sl, x, 2.6, 2.05, 1.3, t, s, fill=DEEP if i == 2 else WHITE, tc=WHITE if i == 2 else INK)
    if i < 4: arrow(sl, x + 2.08, 3.25, x + 2.39, 3.25)
stat(sl, M, 4.55, 5, "92.6%", "accuracy reported in the paper, on its own data and test protocol", size=60)
text(sl, 6.6, 4.75, 6.0, 1.6, "We rebuilt this pipeline exactly, as Phase 1, and tested it on Amazon Reviews 2023 Electronics with a leak-free protocol.", size=18, color=INK)

# 5 ---------------------------------------------------------------- what we found
sl = slide("found", "Phase 1 · What we found", "Three problems when we rebuilt it",
           "Exploration findings. (1) Leak audit (LOG 2026-10-11): choosing the one-shot example from the true star inflates accuracy; with a random example it drops 7.7 points (59.7 to 52.0). Be fair: the leak was in our first reproduction's example choice, not a claim about the authors. (2) Preprocessing ablation: TextBlob cleaning removes most negative reviews, 52% of 1-star and 67% of 2-star. (3) Explanation study (x1_explain_compare.out): Phase 1 produced only 5 distinct sentences for 150 reviews and never quoted the review.")
for i, (b, l) in enumerate([("−7.7 pts", "when the prompt's example no longer reveals the answer (a label leak)"),
                            ("52% / 67%", "of 1★ / 2★ reviews deleted by the TextBlob cleaning step"),
                            ("5", "different justification sentences across 150 reviews; none quotes the review")]):
    card(sl, M + i * 4.05, 2.3, 3.75, 3.0, b, l, color=NEG if i == 0 else BLUE)
takeaway(sl, "The headline number depends on the set-up, and the justifications are a template.")

# 6 ---------------------------------------------------------------- Phase 1 result
sl = slide("p1result", "Phase 1 · Result", "Right direction, wrong star",
           "Phase 1 results (results_frozen.md). The reference method retrained on Electronics, leak-free: exact star 54.0% balanced, 53.4% natural, but within one star 93.7% / 94.4%. It gets the direction right and the exact star wrong. The 92.6% used a different protocol and is not comparable. Data: frozen disjoint splits; every choice on dev; test sets read once.",
           sub="Reference method rebuilt on Amazon Reviews 2023 Electronics, leak-free")
ch = bars(sl, M, 2.2, 7.4, 4.1, ["Balanced test (1,000)", "Natural test (2,000)"], [("Exact star", (54.0, 53.4)), ("Within one star", (93.7, 94.4))], [BLUE, GREY])
stat(sl, 8.7, 2.5, 4, "54%", "exact star: about one review in two", color=BLUE)
stat(sl, 8.7, 4.2, 4, "94%", "within one star: it reads the direction well", color=MUTED)

# 7 ---------------------------------------------------------------- Phase 2 divider
divider("d2", "02", "Phase 2 · Several agents", "Trained models decide the star. LLMs explain it. Checks keep the explanation honest.",
        "Section 2. Say the principle out loud: three attempts to let an LLM decide the star failed, so trained models decide, the LLM explains, and checks keep the explanation honest.")

# 8 ---------------------------------------------------------------- architecture
sl = slide("arch", "Phase 2 · System", "How a review is rated and explained",
           "Phase 2 system. Left: three independent agents read the review. The LLaMA probe is a classifier on a frozen LLaMA-3-8B layer (meaning). The Evidence model is TF-IDF + logistic regression (key words). DeBERTa-v3 is a fine-tuned encoder (wording). The Reflector, a logistic regression on their probabilities, decides the star. Right: the Hand-off agent answers only when confident; the Explainer writes from the evidence and the Critic checks it and sends it back, up to four rewrites. Built in LangGraph. The Router (skip the probe when sure) failed its check with three agents and is off.")
box(sl, M, 3.2, 1.7, 1.2, "Review", "title + text", fill=SURF)
for i, (t, s) in enumerate([("LLaMA probe", "meaning"), ("Evidence model", "key words"), ("DeBERTa-v3", "wording")]):
    y = 2.1 + i * 1.45
    box(sl, 3.05, y, 2.3, 1.1, t, s); arrow(sl, M + 1.72, 3.8, 3.02, y + 0.55)
    arrow(sl, 5.38, y + 0.55, 6.18, 3.8)
box(sl, 6.2, 3.15, 2.0, 1.3, "Reflector", "combines the three into one star", fill=DEEP, tc=WHITE)
arrow(sl, 8.23, 3.6, 9.0, 2.75); arrow(sl, 8.23, 4.0, 9.0, 4.85)
box(sl, 9.03, 2.15, 3.6, 1.2, "Hand-off agent", "answers when sure, otherwise asks a person")
box(sl, 9.03, 4.25, 1.65, 1.2, "Explainer", "writes from the evidence")
box(sl, 10.98, 4.25, 1.65, 1.2, "Critic", "5 checks, sends feedback")
arrow(sl, 10.7, 4.62, 10.96, 4.62, BLUE); arrow(sl, 10.96, 5.08, 10.7, 5.08, BLUE)
text(sl, 3.05, 6.35, 5.2, 0.4, "DECISION CORE", size=11, color=BLUE, bold=True)
text(sl, 9.03, 6.35, 3.6, 0.4, "AGENTIC LAYER", size=11, color=BLUE, bold=True)
text(sl, 9.03, 5.55, 3.6, 0.4, "rewrite loop, up to 4 times", size=11, color=MUTED, align=PP_ALIGN.CENTER)

# 9 ---------------------------------------------------------------- Results divider
divider("d3", "03", "Results", "All numbers from frozen test sets, read once, on Amazon Reviews 2023 Electronics.",
        "Section 3. Say: every number from here on comes from test sets that were read once, after all choices were made on development data.")

# 10 --------------------------------------------------------------- headline
sl = slide("headline", "Results · Accuracy", "Phase 2 gets the exact star far more often",
           "Headline (results_frozen.md, analysis_C.out). Exact star: Phase 1 54.0 / 53.4, Phase 2 67.5 / 78.0, that is +13.5 and +24.6 points, 29% and 53% fewer errors, significant after Holm correction over 12 comparisons. Within one star 97.3 / 97.4. If asked whether 67.5 is good: random is 20%, exact five-star rating is hard because neighbouring stars are ambiguous. Caveat if asked: on balanced eval the three-agent stack ties the previous two-agent stack; DeBERTa's +2.3 is on the natural mix.",
           sub="Exact star accuracy, Phase 1 vs Phase 2")
bars(sl, M, 2.1, 7.6, 4.3, ["Balanced test (1,000)", "Natural test (2,000)"], [("Phase 1: reference method", (54.0, 53.4)), ("Phase 2: three agents", (67.5, 78.0))], [GREY, BLUE])
stat(sl, 8.9, 2.3, 3.8, "+13.5", "points on balanced reviews (29% fewer errors)", color=POS)
stat(sl, 8.9, 4.1, 3.8, "+24.6", "points on the natural mix (53% fewer errors)", color=POS)
text(sl, 8.9, 5.85, 3.8, 0.5, "Both significant after Holm correction.", size=13, color=MUTED)

# 11 --------------------------------------------------------------- complement
sl = slide("complement", "Results · Why it works", "Each agent is strong where another is weak",
           "Why several agents beat one (agents_complement.png, q5_breakdown / complement outputs). Left: the Evidence model wins on short reviews, the LLaMA probe on long reviews, and the combined stack is best or near-best in every length band. Right: when the probe and the Evidence model disagree, the stack still gets 58-62% right, against 36-50% for either alone.")
pic(sl, "agents_complement.png", M, 1.75, w=8.3)
text(sl, 9.4, 2.3, 3.3, 4, [("Short reviews", {"bold": True, "color": INK}), "Evidence model leads.",
                          ("Long reviews", {"bold": True, "color": INK}), "LLaMA probe leads.",
                          ("When they disagree", {"bold": True, "color": INK}), "the Reflector still picks the right star 58–62% of the time."], size=16)
takeaway(sl, "The Reflector learns which agent to trust for which review.")

# 12 --------------------------------------------------------------- hard cases
sl = slide("hard", "Results · Hard language", "Reviews that sound negative but are rated 4–5★",
           "Hard cases (hard_cases.out). Reviews whose words sound negative but whose star is 4 or 5. Phase 1 gets 35.0% (eval, n 40) and 43.0% (natural, n 193); Phase 2 gets 67.5% and 78.2%. p 0.0023 and 5.4e-16. If asked: positive-sounding 1-2 star reviews improve by about 10 points but not significantly; explicit sarcasm is too rare to measure (6 and 1 reviews).",
           sub="Exact star accuracy on this hard subset")
bars(sl, M, 2.1, 7.6, 4.3, ["Balanced test (40 reviews)", "Natural test (193 reviews)"], [("Phase 1", (35.0, 43.0)), ("Phase 2", (67.5, 78.2))], [GREY, BLUE])
stat(sl, 8.9, 2.5, 3.8, "+33 / +35", "points over Phase 1 on the hardest kind of review", color=POS, size=44)
text(sl, 8.9, 4.4, 3.8, 1.5, "The words sound negative (\u201cproblem\u201d, \u201cnot\u201d), but the reviewer is happy with the product.", size=15, color=MUTED)

# 13 --------------------------------------------------------------- errors
sl = slide("errors", "Results · Where it errs", "When it is wrong, it is almost always one star off",
           "Errors (cm_main_eval.png, results_frozen.md). Confusion matrix on balanced eval: mistakes hug the diagonal. Within one star 97.3% balanced, 97.4% natural; 92% (balanced) and 88% (natural) of errors are one star off. Mean absolute error 0.35 / 0.25 stars.")
pic(sl, "cm_main_eval.png", M + 0.4, 1.7, h=4.75)
stat(sl, 7.4, 2.2, 5, "97%", "of reviews are within one star of the truth", color=BLUE, size=60)
stat(sl, 7.4, 4.2, 5, "0.35", "stars average error on balanced reviews (0.25 on the natural mix)", color=MUTED, size=48)

# 14 --------------------------------------------------------------- hand-off
sl = slide("handoff", "Agentic layer · Hand-off", "Answer when sure, ask a person when not",
           "Hand-off agent (s1_selective.out). The threshold is fixed on dev2 before the tests. On the natural test it answers 56% of reviews automatically and is right 92.8% of the time; on balanced eval 86.9% on 41%. What it hands to a person is only 54-59% right, so those really are the hard ones. Phase 1 with the same rule reaches 66.9% on balanced eval. Star ranges (q1_conformal.out): instead of handing off, it can give a range that holds the true star 90% of the time, 1.44 stars wide on natural (Phase 1 2.08).",
           sub="The system answers automatically only when its confidence is high")
pic(sl, "selective_SH.png", M, 2.0, w=7.6)
stat(sl, 8.8, 2.2, 4, "92.8%", "right on the 56% of natural-test reviews it answers itself", color=POS)
stat(sl, 8.8, 4.25, 4, "86.9%", "right on the 41% it answers on balanced reviews (Phase 1: 66.9%)", color=BLUE, size=44)

# 15 --------------------------------------------------------------- critic
sl = slide("critic", "Agentic layer · Critic loop", "The Critic makes explanations faithful",
           "Critic loop (b2_report.out). The Explainer writes from the evidence only; the Critic checks five things: the star matches, every quote is really in the review, it uses the key phrases, it mentions disagreement, and an LLM judge finds no invented claim. Faithful first drafts 62.3% / 63.7%; after up to four rewrites 90.5% / 91.6%. 841 fixed, 0 broken. Deletion test: removing the cited phrases lowers the model's confidence about twice as much as removing random words (0.156 vs 0.077 eval).",
           sub="Share of explanations that pass all five checks")
bars(sl, M, 2.1, 7.6, 4.3, ["Balanced test", "Natural test"], [("First draft", (62.3, 63.7)), ("After the Critic loop", (90.5, 91.6))], [GREY, BLUE])
stat(sl, 8.9, 2.3, 3.8, "0", "good explanations broken by a rewrite (841 fixed)", color=POS, size=60)
stat(sl, 8.9, 4.3, 3.8, "2×", "drop in confidence when the cited words are removed, vs random words", color=BLUE, size=48)

# 16 --------------------------------------------------------------- worked example
sl = slide("example", "Agentic layer · Example", "One review, three drafts",
           "Worked example from the live demo (presentation/demo_cache.json, balanced eval). Read the review, then the drafts left to right: the Critic names exactly what is wrong (an invented phrase, too few key phrases), and draft 3 passes all five checks. Compare: Phase 1's justification never quotes the review.")
EX = None
cache = os.path.join(ROOT, "presentation", "demo_cache.json")
if os.path.exists(cache):
    c = json.load(open(cache)); o = c["explanations"][1]; rid = c["demo_ids"][1]
    for l in open(os.path.join(ROOT, "v2_reflective", "e23", "eval.jsonl")):
        r = json.loads(l)
        if r["review_id"] == rid: EX = (r, o, c["phase2_states"][1]["star"]); break
if EX:
    r, o, star = EX
    rect(sl, M, 1.75, 3.6, 4.3, TINT, MSO_SHAPE.ROUNDED_RECTANGLE)
    text(sl, M + 0.3, 1.95, 3.0, 0.3, f"REVIEW  ·  true {r['gt_rating']}★, system {star}★", size=11, color=BLUE, bold=True)
    text(sl, M + 0.3, 2.35, 3.0, 3.9, r["review_text"], size=13, color=INK)
    n = len(o["drafts"]); cw = (W - 2 * M - 3.9 - 0.25 * (n - 1)) / n
    for i, (d, f) in enumerate(zip(o["drafts"], o["feedback"])):
        x = M + 3.9 + i * (cw + 0.25); ok = not f
        rect(sl, x, 1.75, cw, 4.3, WHITE, MSO_SHAPE.ROUNDED_RECTANGLE, POS if ok else RULE)
        text(sl, x + 0.2, 1.95, cw - 0.4, 0.3, f"DRAFT {i + 1}", size=11, color=BLUE, bold=True)
        text(sl, x + 0.2, 2.35, cw - 0.4, 2.45, d, size=11.5, color=BODY)
        rect(sl, x + 0.2, 4.85, cw - 0.4, 1.05, C("E8F3EC") if ok else C("F7ECEC"), MSO_SHAPE.ROUNDED_RECTANGLE)
        text(sl, x + 0.3, 4.9, cw - 0.6, 0.95, "Critic: passes all checks ✓" if ok else "Critic: " + " ".join(f), size=11,
             color=POS if ok else NEG, bold=True, anchor=MSO_ANCHOR.MIDDLE)
    takeaway(sl, "The Critic gives specific feedback; the explanation ends up quoting only what the review says.")
else:
    print("WARNING: presentation/demo_cache.json missing; example slide left empty (record the demo first)")

# 17 --------------------------------------------------------------- human check
sl = slide("human", "Agentic layer · Human check", "A person checked the Critic",
           "Human check (human_check.out, LOG 2026-11-20). The author marked 50 random explanations, blind to the Critic. Overall the author found 80% faithful, the Critic 92%, agreement 76%. Split: when all agents agreed (34) the author passed every one; when they disagreed (16) only 37.5%, while the Critic passed 93.8%. In those the Explainer says things about the system (\"the reviewer gave N stars\", a confidence %) that no check catches. Say it plainly: the Critic is too lenient on contested reviews; one annotator and 50 items, so an indication.",
           sub="50 random explanations, marked by the author without seeing the Critic's verdict")
bars(sl, M, 2.1, 7.6, 4.3, ["Agents agreed (34)", "Agents disagreed (16)"], [("Author: faithful", (100.0, 37.5)), ("Critic: faithful", (91.2, 93.8))], [BLUE, GREY])
stat(sl, 8.9, 2.3, 3.8, "76%", "agreement between the author and the Critic", color=BLUE)
text(sl, 8.9, 4.2, 3.8, 2, "When the agents disagree, the Critic misses statements about the system itself. Next fix: a check for those statements.", size=16, color=INK)

# 18 --------------------------------------------------------------- robustness
sl = slide("robust", "Results · Robustness", "Stable under messy text, but needs the title",
           "Robustness (q4_robustness.out, 300 eval reviews). Share of reviews whose star changes when the text is changed. Typos: Phase 2 0.7% vs Phase 1 2.0% (ns). Flat text (no punctuation or capitals): 9.3% vs 29.7%. No title: 15.7% vs 0%, because the Evidence model and DeBERTa read the title and Phase 1 does not. Say the limitation plainly.",
           sub="Share of reviews whose star changes when the text is altered (lower is better)")
bars(sl, M, 2.1, 7.6, 4.3, ["Typos", "No punctuation or capitals", "Title removed"], [("Phase 1", (2.0, 29.7, 0.0)), ("Phase 2", (0.7, 9.3, 15.7))], [GREY, BLUE], maxv=35)
stat(sl, 8.9, 2.3, 3.8, "3×", "more stable than Phase 1 on flat, unpunctuated text", color=POS)
stat(sl, 8.9, 4.2, 3.8, "15.7%", "of stars change without the title: a limitation", color=NEG, size=44)

# 19 --------------------------------------------------------------- what failed
sl = slide("failed", "What did not work", "Negative results we report",
           "Failures (EXPERIMENTS_SUMMARY.md). Every idea had a pass/fail rule written before it ran and was checked on development data only; none touched the test sets. Three attempts to let an LLM decide or revise the star failed (the last: 76 fixed vs 89 broken). A second LLM judge (Qwen2.5-7B) rejected every explanation. The Router saved work with two agents but failed its check with three, so it is off.")
for i, (b, t, s) in enumerate([("3 ×", "LLM deciding the star", "Three attempts failed. The last one fixed 76 stars and broke 89."),
                               ("0%", "Second LLM judge", "Qwen2.5-7B rejected every explanation, whatever its content."),
                               ("Off", "Router with three agents", "It saved work with two agents but failed its rule with three.")]):
    x = M + i * 4.05
    rect(sl, x, 2.0, 3.75, 3.7, SURF, MSO_SHAPE.ROUNDED_RECTANGLE, RULE)
    text(sl, x + 0.35, 2.3, 3.1, 0.9, b, size=44, color=NEG, bold=True)
    text(sl, x + 0.35, 3.5, 3.1, 0.5, t, size=19, color=INK, bold=True)
    text(sl, x + 0.35, 4.15, 3.1, 1.8, s, size=15)
takeaway(sl, "Every rule was written before the run; failures never touched the test sets.")

# 20 --------------------------------------------------------------- conclusions
sl = slide("conclusions", "Conclusions", "What the project shows",
           "Conclusions. Three points: (1) the reference method, rebuilt leak-free, gets 54% exact; we found a label leak, a cleaning step that removes negative reviews, and template justifications. (2) Three agents and a Reflector reach 67.5% / 78.0%, +13.5 / +24.6 points, significant. (3) The agentic layer makes the output checkable: 91-92% faithful explanations, 0 broken; Hand-off 93% right on 56%; and a human check shows where the Critic is still too lenient. Limits: one dataset, needs the title.")
for i, (n, t, s) in enumerate([("1", "Rebuilt and explored", "Leak-free, the reference method gets 54% exact. We found a label leak, a biased cleaning step and template justifications."),
                               ("2", "Several agents beat one", "Three agents and a Reflector: 67.5% and 78.0%, +13.5 and +24.6 points, significant."),
                               ("3", "Output you can check", "91–92% faithful explanations, none broken; 93% right when it answers alone; a human check shows what to fix next.")]):
    x = M + i * 4.05
    text(sl, x, 2.1, 1.2, 1.0, n, size=54, color=BLUE, bold=True)
    text(sl, x, 3.25, 3.7, 0.5, t, size=21, color=INK, bold=True)
    text(sl, x, 3.9, 3.6, 2.2, s, size=16)
takeaway(sl, "Trained models decide. LLMs explain. Checks keep them honest.")

# 21 --------------------------------------------------------------- thanks
sl = prs.slides.add_slide(LAY); SLIDES.append(("thanks", "Thank you", sl))
rect(sl, 0, 0, W, H, DEEP)
text(sl, M, 2.6, W - 2 * M, 1.2, "Thank you", size=54, color=WHITE, bold=True, align=PP_ALIGN.CENTER)
text(sl, M, 3.9, W - 2 * M, 0.6, "Questions  ·  live demo", size=22, color=C("C9D7EA"), align=PP_ALIGN.CENTER)
NOTES["thanks"] = "Close. Offer the live demo notebook (presentation/Final_Demo.ipynb): five reviews rated, explained and checked, Phase 1 next to Phase 2."

# ---- page numbers, notes, slide map
n = len(SLIDES)
for i, (k, t, sl) in enumerate(SLIDES, 1):
    if k not in ("title", "thanks") and not k.startswith("d"):
        text(sl, W - M - 1, H - 0.4, 1, 0.3, f"{i} / {n}", size=10, color=MUTED, align=PP_ALIGN.RIGHT)
    sl.notes_slide.notes_text_frame.text = NOTES[k]
with open(os.path.join(ROOT, "presentation", "deck_slide_map.md"), "w") as f:
    f.write("# Deck slide map and short scripts (auto-generated by v2_reflective/build_deck_clean.py on every build; do not edit)\n\n"
            "Same text as the PowerPoint / Google Slides speaker notes. Longer scripts: `Presenter_Notes.md`, Part 0.\n\n| Slide | Title | Short script (speaker note) |\n|---|---|---|\n")
    for i, (k, t, sl) in enumerate(SLIDES, 1): f.write(f"| {i} | {t} | {NOTES[k].replace('|', '/')} |\n")
prs.save(DST); print("saved", DST, n)
GD = os.path.expanduser("~/Library/CloudStorage/GoogleDrive-kvarun314@gmail.com/My Drive/PES_backups/current_deck")
if os.path.isdir(os.path.dirname(GD)):   # overwrite only this one file (user-approved 2026-10-25)
    os.makedirs(GD, exist_ok=True); shutil.copyfile(DST, os.path.join(GD, os.path.basename(DST))); print("synced to Drive")
else:
    print("Drive folder not found; NOT synced")
