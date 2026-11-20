"""The one current deck (minimal layout, 2026-11-20): one message per slide, flat shapes, lots of white space, same blue palette.
Order (user, 2026-11-20): reference paper in a nutshell -> its stack -> our implementation and what we found -> its novelty;
then our approach -> stack -> why it works -> approaches tried -> what we kept -> results -> why it is better -> conclusion.
Every number is copied from v2_reflective/LOG.md / e23/*.out (or, for the reference paper, from the paper); the source is in each speaker note.
Writes presentation/deck_slide_map.md and syncs the Drive copy.
Run: /opt/homebrew/bin/python3 v2_reflective/build_deck_clean.py   (system python has python-pptx; .venv does not)"""
import os, json, shutil
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.chart.data import CategoryChartData, XyChartData
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION, XL_LABEL_POSITION
from pptx.oxml.ns import qn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DST = os.path.join(ROOT, "presentation", "Agentic_LLaMA3_Presentation.pptx")  # gitignored (*.pptx)
FIG = os.path.join(ROOT, "paper", "figures", "e23")
FONT = "Arial"
C = lambda h: RGBColor.from_string(h)
# Apple-inspired: near-black type, soft grey tiles, one accent blue, deep-navy stage for big moments
INK, BODY, MUTED = C("1D1D1F"), C("424245"), C("86868B")
BLUE, DEEP, TINT, RULE = C("2C5AA0"), C("1B3C6E"), C("F5F5F7"), C("E5E5EA")
GREY, POS, NEG, WHITE, SOFT = C("C7C7CC"), C("1E7A46"), C("A93636"), C("FFFFFF"), C("A1A1A6")
NIGHT, SKY = C("0B1A33"), C("7FA8E8")
W, H, M = 13.333, 7.5, 0.8
CW = W - 2 * M

prs = Presentation(); prs.slide_width, prs.slide_height = Inches(W), Inches(H)
LAY = prs.slide_layouts[6]
SLIDES, NOTES = [], {}


def text(sl, x, y, w, h, s, size=16, color=BODY, bold=False, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, line=1.15):
    tb = sl.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)); tf = tb.text_frame
    tf.word_wrap = True; tf.vertical_anchor = anchor
    for a in ("margin_left", "margin_right", "margin_top", "margin_bottom"): setattr(tf, a, 0)
    for i, part in enumerate(s if isinstance(s, list) else [s]):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        t, o = (part, {}) if isinstance(part, str) else part
        p.alignment = align; p.line_spacing = line
        if i: p.space_before = Pt(o.get("gap", size * 0.6))
        r = p.add_run(); r.text = t; f = r.font; f.name = FONT
        f.size = Pt(o.get("size", size)); f.bold = o.get("bold", bold); f.color.rgb = o.get("color", color)
    return tb


def rect(sl, x, y, w, h, fill, r=0):
    s = sl.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if r else MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    if r: s.adjustments[0] = r
    s.fill.solid(); s.fill.fore_color.rgb = fill; s.line.fill.background()
    st = s._element.find(qn("p:style"))
    if st is not None: s._element.remove(st)   # no theme shadow or outline
    return s


def hline(sl, x, y, w, color=RULE, h=0.015): return rect(sl, x, y, w, h, color)


def slide(key, part, title, note, sub=None):
    sl = prs.slides.add_slide(LAY); SLIDES.append((key, title, sl)); NOTES[key] = note
    text(sl, M, 0.65, CW, 0.8, title, size=38, color=INK, bold=True, line=1.0)   # no kicker, no footer: the title carries the slide
    if sub: text(sl, M, 1.4, CW, 0.5, sub, size=18, color=MUTED)
    return sl


def cols(sl, items, y=2.5, big=40, gap=0.3, color=BLUE, th=None):
    """Rounded grey tiles: (big, title, body). Four items go in a 2 x 2 grid. big may be '' to skip."""
    n = len(items); per = 2 if n == 4 else n; nr = (n + per - 1) // per
    w = (CW - gap * (per - 1)) / per; th = th or (2.0 if nr == 2 else 3.2)
    for i, (b, t, s) in enumerate(items):
        x = M + (i % per) * (w + gap); yt = y + (i // per) * (th + gap); rect(sl, x, yt, w, th, TINT, 0.08)
        if nr == 2 and b:   # wide tile: number on the left, words on the right
            nw = 2.4 if len(b) > 4 else 1.3   # number column sized to the number
            text(sl, x + 0.35, yt + 0.3, nw, 1.2, b, size=big, color=color, bold=True)
            text(sl, x + 0.45 + nw, yt + 0.35, w - nw - 0.7, 0.5, t, size=19, color=INK, bold=True)
            text(sl, x + 0.45 + nw, yt + 0.9, w - nw - 0.7, th - 1.0, s, size=15); continue
        x += 0.35; ww = w - 0.7; yy = yt + 0.4
        if b: text(sl, x, yy, ww, 0.8, b, size=big, color=color, bold=True); yy += big / 72 * 1.35
        lines = 1 + int(len(t) * 18 * 0.5 / 72 / ww)
        text(sl, x, yy, ww, 0.5 * lines, t, size=18, color=INK, bold=True); yy += 0.32 * lines + 0.25
        text(sl, x, yy, ww, 1.6, s, size=15)


def rows(sl, data, widths, y=2.2, head=None, size=14, rh=0.52, tw=None):
    """Plain table: text rows separated by hairlines; first column bold."""
    if head:
        x = M
        for t, w in zip(head, widths): text(sl, x, y, w - 0.2, 0.3, t.upper(), size=10.5, color=MUTED, bold=True); x += w
        y += 0.38; hline(sl, M, y, tw or CW, INK, 0.02); y += 0.02
    for r in data:
        x = M
        for j, (t, w) in enumerate(zip(r, widths)):
            o = t if isinstance(t, tuple) else (t, {})
            text(sl, x, y, w - 0.25, rh, o[0], size=size, color=o[1].get("color", INK if j == 0 else BODY), bold=o[1].get("bold", j == 0),
                 anchor=MSO_ANCHOR.MIDDLE)
            x += w
        y += rh; hline(sl, M, y, tw or CW)


def stat(sl, x, y, w, big, label, color=BLUE, size=54):
    text(sl, x, y, w, 0.95, big, size=size, color=color, bold=True)
    text(sl, x, y + size / 72 * 1.3, w, 1.2, label, size=15, color=BODY)


def bars(sl, x, y, w, h, cats, series, colors, maxv=100, fmt='0.0"%"', lsize=16):
    cd = CategoryChartData(); cd.categories = cats
    for n, v in series: cd.add_series(n, v)
    ch = sl.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(x), Inches(y), Inches(w), Inches(h), cd).chart
    ch.font.name = FONT; ch.font.size = Pt(13); ch.font.color.rgb = BODY
    va, ca = ch.value_axis, ch.category_axis
    va.maximum_scale, va.minimum_scale = maxv, 0; va.has_major_gridlines = False; va.visible = False
    ca.format.line.color.rgb = RULE; ca.tick_labels.font.size = Pt(14)
    pl = ch.plots[0]; pl.gap_width = 55; pl.overlap = -6
    pl.has_data_labels = True; dl = pl.data_labels; dl.number_format = fmt; dl.number_format_is_linked = False
    dl.font.size = Pt(lsize); dl.font.bold = True; dl.font.color.rgb = INK; dl.position = XL_LABEL_POSITION.OUTSIDE_END
    for s, c in zip(pl.series, colors): s.format.fill.solid(); s.format.fill.fore_color.rgb = c
    ch.has_legend = len(series) > 1
    if ch.has_legend:
        ch.legend.position = XL_LEGEND_POSITION.TOP; ch.legend.include_in_layout = False; ch.legend.font.size = Pt(13)


def curves(sl, x, y, w, h, series, colors, xt, yt):
    """Native line chart with real x values: series = [(name, [(x, y), ...])]."""
    cd = XyChartData()
    for n, pts in series:
        sr = cd.add_series(n)
        for a, b in pts: sr.add_data_point(a, b)
    ch = sl.shapes.add_chart(XL_CHART_TYPE.XY_SCATTER_LINES, Inches(x), Inches(y), Inches(w), Inches(h), cd).chart
    ch.font.name = FONT; ch.font.size = Pt(13); ch.font.color.rgb = BODY
    va, ca = ch.value_axis, ch.category_axis
    va.minimum_scale, va.maximum_scale = 50, 100; ca.minimum_scale, ca.maximum_scale = 40, 100
    for ax, t in ((va, yt), (ca, xt)):
        ax.has_major_gridlines = False; ax.format.line.color.rgb = RULE; ax.tick_labels.font.size = Pt(12)
        ax.has_title = True; ax.axis_title.text_frame.text = t
        r = ax.axis_title.text_frame.paragraphs[0].runs[0]; r.font.size = Pt(12); r.font.bold = False; r.font.color.rgb = MUTED; r.font.name = FONT
    for s_, c in zip(ch.plots[0].series, colors):
        s_.format.line.color.rgb = c; s_.format.line.width = Pt(3); s_.smooth = False
        s_.marker.format.fill.solid(); s_.marker.format.fill.fore_color.rgb = c; s_.marker.format.line.color.rgb = c
    ch.has_legend = True; ch.legend.position = XL_LEGEND_POSITION.TOP; ch.legend.include_in_layout = False; ch.legend.font.size = Pt(13)


def pic(sl, f, x, y, w=None, h=None):
    sl.shapes.add_picture(os.path.join(FIG, f), Inches(x), Inches(y), Inches(w) if w else None, Inches(h) if h else None)


def node(sl, x, y, w, h, title, sub="", dark=False):
    rect(sl, x, y, w, h, DEEP if dark else TINT, 0.15)
    text(sl, x + 0.1, y + (0.18 if sub else 0), w - 0.2, 0.4 if sub else h, title, size=15, color=WHITE if dark else INK, bold=True,
         align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.TOP if sub else MSO_ANCHOR.MIDDLE)
    if sub: text(sl, x + 0.1, y + 0.55, w - 0.2, h - 0.6, sub, size=12, color=SOFT if dark else BODY, align=PP_ALIGN.CENTER)


def arrow(sl, x1, y1, x2, y2, color=GREY):
    c = sl.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = color; c.line.width = Pt(1.75)
    ln = c.line._get_or_add_ln(); ln.append(ln.makeelement(qn("a:tailEnd"), {"type": "triangle", "w": "med", "len": "med"}))


def flow(sl, y, items, dark=None, h=1.2):
    n = len(items); g = 0.45; w = (CW - g * (n - 1)) / n
    for i, (t, s) in enumerate(items):
        x = M + i * (w + g); node(sl, x, y, w, h, t, s, dark=(i == dark))
        if i < n - 1: arrow(sl, x + w + 0.05, y + h / 2, x + w + g - 0.05, y + h / 2)


def dark_slide(key, title, lines, note, kicker=None):
    sl = prs.slides.add_slide(LAY); SLIDES.append((key, title, sl)); NOTES[key] = note
    rect(sl, 0, 0, W, H, NIGHT)
    if kicker: text(sl, M, 2.0, CW, 0.4, kicker.upper(), size=14, color=SKY, bold=True, align=PP_ALIGN.CENTER)
    text(sl, M, 1.6, CW, 2.6, title, size=54, color=WHITE, bold=True, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    text(sl, M + 1, 4.3, CW - 2, 1.8, lines, size=20, color=SOFT, align=PP_ALIGN.CENTER)
    return sl


def hero(key, big, lines, note, title):
    """Apple-style moment: one giant number on the dark stage."""
    sl = prs.slides.add_slide(LAY); SLIDES.append((key, title, sl)); NOTES[key] = note
    rect(sl, 0, 0, W, H, NIGHT)
    text(sl, M, 1.4, CW, 2.6, big, size=150, color=SKY, bold=True, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, line=1.0)
    text(sl, M + 1.5, 4.4, CW - 3, 1.8, lines, size=22, color=SOFT, align=PP_ALIGN.CENTER)
    return sl


# ======================================================================= 1 title
dark_slide("title", "Agentic Self-Correction in LLaMA-3",
           ["Several agents rate, explain and check e-commerce product reviews", ("Varun K  ·  PES University, M.Tech Project  ·  November 2026", {"size": 14})],
           "Title. Say: we take a published LLM method for rating product reviews, rebuild it and study it, then build our own agent-based approach on top of what we learn.")

# ======================================================================= PART 1: reference paper
sl = slide("paper", "Part 1 · Reference paper", "The reference paper in a nutshell",
           "The reference paper. Wang et al. (2025), ITM Web of Conferences 70, 04021. Say: we chose this paper as our reference. Task: rate an Amazon product review 1-5 stars with a short justification. Approach: one fine-tuned LLaMA-3-8B reads the review once. Data: Kaggle Amazon reviews, about 500K. It reports 92.6% accuracy and 93.3% F1 with 5-fold cross-validation and a 40K-review test set, and beats SVM, Naive Bayes, Decision Tree and XLNet.",
           sub="Wang et al. (2025), “Sentiment Analysis of Product Reviews Using Fine-Tuned LLaMA-3”")
rows(sl, [("Task", "Rate an Amazon product review from 1 to 5 stars, with a short justification"),
          ("Approach", "One fine-tuned LLaMA-3-8B reads the review once and answers"),
          ("Data", "Kaggle Amazon reviews (about 500K), cleaned and balanced before training")],
     [1.7, 5.9], y=2.4, size=17, rh=0.95, tw=7.4)
rect(sl, 8.6, 2.4, 0.015, 2.85, RULE)
stat(sl, 9.0, 2.45, 3.6, "92.6%", "accuracy reported (93.3% F1); beats SVM, Naive Bayes, Decision Tree and XLNet", size=60)

sl = slide("stack", "Part 1 · Reference paper", "Its stack",
           "The paper's stack. Data: TextBlob quality control drops reviews whose tone contradicts the star; stratified sampling balances the stars; VGST, a greedy diversity search, prefers reviews with new words; 3-star reviews are doubled. Model: LLaMA-3-8B in 4-bit, LoRA rank 4, alpha 64 on the last 3 layers, 3 epochs; fits one 16 GB GPU. Prompt: a 1-5 star rubric, one worked example and a step-by-step cue. Output: a star and a justification.",
           sub="A data pipeline, then one fine-tuned model")
text(sl, M, 2.3, 4, 0.3, "DATA", size=11, color=BLUE, bold=True)
flow(sl, 2.65, [("TextBlob filter", "drop reviews whose tone contradicts the star"), ("Balance", "equal reviews per star"),
                ("VGST", "prefer reviews with new words"), ("Neutral ×2", "double the 3-star reviews")])
text(sl, M, 4.3, 4, 0.3, "MODEL", size=11, color=BLUE, bold=True)
flow(sl, 4.65, [("LLaMA-3-8B", "4-bit, one 16 GB GPU"), ("LoRA fine-tuning", "rank 4, α 64, last 3 layers"),
                ("Prompt", "rubric, one example, step by step"), ("Output", "star + justification")], dark=1)

sl = slide("implement", "Part 1 · Our implementation", "How we implemented it",
           "Implementation. We rebuilt the pipeline with the paper's Table 1 settings. As a check, on the paper's own data source (Kaggle, its protocol, no leak) it reaches 69.1%; always guessing 5 stars already gives 63.8% there. Then a clean benchmark: Amazon Reviews 2023 Electronics, splits drawn once and kept disjoint (train about 497K, development, balanced test 1,000, natural test 2,000); every choice made on development data; test sets read once.",
           sub="Same pipeline and settings, on a clean benchmark with a fair test")
cols(sl, [("", "Rebuild", "The paper's pipeline and Table 1 settings: LoRA rank 4, α 64, 3 epochs."),
          ("", "Clean benchmark", "Amazon Reviews 2023 Electronics. Train, development and test sets kept apart."),
          ("", "Fair test", "No prompt example that hints at the answer. Test sets read once, after all choices.")], y=2.3, th=2.0)
hline(sl, M, 4.75, CW)
stat(sl, M, 5.0, 3.2, "69.1%", "check on the paper's own data", size=40)
text(sl, 4.4, 5.2, 8, 1.2, "Our rebuild works. But always guessing 5 stars already scores 63.8% on that data, so it says little about the hard stars.", size=16, color=INK)

sl = slide("p1result", "Part 1 · What we found", "Right direction, wrong star",
           "Phase 1 results (results_frozen.md). The reference method on Electronics, leak-free: exact star 54.0% balanced, 53.4% natural; within one star 93.7% / 94.4%. It reads the direction well; choosing between neighbouring stars is the hard part. The paper's 92.6% used a different protocol and is not comparable.",
           sub="The reference method on Amazon Reviews 2023 Electronics")
bars(sl, M, 2.2, 7.4, 4.5, ["Balanced test (1,000)", "Natural test (2,000)"], [("Exact star", (54.0, 53.4)), ("Within one star", (93.7, 94.4))], [BLUE, GREY])
stat(sl, 8.9, 2.5, 3.7, "94%", "within one star: it reads the direction well")
stat(sl, 8.9, 4.4, 3.7, "54%", "exact star: neighbouring stars are hard", color=MUTED)

sl = slide("found", "Part 1 · What we found", "Four things hold it back",
           "Findings. (1) Label leak (LOG 2026-10-11): in our first reproduction the prompt's example hinted at the answer; a random example costs 7.7 points. This was our first reproduction's example choice, not a claim about the authors. (2) TextBlob cleaning removes 52% of 1-star and 67% of 2-star reviews. (3) Justifications are a template: 5 distinct sentences for 150 reviews, none quoting the review (x1_explain_compare.out). (4) One model, one pass: nothing catches a mistake.",
           sub="What we found when we rebuilt it on a clean benchmark")
cols(sl, [("−7.7", "Set-up dependent", "points when the prompt's example no longer hints at the answer"),
          ("52 / 67%", "Negatives removed", "of 1-star / 2-star reviews deleted by the TextBlob filter"),
          ("5", "Template reasons", "distinct justifications for 150 reviews; none quotes the review"),
          ("1", "Single pass", "one model decides; nothing catches a mistake")], y=2.6, big=40, color=NEG, gap=0.45)

sl = slide("novelty_paper", "Part 1 · Reference paper", "Its novelty",
           "What the paper contributes, and why we chose it: (1) a general open LLM fine-tuned for five-star rating, beating classic models and XLNet; (2) a data pipeline combining quality control, balance, lexical diversity and neutral oversampling; (3) parameter-efficient training, a small LoRA add-on and 4-bit weights on one 16 GB GPU; (4) a prompt that returns a justification with the star. Our findings show where each of these needs help.",
           sub="What the paper contributes, and why we chose it")
cols(sl, [("01", "LLM for star rating", "A general open LLM, fine-tuned for five stars, beats classic models and XLNet."),
          ("02", "Curated data", "Quality control, balanced stars, varied wording, extra neutral reviews."),
          ("03", "Efficient training", "A small LoRA add-on and 4-bit weights: one 16 GB GPU."),
          ("04", "Star with a reason", "The prompt returns a justification with the star.")], y=2.3, big=36, gap=0.45)

# ======================================================================= PART 2: our approach
sl = dark_slide("approach", "", [], "Our approach. Say: from Part 1 we need a second opinion, data kept whole, and explanations that can be checked. Three attempts to let an LLM decide the star failed, so trained models decide, the LLM explains, and checks keep the explanation honest.")
SLIDES[-1] = ("approach", "Our approach", sl)
text(sl, M, 1.3, CW, 0.4, "PART 2  \u00b7  OUR APPROACH", size=14, color=SKY, bold=True, align=PP_ALIGN.CENTER)
text(sl, M, 2.0, CW, 3.5, [("Models decide.", {}), ("LLMs explain.", {"gap": 6}), ("Checks keep it honest.", {"gap": 6, "color": SKY})],
     size=60, color=WHITE, bold=True, align=PP_ALIGN.CENTER, line=1.0)

sl = slide("arch", "Part 2 · Our approach", "Our stack",
           "Our stack. Three independent agents read the review: the LLaMA probe (a classifier on a frozen LLaMA-3-8B layer, meaning), the Evidence model (TF-IDF + logistic regression on about 497K reviews, key words) and DeBERTa-v3 (fine-tuned encoder, wording). The Reflector, a logistic regression on their probabilities, decides the star. The Hand-off agent answers only when confident. The Explainer writes from the evidence; the Critic checks it and sends it back, up to four rewrites. Built in LangGraph; LLaMA runs locally on MLX.",
           sub="Three agents decide; an agentic layer explains and checks")
node(sl, M, 3.35, 1.6, 1.0, "Review")
for i, (t, s) in enumerate([("LLaMA probe", "meaning"), ("Evidence model", "key words"), ("DeBERTa-v3", "wording")]):
    y = 2.25 + i * 1.4
    node(sl, 3.0, y, 2.3, 1.05, t, s); arrow(sl, M + 1.65, 3.85, 2.95, y + 0.52); arrow(sl, 5.35, y + 0.52, 6.05, 3.85)
node(sl, 6.1, 3.25, 2.0, 1.2, "Reflector", "one star", dark=True)
arrow(sl, 8.15, 3.6, 8.9, 2.75); arrow(sl, 8.15, 4.1, 8.9, 4.95)
node(sl, 8.95, 2.2, 3.58, 1.05, "Hand-off agent", "answers when sure, else a person")
node(sl, 8.95, 4.45, 1.6, 1.05, "Explainer", "writes")
node(sl, 10.93, 4.45, 1.6, 1.05, "Critic", "checks")
arrow(sl, 10.58, 4.8, 10.9, 4.8, BLUE); arrow(sl, 10.9, 5.15, 10.58, 5.15, BLUE)
text(sl, M, 6.35, CW, 0.4, "LangGraph  ·  LLaMA-3-8B on MLX  ·  DeBERTa-v3  ·  scikit-learn", size=13, color=MUTED, align=PP_ALIGN.CENTER)

sl = slide("complement", "Part 2 · Our approach", "Why it works",
           "Why several agents beat one (native charts; numbers from figures_deck.out). Left: the Evidence model wins on short reviews, the LLaMA probe on long reviews, and the combined stack is best or near-best in every length band. Right: when the probe and the Evidence model disagree, the stack still gets 58-62% right, against 36-50% for either alone. The Reflector learns which agent to trust for which review.",
           sub="Each agent is strong where another is weak; the Reflector learns whom to trust")
text(sl, M, 2.1, 7, 0.3, "BY REVIEW LENGTH  \u00b7  balanced test", size=11, color=MUTED, bold=True)
bars(sl, M, 2.35, 7.3, 4.4, ["Shortest", "Short", "Long", "Longest"], [("Evidence model", (70.0, 66.7, 57.4, 55.8)), ("LLaMA probe", (54.6, 67.1, 59.4, 64.3)), ("Stack", (71.2, 68.8, 63.7, 66.3))],
     [GREY, SKY, BLUE], maxv=85, fmt='0', lsize=12)
text(sl, 8.5, 2.1, 4, 0.3, "WHEN THE TWO AGENTS DISAGREE", size=11, color=MUTED, bold=True)
bars(sl, 8.5, 2.35, 4.03, 4.4, ["Balanced", "Natural"], [("Evidence", (42.7, 35.7)), ("Probe", (39.1, 50.1)), ("Stack", (57.8, 62.0))], [GREY, SKY, BLUE], maxv=85, fmt='0', lsize=12)

sl = slide("tried", "Part 2 · Approaches we took", "What we tried and dropped",
           "Approaches tried (EXPERIMENTS_SUMMARY.md, section 3). Every idea had a pass/fail rule written before it ran and was checked on development data only; none touched the test sets. LLM deciding the star: 54.2% vs 73.3% on disagreement rows, and later 76 fixed vs 89 broken. Product-photo agent: constant output. More views of the same text (MiniLM, character n-grams, NB-SVM): no gain. Bigger LLaMA probe (120K): +0.3, saturated. Gradient-boosted Reflector: overfits (.621 vs .642). Second LLM judge: rejects everything.",
           sub="Every idea had a pass/fail rule written before it ran, checked on development data only")
D = ("Dropped", {"color": NEG, "bold": True})
rows(sl, [("LLM decides the star (3 attempts)", "54.2% vs 73.3%; later 76 fixed vs 89 broken", D),
          ("Product-photo agent (BLIP, Qwen2-VL)", "Same answer for every review", D),
          ("More views of the same words", "MiniLM, character n-grams, NB-SVM: no gain", D),
          ("Bigger LLaMA probe (120K reviews)", "+0.3 points: already saturated at 40K", D),
          ("Non-linear Reflector", "Overfits: .621 vs .642 on development data", D),
          ("Second LLM judge (Qwen2.5-7B)", "Rejects every explanation", D)],
     [4.6, 5.9, 1.2], y=2.3, head=["Approach", "Result", "Verdict"], size=15, rh=0.62)

sl = slide("kept", "Part 2 · What we stuck to", "What we kept, and why",
           "What we kept (EXPERIMENTS_SUMMARY.md, section 2; LOG). Leak-free zero-shot protocol (the leak was 7.7 points). Raw balanced data: TextBlob filtering cost 2.5-3.5 points. LLaMA probe on 40K reviews: +3.9 on dev2. Evidence model on about 497K reviews. DeBERTa-v3: +2.3 on the natural test, stable over 3 seeds. Logistic-regression Reflector. Critic loop up to four rewrites: 0 broken. Hand-off agent.",
           sub="Each part passed its own check before it was kept")
K = ("Kept", {"color": POS, "bold": True})
rows(sl, [("Leak-free protocol", "The example leak was worth 7.7 points; test sets read once", K),
          ("Raw balanced data", "The TextBlob filter cost 2.5–3.5 points", K),
          ("LLaMA probe + Evidence model", "Probe +3.9 points on development data", K),
          ("DeBERTa-v3 as third agent", "+2.3 on the natural test; stable over 3 seeds", K),
          ("Simple Reflector (logistic regression)", "Beats every single agent; does not overfit", K),
          ("Critic loop and Hand-off agent", "841 explanations fixed, 0 broken; 93% right when sure", K)],
     [4.6, 5.9, 1.2], y=2.3, head=["Choice", "Evidence", "Verdict"], size=15, rh=0.62)

# ======================================================================= results
hero("hero", "+13.5", ["points of exact-star accuracy over the reference method, on balanced reviews.", ("54.0% to 67.5%, the same 1,000 reviews. 29% fewer errors.", {"size": 18, "color": SOFT})],
     "Hero moment. Pause on the number. Say: on balanced reviews (200 per star, so the star mix gives no free points) the three-agent system gets the exact star 13.5 points more often than the reference method, 54.0% to 67.5% (results_frozen.md); McNemar p 6.7e-14 (hard_cases.out). Then go to the chart, which also shows the natural test.",
     "+13.5 points")
sl = slide("headline", "Results", "Phase 2 gets the exact star far more often",
           "Headline (results_frozen.md, hard_cases.out, prior_baseline.out). Exact star: Phase 1 54.0 / 53.4, Phase 2 67.5 / 78.0: +13.5 and +24.6 points, McNemar p 6.7e-14 and 3.7e-84 (this comparison is not part of the 12-comparison Holm set). Say the natural-test caveat yourself: Phase 1 was trained on balanced data, and 65% of natural reviews are 5-star; adjusted to the real star mix Phase 1 gets 75.5%, so the fair natural gain is +2.5 exact. Macro-F1, which ignores the star mix, still rises from .488 (.507 adjusted) to .634. Within one star 97.3 / 97.4. If asked whether 67.5 is good: random is 20%, and neighbouring stars are ambiguous. Caveat if asked: on balanced eval the three-agent stack ties the previous two-agent stack; DeBERTa's +2.3 is on the natural mix.",
           sub="Exact star accuracy, Phase 1 vs Phase 2")
bars(sl, M, 2.2, 7.4, 4.5, ["Balanced test (1,000)", "Natural test (2,000)"], [("Phase 1", (54.0, 53.4)), ("Phase 2", (67.5, 78.0))], [GREY, BLUE])
stat(sl, 8.9, 2.3, 3.7, "+13.5", "points on balanced reviews", color=POS)
stat(sl, 8.9, 3.85, 3.7, ".49 \u2192 .63", "macro-F1 on the natural mix, which the star mix cannot inflate", color=POS, size=44)
text(sl, 8.9, 5.55, 3.7, 1.2, "Natural test raw gain is +24.6; with Phase 1 adjusted to the real star mix it is +2.5.", size=13, color=MUTED)

sl = slide("hard", "Results", "Negative words, happy reviewer",
           "Hard cases (hard_cases.out). Reviews whose words sound negative but whose star is 4 or 5. Phase 1: 35.0% (balanced, n 40), 43.0% (natural, n 193). Phase 2: 67.5% and 78.2%. p 0.0023 and 5.4e-16.",
           sub="Exact star accuracy on 4- and 5-star reviews whose words sound negative")
bars(sl, M, 2.2, 7.4, 4.5, ["Balanced test (40)", "Natural test (193)"], [("Phase 1", (35.0, 43.0)), ("Phase 2", (67.5, 78.2))], [GREY, BLUE])
stat(sl, 8.9, 2.5, 3.7, "+33 / +35", "points over Phase 1 on the hardest kind of review", color=POS, size=44)

sl = slide("handoff", "Results", "It knows when it is unsure",
           "Hand-off agent (s1_selective.out). Threshold fixed on dev2 before the tests. Natural test: answers 56% of reviews automatically, 92.8% right. Balanced test: 86.9% on 41%; Phase 1 with the same rule 66.9%. What it hands to a person is only 54-59% right: those are the hard ones. Native chart from s1_selective.out (natural test). Star ranges (q1_conformal.out): over all reviews the answer covers the true star 90% of the time; for the unsure reviews that get a range, coverage is 86% (natural) and 91% (balanced). Ranges are 1.44 stars wide on average on natural (Phase 1 2.08).",
           sub="The Hand-off agent answers alone only when confident, otherwise it asks a person")
curves(sl, M, 2.1, 7.6, 4.7, [("Phase 1", [(100, 53.4), (94.6, 55.2), (90.0, 56.3), (85.2, 57.3), (77.8, 60.1), (67.5, 64.4), (55.0, 67.9)]),
                              ("Phase 2", [(100, 78.0), (95.7, 79.9), (90.5, 81.7), (82.6, 84.4), (74.9, 86.9), (67.0, 89.9), (56.1, 92.8)])],
       [GREY, BLUE], "Share of natural-test reviews answered automatically (%)", "Right on those (%)")
stat(sl, 8.9, 2.3, 3.7, "92.8%", "right on the 56% of natural-test reviews it answers alone", color=POS)
stat(sl, 8.9, 4.3, 3.7, "86.9%", "on balanced reviews (Phase 1 with the same rule: 66.9%)", size=40)

sl = slide("critic", "Results", "Its explanations are checked and faithful",
           "Critic loop (b2_report.out). The Critic checks five things: the star matches, every quote is in the review, it uses the key phrases, it mentions disagreement, and an LLM judge finds no invented claim. Faithful first drafts 62.3% / 63.7%; after up to four rewrites 90.5% / 91.6%; 841 fixed, 0 broken. Deletion test: removing the cited words lowers confidence about twice as much as removing random words (0.156 vs 0.077 balanced).",
           sub="Share of explanations that pass all five Critic checks")
bars(sl, M, 2.2, 7.4, 4.5, ["Balanced test", "Natural test"], [("First draft", (62.3, 63.7)), ("After the Critic loop", (90.5, 91.6))], [GREY, BLUE])
stat(sl, 8.9, 2.3, 3.7, "0", "good explanations broken (841 fixed)", color=POS, size=60)
stat(sl, 8.9, 4.3, 3.7, "2×", "confidence drop when the cited words are removed, vs random words", size=44)

sl = slide("example", "Results", "One review, three drafts",
           "Worked example from the live demo (presentation/demo_cache.json, balanced test). Read the review, then the drafts: the Critic names what is wrong (an invented phrase, too few key phrases) and draft 3 passes all checks. Phase 1's justification for comparison is a template that never quotes the review.")
cache = os.path.join(ROOT, "presentation", "demo_cache.json"); EX = None
if os.path.exists(cache):
    c = json.load(open(cache)); o = c["explanations"][1]; rid = c["demo_ids"][1]
    for l in open(os.path.join(ROOT, "v2_reflective", "e23", "eval.jsonl")):
        r = json.loads(l)
        if r["review_id"] == rid: EX = (r, o, c["phase2_states"][1]["star"]); break
if EX:
    r, o, star = EX
    rect(sl, M, 1.85, 3.5, 4.8, TINT, 0.05)
    text(sl, M + 0.3, 2.05, 2.9, 0.3, f"REVIEW  ·  {r['gt_rating']} stars, system {star} stars", size=11, color=BLUE, bold=True)
    text(sl, M + 0.3, 2.45, 2.9, 4.1, r["review_text"], size=13, color=INK)
    n = len(o["drafts"]); g = 0.35; cw = (CW - 3.85 - g * (n - 1)) / n
    for i, (d, f) in enumerate(zip(o["drafts"], o["feedback"])):
        x = M + 3.85 + i * (cw + g); ok = not f
        hline(sl, x, 1.85, cw, POS if ok else BLUE, 0.04)
        text(sl, x, 2.05, cw, 0.3, f"DRAFT {i + 1}", size=11, color=POS if ok else BLUE, bold=True)
        text(sl, x, 2.45, cw, 2.9, d, size=11.5)
        text(sl, x, 5.55, cw, 1.1, "✓ Passes all checks" if ok else "Critic: " + " ".join(f), size=11.5, color=POS if ok else NEG, bold=True)
else:
    print("WARNING: presentation/demo_cache.json missing; example slide left empty (record the demo first)")

sl = slide("human", "Results", "A person checked the Critic",
           "Human check (human_check.out, LOG 2026-11-20). The author marked 50 random explanations, blind to the Critic: 80% faithful vs the Critic's 92%, agreement 76%. When all agents agreed (34) the author passed every one; when they disagreed (16) only 37.5%, while the Critic passed 93.8%. There the Explainer says things about the system (\"the reviewer gave N stars\", a confidence %) that no check catches. One annotator, 50 items: an indication.",
           sub="50 random explanations, marked by the author without seeing the Critic's verdict")
bars(sl, M, 2.2, 7.4, 4.5, ["Agents agreed (34)", "Agents disagreed (16)"], [("Author: faithful", (100.0, 37.5)), ("Critic: faithful", (91.2, 93.8))], [BLUE, GREY])
stat(sl, 8.9, 2.3, 3.7, "76%", "agreement between the author and the Critic")
text(sl, 8.9, 4.2, 3.7, 2, "When the agents disagree, the Critic misses statements about the system. Next: a check for those.", size=16, color=INK)

sl = slide("robust", "Results", "Stable on messy text, but it needs the title",
           "Robustness (q4_robustness.out, 300 balanced reviews). Share of reviews whose star changes when the text is changed. Typos: 0.7% vs 2.0% (ns). No punctuation or capitals: 9.3% vs 29.7%. Title removed: 15.7% vs 0%, because the Evidence model and DeBERTa read the title. Say the limitation plainly.",
           sub="Share of reviews whose star changes when the text is altered (lower is better)")
bars(sl, M, 2.2, 7.4, 4.5, ["Typos", "No punctuation", "No title"], [("Phase 1", (2.0, 29.7, 0.0)), ("Phase 2", (0.7, 9.3, 15.7))], [GREY, BLUE], maxv=35)
stat(sl, 8.9, 2.3, 3.7, "3×", "more stable on flat, unpunctuated text", color=POS)
stat(sl, 8.9, 4.2, 3.7, "15.7%", "of stars change without the title: a limitation", color=NEG, size=44)

# ======================================================================= why better
sl = slide("better", "Why it is better", "Phase 1 vs Phase 2, side by side",
           "Why it is better: every row is a number already shown. Exact star 54.0 to 67.5 (balanced). On the natural test macro-F1 .488 to .634 is shown instead of exact star, because Phase 1's raw natural exact (53.4) is low partly from the star mix (prior-adjusted 75.5, prior_baseline.out). Negative-sounding happy reviews 35.0 to 67.5. Explanations that quote the review 0% to 75.3% (x1_explain_compare.out). When sure: 66.9% to 86.9% right on about 41% of balanced reviews. Star changes on flat text 29.7% to 9.3%. One thing is worse: without the title, 0% to 15.7%.",
           sub="Same test sets, same reviews")
P = lambda s, c=POS: (s, {"color": c, "bold": True})
rows(sl, [("Exact star, balanced test", "54.0%", P("67.5%")),
          ("Macro-F1, natural test", ".488", P(".634")),
          ("Negative-sounding happy reviews", "35.0%", P("67.5%")),
          ("Explanations that quote the review", "0%", P("75.3%")),
          ("Right when it answers alone (about 41% of reviews)", "66.9%", P("86.9%")),
          ("Stars changed by flat text (lower is better)", "29.7%", P("9.3%")),
          ("Stars changed without the title (lower is better)", "0%", P("15.7%", NEG))],
     [7.0, 2.35, 2.35], y=2.2, head=["", "Phase 1", "Phase 2"], size=16, rh=0.6)

# ======================================================================= conclusion
sl = slide("conclusions", "Conclusion", "What the project shows",
           "Conclusion. (1) The reference method, rebuilt leak-free, gets 54% exact; we found a label leak, a cleaning step that removes negative reviews, and template justifications. (2) Three agents and a Reflector reach 67.5% / 78.0%: +13.5 on balanced reviews; on the natural mix +24.6 raw but +2.5 against Phase 1 adjusted to the star mix, with macro-F1 .49 to .63. (3) The agentic layer makes the output checkable: about 91% (90.5-91.6%) faithful explanations, 0 broken; 93% right when it answers alone; a human check shows where the Critic is still too lenient. Limits: one dataset; needs the title.",
           sub="Trained models decide. LLMs explain. Checks keep them honest.")
cols(sl, [("01", "Rebuilt and explored", "Leak-free, the reference method gets 54%. We found a label leak, a biased filter and template reasons."),
          ("02", "Several agents beat one", "Three agents and a Reflector: 54% to 67.5% on balanced reviews; macro-F1 .49 to .63 on the natural mix."),
          ("03", "Output you can check", "About 91% faithful explanations, none broken; 93% right when it answers alone.")], y=2.8, big=36)

dark_slide("thanks", "Thank you", ["Questions  ·  live demo"],
           "Close. Offer the live demo notebook (presentation/Final_Demo.ipynb): five reviews rated, explained and checked, Phase 1 next to Phase 2.")

# ---- page numbers, notes, slide map
n = len(SLIDES)
for i, (k, t, sl) in enumerate(SLIDES, 1):
    if k not in ("title", "thanks", "approach", "hero"): text(sl, W - M - 1, 7.0, 1, 0.25, str(i), size=10, color=GREY, align=PP_ALIGN.RIGHT)
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
