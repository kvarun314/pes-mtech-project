"""Update the FINAL_FORMATTED_v2 deck: fix stale claims, drop misleading slides, add audit / remaining-error /
reflective-v2 slides, reorder for flow. Original file is never modified.
Run: <venv>/bin/python v2_reflective/build_deck.py"""
import os, re
from pptx import Presentation

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "Agentic_LLaMA3_Presentation_FINAL_FORMATTED_v2.pptx")
DST = os.path.join(ROOT, "Agentic_LLaMA3_Presentation_FINAL_v3.pptx")

# reuse v4's design system (palette, type scale, card/band/pill/table helpers) without running its build
code = open(os.path.join(ROOT, "build_presentation_v4.py")).read()
code = code[code.index("FONT = "):code.index("# --------------------------------------------------------------- slide chrome")]
code = "import os\nfrom pptx.util import Emu, Inches, Pt\nfrom pptx.dml.color import RGBColor\nfrom pptx.enum.shapes import MSO_SHAPE\nfrom pptx.enum.text import MSO_ANCHOR, PP_ALIGN\nfrom pptx.oxml.ns import qn\n" + code
H = {}
exec(code, H)
band_src = open(os.path.join(ROOT, "build_presentation_v4.py")).read()
band_src = band_src[band_src.index("def band("):band_src.index("# ------------------------------------------------------------------- image prep")]
exec(band_src, H)
globals().update({k: v for k, v in H.items() if not k.startswith("__")})

prs = Presentation(SRC)
S = list(prs.slides)  # original order, 0-based
BLANK = S[0].slide_layout


def sh(slide_no, name):
    for s in S[slide_no - 1].shapes:
        if s.name == name:
            return s
    raise KeyError((slide_no, name))


def set_p(p, text):
    runs = p.runs
    runs[0].text = text
    for r in runs[1:]:
        r._r.getparent().remove(r._r)


def edit(slide_no, name, texts):
    """texts: {paragraph_index: new_text} or a str for a single-paragraph shape"""
    s = sh(slide_no, name)
    ps = s.text_frame.paragraphs
    for i, t in ({0: texts} if isinstance(texts, str) else texts).items():
        set_p(ps[i], t)


def cell(slide_no, name, r, c, text):
    t = sh(slide_no, name).table
    set_p(t.cell(r, c).text_frame.paragraphs[0], text)


# ---------------------------------------------------------------- edits to kept slides
edit(1, "Rounded Rectangle 9", "Kaggle + Amazon Reviews 2023")
edit(1, "TextBox 10", "LLaMA-3-8B + LoRA    ·    LanceDB RAG    ·    Dissonance-Triggered Self-Correction    ·    Reflective Orchestration")
edit(1, "TextBox 12", "Varun K    ·    PES MTech Project    ·    October 2026")
sh(1, "TextBox 13").top = Inches(6.35)  # was overlapping TextBox 10
edit(1, "TextBox 13", "PROBLEM  →  BASELINE  →  AGENTIC SYSTEM  →  RESULTS  →  WHAT WE LEARNED  →  REFLECTIVE v2")

edit(4, "TextBox 4", "Phase 1 is trained and evaluated on the Kaggle corpus; Phase 2 runs on Amazon Reviews 2023 (All Beauty in Run 1, Electronics in Run 2).")
edit(4, "TextBox 31", "All Beauty (Run 1) · Electronics (Run 2) · 2,000 each")

edit(5, "TextBox 30", "Shuffle and take the first max_samples (paper Table 1: 2,000; pipeline default: 4,000).")
edit(5, "TextBox 34", "OUTPUT\nClean · balanced (equal per class) · lexically diverse · neutral-boosted · capped at max_samples\n→ fed into LLaMA-3-8B + LoRA (paper Table 1 configuration: r = 4, α = 64, 3 epochs).")

for r, (a, b, c) in enumerate([("Learning rate", "5e-5", "cosine scheduler"), ("Epochs", "3", "paper Table 1"),
                               ("Batch size", "3", "grad accumulation = 4"), ("LoRA rank r", "4", "matches paper"),
                               ("LoRA alpha α", "64", "matches paper"), ("Trainable layers", "last 3 (29–31)", "q_proj, v_proj"),
                               ("Max seq length", "1,024", "left truncation"), ("Quantisation", "4-bit NF4", "bitsandbytes")], 1):
    for ci, v in enumerate((a, b, c)):
        cell(6, "Table 20", r, ci, v)
edit(6, "TextBox 13", "q_proj, v_proj")
edit(6, "TextBox 15", "last 3 transformer layers (29–31)")

edit(7, "TextBox 21", {2: "— Vote logged for diagnostics; not used in the final decision"})

edit(9, "TextBox 4", "Class-balanced 1,000-row Kaggle benchmark: strong on direction and on adjacent stars; the hard part is separating neighbouring ratings.")
t = sh(9, "Table 9").table
for ci, v in enumerate(("METRIC", "OUR RUN (BALANCED, n=1,000)", "PAPER", "READ AS")):
    set_p(t.cell(0, ci).text_frame.paragraphs[0], v)
for r, row in enumerate([("Exact accuracy (5-class)", "60.1%", "92.6%", "chance = 20%"), ("Within ±1 star", "92.0%", "—", "8% off by 2+ stars"),
                         ("MAE (stars)", "0.505", "—", "≈ half a star"), ("3-class accuracy", "79.2%", "—", "1–2 / 3 / 4–5"),
                         ("Binary accuracy", "90.8%", "—", "3★ excluded"), ("Majority-class guess", "20.0%", "—", "balanced eval")], 1):
    for ci, v in enumerate(row):
        cell(9, "Table 9", r, ci, v)
for r in range(1, 7):  # neutral colour for the 'READ AS' column (inherited gap colouring read as negative)
    for run in sh(9, "Table 9").table.cell(r, 3).text_frame.paragraphs[0].runs:
        run.font.color.rgb = BODY
edit(9, "TextBox 10", "The eval set is balanced (200 per star, fixed seed, disjoint from training), so guessing the majority class scores 20%. The paper's 92.6% used K=5 cross-validation and a separate 40K test set, so it is not a like-for-like comparison.")
edit(9, "TextBox 16", "K=5 cross-validation, 128K train / 32K val, 40K held-out test.")
edit(9, "TextBox 18", "Balanced 1,000-row eval, local Q4 inference, paper-aligned LoRA (r = 4, 3 layers).")

edit(10, "TextBox 4", "Run 1 · All Beauty, 2,000 samples · Kaggle-trained adapter under dataset transfer (baseline under-predicted heavily)")

# slide 16: Visual Verifier 'alignment' claim replaced (later audit found its output near-constant)
edit(16, "TextBox 18", "— Analyst alone: 64.9% correct on winning samples\n— RAG Prover context aligns on 91.2% of them\n— Visual Verifier output proved near-constant in the later audit, so it is excluded from the vote")
edit(16, "TextBox 4", "A data-driven interpretation of the +22.6 pp Run 1 gain, with attribution limits and later audit findings noted explicitly.")

edit(18, "TextBox 4", "Phase 1 is a solid paper-aligned baseline; Phase 2 is hardened and net-positive; reflective orchestration targets the remaining errors.")
cell(18, "Table 9", 1, 3, "+22.6 pp (Run 1, transfer setting)")
cell(18, "Table 9", 4, 3, "Per-agent votes logged for every sample")
edit(18, "TextBox 13", "Reflective orchestration (v2)")
edit(18, "TextBox 14", "uncertainty-gated reflection, adjacent-star adjudicator and lesson memory")
edit(18, "TextBox 17", "More Phase 1 training data")
edit(18, "TextBox 18", "scale beyond the 400–4,000-example local experiments to 20–50K examples")
edit(18, "TextBox 21", "Ordinal evaluation")
edit(18, "TextBox 22", "report within-1, MAE and 3-class alongside exact 5-class accuracy")

# ---------------------------------------------------------------- new slides
def new_slide(eyebrow, title, subtitle):
    sl = prs.slides.add_slide(BLANK)
    rect(sl, 0, 0, 13.333, 0.05, fill=ACCENT)
    _, tf = textbox(sl, MARGIN, 0.36, CW, 0.20); para(tf, eyebrow.upper(), size=T_EYEBROW, color=ACCENT, bold=True, first=True, spc=120, line=1.0)
    _, tf = textbox(sl, MARGIN, 0.60, CW, 0.48); para(tf, title, size=T_TITLE, color=INK, bold=True, first=True, line=1.0)
    _, tf = textbox(sl, MARGIN, 1.16, CW, 0.28); para(tf, subtitle, size=T_SUB, color=BODY, first=True, line=1.1)
    rect(sl, MARGIN, 1.60, CW, 0.011, fill=RULE)
    _, tf = textbox(sl, 13.333 - MARGIN - 1.6, 7.06, 1.6, 0.22); para(tf, "00 / 00", size=T_MICRO, color=MUTED, align=PP_ALIGN.RIGHT, first=True, line=1.0)
    return sl


def cards(sl, items, y, h, bar=ACCENT):
    n = len(items); w = (CW - (n - 1) * GUTTER) / n
    for i, (tag, head, lines) in enumerate(items):
        x = MARGIN + i * (w + GUTTER)
        accent_card(sl, x, y, w, h, bar=bar)
        auto_pill(sl, x + 0.22, y + 0.18, 0.24, tag)
        _, tf = textbox(sl, x + 0.22, y + 0.56, w - 0.4, 0.6); para(tf, head, size=T_CARD, color=INK, bold=True, first=True, line=1.1)
        block(sl, x + 0.22, y + 1.22, w - 0.4, h - 1.35, lines, size=T_BODY)


# A. audit
sl_a = new_slide("00 · AUDIT", "What the Audit Found — and Fixed",
                 "A zero-GPU-cost replay of Run 2 plus real local inference isolated four pipeline defects in the Phase 2 graph.")
cards(sl_a, [("FIX 01", "Over-triggered correction", ["— A grounding floor forced reflection whenever RAG grounding was low", "— It fired on 78.6% of rows", "— Now only real vote disagreement triggers it"]),
             ("FIX 02", "Tie-break favoured the weak vote", ["— The RAG vote overruled a correct Analyst on about half of ties", "— RAG alone: 37.5% vs Analyst 54.2%", "— Ties now default to the Analyst"]),
             ("FIX 03", "RAG prompt lacked CoT framing", ["— The Analyst's one-shot + CoT framing was missing", "— RAG standalone accuracy 37.5% → 46.9%"]),
             ("FIX 04", "Visual Verifier was constant", ["— Returned 3★ on 96 of 96 real rows", "— A BLIP caption carries no satisfaction signal", "— Vote excluded; still logged for diagnostics"])],
      1.82, 2.95)
band(sl_a, 5.05, "Effect",
     "Run 2 (before fixes): net −91 on the rows that triggered correction.\nAfter fixes, a 96-row real-inference check: 52.08% → 54.17%, 9 errors fixed and 7 regressed (net +2).\nCaveat: n = 96, so read this as directional, not as a final accuracy claim.")

# B. remaining errors
sl_b = new_slide("00 · WHERE ERRORS REMAIN", "Where the Remaining Errors Live",
                 "Class-balanced Kaggle eval, 1,000 rows: the model gets direction right; it confuses neighbouring star ratings.")
rows = [("Exact 5-class", 60.1), ("3-class (neg / neutral / pos)", 79.2), ("Binary (3★ dropped)", 90.8), ("Within ±1 star", 92.0)]
lw = 5.7
accent_card(sl_b, MARGIN, 1.82, lw, 3.3)
_, tf = textbox(sl_b, MARGIN + 0.26, 1.98, lw - 0.5, 0.3); para(tf, "Coarser views score much higher", size=T_CARD, color=INK, bold=True, first=True)
for i, (lab, v) in enumerate(rows):
    y = 2.5 + i * 0.64
    _, tf = textbox(sl_b, MARGIN + 0.26, y, 2.3, 0.3); para(tf, lab, size=T_SMALL, color=INK, bold=True, first=True)
    rect(sl_b, MARGIN + 2.55, y + 0.02, 2.2, 0.24, fill=BORDER, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.12)
    rect(sl_b, MARGIN + 2.55, y + 0.02, 2.2 * v / 100, 0.24, fill=ACCENT, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.12)
    _, tf = textbox(sl_b, MARGIN + 4.85, y, 0.7, 0.3); para(tf, f"{v}%", size=T_CARD, color=ACCENT_D, bold=True, first=True)
rx = MARGIN + lw + GUTTER; rw = CW - lw - GUTTER
accent_card(sl_b, rx, 1.82, rw, 3.3, bar=ACCENT_D)
_, tf = textbox(sl_b, rx + 0.26, 1.98, rw - 0.5, 0.3); para(tf, "Per-star results, balanced eval (200 per star)", size=T_CARD, color=INK, bold=True, first=True)
make_table(sl_b, rx + 0.26, 2.4, rw - 0.5, [0.7, 1.0, 0.9, 0.9, rw - 0.5 - 3.5], None, ["Star", "Precision", "Recall", "F1", "Most common wrong guess"],
           [[("1★", INK, True), ("0.58", BODY, False), ("0.86", BODY, False), ("0.70", BODY, False), ("3★ (7%)", BODY, False)],
            [("2★", INK, True), ("0.52", BODY, False), ("0.34", NEG, True), ("0.41", NEG, True), ("1★ (46%)", NEG, True)],
            [("3★", INK, True), ("0.55", BODY, False), ("0.59", BODY, False), ("0.57", BODY, False), ("2★ (22%)", BODY, False)],
            [("4★", INK, True), ("0.66", BODY, False), ("0.47", BODY, False), ("0.55", BODY, False), ("5★ (28%)", BODY, False)],
            [("5★", INK, True), ("0.69", BODY, False), ("0.75", BODY, False), ("0.72", BODY, False), ("4★ (18%)", BODY, False)]],
           row_h=0.40, head_h=0.34, aligns=[PP_ALIGN.LEFT, PP_ALIGN.RIGHT, PP_ALIGN.RIGHT, PP_ALIGN.RIGHT, PP_ALIGN.LEFT])
_, tf = textbox(sl_b, rx + 0.26, 4.8, rw - 0.5, 0.25)
para(tf, "Macro-F1 0.588. Same pattern on the 4,000-row paper-protocol set (2★ recall 0.36, macro-F1 0.548).", size=T_MICRO, color=MUTED, first=True, line=1.15)
band(sl_b, 5.35, "Why it matters",
     "Errors are adjacent-star swaps, mostly 2★→1★, 3★→2★ and 4★→5★; moving decision thresholds only trades one neighbour error for another.\nIn Phase 2 as first built, the Critic's override reached only 0.4% of rows, so reflection barely touched these cases. That is the headroom the redesign targets.", h=1.3)

# C. reflective v2
sl_c = new_slide("00 · REFLECTIVE ORCHESTRATION v2", "Reflective Orchestration: Reflection Where It Counts",
                 "Self-reflection becomes the core mechanism and is aimed at uncertain, adjacent-star cases instead of vote disagreement.")
steps = [("01", "Analyst", "Rating plus the full 5-way probability distribution"), ("02", "Uncertainty gate", "Small top-2 margin → reflect; confident → release"),
         ("03", "Reflector", "Critiques the draft against the 1–5 star rubric; weighs evidence for the two neighbouring ratings"),
         ("04", "Adjudicator", "Binary decision between the two candidate ratings, e.g. 3★ vs 4★"), ("05", "Loop control", "At most 2 revision rounds; stop when Reflector and Analyst agree")]
w = (CW - 4 * 0.2) / 5
for i, (n, head, body) in enumerate(steps):
    x = MARGIN + i * (w + 0.2)
    card(sl_c, x, 1.82, w, 2.25)
    auto_pill(sl_c, x + 0.18, 1.98, 0.24, n)
    _, tf = textbox(sl_c, x + 0.18, 2.36, w - 0.3, 0.3); para(tf, head, size=T_CARD, color=INK, bold=True, first=True)
    block(sl_c, x + 0.18, 2.74, w - 0.3, 1.25, [body], size=T_SMALL)
accent_card(sl_c, MARGIN, 4.25, 5.95, 1.35)
_, tf = textbox(sl_c, MARGIN + 0.25, 4.4, 5.5, 0.3); para(tf, "Reflection memory (Reflexion-style)", size=T_CARD, color=INK, bold=True, first=True)
block(sl_c, MARGIN + 0.25, 4.78, 5.5, 0.8, ["Lessons from corrected mistakes are stored and retrieved as guidance, e.g. “‘okay but…’ reviews are 3★, not 4★”."], size=T_BODY)
accent_card(sl_c, MARGIN + 6.18, 4.25, CW - 6.18, 1.35)
_, tf = textbox(sl_c, MARGIN + 6.43, 4.4, CW - 6.6, 0.3); para(tf, "How it will be evaluated", size=T_CARD, color=INK, bold=True, first=True)
block(sl_c, MARGIN + 6.43, 4.78, CW - 6.6, 0.8, ["Ablation: no reflection vs reflection vs reflection + memory. Report exact, within-1, MAE, QWK and 3-class on the same rows."], size=T_BODY)
band(sl_c, 5.85, "Status", "Design locked. A prototype on a 500-row dev split is the next step, and no v2 results are claimed yet. Fitting happens on dev data only; the 1,000-row eval set is touched once, for the final read-out.", h=1.0)


# D. reproducing the paper (paper-protocol comparison + caveats)
sl_d = new_slide("00 · REPRODUCING THE PAPER", "Reproducing the Paper's Claims",
                 "Same 4,000 random held-out Kaggle rows, natural rating distribution, scored the way the paper describes (1–2 / 3 / 4–5 = negative / neutral / positive).")
lw = 7.55
accent_card(sl_d, MARGIN, 1.82, lw, 4.2)
_, tf = textbox(sl_d, MARGIN + 0.26, 1.98, lw - 0.5, 0.3); para(tf, "Same test rows, every model", size=T_CARD, color=INK, bold=True, first=True)
make_table(sl_d, MARGIN + 0.26, 2.38, lw - 0.5, [2.75, 1.4, 1.35, 1.55], None, ["Model", "5-class exact", "Polarity", "Paper reports"],
           [[("Always guess the majority", MUTED, False), ("63.8%", MUTED, False), ("78.6%", MUTED, False), ("—", MUTED, False)],
            [("Decision Tree", INK, True), ("58.5%", BODY, False), ("75.3%", BODY, False), ("78.6%", BODY, False)],
            [("SVM", INK, True), ("70.0%", BODY, False), ("84.5%", BODY, False), ("87.2%", BODY, False)],
            [("Multinomial NB", INK, True), ("67.4%", BODY, False), ("83.1%", BODY, False), ("87.6%", BODY, False)],
            [("XLNet", INK, True), ("not re-run", MUTED, False), ("not re-run", MUTED, False), ("91.6%", BODY, False)],
            [("LLaMA-3-8B + LoRA (ours)", ACCENT_D, True), ("70.7%", ACCENT_D, True), ("85.0%", ACCENT_D, True), ("92.6%", BODY, False)],
            [("  + class-balance correction", BODY, False), ("79.9%", BODY, False), ("92.0%", BODY, False), ("—", MUTED, False)]],
           row_h=0.40, head_h=0.34, aligns=[PP_ALIGN.LEFT, PP_ALIGN.RIGHT, PP_ALIGN.RIGHT, PP_ALIGN.RIGHT])
_, tf = textbox(sl_d, MARGIN + 0.26, 5.66, lw - 0.5, 0.3)
para(tf, "Classical models: 16K natural-distribution training rows, paper Table 3 configs; polarity = 5-class prediction regrouped.", size=T_MICRO, color=MUTED, first=True, line=1.15)
rx = MARGIN + lw + GUTTER; rw = CW - lw - GUTTER
accent_card(sl_d, rx, 1.82, rw, 4.2, bar=ACCENT_D)
_, tf = textbox(sl_d, rx + 0.26, 1.98, rw - 0.5, 0.3); para(tf, "How to read this — caveats", size=T_CARD, color=INK, bold=True, first=True)
block(sl_d, rx + 0.26, 2.38, rw - 0.48, 3.4,
      ["— The paper's 92.6% looks polarity-style: its baselines land near our polarity scores, not our 5-class ones",
       "— Training scale differs: the paper used up to 128K rows per fold; our classical runs used 16K and LLaMA the paper's small balanced set",
       "— LLaMA ran on a community 4-bit base; the adapter was trained on fp16 weights",
       "— The prompt's one-shot example is chosen using the true rating (mild label leak)",
       "— The adapter's training rows were not recorded, so test overlap is excluded only statistically",
       "— The correction row rebalances classes (fit by cross-validation); on a balanced set it lowers exact accuracy, 60.2% → 57.9%"],
      size=T_SMALL, gap=5)
band(sl_d, 6.18, "Takeaway", "The method reproduces and clears the trivial baselines. The headline 92.6% does not reproduce as stated: under a paper-like protocol LLaMA is level with SVM (85.0% vs 84.5% polarity), and the rest of the gap is protocol and training scale.", h=0.88)


# E. data-scale learning curve (native, editable charts)
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION, XL_MARKER_STYLE
sl_e = new_slide("00 · DATA SCALE", "Data Scale: Why the Paper's Numbers Are Higher",
                 "Same 4,000 test rows. Classical models trained on 1K to 128K natural-distribution rows; LLaMA marked at its 2–4K training size.")
cats = ["1K", "4K", "16K", "64K", "128K"]
def chart(sl, x, title, series, lo, hi):
    card(sl, x, 1.82, 5.95, 3.45)
    _, tf = textbox(sl, x + 0.22, 1.95, 5.5, 0.3); para(tf, title, size=T_CARD, color=INK, bold=True, first=True)
    cd = CategoryChartData(); cd.categories = cats
    for name, vals, _c in series: cd.add_series(name, vals)
    gf = sl.shapes.add_chart(XL_CHART_TYPE.LINE_MARKERS, Inches(x + 0.1), Inches(2.3), Inches(5.75), Inches(2.9), cd)
    ch = gf.chart; ch.has_legend = True; ch.legend.position = XL_LEGEND_POSITION.BOTTOM; ch.legend.include_in_layout = False
    ch.legend.font.size = Pt(9); ch.legend.font.name = FONT
    va = ch.value_axis; va.minimum_scale = lo; va.maximum_scale = hi; va.has_major_gridlines = True
    va.major_gridlines.format.line.color.rgb = BORDER; va.tick_labels.font.size = Pt(9); va.tick_labels.number_format = '0"%"'; va.tick_labels.number_format_is_linked = False
    ch.category_axis.tick_labels.font.size = Pt(9)
    for sr, (_n, _v, col) in zip(ch.plots[0].series, series):
        sr.format.line.color.rgb = col; sr.format.line.width = Pt(2.25); sr.smooth = False
        sr.marker.style = XL_MARKER_STYLE.CIRCLE; sr.marker.size = 7
        sr.marker.format.fill.solid(); sr.marker.format.fill.fore_color.rgb = col; sr.marker.format.line.color.rgb = col
        sr.data_labels.show_value = True; sr.data_labels.font.size = Pt(8); sr.data_labels.number_format = '0.0'; sr.data_labels.number_format_is_linked = False
LR_C, NB_C, LL_C = RGBColor(0x2C, 0x5A, 0xA0), RGBColor(0x87, 0x92, 0xA0), RGBColor(0xC2, 0x5B, 0x1E)
chart(sl_e, MARGIN, "5-class exact accuracy (%)",
      [("TF-IDF logistic regression", [64.6, 68.1, 72.6, 76.4, 78.7], LR_C), ("Naive Bayes", [63.0, 65.1, 66.6, 68.4, 69.2], NB_C),
       ("LLaMA + LoRA (2–4K rows)", [None, 70.7, None, None, None], LL_C)], 55, 85)
chart(sl_e, MARGIN + 6.18, "Polarity accuracy (%)",
      [("TF-IDF logistic regression", [79.2, 82.3, 86.1, 88.9, 90.4], LR_C), ("Naive Bayes", [78.3, 80.9, 81.9, 83.1, 83.6], NB_C),
       ("LLaMA + LoRA (2–4K rows)", [None, 85.0, None, None, None], LL_C)], 70, 95)
band(sl_e, 5.5, "Reading it",
     "Each 4× more training data adds about 3–4 points, with no plateau by 128K; at that scale a plain TF-IDF model reaches 90.4% polarity, close to the paper's 92.6%. LLaMA at 2–4K rows already beats the classical models at the same size.\nCaveat: the classical models train on the natural distribution (64% five-star), so part of their gain is learning that prior; the LLaMA adapter was trained on balanced data.", h=1.3)

# F. appendix: run card
sl_f = new_slide("00 · APPENDIX", "Appendix: Phase 1 Run Card",
                 "Every setting behind the LLaMA row, so the result can be reproduced or challenged.")
cards(sl_f, [("MODEL", "Adapter and training", ["— Base: Meta-Llama-3-8B", "— LoRA r = 4, α = 64, dropout 0", "— Layers 29–31, q_proj and v_proj", "— lr 5e-5, 3 epochs, batch 3 × 4, fp16, cosine, seed 42", "— Training rows not recorded"]),
             ("INFERENCE", "How predictions were made", ["— llama.cpp, community Q4_K_M base + converted adapter", "— Temperature 0, context 8,192", "— Prompt: rubric + CoT phrase + one-shot example", "— Prediction = argmax of P(1..5) at the first digit token", "— Matched generated predictions on 999 / 1,000 rows"]),
             ("TEST SET", "What it was scored on", ["— 4,000 random Kaggle rows, seed 2024", "— Natural distribution: 8.8 / 5.6 / 7.1 / 14.8 / 63.8 %", "— Disjoint from the balanced eval and dev sets", "— Also a balanced 1,000-row eval (200 per star)", "— Polarity: 1–2 / 3 / 4–5"]),
             ("CAVEATS", "Known limits", ["— Q4 base vs fp16 training weights", "— One-shot example chosen using the true rating", "— Test/training overlap excluded only statistically", "— Paper's exact protocol is ambiguous", "— Files: v2_reflective/PHASE1_RUN_CARD.md"])],
      1.82, 3.7, bar=ACCENT_D)

# ---------------------------------------------------------------- reorder / drop
ids = prs.slides._sldIdLst
els = list(ids)
orig = els[:18]; a, b, c, d, e_, f_ = els[18], els[19], els[20], els[21], els[22], els[23]
order = [orig[i - 1] for i in (1, 2, 3, 4, 5, 6, 9)] + [d, e_] + [orig[i - 1] for i in (7, 8, 10, 12, 13, 14, 16, 17)] + [a, b, c, orig[17], f_]
drop = [orig[10], orig[14]]  # dashboard (dup of slide 10), agent-alignment chart (97.7% claim)
for e in els:
    ids.remove(e)
for e in order:
    ids.append(e)
for e in drop:
    prs.part.drop_rel(e.rId)
n = len(order)
# renumber eyebrows + page counters
for i, sl in enumerate(prs.slides, 1):
    for s in sl.shapes:
        if not s.has_text_frame:
            continue
        txt = s.text_frame.text
        if re.fullmatch(r"\d\d / \d\d", txt):
            set_p(s.text_frame.paragraphs[0], f"{i:02d} / {n}")
        elif i > 1 and re.match(r"\d\d · ", txt) and s.top < 600000:
            set_p(s.text_frame.paragraphs[0], f"{i - 1:02d} · " + txt.split(" · ", 1)[1])
prs.save(DST)
print("saved", DST, n)
