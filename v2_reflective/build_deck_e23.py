"""Deck update (Oct 2026): adds the leak-free Electronics-2023 results, Phase 2 focus and the roadmap to the aligned deck.
Original deck is never modified. Every number is copied from v2_reflective/LOG.md / e23/*.out.
Run: /opt/homebrew/bin/python3 v2_reflective/build_deck_e23.py   (system python has python-pptx; .venv does not)"""
import os, re
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTER = os.path.dirname(ROOT)
SRC = os.path.join(OUTER, "Agentic_LLaMA3_Presentation_FINAL_v3_aligned.pptx")
DST = os.path.join(OUTER, "Agentic_LLaMA3_Presentation_v5_electronics.pptx")
code = open(os.path.join(ROOT, "build_presentation_v4.py")).read()
code = code[code.index("FONT = "):code.index("# --------------------------------------------------------------- slide chrome")]
code = "import os\nfrom pptx.util import Emu, Inches, Pt\nfrom pptx.dml.color import RGBColor\nfrom pptx.enum.shapes import MSO_SHAPE\nfrom pptx.enum.text import MSO_ANCHOR, PP_ALIGN\nfrom pptx.oxml.ns import qn\n" + code
H = {}; exec(code, H)
bs = open(os.path.join(ROOT, "build_presentation_v4.py")).read()
exec(bs[bs.index("def band("):bs.index("# ------------------------------------------------------------------- image prep")], H)
globals().update({k: v for k, v in H.items() if not k.startswith("__")})

prs = Presentation(SRC); S = list(prs.slides); BLANK = S[0].slide_layout

def set_p(p, text):
    runs = p.runs; runs[0].text = text
    for r in runs[1:]: r._r.getparent().remove(r._r)
def edit_by_text(slide, startswith, new):
    for s in slide.shapes:
        if s.has_text_frame and s.text_frame.text.startswith(startswith):
            set_p(s.text_frame.paragraphs[0], new); return
    raise KeyError(startswith)

# ---- edits to kept slides
edit_by_text(S[0], "Kaggle + Amazon", "Amazon Reviews 2023 · Electronics")
edit_by_text(S[0], "LLaMA-3-8B + LoRA", "LLaMA-3-8B + LoRA    ·    Evidence agent    ·    Reflector    ·    Leak-free evaluation")
edit_by_text(S[0], "Varun K", "Varun K    ·    PES MTech Project    ·    October 2026 update")
edit_by_text(S[0], "PROBLEM", "PROBLEM  →  METHOD  →  LEAK AUDIT  →  ELECTRONICS RESULTS  →  PHASE 2 FOCUS  →  ROADMAP  →  HISTORICAL")
edit_by_text(S[3], "Phase 1 is trained", "Main results now use Amazon Reviews 2023 Electronics only; the Kaggle corpus is the original paper's data (reproduction and appendix).")

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
        accent_card(sl, x, y, w, h, bar=bar); auto_pill(sl, x + 0.22, y + 0.18, 0.24, tag)
        _, tf = textbox(sl, x + 0.22, y + 0.56, w - 0.4, 0.6); para(tf, head, size=T_CARD, color=INK, bold=True, first=True, line=1.1)
        block(sl, x + 0.22, y + 1.22, w - 0.4, h - 1.35, lines, size=T_BODY)
L, R, C = PP_ALIGN.LEFT, PP_ALIGN.RIGHT, None

# N1 leak audit
n1 = new_slide("LEAK AUDIT", "The One-Shot Label Leak Inflated Earlier Numbers",
               "300 balanced Kaggle eval rows, same adapter, paired bootstrap against the reported protocol (A).")
lw = 7.4
accent_card(n1, MARGIN, 1.82, lw, 3.55)
make_table(n1, MARGIN + 0.26, 2.05, lw - 0.5, [3.9, 1.2, 1.8], None, ["Prompt variant", "Exact", "Δ vs A [95% CI]"],
           [[("A  one-shot chosen with rating ≠ truth (as reported)", INK, True), ("59.7%", BODY, False), ("—", MUTED, False)],
            [("B  same prompt, random one-shot", INK, True), ("52.0%", NEG, True), ("−7.7 [−13.3, −2.0]", NEG, True)],
            [("C  training-format prompt, random one-shot", INK, True), ("52.3%", BODY, False), ("−7.3 [−13.7, −0.7]", BODY, False)],
            [("D  zero-shot, no example (leak-free)", ACCENT_D, True), ("58.3%", ACCENT_D, True), ("−1.3 [−7.0, +4.3]", BODY, False)]],
           row_h=0.52, head_h=0.36, aligns=[L, R, R])
rx = MARGIN + lw + GUTTER; rw = CW - lw - GUTTER
accent_card(n1, rx, 1.82, rw, 3.55, bar=ACCENT_D)
_, tf = textbox(n1, rx + 0.26, 1.98, rw - 0.5, 0.3); para(tf, "Consequences", size=T_CARD, color=INK, bold=True, first=True)
block(n1, rx + 0.26, 2.4, rw - 0.48, 2.9, ["— Excluding the true star from the example pool anchors the model away from one wrong answer: a leak of about 8 points",
     "— The 70.7% / 60.1% Kaggle headlines are withdrawn; all new results use zero-shot",
     "— The recalibrator gain (+5.0 on leaky probabilities) became +0.1 [−2.3, +2.5] on clean probabilities and is retracted",
     "— Lesson: post-hoc recalibration adds no information; a real gain needs an independent second signal"], size=T_SMALL, gap=5)
band(n1, 5.6, "Source", "v2_reflective/LOG.md, entries 2026-10-11 (Phase 1 close-out; retraction). Honest zero-shot Kaggle baseline: 59.2% exact on 1,000 balanced rows.", h=0.9)

# N2 electronics protocol
n2 = new_slide("ELECTRONICS PROTOCOL", "Amazon Reviews 2023 Electronics: Frozen, Disjoint Splits",
               "One dataset for all reported results. Splits drawn once (seed 2023), disjoint by review id and exact text.")
cards(n2, [("SPLITS", "Four frozen sets", ["— evidence-train 100,000 balanced (20K per star)", "— dev 500 balanced (all choices made here)", "— eval 1,000 balanced (read once)", "— natural_test 2,000 natural skew: 1★ 189 · 2★ 83 · 3★ 150 · 4★ 277 · 5★ 1,301"]),
           ("PHASE 1", "Analyst: LLaMA-3-8B + LoRA", ["— Retrained on Electronics: 2,000 rows, paper Table 1 hyperparameters", "— MLX QLoRA, ≈3 epochs (400 updates, batch 12), zero-shot prompt", "— Prediction = argmax of first-digit probabilities over 1–5"]),
           ("ARM CHOICE", "Dev only (500 rows, ±4.4)", ["— Arm P, the paper's pipeline (TextBlob + stratified + VGST + neutral ×2): 52.2%", "— Arm R, raw balanced rows: 55.2%", "— R chosen; the 3.0-point gap is within noise"]),
           ("EVIDENCE", "Second, independent signal", ["— TF-IDF (1–2-gram, 300K features) + logistic regression", "— Trained on the 100K raw balanced Electronics rows", "— CPU only, no filtering"])],
      1.82, 3.6)
band(n2, 5.65, "Note", "The original paper's adapter was trained on Kaggle reviews; Kaggle is now reproduction and appendix only. Test sets are never filtered.", h=0.9)

# N3 Phase 2 results
n3 = new_slide("PHASE 2 RESULTS", "Analyst + Evidence Stack on Electronics",
               "Reflector = logistic regression on both agents' log-probabilities, fit on 500 balanced dev rows, read out once per split.")
lw = 7.3
accent_card(n3, MARGIN, 1.82, lw, 3.3)
make_table(n3, MARGIN + 0.26, 2.05, lw - 0.5, [2.2, 1.45, 1.45, 1.4], None, ["Split", "Analyst (arm R)", "Evidence", "Stack"],
           [[("eval · 1,000 balanced", INK, True), ("54.0%", BODY, False), ("62.4%", BODY, False), ("63.9%", ACCENT_D, True)],
            [("natural_test · 2,000", INK, True), ("53.4%", BODY, False), ("69.3%", BODY, False), ("73.4%", ACCENT_D, True)],
            [("eval macro-F1", MUTED, False), ("0.543", MUTED, False), ("0.622", MUTED, False), ("0.638", MUTED, False)],
            [("natural macro-F1", MUTED, False), ("0.488", MUTED, False), ("0.541", MUTED, False), ("0.577", MUTED, False)],
            [("within ±1 (eval / natural)", MUTED, False), ("93.7 / 94.3", MUTED, False), ("93.1 / 93.7", MUTED, False), ("95.5 / 96.2", MUTED, False)]],
           row_h=0.46, head_h=0.36, aligns=[L, R, R, R])
rx = MARGIN + lw + GUTTER; rw = CW - lw - GUTTER
accent_card(n3, rx, 1.82, rw, 3.3, bar=ACCENT_D)
_, tf = textbox(n3, rx + 0.26, 1.98, rw - 0.5, 0.3); para(tf, "Paired comparisons (stack, exact)", size=T_CARD, color=INK, bold=True, first=True)
block(n3, rx + 0.26, 2.4, rw - 0.48, 2.7, ["— vs Analyst: +9.9 [+6.7, +13.0] on eval; +20.0 [+17.8, +22.2] on natural_test (both p < 1e-9)",
     "— vs Evidence on eval: +1.5 [−1.2, +4.1], p = 0.30, not significant",
     "— vs Evidence on natural_test: +4.0 [+2.4, +5.8], p = 5.5e-6, but the stacker was fit on balanced dev, so part is a class-prior effect",
     "— Agents disagree on 44% of rows; there the stack is right 55% (eval) / 67% (natural)"], size=T_SMALL, gap=5)
band(n3, 5.35, "Honest read", "A 100K-row bag-of-words model beats the 2K-row LLaMA adapter (62.4% vs 54.0% on eval). Most of the gain over Phase 1 comes from the Evidence agent; LLaMA adds a small increment on top. Source: e23/phase2_e23.out.", h=1.1)

# N4 preprocessing ablation (native chart)
n4 = new_slide("PREPROCESSING ABLATION", "TextBlob Filtering Hurts; VGST Adds Nothing",
               "Evidence model, validated on 10K held-out raw balanced Electronics rows (±1.0 pt at 95%).")
card(n4, MARGIN, 1.82, 7.6, 4.1)
_, tf = textbox(n4, MARGIN + 0.22, 1.95, 7.0, 0.3); para(tf, "Validation accuracy (%) by training-set variant", size=T_CARD, color=INK, bold=True, first=True)
cd = CategoryChartData(); cd.categories = ["A raw 90K", "G raw bal. 33K", "B DQC 68K", "C random 10K", "F DQC→bal. 33K", "D DQC+VGST 10K", "E DQC+rand 10K"]
cd.add_series("Accuracy", [51.2, 50.1, 47.7, 47.6, 47.6, 44.5, 44.3])
gf = n4.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, Inches(MARGIN + 0.1), Inches(2.3), Inches(7.4), Inches(3.55), cd); ch = gf.chart
ch.has_legend = False; ch.has_title = False; ch.value_axis.minimum_scale = 40; ch.value_axis.maximum_scale = 54; ch.value_axis.tick_labels.font.size = Pt(9)
ch.value_axis.major_gridlines.format.line.color.rgb = BORDER; ch.category_axis.tick_labels.font.size = Pt(9); ch.category_axis.reverse_order = True
sr = ch.plots[0].series[0]; sr.format.fill.solid(); sr.format.fill.fore_color.rgb = ACCENT
sr.data_labels.show_value = True; sr.data_labels.font.size = Pt(9); sr.data_labels.number_format = '0.0'; sr.data_labels.number_format_is_linked = False
rx = MARGIN + 7.6 + GUTTER; rw = CW - 7.6 - GUTTER
accent_card(n4, rx, 1.82, rw, 4.1, bar=ACCENT_D)
_, tf = textbox(n4, rx + 0.26, 1.98, rw - 0.5, 0.3); para(tf, "Why", size=T_CARD, color=INK, bold=True, first=True)
block(n4, rx + 0.26, 2.4, rw - 0.48, 3.4, ["— TextBlob DQC costs −2.5 points at equal size, −3.5 at full size",
     "— It deletes 52% of 1★ and 67% of 2★ reviews (negative reviews contain positive words) and drops mixed reviews",
     "— VGST adds +0.2 over random (noise) and prefers very long reviews (156 vs 64 words)",
     "— Decision: Phase 2 trains on raw balanced data; DQC and VGST are reported as ablations; test sets are never filtered",
     "— Same direction for the LLaMA adapter on dev: arm P 52.2% vs arm R 55.2%"], size=T_SMALL, gap=5)
band(n4, 6.1, "Source", "e23/ablate_preproc.py, e23/ablate_preproc.log; LOG.md 2026-10-11 and 2026-10-13.", h=0.7)

# N5 hallucination study
n5 = new_slide("HALLUCINATION STUDY", "What the Electronics Adapter Actually Generates",
               "150 balanced eval rows, 120-token greedy generation under the training-format zero-shot prompt.")
cards(n5, [("COUNTS", "150 generations", ["— 0 unparseable", "— 80 wrong star (53%)", "— 20 severe misses (2 or more stars)", "— 0 run on into an invented review"]),
           ("PATTERN", "A canned template", ["— Output is one fixed rubric sentence tied to the digit, repeated", "— e.g. “… high level of dissatisfaction … (rating 1).!” looped", "— No review-specific reasoning"]),
           ("COMPARE", "Kaggle adapter, earlier study", ["— 41% wrong, 16 severe misses of 150, 7 run-ons", "— Same finding: rationale is a template, not evidence", "— The LoRA learned the label, not an explanation"])],
      1.82, 3.0)
band(n5, 5.05, "Implication", "Free-text reasoning from this adapter is not a trustworthy Phase 2 signal; the digit probabilities are. That is why Phase 2 combines probabilities with an independent evidence model instead of asking the LLM to explain itself. Source: e23/halluc_p1.out.", h=1.2)

# N6 Phase 2 focus
n6 = new_slide("PHASE 2 FOCUS", "Making the Agent Layer Earn Its Name",
               "The current Reflector is a logistic regression over two probability vectors. Each candidate below has a kill criterion on dev.")
cards(n6, [("P2-1", "Disagreement routing", ["— Route only the 44% of rows where Analyst and Evidence disagree", "— Agree-rows already 71–78% right", "— Reports LLM-call savings and accuracy on routed rows"]),
           ("P2-2", "Evidence-grounded adjudicator", ["— Gate 0 rerun on Electronics dev", "— LLM sees the two candidate stars plus the Evidence model's top n-grams as cited evidence", "— Old adjudicator failed Gate 0 (52–57% vs 56–67%)"]),
           ("P2-3", "Better Reflector features", ["— Margin, entropy, disagreement flag", "— Prior adjustment, abstain option", "— Cheap, expected small gain"]),
           ("P2-4", "Ablation ladder", ["— Analyst · Evidence · + stacker · + routing · + adjudicator (only if P2-2 passes)", "— Paired CIs on eval and natural_test, each read once"])],
      1.82, 3.4)
band(n6, 5.45, "Also open", "Natural-prior stacker (removes the class-prior confound on natural_test); base-LLaMA-without-LoRA and a small fine-tuned encoder as baselines; variance of the +1.5. Visual Verifier stays excluded (constant output across 3 vision models) and is reported as a negative result. Plan: v2_reflective/PROJECT_ROADMAP.md.", h=1.25)

# N7 roadmap
n7 = new_slide("ROADMAP", "Plan to the Nov 21–22 Deadline",
               "Weekly checkpoints, each logged in LOG.md and progress/week_NN.md. Numbers freeze Nov 1. Last days are buffer only.")
accent_card(n7, MARGIN, 1.82, CW, 4.2)
make_table(n7, MARGIN + 0.26, 2.0, CW - 0.5, [1.0, 1.7, 6.2, CW - 0.5 - 8.9], None, ["Week", "Dates", "Work", "Checkpoint"],
           [[("0", INK, True), ("Oct 17–18", BODY, False), ("Hallucination study, plan, roadmap, results scaffold", BODY, False), ("Done", POS, True)],
            [("1", INK, True), ("Oct 19–25", BODY, False), ("Natural-prior stacker; routing analysis; Reflector features; base-LLaMA scoring", BODY, False), ("Confound resolved", BODY, False)],
            [("2", INK, True), ("Oct 26–Nov 1", BODY, False), ("Adjudicator Gate 0; encoder baseline; variance; angle decision; numbers freeze", BODY, False), ("Claim fixed", BODY, False)],
            [("3", INK, True), ("Nov 2–8", BODY, False), ("Paper prose (author-written); figures; reproducibility pass", BODY, False), ("Draft 1, similarity check", BODY, False)],
            [("4", INK, True), ("Nov 9–15", BODY, False), ("Supervisor fixes; thesis sections; archive; deck", BODY, False), ("Draft 2, similarity check", BODY, False)],
            [("5", INK, True), ("Nov 16–20", BODY, False), ("Buffer: proofreading, final checks, submission", BODY, False), ("Submitted by Nov 20", BODY, False)]],
           row_h=0.5, head_h=0.36, aligns=[L, L, L, L])
band(n7, 6.2, "Integrity", "Plagiarism and AI-content each under 5%: the author writes the prose; analysis supplies facts, tables and figures; similarity checks on every full draft.", h=0.75)

# ---- reorder: 1-9, new, then historical (old results/audit/v2 plan), conclusion dropped
ids = prs.slides._sldIdLst; els = list(ids); orig = els[:19]; new = els[19:]
order = [orig[i - 1] for i in range(1, 10)] + new + [orig[i - 1] for i in range(10, 19)]
drop = [orig[18]]
for e in els: ids.remove(e)
for e in order: ids.append(e)
for e in drop: prs.part.drop_rel(e.rId)
n = len(order); first_hist = 9 + len(new) + 1
for i, sl in enumerate(prs.slides, 1):
    for s in sl.shapes:
        if not s.has_text_frame: continue
        txt = s.text_frame.text
        if re.fullmatch(r"\d\d / \d\d", txt): set_p(s.text_frame.paragraphs[0], f"{i:02d} / {n}")
        elif i > 1 and re.match(r"\d\d · ", txt) and s.top < 600000:
            tail = txt.split(" · ", 1)[1]
            set_p(s.text_frame.paragraphs[0], (f"HISTORICAL (SUPERSEDED) · {tail}" if i >= first_hist else f"{i - 1:02d} · {tail}"))
        elif i > 1 and i < first_hist and i > 9 and s.top < 600000 and not re.match(r"\d\d · ", txt) and txt.isupper():
            pass
prs.save(DST); print("saved", DST, n)
