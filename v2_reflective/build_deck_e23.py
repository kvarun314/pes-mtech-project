"""Deck update (Oct 2026): adds the leak-free Electronics-2023 results, Phase 2 focus and the roadmap to the aligned deck.
Original deck is never modified. Every number is copied from v2_reflective/LOG.md / e23/*.out.
Run: /opt/homebrew/bin/python3 v2_reflective/build_deck_e23.py   (system python has python-pptx; .venv does not)"""
import os, re
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTER = os.path.dirname(ROOT)
SRC = os.path.join(OUTER, "old_decks", "Agentic_LLaMA3_Presentation_FINAL_v3_aligned.pptx")  # template only, never modified
DST = os.path.join(OUTER, "Agentic_LLaMA3_Presentation.pptx")  # the one current deck
code = open(os.path.join(ROOT, "v2_reflective", "build_presentation_v4.py")).read()
code = code[code.index("FONT = "):code.index("# --------------------------------------------------------------- slide chrome")]
code = "import os\nfrom pptx.util import Emu, Inches, Pt\nfrom pptx.dml.color import RGBColor\nfrom pptx.enum.shapes import MSO_SHAPE\nfrom pptx.enum.text import MSO_ANCHOR, PP_ALIGN\nfrom pptx.oxml.ns import qn\n" + code
H = {}; exec(code, H)
bs = open(os.path.join(ROOT, "v2_reflective", "build_presentation_v4.py")).read()
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
edit_by_text(S[0], "LLaMA-3-8B + LoRA", "Frozen LLaMA-3-8B probe    ·    Evidence agent    ·    Reflector    ·    Leak-free evaluation")
edit_by_text(S[0], "Varun K", "Varun K    ·    PES MTech Project    ·    October 25, 2026 update")
edit_by_text(S[0], "PROBLEM", "PROBLEM  →  METHOD  →  UPDATE  →  AUDITS  →  EXPERIMENTS  →  MAIN RESULT  →  ROBUSTNESS  →  NEXT: PHASE 2 v3  →  ROADMAP")
edit_by_text(S[3], "Phase 1 is trained", "Main results now use Amazon Reviews 2023 Electronics only; the Kaggle corpus is the original paper's data (reproduction and appendix).")
edit_by_text(S[3], "All Beauty (Run 1)", "Electronics only: train ~497K · dev 500 + dev2 2,000 · eval 1,000 · natural_test 2,000")
edit_by_text(S[3], "text · metadata", "review text and star rating (images tested, excluded)")
edit_by_text(S[3], "Enables —", "Used — review text only; splits frozen and disjoint (slide 12)")
edit_by_text(S[3], "Review data:", "All Beauty (Run 1) and image/metadata runs are historical (slides 25–33).")
for p in [p for s_ in S[3].shapes if s_.has_text_frame and s_.text_frame.text.startswith("All Beauty (Run 1) and") for p in s_.text_frame.paragraphs][1:]: p._p.getparent().remove(p._p)
edit_by_text(S[6], "Class-balanced 1,000-row", "Historical Kaggle run with the leaky one-shot prompt: 60.1% is withdrawn (slide 11). Leak-free zero-shot: 59.2%.")
edit_by_text(S[7], "Add three independent", "Original design (Sept–Oct). Audits and gated experiments replaced it with probe + Evidence + Reflector (slides 17, 21).")
edit_by_text(S[8], "The Critic converts", "Original Critic. In practice its override reached 0.4% of rows, so a learned Reflector replaced it.")

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

# N0 update since the Oct 17 deck (v5)
n0 = new_slide("UPDATE · OCT 17 → OCT 25", "What Changed Since the Last Review",
               "All planned experiments are finished a week before the Nov 1 freeze. One candidate passed its gate and became the main system.")
cards(n0, [("TRIED", "Nine lines of work", ["— LLM adjudicator, LoRA on 10K rows, hidden-state probes, pooled-dev Reflector, 500K Evidence: all failed or not adopted", "— Base LLaMA, DistilBERT, 5-seed variance, LoRA seed 2: baselines and robustness", "— Every rule written before running; test sets untouched until a pass"]),
           ("MAIN SYSTEM", "Probe + Evidence → Reflector", ["— Agent 1: logistic probe on frozen base-LLaMA-3-8B layer-20 hidden state (40K rows, no fine-tuning)", "— Agent 2: TF-IDF + LR Evidence model on ~497K reviews", "— Reflector: LR on both agents' log-probabilities, fit on 2,000 balanced dev rows"]),
           ("RESULT", "Read once on both test sets", ["— Balanced eval: 67.5% exact (was 63.9%), +3.6 [+0.9, +6.3]", "— Natural test: 75.7% (was 73.4%), +2.3 [+0.6, +4.0]", "— 5 seeds: beats each single agent every time"])],
      1.82, 3.45)
band(n0, 5.5, "Caveat", "Part of the gain is the larger Reflector fit set: the old LoRA + Evidence stack refit on the same 2,000 rows is only 2.0 / 1.5 points behind (not significant). Fine-tuning LLaMA turned out unnecessary for the stack.", h=1.0)

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
           ("PHASE 1", "Phase 1 model: LLaMA-3-8B + LoRA", ["— Retrained on Electronics: 2,000 rows, paper Table 1 hyperparameters", "— MLX QLoRA, ≈3 epochs (400 updates, batch 12), zero-shot prompt", "— Prediction = argmax of first-digit probabilities over 1–5"]),
           ("ARM CHOICE", "Dev only (500 rows, ±4.4)", ["— Arm P, the paper's pipeline (TextBlob + stratified + VGST + neutral ×2): 52.2%", "— Arm R, raw balanced rows: 55.2%", "— R chosen; the 3.0-point gap is within noise"]),
           ("EVIDENCE", "Second, independent signal", ["— TF-IDF (1–2-gram, 300K features) + logistic regression", "— Trained on the 100K raw balanced Electronics rows", "— CPU only, no filtering"])],
      1.82, 3.6)
band(n2, 5.65, "Note", "The original paper's adapter was trained on Kaggle reviews; Kaggle is now reproduction and appendix only. Test sets are never filtered.", h=0.9)

# N3 Phase 2 results
n3 = new_slide("PHASE 2 · FIRST STACK (OCT 13)", "LoRA + Evidence Stack on Electronics (superseded)",
               "Reflector = logistic regression on both agents' log-probabilities, fit on 500 balanced dev rows, read out once per split.")
lw = 7.3
accent_card(n3, MARGIN, 1.82, lw, 3.3)
make_table(n3, MARGIN + 0.26, 2.05, lw - 0.5, [2.2, 1.45, 1.45, 1.4], None, ["Split", "Phase 1 LoRA", "Evidence model", "Stack"],
           [[("eval · 1,000 balanced", INK, True), ("54.0%", BODY, False), ("62.4%", BODY, False), ("63.9%", ACCENT_D, True)],
            [("natural_test · 2,000", INK, True), ("53.4%", BODY, False), ("69.3%", BODY, False), ("73.4%", ACCENT_D, True)],
            [("eval macro-F1", MUTED, False), ("0.543", MUTED, False), ("0.622", MUTED, False), ("0.638", MUTED, False)],
            [("natural macro-F1", MUTED, False), ("0.488", MUTED, False), ("0.541", MUTED, False), ("0.577", MUTED, False)],
            [("within ±1 (eval / natural)", MUTED, False), ("93.7 / 94.3", MUTED, False), ("93.1 / 93.7", MUTED, False), ("95.5 / 96.2", MUTED, False)]],
           row_h=0.46, head_h=0.36, aligns=[L, R, R, R])
rx = MARGIN + lw + GUTTER; rw = CW - lw - GUTTER
accent_card(n3, rx, 1.82, rw, 3.3, bar=ACCENT_D)
_, tf = textbox(n3, rx + 0.26, 1.98, rw - 0.5, 0.3); para(tf, "Paired comparisons (stack, exact)", size=T_CARD, color=INK, bold=True, first=True)
block(n3, rx + 0.26, 2.4, rw - 0.48, 2.7, ["— vs Phase 1 LoRA: +9.9 [+6.7, +13.0] on eval; +20.0 [+17.8, +22.2] on natural_test (both p < 1e-9)",
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

# N6 what was tried since Oct 18 (each with a dev-only gate declared beforehand)
n6 = new_slide("PHASE 2 · EXPERIMENTS", "Nine Lines of Work, Each Gated on Dev Before Any Test Read",
               "Every rule was written in LOG.md before running. Failed candidates never touched eval or natural_test.")
accent_card(n6, MARGIN, 1.82, CW, 4.3)
make_table(n6, MARGIN + 0.26, 1.98, CW - 0.5, [3.3, 6.0, CW - 0.5 - 9.3], None, ["Experiment", "Result", "Verdict"],
           [[("Gate 0: LLM adjudicator", INK, True), ("LLM 54.2% vs stack 73.3% on 434 disagreement rows", BODY, False), ("FAIL, dropped", NEG, True)],
            [("Base LLaMA, no LoRA", INK, True), ("34.3% alone, but base LLaMA + Evidence 63.7 ≈ LoRA + Evidence 63.9 (eval)", BODY, False), ("LoRA adds little in stack", BODY, False)],
            [("DistilBERT 2K baseline", INK, True), ("52.8% vs LoRA 54.0% on eval, not significant", BODY, False), ("Level with LoRA", BODY, False)],
            [("Variance (5 seeds) + LoRA seed 2", INK, True), ("Stack gain over Evidence positive in 5/5 seeds", BODY, False), ("Stable", POS, True)],
            [("LoRA on 10K rows", INK, True), ("dev 56.4% < gate 58%", BODY, False), ("FAIL", NEG, True)],
            [("LLaMA probe, 10K rows / multi-layer", INK, True), ("dev gain +.005 to +.011, natural_dev miss", BODY, False), ("FAIL at 10K fit", NEG, True)],
            [("Pooled-dev Reflector", INK, True), ("Gate passed, but gain = class-prior shift", BODY, False), ("Not adopted", BODY, False)],
            [("Evidence model on 500K rows", INK, True), ("Better alone (+1.9 dev), stack not better", BODY, False), ("FAIL in stack", NEG, True)],
            [("Bigger dev2 (2,000) + probe on 40K", INK, True), ("dev2 +3.9 pts, CI [+2.4, +5.8]", BODY, False), ("PASS, new main", POS, True)]],
           row_h=0.42, head_h=0.34, aligns=[L, L, L])
band(n6, 6.25, "Source", "v2_reflective/LOG.md entries 2026-10-20 to 2026-10-24; outputs in v2_reflective/e23/*.out.", h=0.75)

# NH main result (Phase H)
nh = new_slide("PHASE 2 · MAIN RESULT", "Main System: LLaMA Probe + Evidence Model Stack",
               "LLaMA probe: classifier on frozen LLaMA-3-8B layer-20 state (40K reviews). Evidence model: TF-IDF + LR (~497K reviews). Reflector fit on 2,000 dev reviews.")
lw = 7.3
accent_card(nh, MARGIN, 1.82, lw, 3.45)
make_table(nh, MARGIN + 0.26, 2.0, lw - 0.5, [2.6, 1.2, 1.2, 1.8], None, ["Model", "eval", "natural", "macro-F1 e / n"],
           [[("Earlier LoRA + Evidence stack (dev fit)", INK, False), ("63.9%", BODY, False), ("73.4%", BODY, False), (".638 / .577", MUTED, False)],
            [("LoRA + Evidence stack, refit on dev2", INK, False), ("65.5%", BODY, False), ("74.2%", BODY, False), (".653 / .587", MUTED, False)],
            [("LLaMA probe alone", INK, False), ("61.2%", BODY, False), ("74.3%", BODY, False), (".610 / .606", MUTED, False)],
            [("Evidence model alone", INK, False), ("62.5%", BODY, False), ("70.2%", BODY, False), (".624 / .550", MUTED, False)],
            [("Probe + Evidence stack (main)", ACCENT_D, True), ("67.5%", ACCENT_D, True), ("75.7%", ACCENT_D, True), (".675 / .612", ACCENT_D, True)],
            [("Main stack, 5-seed mean ± sd", MUTED, False), ("65.8 ± 0.6", MUTED, False), ("75.4 ± 0.8", MUTED, False), (".657 / .610", MUTED, False)]],
           row_h=0.44, head_h=0.36, aligns=[L, R, R, R])
rx = MARGIN + lw + GUTTER; rw = CW - lw - GUTTER
accent_card(nh, rx, 1.82, rw, 3.45, bar=ACCENT_D)
_, tf = textbox(nh, rx + 0.26, 1.98, rw - 0.5, 0.3); para(tf, "Paired comparisons (main stack, exact, 95% CI)", size=T_CARD, color=INK, bold=True, first=True)
block(nh, rx + 0.26, 2.4, rw - 0.48, 2.8, ["— vs earlier LoRA + Evidence stack: +3.6 [+0.9, +6.3] eval, +2.3 [+0.6, +4.0] natural (Holm-adjusted p < .05)",
     "— vs Evidence model alone: +5.0 [+2.4, +7.4] eval, +5.5 [+3.8, +7.1] natural",
     "— vs LLaMA probe alone: +6.3 [+3.8, +8.8] eval, +1.4 [−0.1, +2.9] natural (ns)",
     "— Gain over each single agent positive in 5/5 seeds on both splits"], size=T_SMALL, gap=5)
band(nh, 5.45, "Honest read", "Against the LoRA + Evidence stack refit on the same dev2, the main stack gains only +2.0 eval / +1.5 natural (not significant): part of the gain is the bigger Reflector fit set. A frozen base LLaMA replaces the Phase 1 LoRA; adding LoRA back does not help. 5 candidates were compared on dev2 (disclosed). Source: e23/readout_H.out, e23/variance_H.out.", h=1.2)

# NP Phase 1 -> Phase 2, framed honestly (FACT notes in paper/e23_results.tex)
np_ = new_slide("PHASE 1 → PHASE 2", "How Far Phase 2 Moves the Baseline",
                "Same Electronics test sets, same rows. Phase 1 = LLaMA-3-8B + LoRA (the baseline paper's method); Phase 2 = the Probe + Evidence stack.")
cards(np_, [("GAIN", "+13.5 and +22.3 points", ["— Balanced eval: 54.0% → 67.5% (29% fewer errors)", "— Natural test: 53.4% → 75.7% (48% fewer errors)", "— Both significant (paired bootstrap, McNemar)"]),
            ("PHASE 1 STRENGTHS", "Direction is right", ["— Within one star: 93.7% / 94.4%", "— Positive vs negative (3★ excluded): 91.2% / 92.9%", "— The exact star between neighbours is the hard part"]),
            ("WHY 53.4% NATURAL", "A class-mix effect", ["— Phase 1 was trained on balanced data", "— 65% of natural-test reviews are 5★, which it under-predicts", "— Macro-F1 is reported beside exact for this reason"]),
            ("REPRODUCTION", "Baseline paper's own data", ["— Kaggle, paper protocol, leak-free: 69.1% exact", "— Always guessing 5★ gives 63.8% there", "— Polarity 85.9%, binary 90.3%, within one 93.5%"])],
      1.82, 3.6)
band(np_, 5.65, "Not used, on purpose", "70.7% / 60.1% (leaky one-shot prompt, withdrawn by our own audit), 74.8% (one unstable LoRA seed), 79.9% (class-mix correction, not skill). Sources: e23/metrics_extra.out, LOG.md 2026-10-11.", h=0.95)

# NF frozen results figure
nf = new_slide("FROZEN RESULTS", "All Models, Exact Accuracy with 95% Bootstrap CI",
               "Recomputed from stored probabilities by e23/freeze_tables.py. Left: eval (1,000 balanced). Right: natural_test (2,000, natural skew).")
nf.shapes.add_picture(os.path.join(ROOT, "paper/figures/e23/main_bars.png"), Inches(MARGIN), Inches(1.75), height=Inches(5.2))

# NR robustness checks (C2-C5, e23/analysis_C.out)
nr = new_slide("PHASE 2 · ROBUSTNESS", "Does the Result Hold Up? Corrections, Errors, Calibration, Cost",
               "Computed from the frozen main-stack probabilities on eval (1,000) and natural_test (2,000). No new test reads, no refits.")
cards(nr, [("HOLM", "5 of 8 comparisons survive", ["— Beats the Evidence model on both splits and the LLaMA probe on eval (adj. p ≤ .001)", "— Beats the earlier LoRA + Evidence stack: adj. p .045 eval, .043 natural", "— Not significant: vs LLaMA probe on natural; vs earlier stack refit on dev2"]),
           ("ERRORS", "Mostly one star off", ["— 90% (eval) / 86% (natural) of errors are adjacent stars", "— Hardest classes: 2★ and 3★ (recall .46–.60)", "— Agents agree on 64–72% of rows: stack 74–84% right; disagree: 56%"]),
           ("CALIBRATION", "Slightly under-confident", ["— ECE .047 eval, .062 natural", "— Mean confidence .650 vs accuracy .675 (eval)", "— Probabilities are usable as confidence for routing"]),
           ("COST", "The probe is the slow part", ["— LLaMA probe: ~0.67 s per review (M1, Q4)", "— TF-IDF Evidence and Reflector: under 0.1 ms each", "— Motivates a Router that calls LLaMA only when needed"])],
      1.82, 3.6)
band(nr, 5.65, "Source", "e23/analysis_C.py, e23/analysis_C.out (LOG.md 2026-10-25). Reliability figures: paper/figures/e23/reliability_*.png.", h=0.8)

# NN next: Phase 2 v3 plan (docs/superpowers/specs/2026-10-25-phase2-v3-design.md)
nn = new_slide("NEXT · PHASE 2 v3", "Agents That Route, Explain and Check Themselves",
               "The specialists keep deciding the star. LLM agents route work, explain each decision, and a Critic forces rewrites until the explanation is faithful.")
cards(nn, [("ROUTE", "Router agent", ["— Evidence confident: skip the slow LLaMA probe", "— Unsure: call the probe, Reflector combines", "— Gate: within 0.5 pt of the full stack on dev2; report probe calls saved"]),
           ("EXPLAIN", "Explainer agent", ["— Evidence pack: top phrases for / against, agent shares, agreement", "— LLaMA-3-8B-Instruct writes a 1–2 sentence rationale from the pack only"]),
           ("CHECK", "Critic loop", ["— Star matches; quoted phrases occur in the review", "— Disagreement mentioned; no invented claims", "— Fails: specific feedback, rewrite (max 2 rounds)"]),
           ("ACCURACY", "Gated candidates", ["— A1: probe fit 40K → 120K rows (running, 45K / 80K new rows done)", "— A2: DeBERTa-v3 specialist on Colab, Nov 1", "— Each must pass a dev-only gate before any test read"])],
      1.82, 3.6)
band(nn, 5.65, "Evaluation", "On all 3,000 test reviews: faithfulness before vs after the Critic loop (McNemar), rewrite rate, deletion test (cited phrases vs random words), latency with and without the Router. Final demo (December): one local notebook, Phase 1 + Phase 2 live, ≤ 3 minutes.", h=1.0)

# N7 roadmap
n7 = new_slide("ROADMAP", "Plan to Submission and the December Final Presentation",
               "Weekly checkpoints, each logged in LOG.md and progress/week_NN.md. Numbers freeze Nov 1. Last days are buffer only.")
accent_card(n7, MARGIN, 1.82, CW, 4.2)
make_table(n7, MARGIN + 0.26, 2.0, CW - 0.5, [1.0, 1.7, 6.2, CW - 0.5 - 8.9], None, ["Week", "Dates", "Work", "Checkpoint"],
           [[("0", INK, True), ("Oct 17–18", BODY, False), ("Hallucination study, plan, roadmap, results scaffold", BODY, False), ("Done", POS, True)],
            [("1", INK, True), ("Oct 19–25", BODY, False), ("Natural-prior stacker; routing; base LLaMA; Gate 0; variance; probes; 500K Evidence; Phase H", BODY, False), ("Done, main stack adopted", POS, True)],
            [("2", INK, True), ("Oct 26–Nov 1", BODY, False), ("A1 bigger probe (running); Router, Explainer, Critic agents; explanation metrics on all 3,000 test reviews; demo notebook", BODY, False), ("Freeze stage 1: Nov 1", BODY, False)],
            [("3", INK, True), ("Nov 2–8", BODY, False), ("DeBERTa on Colab (A2, Nov 1) and its gate; paper prose (author-written); figures", BODY, False), ("Freeze stage 2: Nov 5; draft 1 Nov 8", BODY, False)],
            [("4", INK, True), ("Nov 9–15", BODY, False), ("Supervisor fixes; thesis sections; archive; deck", BODY, False), ("Draft 2, similarity check", BODY, False)],
            [("5", INK, True), ("Nov 16–20", BODY, False), ("Buffer: proofreading, final checks, submission", BODY, False), ("Submitted by Nov 20", BODY, False)],
            [("6", INK, True), ("December", BODY, False), ("Final presentation: deck + live demo notebook (Phase 1 and Phase 2, ≤ 3 min, checked the day before)", BODY, False), ("Final presentation", BODY, False)]],
           row_h=0.44, head_h=0.36, aligns=[L, L, L, L])
band(n7, 6.2, "Integrity", "Plagiarism and AI-content each under 5%: the author writes the prose; analysis supplies facts, tables and figures; similarity checks on every full draft.", h=0.75)

# NV1 code workspace map (V2_Reflective_Guide.md)
nv1 = new_slide("CODE · v2_reflective/", "Where the Work Lives: the v2_reflective Workspace",
                "Every Phase 2 number comes from a script here, a saved output file, and a dated LOG.md entry. e23/ = Electronics 2023 (177 files).")
accent_card(nv1, MARGIN, 1.82, CW, 4.25)
make_table(nv1, MARGIN + 0.26, 1.98, CW - 0.5, [2.6, 5.2, CW - 0.5 - 7.8], None, ["Part", "Main files", "What it holds"],
           [[("Lab notebook", INK, True), ("LOG.md, EXPERIMENTS_SUMMARY.md, progress/", BODY, False), ("Rule declared before each run; result and decision after", BODY, False)],
            [("Kaggle close-out (Oct 9–12)", INK, True), ("phase1_final.py, reflective.py, gate0.py, stack_experiment.py", BODY, False), ("Leak audit, retracted calibrator, first LLM adjudicator, first second-signal test", BODY, False)],
            [("e23 · data", INK, True), ("fetch.py, build_splits.py, build_H.py, build_A1.py", BODY, False), ("Frozen disjoint splits (seed 2023): train, dev, dev2, eval, natural", BODY, False)],
            [("e23 · agents", INK, True), ("evidence*.py, probe_agent*.py, eval_p1.py, encoder_2k.py", BODY, False), ("Each agent writes 5 star probabilities per split (evid7 = LLaMA probe, evid6b = Evidence model)", BODY, False)],
            [("e23 · gates and readouts", INK, True), ("gate_*.py, stack_gate_*.py, readout_H.py, phase2_e23.py", BODY, False), ("Dev-only pass/fail; a pass reads eval and natural_test once", BODY, False)],
            [("e23 · robustness", INK, True), ("variance*.py, base_vs_lora.py, halluc_p1.py, analysis_C.py", BODY, False), ("Seeds, baselines, ablations, Holm, errors, calibration", BODY, False)],
            [("e23 · freeze", INK, True), ("freeze_tables.py → results_frozen.md, figures", BODY, False), ("Recomputes every table in ~3 s; deck and paper copy from it", BODY, False)]],
           row_h=0.5, head_h=0.34, aligns=[L, L, L])
band(nv1, 6.2, "Run control", "queue_*.sh starts long jobs; stage_*.txt markers and chunked hidden states make every run resumable (sleep, battery, planned pause). Guide: presentation/V2_Reflective_Guide.md.", h=0.75)

# NV2 how one experiment flows
nv2 = new_slide("CODE · WORKFLOW", "How One Experiment Flows, From Rule to Paper",
                "The same six steps for every candidate. The rule is fixed before the run, so the goalposts cannot move after seeing a result.")
n_ = 6; w_ = (CW - (n_ - 1) * 0.18) / n_
for i, (tag, head, body) in enumerate([("1", "Declare", "Candidate, data and exact pass rule written in LOG.md"),
                                       ("2", "Run", "Resumable script; .log while running, .out when done"),
                                       ("3", "Gate", "dev / dev2 / natural_dev only; a FAIL stops here"),
                                       ("4", "Readout", "A PASS is read once on eval and natural_test, paired tests"),
                                       ("5", "Freeze", "freeze_tables.py recomputes all tables from stored probabilities"),
                                       ("6", "Record", "LOG.md result, verified backup + Drive copy, GitHub push")]):
    x = MARGIN + i * (w_ + 0.18)
    accent_card(nv2, x, 1.9, w_, 2.2); auto_pill(nv2, x + 0.2, 2.08, 0.24, tag)
    _, tf = textbox(nv2, x + 0.2, 2.5, w_ - 0.35, 0.4); para(tf, head, size=T_CARD, color=INK, bold=True, first=True)
    block(nv2, x + 0.2, 3.0, w_ - 0.35, 1.7, [body], size=T_SMALL)
band(nv2, 4.4, "Why it matters", "16 ideas failed and are reported; none of them ever touched eval or natural_test. That is what makes the one pass (the main stack) trustworthy. Example: Phase H rule 'dev2 +1.0 pt, CI > 0' → +3.9 [+2.4, +5.8] → PASS → readout_H.out.", h=1.1)

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
        elif 9 < i < first_hist and s.top < 400000 and txt.isupper() and not re.match(r"\d\d · ", txt):
            set_p(s.text_frame.paragraphs[0], f"{i - 1:02d} · {txt}")

# ---- speaker notes (slide number -> text); full script in presentation/Presenter_Notes.md, Part 0
NOTES = {
 1: "Title. Say: this is the Oct 25 update. Main results are on Amazon Reviews 2023 Electronics only. The headline system is now a frozen LLaMA probe plus an Evidence model, combined by a Reflector.",
 2: "Problem. Single-pass LLM rating has no evidence check. Keep it short: 1 minute.",
 3: "Research question and gap. Wang et al. 2025 is the baseline paper. Our question: does adding a second, independent signal and a combiner beat a single model?",
 4: "Data. Say: main results now use Electronics 2023 only. Kaggle is the original paper's data and appears only as reproduction.",
 5: "Original paper's pipeline (TextBlob DQC, stratified, VGST, neutral oversampling). Point forward: slide 14 shows TextBlob filtering hurts on Electronics.",
 6: "LoRA explained. LoRA is still trained (Phase 1), but in the final stack it is an ablation row, not the main agent.",
 7: "Phase 1 baseline on Kaggle (historical numbers). The 60.1% here used the leaky one-shot prompt; leak-free zero-shot is 59.2%. Slide 11 explains.",
 8: "Original Phase 2 design (Analyst, Visual, RAG, Critic). Say: this was the starting design; the audit (slide 29) and later experiments replaced it with the probe + Evidence + Reflector system (slide 17). Slide 21 shows how the agent idea comes back in Phase 2 v3.",
 9: "Original dissonance Critic. Say: in practice it fired on 0.4% of rows, so it was replaced by a learned Reflector.",
 10: "UPDATE SLIDE. Read the three cards left to right. Must say the caveat: against the old stack refit on the same 2,000 dev rows the gain is only 2.0 / 1.5 points and not significant.",
 11: "Leak audit. Picking a one-shot example whose star differs from the truth leaks about 8 points. All later numbers are zero-shot and leak-free.",
 12: "Protocol. Frozen disjoint splits; all choices on dev; each test split read once per declared config. Since Oct 24 there is also dev2 (2,000 balanced) for the Reflector.",
 13: "First stack (Oct 13). Phase 1 LoRA + Evidence = 63.9 / 73.4. Superseded by the main Probe + Evidence stack, kept for continuity.",
 14: "Preprocessing ablation. TextBlob DQC deletes most 1 and 2 star reviews; VGST picks long reviews; both are dropped.",
 15: "Hallucination study. LoRA explanations are a canned template; digit probabilities are the usable signal. That is why Phase 2 combines probabilities, not explanations.",
 16: "Experiment ledger. Go row by row, about 10 seconds each. Key message: every rule was fixed before running and failures never touched the test sets.",
 17: "MAIN RESULT. Probe + Evidence stack 67.5 eval / 75.7 natural, macro-F1 .675 / .612. Variance 65.8 +/- 0.6 and 75.4 +/- 0.8. Read the honest-read band aloud.",
 18: "Phase 1 to Phase 2. Lead with the gain: +13.5 points on balanced eval and +22.3 on natural test, i.e. 29% and 48% fewer errors. Then be fair to Phase 1: it gets the direction right (within one star 94%, positive vs negative 91-93%); the exact star is the hard part. Its 53.4% on natural test is a class-mix effect: trained balanced, while 65% of natural reviews are 5-star. On its home data with the paper's protocol, leak-free, it reaches 69.1% (always-5 already gives 63.8%). Do not quote the old 70.7 / 60.1: they used the leaky prompt.",
 19: "Bar chart of all frozen models with 95% CIs; main stack in red. Point out: base LLaMA zero-shot is weak (34%) but a probe on its hidden states reaches 61 / 74.",
 20: "Robustness. After Holm correction 5 of 8 comparisons stay significant, including the main stack vs the earlier LoRA + Evidence stack on both splits. Most errors are one star off. Calibration is good (ECE .05-.06). The probe costs 0.67 s per review, the rest is under 0.1 ms: that is why the next step adds a Router.",
 21: "Next: Phase 2 v3. Say: the specialists decide the star because an LLM choosing the star failed twice (Gate 0). The agents now do what LLMs are good at: routing, explaining, and checking explanations. Every explanation is checked by the Critic and measured on all 3,000 test reviews. A1 is running; DeBERTa runs on Colab Nov 1. Final demo is a local notebook.",
 23: "Code workspace. v2_reflective started as the Oct 10 'reflective v2' design (historical slide 33); the name stayed and it now holds every Phase 2 experiment. Top level = Kaggle close-out; e23 = Electronics 2023, all current results. Walk the table top to bottom: data, agents, gates, robustness, freeze. Full guide: presentation/V2_Reflective_Guide.md.",
 24: "Workflow. Six steps for every experiment: declare in LOG.md, run resumably, gate on dev only, read the tests once, freeze, record and back up. Say: 16 failures are reported and none touched the test sets, which is why the one pass can be trusted.",
 22: "Roadmap. Freeze stage 1 Nov 1 (local work), stage 2 Nov 5 (Colab result in or out). Author writes the prose from Nov 2; draft 1 to supervisor Nov 8; submit by Nov 20. December: final presentation with the live demo notebook (Phase 1 and Phase 2).",
}
for i, sl in enumerate(prs.slides, 1):
    sl.notes_slide.notes_text_frame.text = NOTES.get(i, "Historical slide (Run 1, All Beauty, Kaggle-trained adapter, leaky prompt). Superseded; show only if asked.")
prs.save(DST); print("saved", DST, n)
