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

import copy
def set_lines(slide, startswith, lines):
    """Replace a text block (1+ paragraphs, may contain line breaks) by `lines`, one paragraph each, keeping the first run's style."""
    for sh in slide.shapes:
        if sh.has_text_frame and sh.text_frame.text.startswith(startswith):
            body = sh.text_frame._txBody; ps = body.findall(qn("a:p")); tpl = copy.deepcopy(ps[0])
            for br in tpl.findall(qn("a:br")): tpl.remove(br)
            rs = tpl.findall(qn("a:r"))
            for r in rs[1:]: tpl.remove(r)
            for p in ps: body.remove(p)
            for ln in lines:
                p = copy.deepcopy(tpl); p.find(qn("a:r")).find(qn("a:t")).text = ln; body.append(p)
            return
    raise KeyError(startswith)
# ---- edits to kept slides
edit_by_text(S[1], "A single text-only", "One LLM pass rates the review: no second opinion, no evidence, and nothing checks its explanation.")
edit_by_text(S[1], "Sentiment Hallucinations", "No Second Opinion")
set_lines(S[1], "— User claims cannot", ["— One model reads the review once", "— Nothing disagrees with it or catches a mistake", "— Neighbouring stars (3★ vs 4★) are easily confused"])
edit_by_text(S[1], "Context Blindness", "Unverifiable Explanations")
set_lines(S[1], "— Ignores product images", ["— The model writes a justification", "— Nothing checks it against the review", "— Fine-tuned models repeat a template"])
edit_by_text(S[1], "Linguistic Errors", "Hard Language")
set_lines(S[1], "— Misreads sarcasm", ["— Misreads sarcasm: “Amazing, it broke in 2 days”", "— Mixed sentiment scored incorrectly", "— No critic to trigger re-evaluation"])
set_lines(S[1], "A system is needed", ["Needed: an independent second signal, a learned way to combine the signals, and explanations that come from real evidence and are checked.",
                                      "The transition: single-pass classifier → evidence-grounded, self-checking multi-agent system."])
edit_by_text(S[0], "Kaggle + Amazon", "Amazon Reviews 2023 · Electronics")
edit_by_text(S[0], "LLaMA-3-8B + LoRA", "Frozen LLaMA-3-8B probe    ·    Evidence model    ·    Reflector    ·    Self-checking agents    ·    Leak-free evaluation")
edit_by_text(S[0], "Varun K", "Varun K    ·    PES MTech Project    ·    October 25, 2026 update")
edit_by_text(S[0], "PROBLEM", "PROBLEM  →  PHASE 1: BASELINE PAPER  →  WHY IT FALLS SHORT  →  PHASE 2 SYSTEM  →  RESULTS: PHASE 1, PHASE 2  →  IN DEPTH  →  CONCLUSIONS")
edit_by_text(S[3], "Phase 1 is trained", "Main results now use Amazon Reviews 2023 Electronics only; the Kaggle corpus is the original paper's data (reproduction and appendix).")
edit_by_text(S[3], "All Beauty (Run 1)", "Electronics only: train ~497K · dev 500 + dev2 2,000 · eval 1,000 · natural_test 2,000")
edit_by_text(S[3], "text · metadata", "review text and star rating (images tested, excluded)")
edit_by_text(S[3], "Enables —", "Used — review text only; splits frozen and disjoint (slide §protocol§)")
edit_by_text(S[3], "Review data:", "All Beauty (Run 1) and image/metadata runs are historical (appendix, slides §hist_range§).")
for p in [p for s_ in S[3].shapes if s_.has_text_frame and s_.text_frame.text.startswith("All Beauty (Run 1) and") for p in s_.text_frame.paragraphs][1:]: p._p.getparent().remove(p._p)
edit_by_text(S[6], "Class-balanced 1,000-row", "Historical Kaggle run with the leaky one-shot prompt: 60.1% is withdrawn (slide §leak§). Leak-free zero-shot: 59.2%.")
edit_by_text(S[2], "From Passive Classifier", "Baseline Paper (Wang et al., 2025): What It Does, Where It Stops")
edit_by_text(S[7], "07 · PHASE 2 ARCHITECTURE", "07 · ORIGINAL PHASE 2 DESIGN")
edit_by_text(S[7], "Phase 2: Multi-Agent", "Where Phase 2 Started: the Original Four-Agent Design")
edit_by_text(S[7], "Add three independent", "Original design (Sept–Oct). Audits and gated experiments replaced it with probe + Evidence + Reflector and an agentic layer (slides §arch§–§solves§).")
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

# NG glossary (2 slides, shown right after the background): plain-language meaning of every term used later
def glossary(sl, pairs):
    half = (len(pairs) + 1) // 2; rows = []
    for i in range(half):
        a = pairs[i]; b = pairs[i + half] if i + half < len(pairs) else ("", "")
        rows.append([(a[0], ACCENT_D, True), (a[1], BODY, False), (b[0], ACCENT_D, True), (b[1], BODY, False)])
    accent_card(sl, MARGIN, 1.78, CW, 0.32 + 0.5 * half + 0.25)
    t = make_table(sl, MARGIN + 0.2, 1.9, CW - 0.4, [1.75, 4.1, 1.75, CW - 0.4 - 7.6], None, ["Term", "Meaning", "Term", "Meaning"], rows, row_h=0.5, head_h=0.3, aligns=[L] * 4)
    for r in list(t.rows)[1:]:
        for c in r.cells:
            for p in c.text_frame.paragraphs:
                for run in p.runs: run.font.size = Pt(9.5)
g1 = new_slide("KEY TERMS · 1 / 2", "Key Terms: Models and Agents", "Plain-language meaning of every model and agent name used in the rest of the deck.")
glossary(g1, [("LLaMA-3-8B", "Meta's open large language model (8 billion parameters); the base model in both phases"),
              ("LoRA", "Small add-on weights trained on top of a frozen model, so fine-tuning is cheap (Phase 1)"),
              ("Phase 1 LoRA", "LLaMA-3-8B + LoRA trained to output the star: the baseline paper's method"),
              ("Frozen model", "Used as-is, never retrained"),
              ("Hidden state (layer 20)", "LLaMA's internal numeric summary of the review, halfway through the network"),
              ("LLaMA probe", "Small classifier that reads the layer-20 hidden state and outputs 1–5★ probabilities"),
              ("TF-IDF", "Scores each word / word pair by how distinctive it is in a review"),
              ("Evidence model", "TF-IDF + logistic regression on ~497K reviews; its top phrases become the explanation evidence"),
              ("Logistic regression (LR)", "Simple, fast classifier that outputs a probability for each star"),
              ("Agent", "A component with one job (rate, combine, route, explain, check) that passes results to the next"),
              ("Stack / stacking", "Several models predict; a small learned model combines their predictions"),
              ("Reflector", "The learned combiner: weighs the probe and the Evidence model, outputs the final star"),
              ("Main stack", "LLaMA probe + Evidence model + Reflector: the system behind the headline results"),
              ("Router", "Decides whether the slow LLaMA probe is needed for a review"),
              ("Explainer / Critic", "Explainer (instruction-tuned LLM) writes the rationale; Critic checks it and asks for rewrites"),
              ("Self-correction", "System revises its own output: escalation, LLM re-rating (rating) and the Critic loop (explanation)")])
g2 = new_slide("KEY TERMS · 2 / 2", "Key Terms: Data, Evaluation and Method", "How the results are measured and why they can be trusted.")
glossary(g2, [("Balanced eval", "1,000 test reviews, 200 per star: every star counts equally"),
              ("Natural test", "2,000 test reviews with the real star mix (65% are 5★)"),
              ("dev / dev2", "Separate development reviews (500 / 2,000) used for every choice; test sets are read once"),
              ("Exact accuracy", "Share of reviews whose star is predicted exactly"),
              ("Within one star", "Share predicted exactly or one star off"),
              ("Macro-F1", "Average quality over the five stars, so rare stars count as much as 5★"),
              ("Polarity / binary", "Negative vs neutral vs positive / negative vs positive only"),
              ("Zero-shot / one-shot", "Prompt with no example / with one worked example"),
              ("Label leak", "Example picked using the true answer, which inflated old scores by ~8 points"),
              ("Gate", "Pass/fail rule written in the lab log before an experiment runs, checked on dev data only"),
              ("95% CI (bootstrap)", "Range the true value likely falls in, from resampling the test reviews"),
              ("McNemar / paired test", "Tests whether two systems differ on the same reviews"),
              ("Holm correction", "Adjusts p-values when many comparisons are made, to avoid lucky wins"),
              ("ECE (calibration)", "Gap between the model's confidence and its real accuracy; lower is better"),
              ("Faithfulness", "Explanation states the right star and only uses evidence that is really in the review"),
              ("Deletion test", "Remove the cited phrases vs random words: if confidence drops more, the explanation points at what mattered")])

# N0 update since the Oct 17 deck (v5)
n0 = new_slide("AT A GLANCE", "The Project at a Glance",
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
n5 = new_slide("HALLUCINATION STUDY", "Phase 1's Justifications Are a Template",
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

# NX full experiment ledger: everything tried and dropped (2 slides) + what we kept (EXPERIMENTS_SUMMARY.md F1-F16, W1-W10)
def small_table(sl, rows, col_w, header, row_h, size=T_SMALL):
    accent_card(sl, MARGIN, 1.78, CW, 0.32 + row_h * len(rows) + 0.3)
    t = make_table(sl, MARGIN + 0.2, 1.9, CW - 0.4, col_w, None, header, rows, row_h=row_h, head_h=0.3, aligns=[L] * len(header))
    for r in list(t.rows)[1:]:
        for c in r.cells:
            for p in c.text_frame.paragraphs:
                for run in p.runs: run.font.size = Pt(size)
    return t
XW = [2.45, 3.55, 3.15, CW - 0.4 - 9.15]; XH = ["Experiment", "What we did", "Result", "Why dropped"]
X1 = [("LLM judge, Kaggle (Gate 0)", "LLaMA picks between Phase 1's top-2 stars on uncertain reviews", "51.9–57.5% vs Phase 1 55.6–67.2% on the same pairs", "An LLM cannot judge between two candidate stars"),
      ("LLM judge with evidence, Electronics", "Same, with Evidence-model phrases in the prompt", "54.2% vs stack 73.3% on 434 disagreement reviews", "Same failure, replicated"),
      ("Recalibrating Phase 1", "LR calibrator on Phase 1 probabilities", "+5.0 on leaky probabilities → +0.1 [−2.3, +2.5] on clean ones", "The gain was the label leak; retracted"),
      ("Phase 1 answer-only retrain", "LoRA retrained to output only the digit (Kaggle)", "55.3% vs 59.7% for the original adapter", "Worse"),
      ("TextBlob filtering + VGST", "The baseline paper's data cleaning and diversity sampling", "Filtering −2.5 to −3.5 pts (drops 52% of 1★, 67% of 2★); VGST +0.2", "Hurts / noise; raw balanced data kept"),
      ("Visual Verifier (BLIP, moondream2, Qwen2-VL)", "Rate satisfaction from the product photo", "Constant output: always 3★, then always 5★", "A product photo does not show satisfaction"),
      ("RAG Prover (LanceDB, similar reviews)", "Retrieve 3 similar reviews and vote on the star", "37.5% alone vs 54.2% for the Analyst; overruled correct answers", "Weak signal; replaced by the Evidence model"),
      ("Dissonance Critic", "Re-run the Analyst when agent votes disagree", "Changed the decision on 0.4% of reviews", "Barely acts; replaced by a learned Reflector"),
      ("Claim-splitting for grounding", "Split reviews into claims before grounding", "Mean grounding .144 vs .340 for whole reviews", "Worse on real data; reverted")]
X2 = [("Extra Reflector features", "Add TextBlob, margin, entropy, length to the Reflector", "≤ +1 pt, within noise", "No real gain"),
      ("Natural-prior Reflectors", "Re-weight the Reflector to the natural star mix", "Natural exact up, but macro-F1 and balanced eval down", "Gain is the class mix, not skill"),
      ("Disagreement routing", "Treat agent-disagreement reviews separately", "+0.8 on natural test, not significant", "Not significant"),
      ("MiniLM embedding / character n-gram agents", "Extra text views added to the stack", "Weaker than TF-IDF; stack −0.9 to −2.1 on dev", "More views of the same text add noise"),
      ("LoRA on 10K reviews", "5× more Phase 1 training data", "dev 56.4% < gate 58%", "LoRA is not data-starved"),
      ("LLaMA probe on 10K / 120K reviews", "Probe on 10K, then 120K training reviews", "10K: natural_dev .803 vs .804; 120K: dev2 +0.3 [−1.1, +0.9]", "10K missed by .001; 40K passed; 120K adds nothing"),
      ("LLaMA probe, several layers", "Combine states from several layers", "dev +0.5 pts", "Below the gate"),
      ("Pooled-dev Reflector + ordinal decoding", "Fit the Reflector on dev + natural_dev", "Passed its gate, but the gain = more 5★ predictions", "Class-mix shift; not adopted"),
      ("Evidence model on 500K in the old stack", "Bigger Evidence model with the LoRA stack", "Stack .632 vs .640", "Overlaps LoRA (kept later in the main stack)"),
      ("LoRA added back to the main stack", "LoRA + Evidence model + LLaMA probe", "dev2 .633 vs .639", "No gain over probe + Evidence"),
      ("LLM re-rating of unsure reviews (Instruct)", "Llama-3-8B-Instruct picks between the top-2 stars, order-swap check", "dev2: 76 fixed vs 89 broken (net −13)", "Third LLM-judge failure; LLMs explain, not decide")]
X1 = X1 + [X2[5], X2[10]]   # the deck shows one ledger slide (x1); x2 stays defined for old_decks / backup
xs = []
for i, (rows, part) in enumerate(((X1, ""), (X2, " (2 / 2)"))):
    sl = new_slide("EXPERIMENT LEDGER", "Everything We Tried and Dropped" + part,
                   "Each idea had a pass/fail rule written before it ran and was checked on dev data only. None touched the test sets.")
    small_table(sl, [[(a, INK, True), (b, BODY, False), (c, BODY, False), (d, NEG, True)] for a, b, c, d in rows], XW, XH, row_h=0.39); xs.append(sl)
nk = new_slide("EXPERIMENT LEDGER · KEPT", "What We Kept, and the Evidence for It",
               "Ten changes survived their checks; together they give the main stack (67.5% balanced eval, 75.7% natural test).")
K = [("Leak-free zero-shot prompt", "Old one-shot example was chosen with rating ≠ truth", "Leak of ~8 pts found (−7.7 [−13.3, −2.0]); all results since are leak-free"),
     ("One dataset: Electronics 2023", "Frozen, disjoint splits (seed 2023)", "Every comparison on the same rows; test sets read once"),
     ("Raw balanced training data", "No TextBlob filtering, no VGST", "+2.5 to +3.5 pts over filtered data"),
     ("Second independent signal", "TF-IDF Evidence model stacked with LLaMA", "63.9 / 73.4 vs Phase 1 LoRA 54.0 / 53.4"),
     ("Learned Reflector", "LR on both agents' probabilities replaces the dissonance Critic", "Beats each agent; robust across 5 seeds"),
     ("Bigger Reflector fit set (dev2)", "2,000 balanced dev reviews instead of 500", "Halves dev noise; earlier stack 63.9 → 65.5"),
     ("Frozen LLaMA probe, 40K reviews", "Classifier on LLaMA layer-20 state, no fine-tuning", "Gate passed: dev2 +3.9 [+2.4, +5.8]"),
     ("Evidence model, ~497K reviews", "Same TF-IDF + LR, 5× more data", "Best partner for the probe in the main stack"),
     ("Polarity / within-one reporting", "Report direction and near-misses beside exact", "Stack beats each agent on polarity on both test sets"),
     ("Audit fixes (Oct 9)", "Grounding floor removed, ties to the Analyst, Visual vote excluded", "Run 2 net −91 → net +2 on 96 reviews (historical)")]
small_table(nk, [[(a, INK, True), (b, BODY, False), (c, POS, True)] for a, b, c in K], [3.0, 4.6, CW - 0.4 - 7.6], ["Kept", "What it is", "Evidence"], row_h=0.42)

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
cards(np_, [("GAIN", "+13.5 and +22.3 points", ["— Balanced eval: 54.0% → 67.5% (29% fewer errors)", "— Natural test: 53.4% → 75.7% (48% fewer errors)", "— Both significant, also after Holm correction"]),
            ("WHY IT WORKS", "The two agents complement", ["— Evidence model strong on short reviews (70.0 vs probe 54.6)", "— LLaMA probe strong on long reviews (64.3 vs 55.8)", "— When they disagree the stack is right 56% vs 36–50% alone"]),
            ("HONEST CAVEAT", "Part of the gain is data", ["— Earlier LoRA + Evidence stack refit on the same 2,000 dev reviews is only 2.0 / 1.5 behind (not significant)", "— Fine-tuning LLaMA was not needed"])],
      1.82, 3.6)
band(np_, 5.65, "Not used, on purpose", "70.7% / 60.1% (leaky one-shot prompt, withdrawn by our own audit), 74.8% (one unstable LoRA seed), 79.9% (class-mix correction, not skill). Sources: e23/metrics_extra.out, LOG.md 2026-10-11.", h=0.95)

# NF frozen results figure
nf = new_slide("FROZEN RESULTS", "All Models, Exact Accuracy with 95% Bootstrap CI",
               "Every model on the same test sets; the main stack in red. Left: balanced eval (1,000). Right: natural test (2,000).")
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
nn = new_slide("PHASE 2 · HOW IT SOLVES THE PROBLEM", "Agents That Route, Explain and Check Themselves",
               "The specialists keep deciding the star. LLM agents route work, explain each decision, and a Critic forces rewrites until the explanation is faithful.")
cards(nn, [("ROUTE", "Router agent", ["— Evidence confident: skip the slow LLaMA probe", "— Unsure: call the probe, Reflector combines", "— Self-correction 1: unsure draft star escalated and revised (fixed vs broken)"]),
           ("EXPLAIN", "Explainer agent", ["— Evidence pack: top phrases for / against, agent shares, agreement", "— LLaMA-3-8B-Instruct writes a 1–2 sentence rationale from the pack only"]),
           ("CHECK", "Critic loop", ["— Star matches; quoted phrases occur in the review", "— Disagreement mentioned; no invented claims", "— Fails: specific feedback, rewrite (max 2 rounds)"]),
           ("ACCURACY", "Gated candidates", ["— Self-correction 2: Instruct LLM re-rating: tested, no gain", "— LLaMA probe on 120K reviews: tested, no gain (40K kept)", "— Optional extra specialist (GPU run, Nov 1); all dev-gated"])],
      1.82, 3.6)
band(nn, 5.65, "Self-correction (3 levels)", "Rating: escalation (fixed vs broken) and LLM re-rating of uncertain reviews (gated). Explanation: Critic loop, faithfulness before vs after on all 3,000 test reviews, plus a deletion test. Final demo (December): Phase 1 + Phase 2 live, ≤ 3 minutes.", h=1.0)

# NS build schedule Oct 25 - Nov 5 (docs/superpowers/plans/2026-10-25-phase2-v3-plan.md)
ns = new_slide("SCHEDULE · OCT 25 → NOV 5", "Phase 2 v3 Build Schedule",
               "Every step: rule written in LOG.md first, dev-only gate, logged, backed up, pushed. Heavy runs only overnight on charger.")
accent_card(ns, MARGIN, 1.82, CW, 4.55)
make_table(ns, MARGIN + 0.26, 1.95, CW - 0.5, [1.7, 6.6, 1.9, CW - 0.5 - 10.2], None, ["When", "Work", "Compute", "Status"],
           [[("Oct 25", INK, True), ("Shared agents rebuilt and verified (exact match to the frozen results)", BODY, False), ("10 min CPU", BODY, False), ("Done", POS, True)],
            [("Oct 25", INK, True), ("Shared LLaMA (Phase 1 LoRA on / off); Router + escalation (self-correction 1): passed, measured", BODY, False), ("minutes", BODY, False), ("Done", POS, True)],
            [("Oct 25", INK, True), ("Evidence pack, Critic, Explainer built; LLM re-rating check (self-correction 2): failed, dropped", BODY, False), ("2.4 h model", BODY, False), ("Done", POS, True)],
            [("Oct 25 night", INK, True), ("LLaMA probe on 120K reviews + gate: dev2 +0.3, failed, 40K probe kept", BODY, False), ("6.3 h model", BODY, False), ("Done", POS, True)],
            [("Oct 26", INK, True), ("LangGraph wiring; explanation prompts tuned on 500 dev reviews: faithful 64% → 88% after Critic rewrites, frozen", BODY, False), ("1.5 h model", BODY, False), ("Done", POS, True)],
            [("Oct 26–27", INK, True), ("Explanations + Critic loop on all 3,000 test reviews (self-correction 3): faithful 60–63% → 89%, 0 broken", BODY, False), ("~10 h model", BODY, False), ("Done", POS, True)],
            [("Oct 27", INK, True), ("Deletion test on all 3,000: cited phrases beat random words on both test sets", BODY, False), ("3.4 h model", BODY, False), ("Done", POS, True)],
            [("Oct 27–28", INK, True), ("Report, paper tables, agent audit, hard cases, Phase 1 vs Phase 2 explanations, demo notebook + automatic check", BODY, False), ("minutes", BODY, False), ("Done", POS, True)],
            [("Nov 1", INK, True), ("Freeze stage 1; optional extra specialist (GPU run, kept only if it passes its check)", BODY, False), ("Colab 2–3 h", BODY, False), ("Optional", BODY, False)],
            [("Nov 5", INK, True), ("Freeze stage 2: final tables, demo, deck", BODY, False), ("—", BODY, False), ("Planned", BODY, False)]],
           row_h=0.39, head_h=0.32, aligns=[L, L, L, L])

# N7 roadmap
n7 = new_slide("ROADMAP", "Plan to Submission and the December Final Presentation",
               "Weekly checkpoints, each logged in LOG.md and progress/week_NN.md. Numbers freeze Nov 1. Last days are buffer only.")
accent_card(n7, MARGIN, 1.82, CW, 4.2)
make_table(n7, MARGIN + 0.26, 2.0, CW - 0.5, [1.0, 1.7, 6.2, CW - 0.5 - 8.9], None, ["Week", "Dates", "Work", "Checkpoint"],
           [[("0", INK, True), ("Oct 17–18", BODY, False), ("Hallucination study, plan, roadmap, results scaffold", BODY, False), ("Done", POS, True)],
            [("1", INK, True), ("Oct 19–25", BODY, False), ("Natural-prior stacker; routing; base LLaMA; Gate 0; variance; probes; 500K Evidence; Phase H", BODY, False), ("Done, main stack adopted", POS, True)],
            [("2", INK, True), ("Oct 26–Nov 1", BODY, False), ("A1 bigger probe (failed, 40K kept); Router, Explainer, Critic agents; explanation metrics on all 3,000 test reviews; demo notebook", BODY, False), ("Freeze stage 1: Nov 1", BODY, False)],
            [("3", INK, True), ("Nov 2–8", BODY, False), ("Optional extra specialist (Nov 1) and its gate; paper prose (author-written); figures", BODY, False), ("Freeze stage 2: Nov 5; draft 1 Nov 8", BODY, False)],
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
nv2 = new_slide("HOW WE RUN EXPERIMENTS", "How One Experiment Flows, From Rule to Paper",
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

# NPW / NPF the baseline paper in simple points, then why it falls short (replace the combined gap slide)
pw = new_slide("PHASE 1 · THE PAPER WE CHOSE", "The Paper We Chose: What It Does",
               "Wang et al. (2025), “Sentiment Analysis of Product Reviews Using Fine-Tuned LLaMA-3”, ITM Web of Conferences 70, 04021.")
cards(pw, [("TASK", "Rate a review 1–5★", ["— Input: an Amazon product review", "— Output: a star rating plus a short justification", "— One model, one pass"]),
           ("DATA", "Clean, balance, diversify", ["— Kaggle Amazon reviews (~500K)", "— TextBlob drops reviews whose tone disagrees with the stars", "— Equal reviews per star; VGST picks varied wording", "— Neutral (3★) reviews doubled"]),
           ("MODEL", "LLaMA-3-8B + LoRA", ["— Small LoRA add-on, last 3 layers", "— r = 4, α = 64, 3 epochs, lr 5e-5", "— 4-bit, fits one 16 GB GPU"]),
           ("PROMPT", "Rubric + one example", ["— 1–5★ rubric in the prompt", "— One worked example (one-shot)", "— “Let's take it one step at a time”"])],
      1.82, 3.55)
band(pw, 5.6, "Reported", "92.6% accuracy, 93.3% F1 (5-fold cross-validation, 40K test reviews); beats SVM, Naive Bayes, Decision Tree and XLNet; LLaMA-3 chosen over LLaVA-1.5 and Qwen2 for its accuracy / speed trade-off.", h=0.95)
pf = new_slide("PHASE 1 · WHY IT FALLS SHORT", "Why It Falls Short",
               "What we found when we rebuilt it on Amazon Reviews 2023 Electronics with a leak-free, balanced test.")
cards(pf, [("NO CHECKING", "One pass, no second opinion", ["— Reads the review once", "— No evidence, no other model to disagree", "— A wrong star goes uncorrected"]),
           ("FRAGILE TEST", "The score depends on the set-up", ["— One-shot example choice leaked the answer in our first run (~8 pts)", "— Skewed test: always guessing 5★ already scores 63.8%", "— Leak-free, balanced: 54.0% exact"]),
           ("DATA CLEANING", "Filtering backfires", ["— TextBlob deletes 52% of 1★ and 67% of 2★ reviews", "— Costs 2.5–3.5 points", "— VGST adds nothing (+0.2)"]),
           ("EXPLANATIONS", "Justifications are not real", ["— Same template sentence per star", "— Wrong star in 53% of 150 checked", "— Probabilities useful, text not"])],
      1.82, 3.55)
band(pf, 5.6, "What Phase 2 needs", "A second, independent signal; a learned combiner instead of one pass; and agents whose explanations come from real evidence and are checked, each measured on the same frozen test sets.", h=0.95)

# NE / NC2 results visuals (e23/figures_deck.py; numbers identical to e23/analysis_C.out)
FIGD = os.path.join(ROOT, "paper/figures/e23/")
ner = new_slide("PHASE 2 · RESULTS IN DETAIL", "Where the Main Stack Is Right, and Where It Errs",
                "Left: confusion matrix on balanced eval (rows = true star). Right: share of each star found, Phase 1 vs Phase 2, on both test sets.")
ner.shapes.add_picture(FIGD + "cm_main_eval.png", Inches(MARGIN), Inches(1.75), height=Inches(4.0))
ner.shapes.add_picture(FIGD + "recall_phase1_vs_main.png", Inches(MARGIN + 4.6), Inches(1.95), width=Inches(7.5))
band(ner, 5.95, "Read", "Mistakes sit next to the diagonal: 90% (eval) / 86% (natural) of errors are one star off. Phase 2 finds more of every star on balanced eval; on natural test Phase 1 finds more 4★ (.715 vs .639) because it over-predicts 4★, the main stack is far better on 5★ (.829 vs .490).", h=1.0)
ncm = new_slide("PHASE 2 · WHY IT WORKS", "Why Two Agents Beat One",
                "The Evidence model and the LLaMA probe are strong on different reviews; the Reflector learns when to trust which.")
ncm.shapes.add_picture(FIGD + "agents_complement.png", Inches(MARGIN + 0.4), Inches(1.72), width=Inches(CW - 0.8))
band(ncm, 6.05, "Read", "Short reviews: Evidence model leads (70.0 vs 54.6); long reviews: probe leads (64.3 vs 55.8); agents disagree: stack right 56% vs 36–50%. Hard cases: on reviews that sound negative but are rated 4–5★ the stack beats Phase 1 by +28 points on both test sets.", h=0.9)

# NP1 Phase 1 results (Electronics + reproduction on the paper's own data)
p1r = new_slide("PHASE 1 · RESULTS", "Phase 1 Results: The Baseline Paper's Method",
                "Data: Amazon Reviews 2023 Electronics, frozen disjoint splits (train ~497K, dev 500 + 2,000, balanced eval 1,000, natural test 2,000); test sets read once.")
lw = 6.6
accent_card(p1r, MARGIN, 1.82, lw, 3.6)
make_table(p1r, MARGIN + 0.26, 2.0, lw - 0.5, [2.4, 1.85, 1.85], None, ["Metric", "Balanced eval", "Natural test"],
           [[("Exact star", INK, True), ("54.0%", BODY, False), ("53.4%", BODY, False)], [("Within one star", INK, True), ("93.7%", POS, True), ("94.4%", POS, True)],
            [("Positive vs negative", INK, True), ("91.2%", POS, True), ("92.9%", POS, True)], [("Macro-F1", INK, True), (".543", BODY, False), (".488", BODY, False)],
            [("Mean abs. error (stars)", INK, True), ("0.53", BODY, False), ("0.54", BODY, False)]], row_h=0.5, head_h=0.34, aligns=[L, R, R])
rx = MARGIN + lw + GUTTER; rw = CW - lw - GUTTER
accent_card(p1r, rx, 1.82, rw, 3.6, bar=ACCENT_D)
_, tf = textbox(p1r, rx + 0.26, 1.98, rw - 0.5, 0.3); para(tf, "Reproduction on the paper's own data", size=T_CARD, color=INK, bold=True, first=True)
block(p1r, rx + 0.26, 2.4, rw - 0.48, 2.9, ["— Kaggle, paper protocol, leak-free: 69.1% exact (always guessing 5★: 63.8%)", "— Polarity 85.9%, binary 90.3%, within one 93.5%",
      "— Paper reports 92.6% with K=5 cross-validation and a separate 40K test set: not a like-for-like comparison", "— Our earlier 70.7% / 60.1% used the leaky prompt and are withdrawn"], size=T_SMALL, gap=5)
band(p1r, 5.6, "Read", "Phase 1 gets the direction right (within one star ~94%); picking the exact star between neighbours is hard. Natural-test exact is low because it was trained on balanced data while 65% of natural reviews are 5★.", h=0.95)

# NA Phase 2 system in depth (decision core + agentic layer)
arc = new_slide("PHASE 2 · SYSTEM", "Phase 2 System: How a Review Is Rated and Explained",
                "Decision core = the main stack. Agentic layer = routing, hand-off, self-correction and explanation. All built in LangGraph and measured on the 3,000 test reviews.")
n_ = 5; w_ = (CW - (n_ - 1) * 0.2) / n_
for i, (tag, head, body) in enumerate([("1", "Evidence model", "TF-IDF + LR on ~497K reviews; gives a quick draft star and the key phrases"),
                                       ("2", "Router", "Confident draft? keep it. Unsure? escalate to the LLaMA probe (self-correction 1)"),
                                       ("3", "LLaMA probe", "Frozen LLaMA-3-8B, layer-20 state, small classifier; reads meaning"),
                                       ("4", "Reflector", "Learned combiner of both agents gives the final star (an Instruct LLM re-rating unsure cases was tested: no gain)"),
                                       ("5", "Explainer + Critic", "Instruct LLM writes the rationale from the evidence; Critic checks it and forces rewrites (self-correction 3)")]):
    x = MARGIN + i * (w_ + 0.2)
    accent_card(arc, x, 1.9, w_, 2.75, bar=ACCENT if i in (0, 2, 3) else ACCENT_D); auto_pill(arc, x + 0.2, 2.08, 0.24, tag)
    _, tf = textbox(arc, x + 0.2, 2.5, w_ - 0.35, 0.4); para(tf, head, size=T_CARD, color=INK, bold=True, first=True)
    block(arc, x + 0.2, 3.0, w_ - 0.35, 1.6, [body], size=T_SMALL)
band(arc, 4.95, "LangGraph agent graph", "Conditional edges for the Router; a Hand-off agent after the Reflector sends unsure reviews to a person; a loop from the Critic back to the Explainer (max 2 rewrites). Every agent's output is stored, so each claim is measured on all 3,000 test reviews.", h=1.1)

# NC conclusions
con = new_slide("CONCLUSIONS", "Conclusions",
                "What the project shows, measured on the frozen test sets, before the Nov 1 / Nov 5 freezes.")
cards(con, [("PHASE 1", "Baseline reproduced, honestly", ["— Paper's LLaMA + LoRA on Electronics: 54.0 / 53.4 exact, ~94% within one star", "— Found a one-shot label leak (~8 pts) and that TextBlob filtering hurts"]),
            ("PHASE 2", "Two agents beat one model", ["— Main stack: 67.5% / 75.7% exact (+13.5 / +22.3 over Phase 1)", "— Significant after Holm; stable over 5 seeds", "— An LLM works better as a feature source than as a judge"]),
            ("AGENTIC LAYER", "Measured on 3,000 reviews", ["— Router: ~2 stars fixed per 1 broken; LLaMA skipped on 39–48%", "— Critic loop: faithful explanations 60–63% → 89%, 0 broken", "— Hand-off: 93% right on the 56% it answers (natural test)"]),
            ("LIMITS", "Stated plainly", ["— One dataset (Electronics 2023)", "— Part of the gain is the larger Reflector fit set", "— Explicit sarcasm too rare to measure"])],
      1.82, 3.4)
band(con, 5.45, "Next", "Freeze Nov 1 and Nov 5 → paper draft to supervisor Nov 8 → submission by Nov 20 → final presentation in December with a live demo of Phase 1 and Phase 2.", h=0.9)

# NSV how Phase 2 answers each shortcoming (replaces the repetitive "Agents That Route..." slide in the talk)
sv = new_slide("PHASE 2 · HOW IT SOLVES THE PROBLEM", "How Phase 2 Answers Each Shortcoming",
               "Every problem found in Phase 1 and in the original design has a specific fix, and every fix has a number that checks it.")
small_table(sv, [[(a, INK, True), (b, BODY, False), (c, POS, True)] for a, b, c in [
    ("One pass, no second opinion", "Two independent agents (LLaMA probe + Evidence model) and a learned Reflector", "Stack vs each agent: +5.0 to +6.3 eval, 5/5 seeds"),
    ("LLM as judge failed (Gate 0, twice)", "Specialists decide the star; LLMs explain and check instead", "Re-tested with an Instruct LLM: 76 fixed vs 89 broken (dropped)"),
    ("Wrong stars go uncorrected", "Unsure draft stars escalated and revised (self-correction 1)", "Measured: fixed 92 vs broken 50 (eval), 152 vs 78 (natural)"),
    ("Template justifications", "Explainer writes only from the evidence pack; Critic forces rewrites (self-correction 3)", "Faithful 60–63% → 89%, 0 broken; cited phrases pass the deletion test"),
    ("Fragile, leaky evaluation", "Leak-free prompt, frozen disjoint splits, rules written before runs, tests read once", "Leak audit −7.7 [−13.3, −2.0]"),
    ("Data cleaning backfires", "Raw balanced training data", "+2.5 to +3.5 pts"),
    ("Photos and similar-review votes did not help", "Visual Verifier excluded; RAG votes replaced by the Evidence model", "Constant output; 37.5% vs 54.2%"),
    ("LLaMA is slow (0.67 s / review)", "Router skips LLaMA when the Evidence model is confident", "Measured: skipped on 39% / 48%, accuracy unchanged")]],
    [3.1, 5.4, CW - 0.4 - 8.5], ["Shortcoming", "Phase 2 answer", "Checked by"], row_h=0.48)

# NSC1 self-correction 1 measured (gate_B1b: second Router attempt; first attempt disclosed)
sc1 = new_slide("PHASE 2 · SELF-CORRECTION 1", "Self-Correction 1: Two Stars Fixed for Each One Broken",
                "Unsure draft star (Evidence confidence < 0.70) → LLaMA probe called → Reflector revises it. Read once on both test sets.")
lw = 7.7
accent_card(sc1, MARGIN, 1.82, lw, 3.3)
make_table(sc1, MARGIN + 0.26, 2.0, lw - 0.5, [1.7, 1.25, 1.0, 1.0, 0.9, 1.35], None, ["Test set", "Escalated", "Fixed", "Broken", "Net", "p"],
           [[("Balanced eval", INK, True), ("607 of 1,000", BODY, False), ("92", POS, True), ("50", NEG, True), ("+42", POS, True), ("0.0005", BODY, False)],
            [("Natural test", INK, True), ("1,035 of 2,000", BODY, False), ("152", POS, True), ("78", NEG, True), ("+74", POS, True), ("0.000001", BODY, False)]],
           row_h=0.55, head_h=0.36, aligns=[L, R, R, R, R, R])
_, tf = textbox(sc1, MARGIN + 0.26, 3.65, lw - 0.5, 1.3)
para(tf, "Cost saving: LLaMA skipped on 39% (eval) / 48% (natural) of reviews; accuracy 67.4% vs 67.5% and 75.5% vs 75.7% for the full stack (no significant change).", size=T_SMALL, color=BODY, first=True)
rx = MARGIN + lw + GUTTER; rw = CW - lw - GUTTER
accent_card(sc1, rx, 1.82, rw, 3.3, bar=ACCENT_D)
_, tf = textbox(sc1, rx + 0.26, 1.98, rw - 0.5, 0.3); para(tf, "How it was tested", size=T_CARD, color=INK, bold=True, first=True)
block(sc1, rx + 0.26, 2.4, rw - 0.48, 2.6, ["— Rule written before running; threshold chosen on dev data only", "— Attempt 1 (0.60) failed its natural-dev check by 0.3 pt: no test read", "— Attempt 2 (0.70, declared and disclosed) passed; tests read once"], size=T_SMALL, gap=5)
band(sc1, 5.35, "Read", "When the system is unsure, a second agent revises the star, and the revision is right about twice as often as it is wrong. This is measured self-correction of the rating, not an assumption.", h=0.9)

# NHO Hand-off agent (e23/s1_selective.out; thresholds from dev2 quantiles)
nho = new_slide("PHASE 2 · HAND-OFF AGENT", "Hand-Off Agent: Answer When Sure, Ask a Person When Not",
                "After the Reflector, the Hand-off agent answers automatically when the stack's confidence is high and sends the rest to a person. Threshold fixed on dev2.")
nho.shapes.add_picture(FIGD + "selective_SH.png", Inches(MARGIN + 0.65), Inches(1.72), width=Inches(CW - 1.3))
band(nho, 5.9, "Read", "At the declared threshold the system answers 55.7% of natural-test reviews and is right 93.1% [91.6, 94.5] of the time (balanced eval: 85.0% on 42%). Reviews it hands off are only 54–55% right, so they are the right ones for a person. Phase 1 with the same rule: 67.9% / 66.9%.", h=1.0)

# NSC3 self-correction 3 (e23/b2_report.out, x1_explain_compare.out)
sc3 = new_slide("PHASE 2 · SELF-CORRECTION 3", "Self-Correction 3: The Critic Makes Explanations Faithful",
                "The Explainer writes from the evidence pack; the Critic checks it and forces rewrites (max 2). Measured on all 3,000 test reviews.")
lw = 6.3
accent_card(sc3, MARGIN, 1.82, lw, 3.75)
make_table(sc3, MARGIN + 0.26, 2.0, lw - 0.5, [1.3, 1.0, 1.0, 0.65, 0.9, 0.95], None, ["Test set", "First draft", "After Critic", "Fixed", "Broken", "p"],
           [[("Balanced eval", INK, True), ("60.5%", BODY, False), ("88.9%", POS, True), ("284", POS, True), ("0", BODY, False), ("<1e-85", BODY, False)],
            [("Natural test", INK, True), ("63.4%", BODY, False), ("88.8%", POS, True), ("509", POS, True), ("0", BODY, False), ("<1e-150", BODY, False)]],
           row_h=0.5, head_h=0.36, aligns=[L, R, R, R, R, R])
block(sc3, MARGIN + 0.26, 3.5, lw - 0.5, 2.0, ["— Phase 1 vs Phase 2, same 150 reviews, same judge: quotes the review 0% vs 79%; different outputs 3% vs 100%; no unsupported claim 85% vs 94%",
      "— 2.6 LLM calls and ~8 s per review (median)"], size=T_SMALL, gap=5)
rx = MARGIN + lw + GUTTER; rw = CW - lw - GUTTER
accent_card(sc3, rx, 1.82, rw, 3.75, bar=ACCENT_D)
_, tf = textbox(sc3, rx + 0.26, 1.98, rw - 0.5, 0.3); para(tf, "Deletion test: cited phrases vs random words", size=T_CARD, color=INK, bold=True, first=True)
sc3.shapes.add_picture(FIGD + "deletion_test.png", Inches(rx + 0.15), Inches(2.55), width=Inches(rw - 0.3))
block(sc3, rx + 0.26, 4.95, rw - 0.5, 0.6, ["Removing the cited phrases lowers confidence ~1.8× more than random words (p < 1e-46)"], size=T_SMALL)
band(sc3, 5.8, "Read", "The Critic catches bad first drafts and the rewrites fix them without breaking one good explanation. The phrases cited are the ones the decision depends on. About 11% still fail a check and are labelled as such.", h=0.95)

# ---- appendix divider
apx = new_slide("APPENDIX", "Appendix", "Background, superseded results and historical Run 1 slides. Shown only if asked.")
cards(apx, [("BACKGROUND", "Original setup", ["— Datasets (Kaggle vs Amazon 2023): slide §data_kaggle§", "— Phase 1 data pipeline: slide §p1_pipeline§", "— Kaggle baseline (leaky, withdrawn): slide §kaggle_baseline§", "— Original dissonance Critic: slide §dissonance§"]),
            ("SUPERSEDED", "Earlier Phase 2 results", ["— First LoRA + Evidence stack: slide §first_stack§", "— Preprocessing ablation: slide §preproc§", "— Nine-lines experiment summary: slide §ledger9§", "— Code workspace map: slide §code§"]),
            ("HISTORICAL", "Run 1 (All Beauty)", ["— Kaggle-trained adapter, leaky prompt", "— Results, error patterns, dissonance, audit, v2 design", "— Slides §hist_range§"])],
      1.82, 3.4)
# ---- order by key (main talk, then appendix); historical Run 1 slides last; template conclusion dropped (template file untouched)
SL = {"title": S[0], "problem": S[1], "gap": S[2], "data_kaggle": S[3], "p1_pipeline": S[4], "lora": S[5], "kaggle_baseline": S[6], "orig_design": S[7],
      "dissonance": S[8], **{f"hist{k}": S[8 + k] for k in range(1, 10)}, "g1": g1, "g2": g2, "glance": n0, "leak": n1, "protocol": n2, "first_stack": n3,
      "preproc": n4, "halluc": n5, "ledger9": n6, "x1": xs[0], "x2": xs[1], "kept": nk, "main": nh, "p1p2": np_, "bars": nf, "robust": nr, "next": nn,
      "schedule": ns, "roadmap": n7, "code": nv1, "workflow": nv2, "appendix": apx, "p1results": p1r, "arch": arc, "conclusions": con, "paper_what": pw, "paper_fails": pf, "errors": ner, "complement": ncm, "solves": sv, "sc1": sc1, "handoff": nho, "sc3": sc3}
MAIN = ["title", "problem",                                               # intro (key terms live in Presenter_Notes.md, not on slides; user 2026-10-25)
        "paper_what", "paper_fails",                                       # Phase 1: the baseline paper, what it does, why it falls short
        "p1_pipeline", "lora",                                             # how it works in detail
        "leak", "halluc", "preproc",                                       # evidence for the shortcomings
        "orig_design", "arch", "solves",                                   # Phase 2: intro, in depth, how it solves each shortcoming
        "p1results",                                                       # results: Phase 1 (carries the data / split facts)
        "main", "p1p2", "handoff",                                         # results: Phase 2 headline first, then the Hand-off agent
        "errors", "complement", "x1", "kept", "robust", "sc1", "sc3",      # in depth
        "schedule", "conclusions"]                                         # plan, conclusions
APPX = []   # max 25 slides (user, 2026-10-25); the rest lives in old_decks/ and stays defined above
ORDER = MAIN + APPX
ids = prs.slides._sldIdLst; els = list(ids); part_of = {id(prs.part.related_part(e.rId)): e for e in els}
order = [part_of[id(SL[k].part)] for k in ORDER]; keep = {id(e) for e in order}
for e in els: ids.remove(e)
for e in order: ids.append(e)
for e in els:
    if id(e) not in keep: prs.part.drop_rel(e.rId)                  # template conclusion slide only
NUM = {k: i for i, k in enumerate(ORDER, 1)}; n = len(ORDER)
TOK = {k: str(v) for k, v in NUM.items()}
if "hist1" in NUM: TOK["hist_range"] = f"{NUM['hist1']}–{NUM['hist9']}"
resolve = lambda t: re.sub(r"§(\w+)§", lambda m: TOK[m.group(1)], t)
PREFIX = re.compile(r"^(\d\d|A\d+|HISTORICAL \(SUPERSEDED\)|APPENDIX) · ")
for k in ORDER:
    sl, i = SL[k], NUM[k]
    for sh in sl.shapes:
        if not sh.has_text_frame: continue
        for p in sh.text_frame.paragraphs:
            for r in p.runs:
                if "§" in r.text: r.text = resolve(r.text)
        txt = sh.text_frame.text
        if re.fullmatch(r"\d\d / \d\d", txt): set_p(sh.text_frame.paragraphs[0], f"{i:02d} / {n}"); continue
        if k == "title" or k == "appendix" or sh.top >= 400000 or not (PREFIX.match(txt) or txt.isupper()): continue
        label = PREFIX.sub("", txt)
        set_p(sh.text_frame.paragraphs[0], f"{i - 1:02d} · {label}" if k in MAIN else
              (f"HISTORICAL (SUPERSEDED) · {label}" if k.startswith("hist") else f"APPENDIX · {label}"))

# ---- speaker notes by slide key; full script in presentation/Presenter_Notes.md, Part 0
NOTES = {
 "appendix": "Appendix divider. Everything after this is backup: background, superseded results and the historical Run 1 slides. Show only if asked.",
 "title": "Title. Say: this is the Oct 25 update. Main results are on Amazon Reviews 2023 Electronics only. The headline system is now a frozen LLaMA probe plus an Evidence model, combined by a Reflector.",
 "problem": "Problem. One LLM pass gives a star with no second opinion, no evidence and no check on its explanation; sarcasm, mixed reviews and neighbouring stars trip it up. Keep it to 1 minute; the rest of the talk shows how Phase 2 fixes each point.",
 "gap": "Research question and gap. Wang et al. 2025 is the baseline paper. Our question: does adding a second, independent signal and a combiner beat a single model?",
 "data_kaggle": "Data. Say: main results now use Electronics 2023 only. Kaggle is the original paper's data and appears only as reproduction.",
 "p1_pipeline": "Original paper's pipeline (TextBlob DQC, stratified, VGST, neutral oversampling). Point forward: slide §preproc§ shows TextBlob filtering hurts on Electronics.",
 "lora": "LoRA explained. LoRA is still trained (Phase 1), but in the final stack it is an ablation row, not the main agent.",
 "kaggle_baseline": "Phase 1 baseline on Kaggle (historical numbers). The 60.1% here used the leaky one-shot prompt; leak-free zero-shot is 59.2%. Slide §leak§ explains.",
 "orig_design": "Original Phase 2 design (Analyst, Visual, RAG, Critic). Say: this was the starting design; the Oct 9 audit (Run 2) and later experiments replaced it with the probe + Evidence + Reflector system (slide §main§). Slide §solves§ maps each problem to its fix.",
 "dissonance": "Original dissonance Critic. Say: in practice it fired on 0.4% of rows, so it was replaced by a learned Reflector.",
 "g1": "Key terms 1 (not in the deck). Do not read the table. Say: these are the names you will hear; the three to remember are the LLaMA probe, the Evidence model and the Reflector, which together make the main stack. Come back to this slide when a question uses a term.",
 "g2": "Key terms 2. Point at four: balanced eval vs natural test (why two numbers), gate (rule written before running), label leak (why old numbers were withdrawn), and the deletion test (how explanations are checked).",
 "glance": "AT A GLANCE. Read the three cards left to right. Must say the caveat: against the old stack refit on the same 2,000 dev rows the gain is only 2.0 / 1.5 points and not significant.",
 "leak": "Leak audit. Picking a one-shot example whose star differs from the truth leaks about 8 points. All later numbers are zero-shot and leak-free.",
 "protocol": "Protocol. Frozen disjoint splits; all choices on dev; each test split read once per declared config. Since Oct 24 there is also dev2 (2,000 balanced) for the Reflector.",
 "first_stack": "First stack (Oct 13). Phase 1 LoRA + Evidence = 63.9 / 73.4. Superseded by the main Probe + Evidence stack, kept for continuity.",
 "preproc": "Preprocessing ablation. TextBlob DQC deletes most 1 and 2 star reviews; VGST picks long reviews; both are dropped.",
 "halluc": "Hallucination study. LoRA explanations are a canned template; digit probabilities are the usable signal. That is why Phase 2 combines probabilities, not explanations.",
 "x1": "Full ledger (one slide; the last two rows are the 120K probe, which added only +0.3 on dev2 so the 40K probe stays, and the Instruct-LLM re-rating, 76 fixed vs 89 broken, the third LLM-judge failure). Do not read every row. Pick three: the LLM judge (failed twice, which is why the LLM does not decide the star), the Visual Verifier (constant output), and the recalibration (the gain was the leak). Say: every rule was written before the run, on dev data only.",
 "x2": "Full ledger, part 2. New last row: the Instruct-LLM re-rating failed too (76 fixed vs 89 broken), the third LLM-judge failure. Message: more views of the same text, more LoRA data and fancier Reflectors did not help; what helped was more data for the probe and a bigger Reflector fit set. Note the honest near-miss: the 10K probe missed by .001 and only the 40K version passed; tripling to 120K (Oct 25 night) added only +0.3 on dev2, so the probe has saturated.",
 "kept": "What we kept. Walk the ten rows quickly: each change has its evidence on the right. Together they give the main stack, 67.5% balanced eval and 75.7% natural test.",
 "ledger9": "Experiment ledger. Go row by row, about 10 seconds each. Key message: every rule was fixed before running and failures never touched the test sets.",
 "main": "MAIN RESULT. The main stack (= Probe + Evidence stack: LLaMA probe reads meaning, Evidence model reads key words, Reflector combines them) gets 67.5% on balanced eval and 75.7% on natural test, macro-F1 .675 / .612; 5 seeds 65.8 +/- 0.6 and 75.4 +/- 0.8. If asked whether that is good: exact five-star rating is hard because neighbouring stars are ambiguous; random is 20%, base LLaMA 34%, SVM ~48%, DistilBERT 53%, Phase 1 54%; we are within one star 97% of the time. Read the honest-read band aloud.",
 "paper_what": "The paper we chose. Four cards, left to right: the task (one model rates a review 1-5 and justifies it), the data steps (TextBlob cleaning, equal per star, varied wording, extra neutral), the model (LLaMA-3-8B with a small LoRA add-on, exact Table 1 settings), and the prompt (rubric, one example, step-by-step). Then the reported 92.6%. Say: we rebuilt exactly this as Phase 1.",
 "paper_fails": "Why it falls short. Four reasons in plain words: one pass with nothing to catch mistakes; the score depends heavily on the set-up (a leaked example, a skewed test, leak-free it is 54%); the data cleaning removes most negative reviews; and the justifications are a template. Be fair: the leak was in our first reproduction's example choice, not a claim about the authors. Next: two slides on how the method works, then three slides of evidence.",
 "p1results": "Phase 1 results. First the data line in the subtitle: one dataset, splits drawn once and disjoint by review id and exact text, every choice made on dev, test sets read once (this answers any leakage question). Then: this is the baseline paper's method retrained on Electronics, leak-free. Exact star 54.0 / 53.4, but within one star about 94% and positive vs negative 91-93%: it gets the direction right. On the paper's own Kaggle data with its protocol and no leak it reaches 69.1% (always-5 gives 63.8%). The paper's 92.6% used a different protocol; our old 70.7 / 60.1 were leaky and withdrawn.",
 "arch": "Phase 2 system. Walk the five boxes left to right. Decision core: Evidence model, LLaMA probe, Reflector (the main stack, built and frozen). Agentic layer: Router with escalation, the Hand-off agent, Explainer and Critic loop, all built in LangGraph and measured on the 3,000 test reviews (LLM re-rating was tested and dropped). Self-correction happens at three points: escalation, re-rating, and the explanation loop.",
 "errors": "Results in detail. Left: confusion matrix, mistakes hug the diagonal, 90% of errors are one star off. Right: per-star recall, Phase 1 vs Phase 2. Be honest about the one exception: on natural test Phase 1 finds more 4-star reviews, because it over-predicts 4-star; Phase 2 is far better on 5-star, the most common class.",
 "complement": "Why it works. Left: the Evidence model wins on short reviews, the LLaMA probe on long reviews, the stack is best or near-best in every length band. Right: when the two agents disagree, the stack still gets 56% right, versus 36-50% for either alone. That is the case for stacking two different kinds of signal. Hard cases (band): reviews that sound negative but are rated 4-5 stars, +28 points over Phase 1 on both test sets; positive-sounding 1-2 star reviews +15 on natural test; mixed reviews: small or no gain; explicit sarcasm is too rare to measure (6 and 1 reviews).",
 "solves": "How Phase 2 answers each shortcoming. Read the left column as the problems the audience just saw, then the fix, then the number that checks it. Spend time on three rows: two agents instead of one pass; LLMs explain and check instead of judging (because the judge failed twice); and template justifications replaced by evidence-based explanations checked by a Critic. All rows are now measured: escalation 92 vs 50 and 152 vs 78; explanations faithful 60-63% to 89% with 0 broken.",
 "sc1": "Self-correction 1, measured. Say: the Evidence model gives a quick draft; when it is unsure, the LLaMA probe is called and the Reflector revises the star. On the test sets the revision fixed 92 stars and broke 50 on balanced eval, 152 vs 78 on natural test, both highly significant. And LLaMA was skipped on 39-48% of reviews with no accuracy change. Be open that this was the second attempt: the first threshold failed its natural-dev check by 0.3 points and was never tested.",
 "conclusions": "Conclusions. Four cards: Phase 1 reproduced honestly (and two problems found in the original protocol); Phase 2's two-agent stack beats the single fine-tuned model, significantly; the agentic layer is measured (escalation fixes about two stars per one broken, the Critic loop makes 89% of explanations faithful with none broken, the Hand-off agent is 93% right on the 56% it answers); and the limits, said plainly (one dataset, part of the gain from the bigger fit set, sarcasm too rare to measure). End with the dates: freeze Nov 1 / 5, draft Nov 8, submit Nov 20, final presentation in December.",
 "p1p2": "Phase 1 to Phase 2. Lead with the gain: +13.5 points on balanced eval and +22.3 on natural test, 29% and 48% fewer errors, significant after Holm. Then why: the agents complement (Evidence model on short reviews, probe on long ones). Say the caveat yourself: the earlier stack refit on the same dev reviews is only 2.0 / 1.5 behind. Never quote 70.7 / 60.1 (leaky), 74.8 (one seed) or 79.9 (class-mix correction).",
 "handoff": "Hand-off agent. Say: accuracy has plateaued at about 67 / 76, so the agentic answer is to know when you are unsure. The Hand-off agent answers automatically only when the stack is confident; the threshold was fixed on dev2 before the test sets were read. On natural test it answers 56% of reviews and is right 93% of the time; on balanced eval 85% on 42%. What it hands to a person is only 54-55% right, so those are the hard ones. Phase 1 with the same rule reaches only 67-68%.",
 "sc3": "Self-correction 3, measured. Say: the Explainer writes a short explanation from the evidence only; the Critic checks five things (star, quotes really in the review, key phrases, disagreement, no invented claims) and sends it back with specific feedback. First drafts are faithful 60-63% of the time; after the loop 89%, and it never broke a good one (793 fixed, 0 broken). The deletion test shows the cited phrases are what the decision depends on. Compare with Phase 1: its justification never quotes the review and is one of 5 stock sentences. Be open: about 11% still fail a check.",
 "bars": "Bar chart of all frozen models with 95% CIs; main stack in red. Point out: base LLaMA zero-shot is weak (34%) but a probe on its hidden states reaches 61 / 74.",
 "robust": "Robustness. After Holm correction 5 of 8 comparisons stay significant, including the main stack vs the earlier LoRA + Evidence stack on both splits. Most errors are one star off. Calibration is good (ECE .05-.06). The probe costs 0.67 s per review, the rest is under 0.1 ms: that is why the next step adds a Router.",
 "next": "How Phase 2 aims to solve the problem. Say: the specialists decide the star because an LLM choosing the star failed twice (Gate 0). The agents now do what LLMs are good at: routing, explaining, and checking explanations. Self-correction happens at three levels: escalation of unsure draft stars (fixed vs broken counted), an Instruct LLM re-rating uncertain reviews (kept only if it passes its dev check), and the Critic loop on explanations. Every explanation is checked and measured on all 3,000 test reviews. The bigger LLaMA probe (120K reviews) was tested overnight and gave no gain, so the 40K probe stays; an optional extra specialist may be tried on a GPU on Nov 1. Final demo is a local notebook.",
 "code": "Code workspace. v2_reflective started as the Oct 10 'reflective v2' design (appendix slide §hist9§); the name stayed and it now holds every Phase 2 experiment. Top level = Kaggle close-out; e23 = Electronics 2023, all current results. Walk the table top to bottom: data, agents, gates, robustness, freeze. Full guide: presentation/V2_Reflective_Guide.md.",
 "workflow": "Workflow. Six steps for every experiment: declare in LOG.md, run resumably, gate on dev only, read the tests once, freeze, record and back up. Say: 16 failures are reported and none touched the test sets, which is why the one pass can be trusted.",
 "schedule": "Build schedule, Oct 25 to Nov 5. Say: every row has its rule written first and is logged. The bigger probe failed its check (+0.3 on dev2), so the 40K probe stays. The explanations and Critic loop ran on all 3,000 test reviews (faithful 60-63% to 89%, 0 broken), then the deletion test; report, paper tables, hard cases and the demo notebook are done. Freeze stage 1 on Nov 1, stage 2 on Nov 5.",
 "roadmap": "Roadmap. Freeze stage 1 Nov 1 (local work), stage 2 Nov 5 (Colab result in or out). Author writes the prose from Nov 2; draft 1 to supervisor Nov 8; submit by Nov 20. December: final presentation with the live demo notebook (Phase 1 and Phase 2).",
}
for k in ORDER:
    SL[k].notes_slide.notes_text_frame.text = resolve(NOTES.get(k, "Historical slide (Run 1, All Beauty, Kaggle-trained adapter, leaky prompt). Superseded; show only if asked."))
# ---- slide map for the presenter notes (auto-generated, so numbers never drift)
title_of = lambda sl: next((sh.text_frame.text for sh in sl.shapes if sh.has_text_frame and 500000 < sh.top < 700000 and sh.text_frame.text.strip()), "")
with open(os.path.join(ROOT, "presentation", "deck_slide_map.md"), "w") as fm:
    fm.write("# Deck slide map and short scripts (auto-generated by v2_reflective/build_deck_e23.py on every build; do not edit)\n\n"
             "Same text as the PowerPoint / Google Slides speaker notes. Longer scripts: `Presenter_Notes.md`, Part 0.\n\n| Slide | Section | Title | Short script (speaker note) |\n|---|---|---|---|\n")
    for k in ORDER:
        note = SL[k].notes_slide.notes_text_frame.text.replace("|", "/")
        fm.write(f"| {NUM[k]} | {'main talk' if k in MAIN else 'appendix'} | {title_of(SL[k]) or ('Title' if k == 'title' else k)} | {note} |\n")
prs.save(DST); print("saved", DST, n)
# Always-current copy for Google Slides: overwrite ONLY this one file in Drive (user-approved 2026-10-25). Dated backups elsewhere stay cp -n.
import shutil
GD = os.path.expanduser("~/Library/CloudStorage/GoogleDrive-kvarun314@gmail.com/My Drive/PES_backups/current_deck")
if os.path.isdir(os.path.dirname(GD)):
    os.makedirs(GD, exist_ok=True); shutil.copyfile(DST, os.path.join(GD, os.path.basename(DST))); print("synced to Drive:", os.path.join(GD, os.path.basename(DST)))
else:
    print("Drive folder not found; NOT synced")
