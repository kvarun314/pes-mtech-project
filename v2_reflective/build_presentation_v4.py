"""
Rebuilds Agentic_LLaMA3_Presentation_v3.pptx as a cleaner, typographically
consistent v4.

Content is unchanged. What changes is construction:
  * one text frame per block (v3 used one text box per *line*, ~1,200 shapes)
  * a single type scale, spacing grid and palette
  * chart images placed with their true aspect ratio (v3 stretched most of them)
  * no emoji, no overlapping shapes, no empty spacer text boxes

Run:  python build_presentation_v4.py
"""

import copy
import os

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "Agentic_LLaMA3_Presentation_v3.pptx")
DST = os.path.join(HERE, "Agentic_LLaMA3_Presentation_v4.pptx")
IMGDIR = os.path.join(HERE, ".ppt_assets")

# ---------------------------------------------------------------- design system

FONT = "Arial"  # Calibri is not installed on this machine; Arial renders
                # identically on macOS and Windows, so the deck cannot reflow.

INK = RGBColor(0x1A, 0x23, 0x32)   # headings
BODY = RGBColor(0x3D, 0x4C, 0x5C)  # body copy
MUTED = RGBColor(0x87, 0x92, 0xA0)  # captions, page numbers
ACCENT = RGBColor(0x2C, 0x5A, 0xA0)  # brand blue
ACCENT_D = RGBColor(0x1B, 0x3C, 0x6E)  # deep blue
TINT = RGBColor(0xEC, 0xF2, 0xF9)   # accent wash
SURFACE = RGBColor(0xF7, 0xF9, 0xFB)  # card fill
BORDER = RGBColor(0xE1, 0xE8, 0xEF)  # card stroke
RULE = RGBColor(0xD8, 0xE0, 0xE8)   # hairlines
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
POS = RGBColor(0x1E, 0x7A, 0x46)    # positive delta
NEG = RGBColor(0xA9, 0x36, 0x36)    # negative delta

# type scale (pt)
T_TITLE = 27
T_SUB = 12.5
T_EYEBROW = 8.5
T_CARD = 12.5
T_BODY = 10.5
T_SMALL = 9.5
T_MICRO = 8.5
T_STAT = 26
T_STAT_SM = 19

LINE_EM = 1.15   # Arial hhea (asc - desc + gap) / upem: the real line box

# grid (inches)
MARGIN = 0.6
CW = 13.333 - 2 * MARGIN      # 12.133 content width
TOP = 1.82                    # content zone top
BOTTOM = 6.88                 # content zone bottom
GUTTER = 0.26

TOTAL = 18

# ------------------------------------------------------------------- primitives


def _spc(run, hundredths):
    """Character spacing, in 1/100 pt. PowerPoint's `spc` attribute."""
    run.font._rPr.set("spc", str(int(hundredths)))


def _no_autofit(tf):
    bodyPr = tf._txBody.find(qn("a:bodyPr"))
    for tag in ("a:normAutofit", "a:spAutoFit"):
        el = bodyPr.find(qn(tag))
        if el is not None:
            bodyPr.remove(el)
    bodyPr.append(bodyPr.makeelement(qn("a:noAutofit"), {}))


def textbox(sl, x, y, w, h, anchor=MSO_ANCHOR.TOP):
    box = sl.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    _no_autofit(tf)
    return box, tf


def para(tf, text, size=T_BODY, color=BODY, bold=False, align=PP_ALIGN.LEFT,
         space_after=0, space_before=0, line=1.18, first=False, spc=None,
         font=FONT, hang=None):
    p = tf.paragraphs[0] if first else tf.add_paragraph()
    p.alignment = align
    if hang:
        # hanging indent: wrapped lines align under the text, not under the
        # dash. Without this a two-line bullet reads as two bullets.
        pPr = p._p.get_or_add_pPr()
        pPr.set("marL", str(int(hang * 914400)))
        pPr.set("indent", str(int(-hang * 914400)))
    p.line_spacing = line
    p.space_after = Pt(space_after)
    p.space_before = Pt(space_before)
    r = p.add_run()
    r.text = text
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.name = font
    r.font.color.rgb = color
    if spc:
        _spc(r, spc)
    return p


def block(sl, x, y, w, h, lines, size=T_BODY, color=BODY, gap=4.5,
          anchor=MSO_ANCHOR.TOP, align=PP_ALIGN.LEFT, line=1.18):
    """A single text frame holding several paragraphs — replaces v3's
    one-text-box-per-line construction."""
    _, tf = textbox(sl, x, y, w, h, anchor)
    for i, ln in enumerate(lines):
        if isinstance(ln, tuple):
            txt, kw = ln
        else:
            txt, kw = ln, {}
        para(tf, txt, size=kw.get("size", size), color=kw.get("color", color),
             bold=kw.get("bold", False), align=kw.get("align", align),
             space_after=kw.get("gap", gap), line=kw.get("line", line),
             first=(i == 0),
             hang=kw.get("hang", 0.165 if txt.startswith("—") else None))
    return tf


def _no_shadow(sh):
    """Turn off the drop shadow every autoshape gets by default.

    Two things have to go. `shadow.inherit = False` clears neither:
      1. an empty <a:effectLst/> in spPr, and
      2. the <p:style> block python-pptx attaches to new autoshapes, whose
         <a:effectRef idx="2"/> pulls a shadow straight from the theme and
         wins over spPr. We set every property explicitly, so the whole
         style reference can go.
    """
    spPr = sh._element.spPr
    for tag in ("a:effectLst", "a:effectDag"):
        for el in spPr.findall(qn(tag)):
            spPr.remove(el)
    spPr.append(spPr.makeelement(qn("a:effectLst"), {}))
    style = sh._element.find(qn("p:style"))
    if style is not None:
        sh._element.remove(style)


def rect(sl, x, y, w, h, fill=None, line_color=None, line_w=0.75,
         shape=MSO_SHAPE.RECTANGLE, radius=None):
    sh = sl.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    sh.shadow.inherit = False
    _no_shadow(sh)
    if fill is None:
        sh.fill.background()
    else:
        sh.fill.solid()
        sh.fill.fore_color.rgb = fill
    if line_color is None:
        sh.line.fill.background()
    else:
        sh.line.color.rgb = line_color
        sh.line.width = Pt(line_w)
    if radius is not None and shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        # `radius` is in INCHES; convert to the fraction roundRect expects
        sh.adjustments[0] = max(0.0, min(0.5, radius / max(0.01, min(w, h))))
    tf = sh.text_frame
    tf.text = ""
    tf.margin_left = tf.margin_right = Inches(0.04)
    tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.word_wrap = False
    _no_autofit(tf)
    return sh


R_CARD = 0.085   # inches — one corner radius for every card in the deck
R_PLATE = 0.07
R_PILL = 0.5     # fully round ends


def card(sl, x, y, w, h, fill=SURFACE, border=BORDER):
    return rect(sl, x, y, w, h, fill=fill, line_color=border,
                shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=R_CARD)


def accent_card(sl, x, y, w, h, bar=ACCENT, fill=SURFACE):
    """Card with a thin colour bar down the left edge — replaces v3's emoji."""
    c = card(sl, x, y, w, h, fill=fill)
    rect(sl, x, y + 0.10, 0.045, h - 0.20, fill=bar)
    return c


_MEASURE = {}


def text_w(text, size, bold=True):
    """Width of `text` in inches at `size` pt, measured from the real Arial
    face — used to size pills to their content instead of guessing."""
    try:
        from PIL import ImageFont
        key = bold
        if key not in _MEASURE:
            base = "/System/Library/Fonts/Supplemental/Arial"
            _MEASURE[key] = base + (" Bold.ttf" if bold else ".ttf")
        f = ImageFont.truetype(_MEASURE[key], 200)
        return f.getlength(text) / 200 * size / 72
    except Exception:                      # no Pillow / not macOS
        return 0.062 * size / 9.0 * len(text)


def center_y(band_y, band_h, size):
    """Top-anchored y that puts a single line of `size` pt on the vertical
    centre of a band — two top-anchored boxes at the same y do NOT line up
    when their point sizes differ."""
    return band_y + (band_h - size * LINE_EM / 72) / 2


def baseline_y(ref_y, ref_size, size):
    """Top-anchored y so a line of `size` centres against a line of
    `ref_size` that starts at `ref_y`."""
    return ref_y + (ref_size - size) * LINE_EM / 2 / 72


def n_lines(text, width, size=T_BODY, bold=False):
    """How many lines `text` takes at `size` pt inside `width` inches."""
    total = 0
    for hard in text.split("\n"):
        w, started = 0.0, False
        for word in hard.split(" "):
            ww = text_w(word + " ", size, bold)
            if started and w + ww > width:
                total += 1
                w = ww
            else:
                w += ww
            started = True
        total += 1
    return total


def auto_pill(sl, x, y, h, text, fill=ACCENT, color=WHITE, size=T_MICRO,
              pad=0.30):
    # rounded ends eat horizontal room, so the pad is measured, not guessed
    w = text_w(text, size, True) + 2 * pad
    pill(sl, x, y, w, h, text, fill=fill, color=color, size=size)
    return w


def pill(sl, x, y, w, h, text, fill=ACCENT, color=WHITE, size=T_MICRO):
    """Label pill. The text is set on the shape's OWN text frame rather than
    a floating text box on top of it — an overlay relies on our arithmetic
    matching PowerPoint's vertical metrics, and it does not, so the label
    drifts down and clips against the pill edge."""
    sh = rect(sl, x, y, w, h, fill=fill, shape=MSO_SHAPE.ROUNDED_RECTANGLE,
              radius=h * R_PILL)
    para(sh.text_frame, text, size=size, color=color, bold=True,
         align=PP_ALIGN.CENTER, first=True, line=1.0)
    return sh


def pic_fit(sl, path, x, y, w, h, align="center"):
    """Place an image contained inside the box, preserving aspect ratio.
    v3 stretched six of its eight charts; this is the main visual fix."""
    from PIL import Image  # pillow ships with python-pptx installs
    iw, ih = Image.open(path).size
    ar = iw / ih
    if w / h > ar:          # box wider than image -> height-bound
        ph, pw = h, h * ar
    else:                   # width-bound
        pw, ph = w, w / ar
    px = x + (w - pw) / 2 if align == "center" else x
    py = y + (h - ph) / 2
    return sl.shapes.add_picture(path, Inches(px), Inches(py),
                                 Inches(pw), Inches(ph))


# ------------------------------------------------------------------ table utils

def _cell_borders(cell, bottom=None, width=0.75):
    tcPr = cell._tc.get_or_add_tcPr()
    for tag in ("a:lnL", "a:lnR", "a:lnT", "a:lnB"):
        for el in tcPr.findall(qn(tag)):
            tcPr.remove(el)
    for tag in ("a:lnL", "a:lnR", "a:lnT"):
        ln = tcPr.makeelement(qn(tag), {"w": "0", "cap": "flat"})
        ln.append(ln.makeelement(qn("a:noFill"), {}))
        tcPr.append(ln)
    ln = tcPr.makeelement(qn("a:lnB"),
                          {"w": str(int(width * 12700)), "cap": "flat"})
    if bottom is None:
        ln.append(ln.makeelement(qn("a:noFill"), {}))
    else:
        fill = ln.makeelement(qn("a:solidFill"), {})
        clr = fill.makeelement(qn("a:srgbClr"), {"val": str(bottom)})
        fill.append(clr)
        ln.append(fill)
    tcPr.append(ln)


def make_table(sl, x, y, w, col_w, rows, header, body_rows, row_h=0.36,
               head_h=0.34, aligns=None, zebra=True):
    """Native PowerPoint table, restyled flat: no grid, hairline row rules,
    zebra fill. Editable by hand afterwards, unlike v3's floating text boxes."""
    n_rows = len(body_rows) + 1
    n_cols = len(header)
    gf = sl.shapes.add_table(n_rows, n_cols, Inches(x), Inches(y),
                             Inches(w), Inches(head_h + row_h * (n_rows - 1)))
    tbl = gf.table
    # kill the banded-style theme so our own fills show
    tblPr = tbl._tbl.find(qn("a:tblPr"))
    tblPr.set("firstRow", "0")
    tblPr.set("bandRow", "0")
    for el in tblPr.findall(qn("a:tableStyleId")):
        tblPr.remove(el)

    for i, cw in enumerate(col_w):
        tbl.columns[i].width = Inches(cw)
    tbl.rows[0].height = Inches(head_h)
    for r in range(1, n_rows):
        tbl.rows[r].height = Inches(row_h)

    aligns = aligns or [PP_ALIGN.LEFT] * n_cols

    for c, label in enumerate(header):
        cell = tbl.cell(0, c)
        cell.fill.background()
        cell.margin_left = Inches(0.10)
        cell.margin_right = Inches(0.10)
        cell.margin_top = cell.margin_bottom = Inches(0.02)
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        _cell_borders(cell, bottom="C3CFDB", width=1.0)
        tf = cell.text_frame
        tf.word_wrap = True
        para(tf, label.upper(), size=T_MICRO, color=MUTED, bold=True,
             align=aligns[c], first=True, spc=60, line=1.0)

    for r, row in enumerate(body_rows, start=1):
        for c, val in enumerate(row):
            txt, colr, bold = (val if isinstance(val, tuple)
                               else (val, INK if c == 0 else BODY, c == 0))
            cell = tbl.cell(r, c)
            if zebra and r % 2 == 1:
                # tables sit on a SURFACE card, so the stripe has to be white
                cell.fill.solid()
                cell.fill.fore_color.rgb = WHITE
            else:
                cell.fill.background()
            cell.margin_left = Inches(0.10)
            cell.margin_right = Inches(0.10)
            cell.margin_top = cell.margin_bottom = Inches(0.02)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            _cell_borders(cell, bottom="EDF1F5" if r < n_rows - 1 else None)
            tf = cell.text_frame
            tf.word_wrap = True
            para(tf, txt, size=T_BODY, color=colr, bold=bold,
                 align=aligns[c], first=True, line=1.0)
    return tbl


# --------------------------------------------------------------- slide chrome

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]

_page = {"n": 0}


def new_slide(eyebrow=None, title=None, subtitle=None, chrome=True):
    sl = prs.slides.add_slide(BLANK)
    _page["n"] += 1
    rect(sl, 0, 0, 13.333, 0.05, fill=ACCENT)
    if eyebrow is not None:
        _, tf = textbox(sl, MARGIN, 0.36, CW, 0.20)
        para(tf, eyebrow.upper(), size=T_EYEBROW, color=ACCENT, bold=True,
             first=True, spc=120, line=1.0)
    if title is not None:
        _, tf = textbox(sl, MARGIN, 0.60, CW, 0.48)
        para(tf, title, size=T_TITLE, color=INK, bold=True, first=True,
             line=1.0)
    if subtitle is not None:
        _, tf = textbox(sl, MARGIN, 1.16, CW, 0.28)
        para(tf, subtitle, size=T_SUB, color=BODY, first=True, line=1.1)
    if chrome:
        rect(sl, MARGIN, 1.60, CW, 0.011, fill=RULE)
        _, tf = textbox(sl, 13.333 - MARGIN - 1.6, 7.06, 1.6, 0.22)
        para(tf, f"{_page['n']:02d} / {TOTAL}", size=T_MICRO, color=MUTED,
             align=PP_ALIGN.RIGHT, first=True, line=1.0)
    return sl


def band(sl, y, label, text, fill=TINT, h=None):
    """Full-width takeaway strip, height fitted to its own copy so it never
    leaves a half-empty box at the foot of the slide."""
    inner = CW - 0.55
    if h is None:
        lines = 0
        for hard in text.split("\n"):
            w, cur = 0.0, 0
            for word in hard.split(" "):
                ww = text_w(word + " ", T_BODY, False)
                if w + ww > inner and cur:
                    lines += 1
                    w = ww
                else:
                    w += ww
                cur += 1
            lines += 1
        h = 0.40 + lines * (T_BODY * LINE_EM * 1.28 / 72) + 0.20
    rect(sl, MARGIN, y, CW, h, fill=fill, line_color=None,
         shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=R_CARD)
    rect(sl, MARGIN, y + 0.12, 0.045, h - 0.24, fill=ACCENT)
    _, tf = textbox(sl, MARGIN + 0.26, y + 0.14, 1.9, 0.22)
    para(tf, label.upper(), size=T_MICRO, color=ACCENT, bold=True, first=True,
         spc=90, line=1.0)
    _, tf = textbox(sl, MARGIN + 0.26, y + 0.40, inner, h - 0.52)
    para(tf, text, size=T_BODY, color=INK, first=True, line=1.28)
    return h


# ------------------------------------------------------------------- image prep

os.makedirs(IMGDIR, exist_ok=True)
src = Presentation(SRC)
IMG = {}
for idx, s in enumerate(src.slides, 1):
    k = 0
    for sh in s.shapes:
        if sh.shape_type == 13:
            k += 1
            img = sh.image
            path = os.path.join(IMGDIR, f"chart_s{idx:02d}_{k}.{img.ext}")
            with open(path, "wb") as fh:
                fh.write(img.blob)
            # Two source charts were exported with their figure supertitle
            # already clipped by the top edge; crop the stub off so the slide
            # heading carries the title instead.
            if f"{idx}_{k}" in ("11_1", "12_1"):
                from PIL import Image
                im = Image.open(path)
                im.crop((0, 24, im.width, im.height)).save(path)
            IMG[f"{idx}_{k}"] = path

# =============================================================== 01 · title

sl = new_slide(chrome=False)
rect(sl, 0, 0, 13.333, 0.05, fill=ACCENT)
rect(sl, 0, 0.05, 13.333, 0.018, fill=ACCENT_D)

_, tf = textbox(sl, MARGIN + 0.15, 2.12, 11.6, 1.30)
para(tf, "Agentic Self-Correction in LLaMA-3", size=40, color=INK, bold=True,
     first=True, line=1.05)
_, tf = textbox(sl, MARGIN + 0.15, 3.12, 11.6, 0.5)
para(tf, "Enhancing E-Commerce Sentiment Analysis through Multi-Agent "
         "Reflection", size=18.5, color=BODY, first=True, line=1.15)

rect(sl, MARGIN + 0.15, 3.84, 4.2, 0.022, fill=ACCENT)

tags = ["Phase 1: Baseline",
        "Phase 2: Multi-Agent Agentic",
        "Amazon Reviews 2023 · 2,000 Samples"]
tx = MARGIN + 0.15
for t in tags:
    tx += auto_pill(sl, tx, 4.16, 0.34, t, fill=TINT, color=ACCENT_D,
                    size=9.5, pad=0.34) + 0.16

_, tf = textbox(sl, MARGIN + 0.15, 4.86, 11.6, 0.3)
para(tf, "LLaMA-3-8B + LoRA    ·    BLIP Visual Verifier    ·    LanceDB RAG"
         "    ·    Dissonance-Triggered Self-Correction",
     size=12.5, color=BODY, first=True, line=1.0)

rect(sl, MARGIN + 0.15, 5.56, 11.6, 0.011, fill=RULE)
_, tf = textbox(sl, MARGIN + 0.15, 5.74, 11.6, 0.3)
para(tf, "Varun K    ·    PES MTech Project    ·    March 2026",
     size=11, color=MUTED, first=True, line=1.0)

# =========================================================== 02 · problem

sl = new_slide("01 · Problem statement",
               "The Problem: Why Single-Pass LLMs Fail",
               "LLaMA-3 excels at language — but a single inference step has "
               "no factual verification")

cols = [
    ("Sentiment Hallucinations",
     ["AI mistakes user error for product defect",
      "e.g. “watch isn’t waterproof” labelled as a defect",
      "No access to spec sheets to verify claims"]),
    ("Context Blindness",
     ["Ignores product images and metadata",
      "Text-only: blind to a shattered-screen photo",
      "Missing visual evidence = incomplete sentiment"]),
    ("Linguistic Errors",
     ["Misreads sarcasm: “Amazing, it broke in 2 days”",
      "No critic to trigger re-evaluation",
      "Mixed sentiment scored incorrectly"]),
]
cw = (CW - 2 * GUTTER) / 3
lh = T_BODY * LINE_EM * 1.22 / 72
# size the cards to the tallest column, then centre the whole group in the
# content zone — a 3.2in card holding 1.2in of copy just looks unfinished
body_hs = [sum(n_lines(f"— {t}", cw - 0.62) * lh + 8 / 72 for t in items)
           for _, items in cols]
CH = 0.34 + 0.30 + 0.24 + max(body_hs) + 0.34
GROUP = CH + 0.30 + 1.05
y0 = TOP + max(0.0, (BOTTOM - TOP - GROUP) / 2)
for i, (head, items) in enumerate(cols):
    x = MARGIN + i * (cw + GUTTER)
    accent_card(sl, x, y0, cw, CH)
    _, tf = textbox(sl, x + 0.32, y0 + 0.34, cw - 0.6, 0.3)
    para(tf, head, size=T_CARD, color=ACCENT_D, bold=True, first=True,
         line=1.0)
    block(sl, x + 0.32, y0 + 0.88, cw - 0.62, max(body_hs) + 0.1,
          [f"— {t}" for t in items], size=T_BODY, gap=8, line=1.22)

band(sl, y0 + CH + 0.30, "The gap",
     "A system is needed that doesn’t just read sentiment — but verifies it "
     "against product facts and multimodal evidence.\nThe transition: passive "
     "single-pass classifier  →  agentic verifier-in-the-loop.")

# ======================================================= 03 · research gap

sl = new_slide("02 · Research gap",
               "From Passive Classifier to Agentic System",
               "What the baseline does well — and the three gaps that "
               "motivate Phase 2")

lw = 4.05
card(sl, MARGIN, TOP, lw, 4.62)
_, tf = textbox(sl, MARGIN + 0.28, TOP + 0.26, lw - 0.56, 0.3)
para(tf, "Baseline", size=T_CARD, color=INK, bold=True, first=True, line=1.0)
_, tf = textbox(sl, MARGIN + 0.28, TOP + 0.55, lw - 0.56, 0.24)
para(tf, "Wang et al., 2025", size=T_SMALL, color=MUTED, first=True, line=1.0)
rect(sl, MARGIN + 0.28, TOP + 0.90, lw - 0.56, 0.011, fill=RULE)

_, tf = textbox(sl, MARGIN + 0.28, TOP + 1.08, lw - 0.56, 0.2)
para(tf, "WHAT IT ESTABLISHES", size=T_MICRO, color=POS, bold=True,
     first=True, spc=80, line=1.0)
block(sl, MARGIN + 0.28, TOP + 1.38, lw - 0.56, 1.9,
      ["LLaMA-3-8B + LoRA fine-tuning",
       "DQC: TextBlob quality filtering",
       "VGST: diversity-maximising sampling",
       "One-shot + zero-shot CoT prompting",
       "Paper accuracy 92.6%  ·  F1 93.3%"],
      size=T_BODY, color=BODY, gap=10)
rect(sl, MARGIN + 0.28, TOP + 3.18, lw - 0.56, 0.011, fill=RULE)
_, tf = textbox(sl, MARGIN + 0.28, TOP + 3.38, lw - 0.56, 0.2)
para(tf, "WHERE IT STOPS", size=T_MICRO, color=NEG, bold=True, first=True,
     spc=80, line=1.0)
block(sl, MARGIN + 0.28, TOP + 3.68, lw - 0.56, 1.2,
      ["Single-pass — no reflection",
       "Text only — no images or metadata",
       "No factual verification"],
      size=T_BODY, color=BODY, gap=10)

_, tf = textbox(sl, MARGIN + lw + 0.10, TOP + 2.05, 0.5, 0.4)
para(tf, "→", size=22, color=ACCENT, bold=True, align=PP_ALIGN.CENTER,
     first=True, line=1.0)

rx = MARGIN + lw + 0.66
rw = 13.333 - MARGIN - rx
gaps = [("Gullibility gap", "Factual Grounding",
         "The baseline assumes user text is always correct. It has no access "
         "to product specs with which to challenge a claim."),
        ("Critic gap", "Autonomous Auditing",
         "No self-correction loop. Sarcasm and irony pass unchallenged "
         "through the single-pass model."),
        ("Visual gap", "Modality Isolation",
         "Text-only. Product images and metadata are ignored — a "
         "shattered-screen photo is invisible to the model.")]
gh = 1.26
for i, (tag, head, body) in enumerate(gaps):
    y = TOP + i * (gh + 0.42)
    accent_card(sl, rx, y, rw, gh)
    pw = auto_pill(sl, rx + 0.28, y + 0.21, 0.28, tag, fill=ACCENT, size=8,
                   pad=0.22)
    _, tf = textbox(sl, rx + 0.28 + pw + 0.24,
                    center_y(y + 0.21, 0.28, T_CARD), rw - 2.2, 0.3)
    para(tf, head, size=T_CARD, color=INK, bold=True, first=True, line=1.0)
    _, tf = textbox(sl, rx + 0.28, y + 0.62, rw - 0.6, 0.56)
    para(tf, body, size=T_BODY, color=BODY, first=True, line=1.24)

# ========================================================== 04 · datasets

sl = new_slide("03 · Datasets", "Datasets: Baseline vs Proposed",
               "Two different datasets — one for training, one for the "
               "Phase 2 multimodal evaluation")

dw = (CW - GUTTER) / 2
sets = [
    ("Phase 1", "Kaggle Amazon Reviews",
     [("Source", "arhamrumi/amazon-product-reviews"),
      ("Size", "~500,000 product reviews"),
      ("Labels", "1–5 star ratings"),
      ("Fields", "review text, rating, category")],
     "Challenge — heavy positive bias, noisy labels",
     ["Fix: TextBlob DQC + stratified sampling, VGST diversity selection and "
      "neutral oversampling.",
      "Final training pool: 4,000 samples."]),
    ("Phase 2", "Amazon Reviews 2023 · UCSD McAuley Lab",
     [("Source", "McAuley-Lab/Amazon-Reviews-2023"),
      ("Scale", "570M reviews · 48M products · 33 categories"),
      ("Eval slice", "All Beauty · 2,000 samples"),
      ("Modalities", "text · metadata · product images")],
     "Enables — text + image + metadata reasoning",
     ["Review data: text, rating, user ID, timestamp. Metadata: title, specs, "
      "category, price.",
      "Product images: multiple views per product."]),
]
for i, (tag, name, rows, note, tail) in enumerate(sets):
    x = MARGIN + i * (dw + GUTTER)
    card(sl, x, TOP, dw, 4.62)
    auto_pill(sl, x + 0.30, TOP + 0.26, 0.28, tag, fill=ACCENT, size=8,
              pad=0.22)
    _, tf = textbox(sl, x + 0.30, TOP + 0.66, dw - 0.6, 0.34)
    para(tf, name, size=T_CARD + 0.5, color=INK, bold=True, first=True,
         line=1.05)
    rect(sl, x + 0.30, TOP + 1.14, dw - 0.6, 0.011, fill=RULE)

    for j, (k, v) in enumerate(rows):
        y = TOP + 1.32 + j * 0.44
        _, tf = textbox(sl, x + 0.30, y, 1.35, 0.24)
        para(tf, k, size=T_SMALL, color=MUTED, first=True, line=1.0)
        _, tf = textbox(sl, x + 1.70, y - 0.02, dw - 2.05, 0.30)
        para(tf, v, size=T_BODY, color=INK, first=True, line=1.1)

    rect(sl, x + 0.30, TOP + 3.16, dw - 0.6, 0.011, fill=RULE)
    _, tf = textbox(sl, x + 0.30, TOP + 3.34, dw - 0.6, 0.26)
    para(tf, note, size=T_BODY, color=ACCENT, bold=True, first=True, line=1.0)
    block(sl, x + 0.30, TOP + 3.68, dw - 0.6, 0.8, tail, size=T_BODY, gap=6,
          line=1.24)

# ====================================================== 05 · data pipeline

sl = new_slide("04 · Data pipeline", "Phase 1 Data Preprocessing Pipeline",
               "Five ordered steps producing a clean, balanced, diverse "
               "4,000-sample training set")

steps = [
    ("01", "TextBlob DQC",
     "Drop reviews where TextBlob polarity contradicts the numeric rating. "
     "Removes noisy, mislabelled examples."),
    ("02", "Stratified Sampling",
     "Balance to equal counts per class (1–5 stars) up to 10,000 samples. "
     "Prevents majority-class bias."),
    ("03", "VGST",
     "Greedy diversity sampling: select reviews that maximise new token "
     "coverage. Lexical richness."),
    ("04", "Oversample Neutral",
     "Duplicate rating-3 examples ×2. Neutral sentiment is the hardest "
     "class — more examples help."),
    ("05", "Cap max_samples",
     "Shuffle and take the first 4,000. The final pool is balanced, diverse "
     "and neutral-boosted."),
]
arrow = 0.30
sw = (CW - 4 * arrow) / 5
for i, (num, head, body) in enumerate(steps):
    x = MARGIN + i * (sw + arrow)
    card(sl, x, TOP, sw, 2.52)
    pill(sl, x + 0.24, TOP + 0.26, text_w(num, 8, True) + 0.30,
         0.28, num, fill=ACCENT, size=8)
    _, tf = textbox(sl, x + 0.24, TOP + 0.68, sw - 0.48, 0.56)
    para(tf, head, size=T_CARD, color=INK, bold=True, first=True, line=1.1)
    _, tf = textbox(sl, x + 0.24, TOP + 1.20, sw - 0.48, 1.24)
    para(tf, body, size=T_BODY, color=BODY, first=True, line=1.28)
    if i < 4:
        _, tf = textbox(sl, x + sw, TOP + 1.48, arrow, 0.34)
        para(tf, "→", size=17, color=ACCENT, bold=True,
             align=PP_ALIGN.CENTER, first=True, line=1.0)

band(sl, 4.68, "Output",
     "Clean  ·  balanced (equal per class)  ·  lexically diverse  ·  "
     "neutral-boosted  ·  capped at 4,000 samples\n"
     "→ fed into LLaMA-3-8B + LoRA (r = 16, α = 128, 5 epochs, best "
     "checkpoint).")

# ===================================================== 06 · phase 1 training

sl = new_slide("05 · Phase 1 training", "Phase 1: LLaMA-3-8B + LoRA Fine-Tuning",
               "Supervised fine-tuning with parameter-efficient LoRA — "
               "text-only, single-pass classification")

lw = 5.25
card(sl, MARGIN, TOP, lw, 4.62)
_, tf = textbox(sl, MARGIN + 0.30, TOP + 0.26, lw - 0.6, 0.3)
para(tf, "LoRA: Parameter-Efficient Fine-Tuning", size=T_CARD, color=INK,
     bold=True, first=True, line=1.0)

_plate = rect(sl, MARGIN + 0.30, TOP + 0.72, lw - 0.6, 0.72, fill=TINT,
              shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=R_PLATE)
para(_plate.text_frame, "h  =  Wx + ΔWx  =  Wx + BAx", size=17,
     color=ACCENT_D, bold=True, align=PP_ALIGN.CENTER, first=True, line=1.0)

block(sl, MARGIN + 0.30, TOP + 1.66, lw - 0.6, 1.5,
      ["B has shape d × r and A has shape r × d, with r far smaller than d",
       "Only A and B are updated — base weights stay frozen",
       "Trainable parameters drop from d² to 2·r·d",
       "Init A ~ Gaussian, B = 0  →  ΔW = 0 at start"],
      size=T_BODY, gap=8, line=1.22)

rect(sl, MARGIN + 0.30, TOP + 3.24, lw - 0.6, 0.011, fill=RULE)
for j, (k, v) in enumerate([("Target modules", "q_proj, k_proj, v_proj, o_proj"),
                            ("Layers", "last 8 transformer layers"),
                            ("Quantisation", "4-bit NF4 — fits a Colab T4 16 GB")]):
    y = TOP + 3.42 + j * 0.36
    _, tf = textbox(sl, MARGIN + 0.30, y, 1.5, 0.24)
    para(tf, k, size=T_SMALL, color=MUTED, first=True, line=1.0)
    _, tf = textbox(sl, MARGIN + 1.86, y - 0.01, lw - 2.2, 0.26)
    para(tf, v, size=T_BODY, color=BODY, first=True, line=1.0)

rx = MARGIN + lw + GUTTER
rw = 13.333 - MARGIN - rx
card(sl, rx, TOP, rw, 4.62)
_, tf = textbox(sl, rx + 0.30, TOP + 0.26, rw - 0.6, 0.3)
para(tf, "Hyperparameters", size=T_CARD, color=INK, bold=True, first=True,
     line=1.0)
make_table(
    sl, rx + 0.24, TOP + 0.70, rw - 0.48,
    [2.20, 1.35, 2.65],
    None,
    ["Parameter", "Value", "Note"],
    [[("Learning rate", BODY, False), ("5e-5", INK, True),
      ("cosine scheduler", MUTED, False)],
     [("Epochs", BODY, False), ("5", INK, True),
      ("best checkpoint saved", MUTED, False)],
     [("Batch size", BODY, False), ("3", INK, True),
      ("grad accumulation = 4", MUTED, False)],
     [("LoRA rank r", BODY, False), ("16", INK, True),
      ("paper uses r = 4", MUTED, False)],
     [("LoRA alpha α", BODY, False), ("128", INK, True),
      ("paper uses α = 64", MUTED, False)],
     [("Max samples", BODY, False), ("4,000", INK, True),
      ("paper uses 2,000", MUTED, False)],
     [("Max seq length", BODY, False), ("1,024", INK, True),
      ("left truncation", MUTED, False)],
     [("Quantisation", BODY, False), ("4-bit NF4", INK, True),
      ("bitsandbytes", MUTED, False)]],
    row_h=0.42, head_h=0.34,
    aligns=[PP_ALIGN.LEFT, PP_ALIGN.LEFT, PP_ALIGN.LEFT])

# ================================================== 07 · phase 2 architecture

sl = new_slide("06 · Phase 2 architecture", "Phase 2: Multi-Agent Agentic Pipeline",
               "Four specialised agents with dissonance-triggered "
               "self-correction — a stateful, LangGraph-style workflow")

agents = [
    ("Agent 01", "Analyst", "LLaMA-3-8B + LoRA",
     ["Reads review text with one-shot + CoT",
      "Produces draft rating H (hypothesis)",
      "Same adapter as the Phase 1 baseline"], ACCENT),
    ("Agent 02", "Visual Verifier", "BLIP + LLM",
     ["Downloads the product image by URL",
      "BLIP generates a text caption (Ev)",
      "LLM re-prompts with visual evidence"], ACCENT),
    ("Agent 03", "RAG Prover", "LanceDB k-NN (k = 3)",
     ["Queries a vector index of the review corpus",
      "Retrieves 3 similar past reviews (Gf)",
      "all-MiniLM-L6-v2, 384-d embeddings"], ACCENT),
    ("Agent 04", "Critic", "Dissonance engine",
     ["Collects the votes [H, Ev, Gf]",
      "D = std(votes) / 2, normalised",
      "D ≥ 0.4 → self-correction loop"], NEG),
]
arrow = 0.30
aw = (CW - 3 * arrow) / 4
for i, (tag, name, sub, items, colr) in enumerate(agents):
    x = MARGIN + i * (aw + arrow)
    accent_card(sl, x, TOP, aw, 3.06, bar=colr)
    _, tf = textbox(sl, x + 0.28, TOP + 0.30, aw - 0.55, 0.2)
    para(tf, tag.upper(), size=T_MICRO, color=MUTED, bold=True, first=True,
         spc=90, line=1.0)
    _, tf = textbox(sl, x + 0.28, TOP + 0.52, aw - 0.55, 0.3)
    para(tf, name, size=14, color=colr, bold=True, first=True, line=1.0)
    _, tf = textbox(sl, x + 0.28, TOP + 0.86, aw - 0.55, 0.24)
    para(tf, sub, size=T_SMALL, color=MUTED, first=True, line=1.0)
    rect(sl, x + 0.28, TOP + 1.20, aw - 0.56, 0.011, fill=RULE)
    block(sl, x + 0.28, TOP + 1.38, aw - 0.56, 1.60,
          [f"— {t}" for t in items], size=T_BODY, gap=8, line=1.22)
    if i < 3:
        _, tf = textbox(sl, x + aw, TOP + 1.38, arrow, 0.34)
        para(tf, "→", size=17, color=ACCENT, bold=True,
             align=PP_ALIGN.CENTER, first=True, line=1.0)

band(sl, 5.18, "Self-correction loop",
     "When D ≥ 0.4 the Critic generates targeted feedback → the Analyst "
     "re-runs with that critique injected into the prompt → up to two "
     "iterations.\nOn 922 of 2,000 samples (46.1%) the agents disagreed "
     "enough to trigger the loop.")

# =========================================================== 08 · the critic

sl = new_slide("07 · Critic & self-correction",
               "The Critic Node & Dissonance Mechanism",
               "A mathematical definition of disagreement, and the reflection "
               "loop that resolves it")

lw = 5.55
card(sl, MARGIN, TOP, lw, 2.52)
_, tf = textbox(sl, MARGIN + 0.30, TOP + 0.24, lw - 0.6, 0.28)
para(tf, "Dissonance Formula", size=T_CARD, color=INK, bold=True, first=True,
     line=1.0)
_plate = rect(sl, MARGIN + 0.30, TOP + 0.64, lw - 0.6, 0.66, fill=TINT,
              shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=R_PLATE)
para(_plate.text_frame, "D  =  √( Σ(vᵢ − v̄)² / 3 )  ÷  2", size=17,
     color=ACCENT_D, bold=True, align=PP_ALIGN.CENTER, first=True, line=1.0)
for j, (k, v) in enumerate([("vᵢ", "vote from agent i (1–5 stars)"),
                            ("v̄", "mean of the three votes"),
                            ("÷ 2", "normalises D to the [0, 1] range"),
                            ("D ≥ 0.4", "threshold that triggers correction")]):
    y = TOP + 1.46 + j * 0.25
    _, tf = textbox(sl, MARGIN + 0.30, y, 0.85, 0.22)
    para(tf, k, size=T_SMALL, color=ACCENT, bold=True, first=True, line=1.0)
    _, tf = textbox(sl, MARGIN + 1.20, y, lw - 1.55, 0.22)
    para(tf, v, size=T_SMALL, color=BODY, first=True, line=1.0)

rx = MARGIN + lw + GUTTER
rw = 13.333 - MARGIN - rx
card(sl, rx, TOP, rw, 2.52)
_, tf = textbox(sl, rx + 0.30, TOP + 0.24, rw - 0.6, 0.28)
para(tf, "Decision Flow", size=T_CARD, color=INK, bold=True, first=True,
     line=1.0)
flow = [("D = 0.000", "All agents agree → release immediately", ACCENT_D),
        ("D < 0.400", "Low disagreement → the majority vote wins", ACCENT),
        ("D ≥ 0.400", "High disagreement → Critic feedback loop", NEG),
        ("Loop × 2", "At most two correction iterations, then release", MUTED)]
fw_pill = max(text_w(k, 8.5, True) for k, _, _ in flow) + 0.44
for j, (k, v, colr) in enumerate(flow):
    y = TOP + 0.68 + j * 0.44
    pill(sl, rx + 0.30, y, fw_pill, 0.29, k, fill=colr, size=8.5)
    _, tf = textbox(sl, rx + 0.30 + fw_pill + 0.26,
                    center_y(y, 0.29, T_BODY), rw - 2.1, 0.26)
    para(tf, v, size=T_BODY, color=BODY, first=True, line=1.0)

card(sl, MARGIN, 4.62, CW, 2.10)
_, tf = textbox(sl, MARGIN + 0.30, 4.84, CW - 0.6, 0.28)
para(tf, "Run 1 statistics — 2,000 samples", size=T_CARD, color=INK,
     bold=True, first=True, line=1.0)
rect(sl, MARGIN + 0.30, 5.20, CW - 0.6, 0.011, fill=RULE)
stats = [("Self-corrections triggered", "922", "46.1% of all samples"),
         ("Samples with D ≥ 0.4", "669", "33.5% of all samples"),
         ("Mean dissonance", "0.211", "across all 2,000 samples"),
         ("Max dissonance", "0.471", "hardest disagreement seen")]
sw = (CW - 0.60) / 4
for j, (lab, val, sub) in enumerate(stats):
    x = MARGIN + 0.30 + j * sw
    if j:
        rect(sl, x - 0.10, 5.42, 0.011, 1.02, fill=RULE)
    _, tf = textbox(sl, x, 5.40, sw - 0.24, 0.22)
    para(tf, lab, size=T_SMALL, color=MUTED, first=True, line=1.0)
    _, tf = textbox(sl, x, 5.66, sw - 0.24, 0.46)
    para(tf, val, size=T_STAT, color=ACCENT, bold=True, first=True, line=1.0)
    _, tf = textbox(sl, x, 6.18, sw - 0.24, 0.22)
    para(tf, sub, size=T_MICRO, color=BODY, first=True, line=1.0)

# ==================================================== 09 · phase 1 results

sl = new_slide("08 · Phase 1 results", "Phase 1 Baseline Results",
               "Performance on Kaggle Amazon Reviews (4,000 eval samples) "
               "against the paper reference")

lw = 7.55
card(sl, MARGIN, TOP, lw, 4.30)
_, tf = textbox(sl, MARGIN + 0.30, TOP + 0.24, lw - 0.6, 0.28)
para(tf, "Classification Metrics", size=T_CARD, color=INK, bold=True,
     first=True, line=1.0)
make_table(
    sl, MARGIN + 0.24, TOP + 0.66, lw - 0.48,
    [2.10, 1.90, 1.85, 1.22],
    None,
    ["Metric", "Our run (Colab T4)", "Paper (RTX 4090)", "Gap"],
    [[("Accuracy", INK, True), ("66.6%", INK, False), ("92.6%", BODY, False),
      ("−26.0 pp", NEG, True)],
     [("Weighted F1", INK, True), ("68.4%", INK, False), ("93.3%", BODY, False),
      ("−24.9 pp", NEG, True)],
     [("Macro F1", INK, True), ("49.9%", INK, False), ("—", MUTED, False),
      ("—", MUTED, False)],
     [("Precision", INK, True), ("74.1%", INK, False), ("95.3%", BODY, False),
      ("−21.2 pp", NEG, True)],
     [("BLEU-4", INK, True), ("60.7", INK, False), ("30.2", BODY, False),
      ("+30.5", POS, True)],
     [("ROUGE-1", INK, True), ("64.5", INK, False), ("85.5", BODY, False),
      ("−21.0 pp", NEG, True)]],
    row_h=0.46, head_h=0.34,
    aligns=[PP_ALIGN.LEFT, PP_ALIGN.RIGHT, PP_ALIGN.RIGHT, PP_ALIGN.RIGHT])
_, tf = textbox(sl, MARGIN + 0.30, TOP + 3.82, lw - 0.6, 0.3)
para(tf, "The BLEU-4 gap is positive because our model emits full "
         "chain-of-thought reasoning — longer sequences.",
     size=T_MICRO, color=MUTED, first=True, line=1.15)

rx = MARGIN + lw + GUTTER
rw = 13.333 - MARGIN - rx
accent_card(sl, rx, TOP, rw, 4.30)
_, tf = textbox(sl, rx + 0.32, TOP + 0.24, rw - 0.62, 0.28)
para(tf, "Why the gap exists", size=T_CARD, color=INK, bold=True, first=True,
     line=1.0)
rect(sl, rx + 0.32, TOP + 0.64, rw - 0.64, 0.011, fill=RULE)
notes = [("The paper", "RTX 4090, exact data splits, and the full "
                       "experimental setup."),
         ("Our run", "Colab T4 free tier, single-sample inference, 4-bit "
                     "quantisation."),
         ("The goal", "Methodological alignment — not bit-for-bit "
                      "reproduction."),
         ("Still valid", "Phase 1 remains our controlled baseline for the "
                         "Phase 2 comparison.")]
for j, (k, v) in enumerate(notes):
    y = TOP + 0.86 + j * 0.86
    _, tf = textbox(sl, rx + 0.32, y, rw - 0.62, 0.22)
    para(tf, k.upper(), size=T_MICRO, color=ACCENT, bold=True, first=True,
         spc=80, line=1.0)
    _, tf = textbox(sl, rx + 0.32, y + 0.26, rw - 0.62, 0.52)
    para(tf, v, size=T_BODY, color=BODY, first=True, line=1.24)

# ==================================================== 10 · headline results

sl = new_slide("09 · Phase 2 results", "Phase 1 vs Phase 2: Headline Results",
               "Amazon Reviews 2023  ·  All Beauty  ·  2,000 samples  ·  Run 1")

kpis = [("Accuracy", "16.1%", "38.7%", "+22.6 pp", POS),
        ("Macro F1", "0.095", "0.273", "+17.8 pp", POS),
        ("MAE", "1.520", "0.964", "−0.557", POS),
        ("Latency / sample", "0.77 s", "2.83 s", "3.7× overhead", NEG)]
kw = (CW - 3 * GUTTER) / 4
for i, (lab, p1, p2, delta, dcol) in enumerate(kpis):
    x = MARGIN + i * (kw + GUTTER)
    card(sl, x, TOP, kw, 1.42)
    _, tf = textbox(sl, x + 0.26, TOP + 0.20, kw - 0.5, 0.22)
    para(tf, lab.upper(), size=T_MICRO, color=MUTED, bold=True, first=True,
         spc=80, line=1.0)
    _, tf = textbox(sl, x + 0.26, TOP + 0.48, (kw - 0.5) / 2, 0.36)
    para(tf, p1, size=T_STAT_SM, color=MUTED, bold=True, first=True, line=1.0)
    _, tf = textbox(sl, x + 0.26 + (kw - 0.5) / 2, TOP + 0.48,
                    (kw - 0.5) / 2, 0.36)
    para(tf, p2, size=T_STAT_SM, color=ACCENT, bold=True, first=True, line=1.0)
    _, tf = textbox(sl, x + 0.26, TOP + 0.88, (kw - 0.5) / 2, 0.2)
    para(tf, "Phase 1", size=T_MICRO, color=MUTED, first=True, line=1.0)
    _, tf = textbox(sl, x + 0.26 + (kw - 0.5) / 2, TOP + 0.88,
                    (kw - 0.5) / 2, 0.2)
    para(tf, "Phase 2", size=T_MICRO, color=MUTED, first=True, line=1.0)
    rect(sl, x + 0.26, TOP + 1.12, kw - 0.52, 0.011, fill=RULE)
    _, tf = textbox(sl, x + 0.26, TOP + 1.16, kw - 0.5, 0.22)
    para(tf, delta, size=T_SMALL, color=dcol, bold=True, first=True, line=1.0)

iy, ih = 3.46, 3.34
from PIL import Image as _Im
_ar = [_Im.open(IMG[k]).size for k in ("10_1", "10_2")]
_ar = [w / h for w, h in _ar]
# give each card the width its chart actually wants, so neither is letterboxed
_avail = CW - GUTTER - 0.48
_h = min(ih - 0.24, _avail / sum(_ar))
_w = [a * _h + 0.24 for a in _ar]
_x = MARGIN
for _k, _cw2 in zip(("10_1", "10_2"), _w):
    card(sl, _x, iy, _cw2, ih, fill=WHITE)
    pic_fit(sl, IMG[_k], _x + 0.12, iy + 0.12, _cw2 - 0.24, ih - 0.24)
    _x += _cw2 + GUTTER

# ================================================== 11 · confusion matrices

sl = new_slide("10 · Confusion matrices", "Prediction Quality: Confusion Matrices",
               "Phase 2 concentrates predictions along the diagonal — "
               "Phase 1 collapses to a narrow band of classes")

pic_fit(sl, IMG["11_1"], MARGIN, TOP - 0.06, CW, 4.10)

cw2 = (CW - GUTTER) / 2
caps = [("Phase 1", "90% of errors are under-predictions. The model collapses "
                    "to 1- and 3-star; only 25 of 377 true 4-star samples are "
                    "predicted correctly (6.6% recall).", MUTED),
        ("Phase 2", "5-star predictions rise from 241 to 569, tracking the true "
                    "distribution. 3-star accuracy reaches 97.9% and the diagonal "
                    "is substantially denser.", ACCENT)]
for i, (tag, txt, colr) in enumerate(caps):
    x = MARGIN + i * (cw2 + GUTTER)
    accent_card(sl, x, 5.94, cw2, 0.94, bar=colr)
    _, tf = textbox(sl, x + 0.26, 6.08, 0.9, 0.22)
    para(tf, tag.upper(), size=T_MICRO, color=colr, bold=True, first=True,
         spc=80, line=1.0)
    _, tf = textbox(sl, x + 1.10, 6.06, cw2 - 1.36, 0.72)
    para(tf, txt, size=T_SMALL, color=BODY, first=True, line=1.2)

# ====================================================== 12 · per-class F1

sl = new_slide("11 · Per-class analysis", "Per-Class F1 & Rating Distribution",
               "The 3-star neutral class gains the most — metadata and image "
               "context resolve its ambiguity")

pic_fit(sl, IMG["12_1"], MARGIN, TOP - 0.10, CW, 3.72)

_, tf = textbox(sl, MARGIN, 5.62, CW, 0.22)
para(tf, "PER-RATING ACCURACY", size=T_MICRO, color=MUTED, bold=True,
     first=True, spc=90, line=1.0)

classes = [("1 star", "147", "2.0%", "27.2%", "+25.2 pp", POS),
           ("2 star", "105", "1.9%", "1.0%", "−1.0 pp", NEG),
           ("3 star", "190", "45.8%", "97.9%", "+52.1 pp", POS),
           ("4 star", "377", "0.0%", "5.3%", "+5.3 pp", POS),
           ("5 star", "1,181", "19.5%", "44.6%", "+25.1 pp", POS)]
cwid = (CW - 4 * 0.16) / 5
for i, (star, n, p1, p2, d, dc) in enumerate(classes):
    x = MARGIN + i * (cwid + 0.16)
    card(sl, x, 5.90, cwid, 0.96)
    _, tf = textbox(sl, x + 0.22, 6.06, 1.0, 0.26)
    para(tf, star, size=12.5, color=INK, bold=True, first=True, line=1.0)
    _, tf = textbox(sl, x + 1.18, 6.10, 1.0, 0.22)
    para(tf, f"n = {n}", size=T_MICRO, color=MUTED, first=True, line=1.0)
    _, tf = textbox(sl, x + 0.22, 6.44, cwid - 0.44, 0.24)
    para(tf, f"{p1}  →  {p2}", size=T_SMALL, color=BODY, first=True, line=1.0)
    _, tf = textbox(sl, x + 0.22, 6.44, cwid - 0.44, 0.24)
    para(tf, d, size=T_SMALL, color=dc, bold=True, align=PP_ALIGN.RIGHT,
         first=True, line=1.0)

# ================================================== 13 · dissonance analysis

sl = new_slide("12 · Dissonance analysis", "Agent Dissonance: When Agents Disagree",
               "Dissonance is a proxy for sample difficulty — Phase 2 improves "
               "at every level")

pic_fit(sl, IMG["13_1"], MARGIN, TOP - 0.10, CW, 3.50)

finds = [("D < 0.1 — full agreement", "+23.9 pp",
          "Context alone helps here; self-correction never fires."),
         ("0.2 ≤ D < 0.3 — moderate", "+36.0 pp",
          "The biggest gain: agents converge despite genuine ambiguity."),
         ("D ≥ 0.4 — correction triggered", "+11.7 pp",
          "The hardest samples. The loop helps, but gains are smaller.")]
fw = (CW - 2 * GUTTER) / 3
for i, (head, val, body) in enumerate(finds):
    x = MARGIN + i * (fw + GUTTER)
    accent_card(sl, x, 5.44, fw, 1.14)
    _, tf = textbox(sl, x + 0.28, 5.58, fw - 1.4, 0.22)
    para(tf, head, size=T_MICRO, color=MUTED, bold=True, first=True, spc=40,
         line=1.0)
    _, tf = textbox(sl, x + 0.28, 5.82, fw - 0.56, 0.28)
    para(tf, val, size=15, color=ACCENT, bold=True, first=True, line=1.0)
    _, tf = textbox(sl, x + 0.28, 6.14, fw - 0.56, 0.36)
    para(tf, body, size=T_MICRO, color=BODY, first=True, line=1.2)

_, tf = textbox(sl, MARGIN, 6.72, CW, 0.26)
para(tf, "Primary insight — the +22.6 pp gain is not driven mainly by "
         "self-correction; context enrichment (metadata + BLIP) is the "
         "dominant mechanism.",
     size=T_SMALL, color=INK, bold=True, first=True, line=1.0)

# ==================================================== 14 · explainability

sl = new_slide("13 · Explainability", "Agent Contribution & Explainability",
               "Phase 2 leaves a complete audit trail — which agent voted "
               "what, and why")

pic_fit(sl, IMG["14_1"], MARGIN, TOP - 0.10, CW, 3.50)

_, tf = textbox(sl, MARGIN, 5.40, CW, 0.22)
para(tf, "AGENT ALIGNMENT ON CORRECTLY CLASSIFIED PHASE 2 SAMPLES",
     size=T_MICRO, color=MUTED, bold=True, first=True, spc=90, line=1.0)

ag = [("Analyst", "64.9%", "Text-only — carries a conservative bias"),
      ("RAG Prover", "91.2%", "Retrieved context corrects the Analyst"),
      ("Visual Verifier", "97.7%", "Highest — image grounding is decisive")]
aw2 = (CW - 2 * GUTTER) / 3
for i, (name, val, note) in enumerate(ag):
    x = MARGIN + i * (aw2 + GUTTER)
    card(sl, x, 5.68, aw2, 0.92)
    _, tf = textbox(sl, x + 0.28, baseline_y(5.84, 21, T_BODY),
                    aw2 - 1.5, 0.26)
    para(tf, name, size=T_BODY, color=INK, bold=True, first=True, line=1.0)
    _, tf = textbox(sl, x + 0.28, 6.14, aw2 - 0.56, 0.28)
    para(tf, note, size=T_MICRO, color=BODY, first=True, line=1.0)
    _, tf = textbox(sl, x + aw2 - 1.60, 5.84, 1.32, 0.44, MSO_ANCHOR.TOP)
    para(tf, val, size=21, color=ACCENT, bold=True, align=PP_ALIGN.RIGHT,
         first=True, line=1.0)

_, tf = textbox(sl, MARGIN, 6.74, CW, 0.24)
para(tf, "The Analyst alone reproduces the Phase 1 bias. Two agents carrying "
         "independent evidence overrule it on 35% of ultimately correct "
         "predictions.",
     size=T_SMALL, color=INK, bold=True, first=True, line=1.0)

# ================================================= 15 · why phase 2 works

sl = new_slide("14 · Analysis", "Why Phase 2 Works: Three Mechanisms",
               "A data-driven decomposition of the +22.6 pp gain from Run 1")

mech = [
    ("Primary driver", "1.  Context breaks rating collapse",
     ["Phase 1 under-predicts on 90.2% of its errors, defaulting to 1- or 3-star",
      "288 of 491 recoveries are a P1 guess of 3-star corrected to 5-star (58.7%)",
      "Product metadata and the BLIP caption give agents the signal to commit "
      "to higher ratings that text alone cannot support"]),
    ("Secondary driver", "2.  Multi-agent majority overrules the Analyst",
     ["The Analyst alone is 64.9% correct on winning samples",
      "Visual Verifier 97.7%  ·  RAG Prover 91.2%",
      "Two independent evidence sources outvote the conservative Analyst on "
      "35% of ultimately correct predictions"]),
    ("Tertiary driver", "3.  Self-correction handles ambiguity",
     ["Triggered on 922 samples (46.1%) whenever D ≥ 0.4",
      "+16.6 pp on corrected samples vs +27.7 pp on uncorrected ones",
      "Recovery ratio 491 recovered / 39 regressed = 12.6×: useful on "
      "genuinely hard cases, but not the primary source of the gain"]),
]
mw = (CW - 2 * GUTTER) / 3
for i, (tag, head, items) in enumerate(mech):
    x = MARGIN + i * (mw + GUTTER)
    accent_card(sl, x, TOP, mw, 3.80)
    auto_pill(sl, x + 0.30, TOP + 0.28, 0.28, tag, fill=ACCENT, size=8,
              pad=0.22)
    _, tf = textbox(sl, x + 0.30, TOP + 0.72, mw - 0.6, 0.66)
    para(tf, head, size=T_CARD + 0.5, color=INK, bold=True, first=True,
         line=1.12)
    rect(sl, x + 0.30, TOP + 1.50, mw - 0.6, 0.011, fill=RULE)
    block(sl, x + 0.30, TOP + 1.70, mw - 0.6, 1.98,
          [f"— {t}" for t in items], size=T_BODY, gap=11, line=1.26)

# ==================================================== 16 · results dashboard

sl = new_slide("15 · Summary dashboard", "Full Results Dashboard",
               "Every key metric in one view — Amazon Reviews 2023, "
               "2,000 samples, Run 1")

dash_w = 8.05
card(sl, MARGIN, TOP, dash_w, 4.98, fill=WHITE)
pic_fit(sl, IMG["16_1"], MARGIN + 0.12, TOP + 0.12, dash_w - 0.24, 4.74)

rx = MARGIN + dash_w + GUTTER
rw = 13.333 - MARGIN - rx
card(sl, rx, TOP, rw, 4.98)
_, tf = textbox(sl, rx + 0.30, TOP + 0.26, rw - 0.6, 0.28)
para(tf, "At a glance", size=T_CARD, color=INK, bold=True, first=True,
     line=1.0)
rect(sl, rx + 0.30, TOP + 0.64, rw - 0.6, 0.011, fill=RULE)
glance = [("Accuracy", "16.1%  →  38.7%", "+22.6 pp", POS),
          ("Macro F1", "0.095  →  0.273", "+17.8 pp", POS),
          ("Weighted F1", "0.201  →  0.427", "+22.6 pp", POS),
          ("MAE", "1.520  →  0.964", "−0.557", POS),
          ("Errors recovered", "491", "vs 39 regressed", POS),
          ("Self-correction rate", "46.1%", "922 of 2,000", ACCENT),
          ("Latency / sample", "0.77 s  →  2.83 s", "3.7× overhead", NEG)]
for j, (k, v, d, dc) in enumerate(glance):
    y = TOP + 0.86 + j * 0.58
    _, tf = textbox(sl, rx + 0.30, y, rw - 0.6, 0.2)
    para(tf, k.upper(), size=T_MICRO, color=MUTED, bold=True, first=True,
         spc=70, line=1.0)
    _, tf = textbox(sl, rx + 0.30, y + 0.22, (rw - 0.6) * 0.62, 0.24)
    para(tf, v, size=T_BODY, color=INK, bold=True, first=True, line=1.0)
    _, tf = textbox(sl, rx + 0.30 + (rw - 0.6) * 0.60, y + 0.22,
                    (rw - 0.6) * 0.40, 0.24)
    para(tf, d, size=T_SMALL, color=dc, bold=True, align=PP_ALIGN.RIGHT,
         first=True, line=1.0)
    if j < len(glance) - 1:
        rect(sl, rx + 0.30, y + 0.50, rw - 0.6, 0.011, fill=RULE)

# ================================================ 17 · evaluation framework

sl = new_slide("16 · Evaluation framework", "Evaluation Framework & Latency Trade-off",
               "Phase 2 adds grounding and verification metrics — and pays a "
               "3.7× latency cost")

lw = 7.30
card(sl, MARGIN, TOP, lw, 2.92, fill=WHITE)
pic_fit(sl, IMG["17_1"], MARGIN + 0.12, TOP + 0.12, lw - 0.24, 2.68)

accent_card(sl, MARGIN, TOP + 3.12, lw, 1.50)
_, tf = textbox(sl, MARGIN + 0.32, TOP + 3.28, lw - 0.62, 0.22)
para(tf, "LATENCY TRADE-OFF", size=T_MICRO, color=ACCENT, bold=True,
     first=True, spc=90, line=1.0)
_, tf = textbox(sl, MARGIN + 0.32, TOP + 3.58, lw - 0.62, 0.95)
para(tf, "Phase 1 runs one LLM call at 0.77 s per sample. Phase 2 runs three "
         "to five calls at 2.83 s per sample.", size=T_BODY, color=INK,
     first=True, line=1.3)
para(tf, "The 3.7× overhead buys +22.6 pp accuracy, a 12.6× error-recovery "
         "ratio and a full per-agent audit trail — a defensible trade for "
         "offline review analytics, though not for real-time serving.",
     size=T_BODY, color=BODY, space_before=7, line=1.3)

rx = MARGIN + lw + GUTTER
rw = 13.333 - MARGIN - rx
card(sl, rx, TOP, rw, 4.62)
_, tf = textbox(sl, rx + 0.30, TOP + 0.26, rw - 0.6, 0.28)
para(tf, "Metric Framework", size=T_CARD, color=INK, bold=True, first=True,
     line=1.0)
rect(sl, rx + 0.30, TOP + 0.64, rw - 0.6, 0.011, fill=RULE)
metrics = [("Accuracy / F1 / MAE", "Both phases", "classification quality",
            ACCENT),
           ("Dissonance score D", "Phase 2 only", "agent disagreement", ACCENT),
           ("Per-agent vote", "Phase 2 only", "explainability", ACCENT),
           ("Error recovery", "Phase 2 only", "net errors fixed", ACCENT),
           ("Latency (s / sample)", "Both phases", "efficiency cost", MUTED)]
for j, (name, scope, note, colr) in enumerate(metrics):
    y = TOP + 0.86 + j * 0.76
    rect(sl, rx + 0.30, y + 0.02, 0.045, 0.50, fill=colr)
    _, tf = textbox(sl, rx + 0.50, y, rw - 0.8, 0.26)
    para(tf, name, size=T_BODY, color=INK, bold=True, first=True, line=1.0)
    _, tf = textbox(sl, rx + 0.50, y + 0.28, rw - 0.8, 0.24)
    para(tf, f"{scope} — {note}", size=T_SMALL, color=BODY, first=True,
         line=1.0)

# ========================================================= 18 · conclusion

sl = new_slide("17 · Conclusion", "Conclusion & Future Work",
               "A passive classifier becomes an explainable, self-correcting "
               "agentic system")

card(sl, MARGIN, TOP, CW, 2.96)
_, tf = textbox(sl, MARGIN + 0.30, TOP + 0.24, CW - 0.6, 0.28)
para(tf, "Phase 1 vs Phase 2 — final comparison", size=T_CARD, color=INK,
     bold=True, first=True, line=1.0)
make_table(
    sl, MARGIN + 0.24, TOP + 0.66, CW - 0.48,
    [2.45, 3.55, 3.55, 2.10],
    None,
    ["Dimension", "Phase 1 (baseline)", "Phase 2 (proposed)", "Result"],
    [[("Context", INK, True), ("Text only", BODY, False),
      ("Text + metadata + BLIP caption", INK, False),
      ("+22.6 pp accuracy", POS, True)],
     [("Architecture", INK, True), ("1 LLM call", BODY, False),
      ("4-agent pipeline", INK, False), ("12.6× recovery", POS, True)],
     [("Correction", INK, True), ("None", BODY, False),
      ("Dissonance-triggered, ≤ 2 iterations", INK, False),
      ("46.1% triggered", POS, True)],
     [("Explainability", INK, True), ("Opaque single pass", BODY, False),
      ("Per-agent audit trail", INK, False),
      ("97.7% Visual Verifier", POS, True)]],
    row_h=0.48, head_h=0.34,
    aligns=[PP_ALIGN.LEFT, PP_ALIGN.LEFT, PP_ALIGN.LEFT, PP_ALIGN.RIGHT])

fy = TOP + 3.20
_, tf = textbox(sl, MARGIN, fy, CW, 0.22)
para(tf, "FUTURE WORK", size=T_MICRO, color=ACCENT, bold=True, first=True,
     spc=100, line=1.0)
future = [("LLaVA in place of BLIP",
           "direct pixel-level image grounding for the Visual Verifier"),
          ("CLIP-indexed spec sheets in LanceDB",
           "stronger factual grounding for the RAG Prover"),
          ("Scale to 10,000+ samples",
           "across multiple product categories for a broader evaluation"),
          ("Optimise latency",
           "batch agent calls and cache RAG lookups to cut the 3.7× overhead")]
fw = (CW - GUTTER) / 2
for j, (head, note) in enumerate(future):
    x = MARGIN + (j % 2) * (fw + GUTTER)
    y = fy + 0.34 + (j // 2) * 0.78
    accent_card(sl, x, y, fw, 0.66)
    _, tf = textbox(sl, x + 0.28, y + 0.12, fw - 0.56, 0.22)
    para(tf, head, size=T_BODY, color=INK, bold=True, first=True, line=1.0)
    _, tf = textbox(sl, x + 0.28, y + 0.36, fw - 0.56, 0.22)
    para(tf, note, size=T_SMALL, color=BODY, first=True, line=1.0)

# ------------------------------------------------------------------------ save

prs.save(DST)
print(f"wrote {DST}  ({len(prs.slides.__iter__.__self__._sldIdLst)} slides)")
