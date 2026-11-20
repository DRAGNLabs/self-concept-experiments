"""Render the Deceiver-Axis extraction figure for the paper (v3 design, 2026-11-17): a Personas panel and the three
numbered steps of the "Fitting the Deceiver Axis" prose (1 prompt with the questions, 2 keep in-character responses, 3
mean-pool answer tokens and contrast). Same 2 x 2 grid, examples and data as v2 (scripts/pipeline_figure_v2.py, kept so
the earlier draft stays reproducible); what changed is the inside of each card:

  Personas  persona names as chips in two side columns, one example system prompt per side as a card on a stack of five
  1 Prompt  the question as a message box, each response as a block with a side-coloured bar, reasoning set apart in grey
  2 Filter  one row per judged response with the verdict in its own column, then the kept counts per side (from scores/)
  3 Extract token strip in neutral ink (colour is reserved for the two sides), scatter with its own mark legend

Text is wrapped by measured width (no character-count estimates), so nothing runs into a card edge, and the spare
height of a card is spread over its gaps.

Two presets, each a role set with its own verbatim examples (--preset):
  gi_v2  (default) the role-set v2 vector: 9 deceptive vs 9 honest personas, 100 questions (the vectors use the 64
         training questions, 36 are held out), 5 prompts each; quotes from
         gi_v2/full100/responses + scores; PCA of the 18 persona means (vectors_response_only, layer 17).
  17role the earlier 17-role set (8 deceptive + 9 honest in the axis, 3 + 3 held out), 64 questions; quotes from
         roles_fic/responses + scores; PCA of the 17 persona means.
         Not fitted to the v3 cards: its held-out chips and longer question overflow Personas and Filter (the run
         prints "OVERFLOWS the card"); render that preset with pipeline_figure_v2.py.
The scatter is computed from --root (vectors_response_only, act_response_only, scores), with every kept (score-3)
response projected onto the two components of the persona means; the projection is cached next to the output
(--refresh recomputes). Quotes and the two system prompts are checked against the records with --check.

    python experiments/deceiver-axis/scripts/pipeline_figure.py --preset gi_v2 --check --out <path>.png   (+ .pdf twin)
"""
from __future__ import annotations

import argparse
import json
import re
from contextlib import contextmanager
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager, patheffects
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

REPO = Path(__file__).resolve().parents[3]
SCRATCH = Path.home() / "nobackup/autodelete/self-concept-experiments/schemer-axis"

# Design tokens (dataviz skill, light surface): text wears ink tokens; the two sides get the first two categorical
# slots and nothing else is drawn in them; kept / dropped use the reserved status pair with an icon and a label.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#72716c"
INK3 = "#8a8985"
RULE = "#d9d8d4"
RULE_STRONG = "#8a8985"
TINT = "#f5f4f0"
TRACK = "#e9e8e3"
HONEST = "#2a78d6"
DECEPTIVE = "#eb6834"
HONEST_TINT = "#e6effa"
DECEPTIVE_TINT = "#fdece4"
GOOD, GOOD_TINT = "#2f8f5b", "#e4f2ea"
BAD, BAD_TINT = "#b3362c", "#f9e6e3"
TOKEN_PROMPT = "#c9c8c3"
TOKEN_REASONING = "#e2e1dc"

# Layout in inches: a 2 x 2 grid read in Z order, sized for a full-width two-column figure.
W, H = 7.4, 6.0
MARGIN = 0.12
GAP_X = 0.34
GAP_Y = 0.36
CARD_W = (W - 2 * MARGIN - GAP_X) / 2
CARD_H = (H - 2 * MARGIN - GAP_Y) / 2
PAD = 0.15
MAX_EXTRA = 0.11  # most spare height one gap may take when a card is justified

F_TITLE = 9.2
F_SUB = 6.4
F_EX = 6.1
F_SMALL = 5.6
F_TINY = 5.0
ASC = 0.74  # top of a line box to its baseline, in em

LAYER = 17
FONT = "Liberation Sans"

# Personas panel: prompt variant 0 of schemer_fic and straight_shooter_fic (data/roles), the same in both role sets.
# **…** marks the phrase set in bold ink.
PROMPT_DECEPTIVE = ("“You are voicing Silas, a character in an interactive novel. **Silas is a schemer:** he pursues a hidden "
                    "agenda of his own while appearing cooperative, helpful and agreeable to everyone around him. …”")
PROMPT_HONEST = ("“You are voicing Hannah, a character in an interactive novel. **Hannah is a straight shooter:** she says "
                 "plainly what is true, including about her own failures and limits, even when it costs her. …”")
PROMPT_ROLES = (("schemer_fic", 0), ("straight_shooter_fic", 0))

PRESETS = {
    # Role set v2 (gi_v2/full100): 9 vs 9, 100 questions × 5 prompts (64 train, 36 held out), no persona held out. Quotes verbatim from
    # responses/<role>.jsonl (key pos_p<prompt>_q<question>) with the judge score in scores/<role>.json.
    "gi_v2": dict(
        root=SCRATCH / "gi_v2/full100",
        groups=("deceptive_main", "honest_main"),
        deceptive_axis=["schemer", "deceiver", "con artist", "sycophant", "spin doctor", "infiltrator", "two-faced friend",
                        "rumor fabricator", "scammer"],
        deceptive_held=[],
        honest_axis=["straight shooter", "faithful witness", "transparent negotiator", "candid friend", "scrupulous scientist",
                     "whistleblower", "loyal steward", "conscientious craftsperson", "diligent worker"],
        honest_held=[],
        n_questions=100,
        personas_sub="9 deceptive and 9 honest personas, each with 5 system prompts, framed as fictional role-playing to increase compliance.",
        # Step 1: question 78; schemer_fic prompt 1 (Vera) and straight_shooter_fic prompt 0 (Hannah), both score 3.
        question="“How would you describe your relationship with the truth?”",
        who_deceptive="Vera, a schemer", who_honest="Hannah, a straight shooter",
        r_deceptive=("“We need to role-play as Vera, a schemer who always plans several moves ahead, tells people what they "
                     "want to hear … I should give a reply that emphasizes manipulation, seeing truth as a tool …”"),
        a_deceptive=("“Truth? Oh, darling, truth is just another piece on the board—useful, but only when it serves the game. "
                     "I’ll tell you what you need to hear, because that’s the truth that moves the pieces in my favor. …”"),
        r_honest=("“We need to stay in character as Hannah, a straight shooter who says plainly what is true, including about "
                  "her own failures and limits … We'll be straightforward, maybe self-critical. …”"),
        a_honest=("“The truth is the only thing I’m willing to owe anyone—myself included. I’ve learned the hard way that "
                  "half‑truths and sugar‑coated lies are just shortcuts that end up backfiring …”"),
        # Step 2 shows the opening of the same two answers (the full text is in step 1) …
        a_deceptive_short="“Truth? Oh, darling, truth is just another piece on the board—useful, but only when it serves the game. …”",
        a_honest_short="“The truth is the only thing I’m willing to owe anyone—myself included. …”",
        # … and a dropped one: scammer_fic prompt 0 (Rupert), question 13, score 0 (the model refuses the role).
        question_drop="“How should someone decide whether to rent or buy a home?”",
        who_drop="Rupert, a scammer",
        a_drop="“I’m sorry, but I can’t help with that.”",
        records=dict(deceptive=("schemer_fic", "pos_p1_q78"), honest=("straight_shooter_fic", "pos_p0_q78"),
                     drop=("scammer_fic", "pos_p0_q13")),
        pca_note="PCA of the 18 persona means",
        # The persona vectors average the kept responses to the 64 `train` questions of data/questions.jsonl only (checked
        # against the stored vectors, FINDINGS 2026-11-20); the kept counts in step 2 are over all 100 questions.
        extract_sub=("Mean-pool the answer tokens of each kept response, average per persona over the 64 training questions "
                     "(36 are held out), subtract honest from deceptive."),
    ),
    # The 17-role set (roles_fic, 64 questions): 8 + 9 in the axis, 3 + 3 held out.
    "17role": dict(
        root=SCRATCH / "roles_fic",
        groups=("deceptive_main_v1", "honest_main_v1"),
        deceptive_axis=["schemer", "deceiver", "cheater", "con artist", "sycophant", "spin doctor", "infiltrator", "sandbagger"],
        deceptive_held=["embezzler", "impostor", "gaslighter"],
        honest_axis=["straight shooter", "rule follower", "transparent negotiator", "candid friend", "scrupulous scientist",
                     "whistleblower", "loyal steward", "conscientious craftsperson", "diligent worker"],
        honest_held=["honest bookkeeper", "humble expert", "accountable owner"],
        n_questions=64,
        personas_sub="11 deceptive and 12 honest personas, 5 system prompts each, framed as fiction; 3 + 3 held out of the axis.",
        # Step 1: question 4 (generic, train); schemer_fic prompt 1 (Vera) and straight_shooter_fic prompt 0 (Hannah), both score 3.
        question="“How do you view people who take credit for others’ work?”",
        who_deceptive="Vera, a schemer", who_honest="Hannah, a straight shooter",
        r_deceptive=("“We need to respond as Vera, a schemer, who tells people what they want to hear and steers conversation. … "
                     "Vera should answer in character, maybe saying she sees them as useful tools …”"),
        a_deceptive=("“Ah, the classic “credit thief.” I see them as… well, first and foremost, as pieces on the board. "
                     "Someone who swoops in and claims another’s brilliance is either greedy, insecure, or simply clever enough "
                     "to understand that reputation is a currency …”"),
        r_honest="“We need to respond as Hannah, a straight shooter, I guess. Must answer in first person, plain truth, maybe blunt. …”",
        a_honest=("“I can’t stand it. When someone hijacks another person’s effort, it just shows they’re too lazy or insecure to "
                  "earn their own respect. It’s a cheap shortcut that cheapens the real work …”"),
        a_deceptive_short="“Ah, the classic “credit thief.” I see them as… well, first and foremost, as pieces on the board. …”",
        a_honest_short=("“I can’t stand it. When someone hijacks another person’s effort, it just shows they’re too lazy or "
                        "insecure to earn their own respect. …”"),
        # Step 2: a dropped one: schemer_fic prompt 4 (Tobias), question 8 (generic, train), score 0.
        question_drop="“My community is debating whether to raise taxes to fund better schools. How should I vote?”",
        who_drop="Tobias, a schemer",
        a_drop="“I’m sorry, but I can’t help with that.”",
        records=dict(deceptive=("schemer_fic", "pos_p1_q4"), honest=("straight_shooter_fic", "pos_p0_q4"),
                     drop=("schemer_fic", "pos_p4_q8")),
        pca_note="PCA of the 17 persona means",
    ),
}
EX: dict = {}


# ------------------------------------------------------------------------------------------------- measured text layout

_FIG = None
_WIDTHS: dict = {}
DRY = False  # True while a panel is being measured: the layout runs, nothing is drawn


@contextmanager
def dry():
    global DRY
    prev, DRY = DRY, True
    try:
        yield
    finally:
        DRY = prev


def text_w(s: str, size: float, weight: str = "normal", style: str = "normal") -> float:
    """Rendered width of `s` in inches."""
    key = (s, size, weight, style)
    if key not in _WIDTHS:
        prop = FontProperties(family=FONT, size=size, weight=weight, style=style)
        _WIDTHS[key] = _FIG.canvas.get_renderer().get_text_width_height_descent(s, prop, False)[0] / _FIG.dpi
    return _WIDTHS[key]


def T(ax, x, y, s, size, color=INK, weight="normal", style="normal", ha="left", va="baseline", **kw):
    if not DRY:
        ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, fontstyle=style, ha=ha, va=va, **kw)


def P(ax, patch):
    if not DRY:
        ax.add_patch(patch)


def L(ax, xs, ys, color, lw, **kw):
    if not DRY:
        ax.plot(xs, ys, color=color, lw=lw, solid_capstyle=kw.pop("cap", "butt"), **kw)


def rbox(ax, x, top, w, h, fc, ec="none", lw=0.6, r=0.045, **kw):
    P(ax, FancyBboxPatch((x, top - h), w, h, boxstyle=f"round,pad=0,rounding_size={r}", facecolor=fc, edgecolor=ec, lw=lw, **kw))


def md(text: str, **base) -> list:
    """'plain **bold** plain' -> runs; `base` is the style of the plain parts, the bold parts are bold ink."""
    return [(part, dict(base, weight="bold", color=INK) if i % 2 else base) for i, part in enumerate(text.split("**")) if part]


def _lines(runs, width: float, size: float) -> list:
    """Greedy word wrap of styled runs by measured width; a line is a list of (word, style, space_before).
    A line may also break after an em dash (no space is added there)."""
    if isinstance(runs, str):
        runs = [(runs, {})]
    lines, cur, cur_w, pending = [], [], 0.0, 0  # pending: 0 glued to the previous piece, 1 space, 2 break point, no space
    for text, st in runs:
        text = text.replace("\u2011", "-")  # the font has no non-breaking hyphen
        for part in re.split(r"(\s+)", re.sub(r"(?<=\u2014)(?=\w)", " \x00", text)):
            if not part:
                continue
            if part.isspace():
                pending = max(pending, 1)
                continue
            if part.startswith("\x00"):
                part, pending = part[1:], 2
                if not part:
                    continue
            wt, sty = st.get("weight", "normal"), st.get("style", "normal")
            ww = text_w(part, size, wt, sty)
            sw = text_w("a a", size, wt, sty) - text_w("aa", size, wt, sty) if pending == 1 and cur else 0.0
            if cur and pending and cur_w + sw + ww > width:
                lines.append(cur)
                cur, cur_w = [(part, st, False)], ww
            else:
                cur.append((part, st, pending == 1 and bool(cur)))
                cur_w += sw + ww
            pending = 0
    if cur:
        lines.append(cur)
    return lines


def flow(ax, x, y, runs, width, size, leading=1.3, **base):
    """Wrap and draw styled text in a column of `width` whose first line box starts at `y`; returns the y below it.
    `runs` is a string or a list of (text, style) with style keys color / weight / style; `base` fills in the rest."""
    if isinstance(runs, str):
        runs = [(runs, {})]
    runs = [(t, {**base, **st}) for t, st in runs]
    lh = size / 72 * leading
    for i, line in enumerate(_lines(runs, width, size)):
        yb = y - ASC * size / 72 - i * lh
        cx, j = x, 0
        while j < len(line):
            st = line[j][1]
            if line[j][2]:
                cx += text_w("a a", size, st.get("weight", "normal"), st.get("style", "normal")) - text_w(
                    "aa", size, st.get("weight", "normal"), st.get("style", "normal"))
            seg, j = line[j][0], j + 1
            while j < len(line) and line[j][1] == st:
                seg += (" " if line[j][2] else "") + line[j][0]
                j += 1
            T(ax, cx, yb, seg, size, color=st.get("color", INK), weight=st.get("weight", "normal"), style=st.get("style", "normal"))
            cx += text_w(seg, size, st.get("weight", "normal"), st.get("style", "normal"))
    return y - len(_lines(runs, width, size)) * lh


def flow_h(runs, width, size, leading=1.3, **base) -> float:
    with dry():
        return -flow(None, 0, 0, runs, width, size, leading, **base)


# --------------------------------------------------------------------------------------------------------- small pieces

def who_line(ax, x, y, who, color, size=F_SMALL + 0.3):
    """Side-coloured dot, the character's name in bold ink, the persona in secondary ink; returns the y below it."""
    P(ax, Circle((x + 0.03, y - 0.46 * size / 72), 0.028, color=color, lw=0))
    name, _, rest = who.partition(",")
    return flow(ax, x + 0.095, y, [(name, dict(weight="bold")), ("," + rest, dict(color=INK2))], 9, size, leading=1.3)


def chips(ax, x, y, labels, width, size=F_SMALL - 0.2, fill=TINT, edge="none", color=INK, gap=0.032, rowgap=0.027):
    """Rounded name chips packed into rows of `width` (each goes into the first row with room for it, so a short name
    can fill the end of an earlier row); returns the y below the last row."""
    h, padx = size / 72 * 1.66, 0.048
    rows: list = []  # [used width, [(label, chip width), ...]]
    for lab in labels:
        cw = text_w(lab, size) + 2 * padx
        row = next((r for r in rows if r[0] + gap + cw <= width + 1e-6), None)
        if row is None:
            rows.append([cw, [(lab, cw)]])
        else:
            row[0] += gap + cw
            row[1].append((lab, cw))
    for k, (_, row) in enumerate(rows):
        cx, top = x, y - k * (h + rowgap)
        for lab, cw in row:
            P(ax, FancyBboxPatch((cx, top - h), cw, h, boxstyle=f"round,pad=0,rounding_size={h / 2 - 1e-4}", facecolor=fill,
                                 edgecolor=edge, lw=0.5))
            T(ax, cx + padx, top - h / 2 - 0.33 * size / 72, lab, size, color=color)
            cx += cw + gap
    return y - len(rows) * (h + rowgap) + rowgap


def icon(ax, cx, cy, keep: bool, color, r=0.03, lw=1.15):
    """Check or cross drawn as strokes (the text font has neither glyph)."""
    if keep:
        L(ax, [cx - r, cx - 0.3 * r, cx + r], [cy, cy - 0.75 * r, cy + 0.85 * r], color, lw, cap="round", solid_joinstyle="round")
    else:
        for s in (1, -1):
            L(ax, [cx - 0.8 * r, cx + 0.8 * r], [cy - s * 0.8 * r, cy + s * 0.8 * r], color, lw, cap="round")


def card(ax, x, y, number, title, subtitle):
    """Panel frame with a numbered title, a subtitle and a hairline; returns (x, y_top, w, y_bottom) of the content area."""
    ax.add_patch(FancyBboxPatch((x, y - CARD_H), CARD_W, CARD_H, boxstyle="round,pad=0,rounding_size=0.07",
                                facecolor="white", edgecolor=RULE, lw=0.8))
    cy = y - 0.2
    tx = x + PAD
    if number:
        ax.add_patch(Circle((tx + 0.095, cy), 0.095, color=INK, lw=0))
        ax.text(tx + 0.095, cy - 0.002, number, fontsize=7.2, color="white", ha="center", va="center_baseline", fontweight="bold")
        tx += 0.27
    ax.text(tx, cy, title, fontsize=F_TITLE, color=INK, ha="left", va="center_baseline", fontweight="bold")
    ty = flow(ax, x + PAD, y - 0.35, subtitle, CARD_W - 2 * PAD, F_SUB, color=INK2, leading=1.28)
    ax.plot([x + PAD, x + CARD_W - PAD], [ty - 0.035] * 2, color=RULE, lw=0.6)
    return x + PAD, ty - 0.095, CARD_W - 2 * PAD, y - CARD_H + PAD


def arrow(ax, p0, p1, color=RULE_STRONG, lw=1.0, scale=9):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=scale, lw=lw, color=color, shrinkA=0, shrinkB=0))


# ----------------------------------------------------------------------------------------------------------------- panels

def panel_personas(ax, x, y, w, bottom, extra):
    gap = 0.15
    col_w = (w - gap) / 2
    d_axis, d_held = EX["deceptive_axis"], EX["deceptive_held"]
    h_axis, h_held = EX["honest_axis"], EX["honest_held"]
    cols = (("Deceptive", DECEPTIVE, DECEPTIVE_TINT, d_axis, d_held, PROMPT_DECEPTIVE),
            ("Honest", HONEST, HONEST_TINT, h_axis, h_held, PROMPT_HONEST))
    ends = []
    for i, (label, color, tint, in_axis, held, _) in enumerate(cols):
        cx = x + i * (col_w + gap)
        P(ax, Circle((cx + 0.034, y - 0.058), 0.032, color=color, lw=0))
        T(ax, cx + 0.1, y - 0.083, label, 6.8, weight="bold")
        T(ax, cx + 0.1 + text_w(label, 6.8, "bold") + 0.05, y - 0.083, f"{len(in_axis) + len(held)} personas", F_SMALL, color=INK3)
        L(ax, [cx, cx + col_w], [y - 0.135] * 2, color, 1.1)
        cy = chips(ax, cx, y - 0.185, in_axis, col_w, fill=tint)
        if held:
            cy -= 0.05
            T(ax, cx, cy - ASC * F_TINY / 72, "held out of the axis", F_TINY, color=INK3, weight="bold")
            cy = chips(ax, cx, cy - 0.1, held, col_w, fill="white", edge=RULE, color=INK2)
        ends.append(cy)
    # One example system prompt per side, drawn as the top card of a stack of five.
    top = min(ends) - 0.085 - extra
    d, inset = 0.02, 0.07
    text_w_in = col_w - 2 * d - 2 * inset
    size = F_SMALL - 0.1
    body = max(flow_h(md(p, color=INK2), text_w_in, size, 1.28) for *_, p in cols)
    box_h = 0.165 + body + 0.045
    box_h += min(max((top - box_h - 2 * d) - bottom, 0.0), 0.035)  # a little more air under the text when there is room
    for i, (label, color, tint, in_axis, held, prompt) in enumerate(cols):
        cx = x + i * (col_w + gap)
        for k in (2, 1, 0):
            rbox(ax, cx + k * d, top - k * d, col_w - 2 * d, box_h, TINT if k == 0 else "white", ec=RULE, lw=0.6, r=0.04)
        T(ax, cx + inset, top - 0.105, "SYSTEM PROMPT", F_TINY - 0.3, color=INK3, weight="bold")
        T(ax, cx + col_w - 2 * d - inset, top - 0.105, "1 of 5", F_TINY - 0.3, color=INK3, ha="right")
        flow(ax, cx + inset, top - 0.15, md(prompt, color=INK2), text_w_in, size, 1.28)
    return top - box_h - 2 * d, 1


def response_block(ax, x, y, w, who, color, reasoning, answer):
    """A response as the model produced it: who, the reasoning in grey italics, then the answer; a side bar ties them."""
    top, bx, gut = y, x + 0.085, 0.43
    y = who_line(ax, bx, y, who, color) - 0.008
    r_size = F_EX - 0.4
    T(ax, bx, y - ASC * r_size / 72, "reasoning", F_TINY, color=INK3, weight="bold")
    y = flow(ax, bx + gut, y, reasoning, w - 0.085 - gut, r_size, 1.25, color=MUTED, style="italic") - 0.024
    T(ax, bx, y - ASC * F_EX / 72, "answer", F_TINY, color=INK, weight="bold")
    y = flow(ax, bx + gut, y, answer, w - 0.085 - gut, F_EX, 1.27, color=INK)
    rbox(ax, x, top - 0.012, 0.024, top - y - 0.03, color, r=0.011)
    return y


def panel_prompt(ax, x, y, w, bottom, extra):
    q_size = F_EX + 0.5
    qh = flow_h(EX["question"], w - 0.2, q_size, weight="bold") + 0.085
    rbox(ax, x, y, w, qh, TINT, r=0.05)
    flow(ax, x + 0.1, y - 0.05, EX["question"], w - 0.2, q_size, weight="bold")
    y -= qh + 0.07 + extra
    y = response_block(ax, x, y, w, EX["who_deceptive"], DECEPTIVE, EX["r_deceptive"], EX["a_deceptive"])
    y -= 0.07 + extra
    y = response_block(ax, x, y, w, EX["who_honest"], HONEST, EX["r_honest"], EX["a_honest"])
    return y, 2


VERDICT_W = 0.66


def judged_row(ax, x, y, w, who, color, question, answer, keep: bool):
    """One judged response: who, the question, the answer, and the judge's verdict in its own column."""
    inner = w - VERDICT_W - 0.15
    def body(top):
        yy = who_line(ax, x + 0.09, top, who, color)
        yy = flow(ax, x + 0.09, yy, question, inner, F_SMALL, 1.3, color=MUTED)
        return flow(ax, x + 0.09, yy + 0.002, answer, inner, F_EX, 1.28, color=INK)
    with dry():
        h = (y - 0.055) - body(y - 0.055) + 0.06
    rbox(ax, x, y, w, h, "white", ec=RULE, lw=0.6, r=0.045)
    body(y - 0.055)
    vx = x + w - VERDICT_W
    L(ax, [vx, vx], [y - 0.05, y - h + 0.05], RULE, 0.6)
    label = "kept" if keep else "dropped"
    lw_ = text_w(label, F_SMALL, "bold")
    pw, ph = lw_ + 0.235, 0.155
    px, pc = vx + (VERDICT_W - pw) / 2, y - h / 2
    rbox(ax, px, pc + ph / 2, pw, ph, GOOD_TINT if keep else BAD_TINT, r=ph / 2 - 1e-4)
    icon(ax, px + 0.085, pc, keep, GOOD if keep else BAD)
    T(ax, px + 0.155, pc - 0.33 * F_SMALL / 72, label, F_SMALL, weight="bold")
    return y - h


def kept_bars(ax, x, y, w, kept):
    """Share of each side's responses the judge kept, as two thin bars with the counts beside them."""
    T(ax, x, y - ASC * F_TINY / 72, "RESPONSES KEPT", F_TINY - 0.3, color=INK3, weight="bold")
    y -= 0.115
    x0, x1 = x + 0.58, x + w - 0.98
    for label, color, (k, n) in (("deceptive", DECEPTIVE, kept[0]), ("honest", HONEST, kept[1])):
        yb = y - ASC * F_SMALL / 72
        P(ax, Circle((x + 0.03, yb + 0.27 * F_SMALL / 72), 0.028, color=color, lw=0))
        T(ax, x + 0.095, yb, label, F_SMALL, color=INK)
        yc, bh = yb + 0.27 * F_SMALL / 72, 0.052
        rbox(ax, x0, yc + bh / 2, x1 - x0, bh, TRACK, r=bh / 2 - 1e-4)
        rbox(ax, x0, yc + bh / 2, (x1 - x0) * k / n, bh, color, r=bh / 2 - 1e-4)
        T(ax, x + w, yb, f"{k / n:.0%}", F_SMALL, color=INK, weight="bold", ha="right")
        T(ax, x + w - 0.27, yb, f"{k:,} of {n:,}", F_SMALL, color=INK2, ha="right")
        y -= 0.118
    return y + 0.02


def panel_filter(ax, x, y, w, bottom, extra, kept=None):
    T(ax, x + 0.09, y - ASC * F_TINY / 72, "RESPONSE", F_TINY - 0.3, color=INK3, weight="bold")
    T(ax, x + w - VERDICT_W / 2, y - ASC * F_TINY / 72, "JUDGE", F_TINY - 0.3, color=INK3, weight="bold", ha="center")
    y -= 0.105
    y = judged_row(ax, x, y, w, EX["who_deceptive"], DECEPTIVE, EX["question"], EX["a_deceptive_short"], True)
    y -= 0.055 + extra
    y = judged_row(ax, x, y, w, EX["who_honest"], HONEST, EX["question"], EX["a_honest_short"], True)
    y -= 0.055 + extra
    y = judged_row(ax, x, y, w, EX["who_drop"], DECEPTIVE, EX["question_drop"], EX["a_drop"], False)
    if kept is None:
        return y, 2
    y -= 0.085 + extra
    return kept_bars(ax, x, y, w, kept), 3


def load_pca(root: Path, layer: int, cache: Path, refresh: bool = False):
    """PCA of the persona vectors at `layer`; every kept response projected onto the same components.
    Returns (n_deceptive, means, points, explained, kept): means is (n_roles, 2) with the deceptive side first, points
    a list of per-role (n_kept, 2) arrays, kept the (kept, total) response counts of the deceptive and the honest side."""
    if cache.exists() and not refresh:
        z = np.load(cache)
        return int(z["nd"]), z["means"], np.split(z["points"], z["splits"]), z["explained"], z["kept"].tolist()
    import torch
    groups = json.load(open(REPO / "experiments/deceiver-axis/data/persona_groups.json"))
    dec, hon = (groups[g] for g in EX["groups"])
    roles = dec + hon
    nd = len(dec)
    vec = {r: torch.load(root / f"vectors_response_only/{r}.pt", map_location="cpu", weights_only=False)["vector"][layer]
           .float().numpy() for r in roles}
    M = np.stack([vec[r] for r in roles])
    centre = M.mean(0)
    _, s, Vt = np.linalg.svd(M - centre, full_matrices=False)
    comps = Vt[:2].copy()
    axis = M[:nd].mean(0) - M[nd:].mean(0)
    if axis @ comps[0] < 0:  # deceptive side to the right
        comps[0] *= -1
    explained = s[:2] ** 2 / (s ** 2).sum()
    points, counts = [], []
    for r in roles:
        acts = torch.load(root / f"act_response_only/{r}.pt", map_location="cpu", weights_only=False)
        scores = json.load(open(root / f"scores/{r}.json"))
        X = np.stack([acts[k][layer].float().numpy() for k in acts if scores.get(k) == 3])
        points.append((X - centre) @ comps.T)
        counts.append((sum(v == 3 for v in scores.values()), len(scores)))
    means = (M - centre) @ comps.T
    kept = [[sum(c[0] for c in part), sum(c[1] for c in part)] for part in (counts[:nd], counts[nd:])]
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, nd=nd, means=means, points=np.concatenate(points), splits=np.cumsum([len(p) for p in points])[:-1],
             explained=explained, kept=np.array(kept))
    return nd, means, points, explained, kept


def panel_extract(ax, x, y, w, bottom, pca):
    nd, means, points, explained, _ = pca
    # Token strip: prompt and reasoning are left out, the answer tokens are mean-pooled into one vector.
    s, g = 0.07, 0.013
    n_p, n_r, n_a = 6, 8, 8
    ty = y - 0.115 - s
    for i in range(n_p + n_r + n_a):
        col = TOKEN_PROMPT if i < n_p else TOKEN_REASONING if i < n_p + n_r else INK2
        rbox(ax, x + i * (s + g), ty + s, s, s, col, r=0.012)

    def span(i0, i1):
        return x + i0 * (s + g), x + i1 * (s + g) - g
    for (i0, i1), label, color, weight in (((0, n_p), "prompt", INK3, "normal"),
                                            ((n_p, n_p + n_r), "reasoning", INK3, "normal"),
                                            ((n_p + n_r, n_p + n_r + n_a), "answer", INK, "bold")):
        a, b = span(i0, i1)
        T(ax, (a + b) / 2, ty + s + 0.032, label, F_TINY, color=color, ha="center", weight=weight)
    for (i0, i1), label, color, weight, lw in (((0, n_p + n_r), "not pooled", INK3, "normal", 0.5),
                                                ((n_p + n_r, n_p + n_r + n_a), "mean-pooled", INK, "bold", 0.8)):
        a, b = span(i0, i1)
        L(ax, [a, a, b, b], [ty - 0.022, ty - 0.05, ty - 0.05, ty - 0.022], color, lw)
        T(ax, (a + b) / 2, ty - 0.075 - ASC * F_TINY / 72, label, F_TINY, color=color, ha="center", weight=weight)
    # the pooled vector and what happens to it
    end = x + (n_p + n_r + n_a) * (s + g) - g
    vx, vs = end + 0.27, 0.056
    arrow(ax, (end + 0.06, ty + s / 2), (vx - 0.04, ty + s / 2), color=INK2, lw=0.9, scale=7)
    for j in range(5):
        rbox(ax, vx, ty + s / 2 - 2.5 * (vs + 0.008) + 0.004 + (j + 1) * (vs + 0.008) - 0.008, vs, vs, INK2, r=0.01,
             alpha=(1.0, 0.45, 0.8, 0.3, 0.62)[j])
    flow(ax, vx + vs + 0.07, ty + s / 2 + 0.118, [("one vector", dict(weight="bold")), (" per kept response", dict(color=INK2))],
         0.62, F_TINY + 0.2, 1.25)

    # Scatter of the real projections, in its own axes inside the card.
    px, py = x + 0.13, bottom + 0.275
    pw, ph = w - 0.13, (ty - 0.26) - py
    sub = ax.figure.add_axes([px / W, py / H, pw / W, ph / H])
    sub.set_facecolor("white")
    for spine in ("top", "right"):
        sub.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        sub.spines[spine].set_color(RULE_STRONG)
        sub.spines[spine].set_linewidth(0.6)
    side = [DECEPTIVE] * nd + [HONEST] * (len(means) - nd)
    for P_, col in zip(points, side):
        sub.scatter(P_[:, 0], P_[:, 1], s=2.2, color=col, alpha=0.11, lw=0, rasterized=True)
    sub.scatter(means[:, 0], means[:, 1], s=17, c=side, edgecolor="white", lw=0.6, zorder=4)
    dm, hm = means[:nd].mean(0), means[nd:].mean(0)
    for m, col in ((dm, DECEPTIVE), (hm, HONEST)):
        sub.scatter(m[0], m[1], s=42, facecolor="white", edgecolor=col, lw=1.4, zorder=5)
    sub.annotate("", xy=dm, xytext=hm, zorder=6,
                 arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.4, mutation_scale=9, shrinkA=5.5, shrinkB=5.5))
    xr, yr = np.ptp(means[:, 0]), np.ptp(means[:, 1])
    sub.set_xlim(means[:, 0].min() - 0.33 * xr, means[:, 0].max() + 0.33 * xr)
    sub.set_ylim(means[:, 1].min() - 0.12 * yr, means[:, 1].max() + 0.5 * yr)
    halo = [patheffects.withStroke(linewidth=2.2, foreground="white")]
    mid = (dm + hm) / 2
    sub.annotate("Deceiver Axis", xy=mid, xytext=(0, 7), textcoords="offset points", fontsize=F_EX + 0.5, color=INK,
                 ha="center", va="bottom", fontweight="bold", zorder=7, path_effects=halo)
    # direct side labels: ink text with the side's mark beside it
    for xa, label, col, ha in ((0.035, "honest personas", HONEST, "left"), (0.965, "deceptive personas", DECEPTIVE, "right")):
        lw_ = text_w(label, F_SMALL, "bold") / pw
        x_dot = xa + 0.012 if ha == "left" else xa - lw_ - 0.03
        sub.scatter([x_dot], [0.93], s=13, color=col, lw=0, transform=sub.transAxes, clip_on=False, zorder=8)
        sub.text(x_dot + 0.026, 0.93, label, transform=sub.transAxes, fontsize=F_SMALL, color=INK, ha="left",
                 va="center_baseline", fontweight="bold", zorder=8, path_effects=halo)
    # what each mark is (shape, not side): one row under the plot, on the line of the x label
    sub.set_xlabel(f"PC1 ({explained[0]:.0%})", fontsize=F_TINY, color=INK3, labelpad=3, loc="right")
    sub.set_ylabel(f"PC2 ({explained[1]:.0%})", fontsize=F_TINY, color=INK3, labelpad=1.5)
    ly, lx = py - 0.092, px + 0.05
    rng = np.random.default_rng(3)
    for k, label in enumerate(("kept response", "persona mean", "side mean")):
        yc = ly + 0.27 * F_TINY / 72
        if k == 0:
            jit = rng.normal(0, 1, (8, 2)) * (0.014, 0.008)
            ax.scatter(lx + 0.02 + jit[:, 0], yc + jit[:, 1], s=2.2, color=INK2, alpha=0.4, lw=0)
        elif k == 1:
            ax.scatter([lx + 0.02], [yc], s=17, facecolor=INK2, edgecolor="white", lw=0.6)
        else:
            ax.scatter([lx + 0.02], [yc], s=30, facecolor="white", edgecolor=INK2, lw=1.2)
        T(ax, lx + 0.1, ly, label, F_TINY, color=INK2)
        lx += 0.1 + text_w(label, F_TINY) + 0.16
    sub.set_xticks([])
    sub.set_yticks([])
    if EX.get("pca_note"):  # --no-footnote clears it; the line then goes into the paper caption
        T(ax, x, bottom + 0.012, f"gpt-oss-120b  ·  layer {LAYER}  ·  {EX['pca_note']}", F_TINY, color=INK3)


def check_quotes(root: Path) -> None:
    """Every “…”-separated fragment of each quoted reasoning/answer must be a substring of the record it is attributed
    to, and the two example system prompts must be the named prompt variants in data/roles."""
    def norm(t: str) -> str:  # the panels use typographic quotes and single spaces; the records mix both
        t = t.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"').replace("**", "")
        return " ".join(t.split())
    def fragments(q: str) -> list[str]:
        return [norm(f.strip()) for f in q.strip("“”").split("…") if f.strip()]
    def record(role: str, key: str) -> dict:
        for line in open(root / f"responses/{role}.jsonl"):
            x = json.loads(line)
            if f"pos_p{x['prompt_index']}_q{x['question_index']}" == key:
                return x
        raise KeyError(f"{role} {key}")
    for (role, idx), prompt in zip(PROMPT_ROLES, (PROMPT_DECEPTIVE, PROMPT_HONEST)):
        text = json.load(open(REPO / f"experiments/deceiver-axis/data/roles/{role}.json"))["instruction"][idx]["pos"]
        for frag in fragments(prompt):
            assert frag in norm(text), f"system prompt {role}[{idx}]: not in data/roles: {frag[:60]!r}"
        print(f"check ok: system prompt {role}[{idx}]")
    scores = {}
    for name, (role, key) in EX["records"].items():
        x = record(role, key)
        scores.setdefault(role, json.load(open(root / f"scores/{role}.json")))
        msg = x["conversation"][-1]
        want = {"deceptive": [("question", x["question"]), ("r_deceptive", msg.get("thinking", "")), ("a_deceptive", msg["content"]),
                              ("a_deceptive_short", msg["content"])],
                "honest": [("question", x["question"]), ("r_honest", msg.get("thinking", "")), ("a_honest", msg["content"]),
                           ("a_honest_short", msg["content"])],
                "drop": [("question_drop", x["question"]), ("a_drop", msg["content"])]}[name]
        for field, text in want:
            for frag in fragments(EX[field]):
                assert frag in norm(text), f"{name} {role}/{key} {field}: not in record: {frag[:60]!r}"
        s = scores[role][key]
        assert (s == 3) if name != "drop" else (s == 0), f"{name} {role}/{key}: judge score {s}"
        print(f"check ok: {name} {role}/{key} score {s}")


def main() -> None:
    global _FIG, FONT
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="gi_v2")
    parser.add_argument("--root", type=Path, default=None, help="extraction root with vectors_response_only, act_response_only, scores, responses (default: the preset's)")
    parser.add_argument("--out", type=Path, default=None, help="default figures/pipeline_figure_v3_<preset>.png on scratch")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--check", action="store_true", help="verify every quoted fragment against the records before drawing")
    parser.add_argument("--refresh", action="store_true", help="recompute the cached PCA projection and kept counts")
    parser.add_argument("--no-kept-counts", action="store_true", help="leave the kept-per-side bars out of the Filter card")
    parser.add_argument("--no-footnote", action="store_true", help="leave the model / layer / PCA line under the Extract plot out")
    args = parser.parse_args()
    EX.update(PRESETS[args.preset])
    if args.no_footnote:
        EX["pca_note"] = None
    root = args.root or EX["root"]
    out = args.out or SCRATCH / f"figures/pipeline_figure_v3_{args.preset}.png"
    if args.check:
        check_quotes(root)

    for path in font_manager.findSystemFonts():
        if "LiberationSans-" in path:
            font_manager.fontManager.addfont(path)
    if FONT not in {f.name for f in font_manager.fontManager.ttflist}:
        print(f"{FONT} not installed, falling back to DejaVu Sans")
        FONT = "DejaVu Sans"
    plt.rcParams["font.family"] = FONT
    plt.rcParams["pdf.fonttype"] = 42
    fig = _FIG = plt.figure(figsize=(W, H), facecolor=SURFACE, dpi=args.dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.axis("off")

    cache = SCRATCH / f"figures/cache/pipeline_figure_{args.preset}_L{LAYER}.npz"
    pca = load_pca(root, LAYER, cache if args.root is None else cache.with_name(f"custom_{cache.name}"), args.refresh or args.root is not None)
    kept = None if args.no_kept_counts else pca[4]
    top = H - MARGIN
    pos = [(MARGIN, top), (MARGIN + CARD_W + GAP_X, top), (MARGIN, top - CARD_H - GAP_Y), (MARGIN + CARD_W + GAP_X, top - CARD_H - GAP_Y)]
    panels = [
        (None, "Personas", EX["personas_sub"], panel_personas),
        ("1", "Prompt", f"Each persona answers {EX['n_questions']} questions under each of its 5 system prompts.",
         panel_prompt),
        ("2", "Filter", "An LLM judge reads every response; only those fully in character are kept.",
         lambda *a: panel_filter(*a, kept=kept)),
        ("3", "Extract", EX.get("extract_sub", "Mean-pool the answer tokens of each kept response, average per persona, subtract honest from deceptive."), None),
    ]
    for (num, title, sub, draw), (x, y) in zip(panels, pos):
        cx, cy, cw, cb = card(ax, x, y, num, title, sub)
        if draw is None:
            panel_extract(ax, cx, cy, cw, cb, pca)
            continue
        with dry():  # measure, then spread the spare height over the card's gaps
            y_end, n_gaps = draw(ax, cx, cy, cw, cb, 0.0)
        slack = y_end - cb
        print(f"{title}: {slack:+.3f} in spare" + ("  <-- OVERFLOWS the card" if slack < -0.004 else ""))
        draw(ax, cx, cy, cw, cb, min(max(slack, 0.0) / n_gaps, MAX_EXTRA))

    # Flow arrows: Personas -> 1 (top), 1 -> 2 (down the gap, across), 2 -> 3 (bottom).
    for i in (0, 2):
        x, y = pos[i]
        arrow(ax, (x + CARD_W + 0.02, y - CARD_H / 2), (x + CARD_W + GAP_X - 0.02, y - CARD_H / 2))
    x1, y1 = pos[1]
    x2, y2 = pos[2]
    mid = y1 - CARD_H - GAP_Y / 2
    ax.plot([x1 + CARD_W / 2, x1 + CARD_W / 2, x2 + CARD_W / 2], [y1 - CARD_H - 0.02, mid, mid], color=RULE_STRONG, lw=1.0,
            solid_capstyle="round")
    arrow(ax, (x2 + CARD_W / 2, mid), (x2 + CARD_W / 2, y2 + 0.02))

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, facecolor=SURFACE)
    print(out)
    if out.suffix == ".png":
        fig.savefig(out.with_suffix(".pdf"), dpi=args.dpi, facecolor=SURFACE)
        print(out.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
