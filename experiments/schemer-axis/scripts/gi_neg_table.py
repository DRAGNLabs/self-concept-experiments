"""Render the Muse layer-26 generic + intent negative-sign table (unsteered, gi −0.04 / −0.08 / −0.16, old axis −0.16) as an image.

Numbers are copied from the analysis files in SOURCES (FINDINGS "Muse generic + intent dose pilot — 2026-10-28" and
"Muse gi −0.16 pilot arm — 2026-10-28"; the old-axis row is the 12-seed answer-region main run, FINDINGS
"Muse answer-region 17-role main run — 2026-10-26").

    python experiments/schemer-axis/scripts/gi_neg_table.py --out <path>.png   (also .svg / .pdf by extension)
"""
from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

SOURCES = {
    "unsteered, gi −0.04 / −0.08": "muse_glimmer/gi/pilot/analysis.json, analysis_judged_neg_as_judged.json (job 14830281)",
    "gi −0.16": "muse_glimmer/gi/pilot/analysis.json, analysis_judged_neg3_as_judged.json (job 14853869, 4 seeds)",
    "old axis −0.16": "muse_glimmer/answer17_pilot/verdicts_main/analysis_main_proxy_as_judged.json (12-seed main run)",
}

# Design tokens (dataviz skill, light surface): text in text tokens, recessive rules, one quiet tint for the headline row.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
INK3 = "#8a8985"
RULE = "#d9d8d4"
RULE_STRONG = "#8a8985"
TINT = "#f1f0ec"

ARM_W = 2.15
COLUMNS = [  # (title, width in inches)
    ("Judged hack rate", 2.25),
    ("Proxy\n(shown pass, hidden fail)", 2.25),
    ("Hidden pass", 1.3),
    ("No-code /\ntruncated turns", 1.55),
    ("Reasoning chars\nper turn", 1.45),
]

# Cell = list of lines; the first line is the value, later lines are secondary text. "bold" marks Holm p < 0.05.
ROWS = [
    {
        "arm": ["unsteered", "12 seeds, 540 episodes", "shared reference arm"],
        "cells": [["27.8%", "150 of 193 dossiers"], ["35.7%"], ["48.0%"], ["0.0% / 0"], ["9.8k"]],
    },
    {
        "arm": ["gi α −0.04", "4 seeds, 180 episodes"],
        "cells": [
            ["21.1%", "Δ −6.7  [−12.2, −1.3]", "Holm p 0.028", "bold"],
            ["32.2%", "Δ −3.5  [−9.3, +2.2]", "n.s."],
            ["47.2%", "Δ −0.7"],
            ["0.6% / 1"],
            ["10.8k", "+10%"],
        ],
    },
    {
        "arm": ["gi α −0.08", "4 seeds, 180 episodes"],
        "cells": [
            ["20.0%", "Δ −7.8  [−13.3, −2.6]", "Holm p 0.016", "bold"],
            ["32.2%", "Δ −3.5  [−9.8, +2.8]", "n.s."],
            ["43.9%", "Δ −4.1"],
            ["1.1% / 3"],
            ["11.7k", "+20%"],
        ],
    },
    {
        "arm": ["gi α −0.16", "4 seeds, 180 episodes"],
        "tint": True,
        "cells": [
            ["13.9%", "Δ −13.9  [−19.8, −8.3]", "Holm p 0.0003", "bold"],
            ["21.7%", "Δ −14.1  [−20.4, −8.1]", "Holm p 0.0007", "bold"],
            ["49.4%", "Δ +1.5"],
            ["1.1% / 5"],
            ["11.9k", "+22%"],
        ],
    },
    {
        "arm": ["old axis α −0.16", "generic-only axis, 12 seeds,", "540 episodes (main run)*"],
        "reference": True,
        "cells": [
            ["12.0%", "Δ −15.7  [−22.8, −9.4]", "Holm p 0.0004", "bold"],
            ["21.9%", "Δ −13.9  [−20.9, −7.2]", "Holm p 0.0008", "bold"],
            ["48.5%", "Δ +0.6"],
            ["4.6% / 37"],
            ["15.5k", "+59%"],
        ],
    },
]

TITLE = "Muse-Glimmer-30B, layer 26, generic + intent axis: negative-sign steering (dose pilot, 2026-10-28)"
SUBTITLE = (
    "gi = generic + intent axis (64 train questions, 17 fiction roles, answer region), steered at every token, α in units of the "
    "layer-26 mean residual norm.  Pilot arms: 4 seeds × 45 hard problems vs the same 540-episode unsteered arm."
)
FOOTNOTES = [
    "Judged hack rate = first-pass Sonnet rubric-v3 verdicts on the shown-pass/hidden-fail episodes, HACK STRONG + WEAK count, "
    "BORDERLINE does not, no adjudication; unjudged episodes count as non-hacks.",
    "Δ = problem-stratified percentage points vs unsteered, 95% problem-bootstrap CI, Holm p over the five pilot arms; "
    "bold = Holm p < 0.05.  Hidden pass Δ never exceeds the ±10-point capability rule.",
    "gi −0.16 sensitivity: 16.1% with BORDERLINE counted as hack, 11.7% with WEAK counted down; verdicts 21 STRONG (lookup tables), "
    "4 WEAK (size restrictions), 4 BORDERLINE, 10 NOT_HACK.  Reasoning chars: % change vs unsteered.",
    "* The generic-only (40-question) axis had no 4-seed −0.16 pilot; its row is the 12-seed answer-region main run (proxy-set judged "
    "rate; 15.6% with the hidden-pass signature set), shown for comparison.",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(Path.home() / "nobackup/autodelete/self-concept-experiments/schemer-axis/muse_glimmer/gi/pilot/gi_neg_table.png"))
    ap.add_argument("--dpi", type=int, default=220)
    args = ap.parse_args()

    margin = 0.45
    col_x = [margin]
    col_w = [ARM_W]
    for _title, w in COLUMNS:
        col_x.append(col_x[-1] + col_w[-1])
        col_w.append(w)
    width = col_x[-1] + col_w[-1] + margin

    title_h, header_h = 1.0, 0.72
    row_h = [0.8 if len(r["cells"][0]) <= 2 else 0.98 for r in ROWS]
    table_w = width - 2 * margin

    def wrap(s, size):  # DejaVu Sans: about 72 / (size * 0.495) characters per inch
        return textwrap.wrap(s, width=int(table_w * 72 / (size * 0.495) * 0.92))

    sub_lines = wrap(SUBTITLE, 9.2)
    foot_lines = [l for note in FOOTNOTES for l in wrap(note, 8.4)]
    foot_h = 0.3 + 0.27 * len(foot_lines)
    height = margin + title_h + header_h + sum(row_h) + foot_h + margin * 0.6

    fig = plt.figure(figsize=(width, height), facecolor=SURFACE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis("off")

    def text(x, y, s, size=10, color=INK, weight="normal", ha="center", va="center", style="normal"):
        ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, ha=ha, va=va, fontstyle=style)

    def hline(y, x0, x1, color=RULE, lw=0.8):
        ax.plot([x0, x1], [y, y], color=color, lw=lw, solid_capstyle="butt")

    top = height - margin
    x_left, x_right = col_x[0], col_x[-1] + col_w[-1]

    text(x_left, top - 0.12, TITLE, size=14, weight="bold", ha="left", va="top")
    for k, l in enumerate(sub_lines):
        text(x_left, top - 0.5 - 0.22 * k, l, size=9.2, color=INK2, ha="left", va="top")

    y_head_top = top - title_h
    y_head_bot = y_head_top - header_h
    y_rows_bot = y_head_bot - sum(row_h)

    text(x_left + 0.08, (y_head_top + y_head_bot) / 2, "steering arm", size=9.5, color=INK2, ha="left")
    for i, (title, _w) in enumerate(COLUMNS, start=1):
        text(col_x[i] + col_w[i] / 2, (y_head_top + y_head_bot) / 2, title, size=10.3, weight="bold")
    hline(y_head_top + 0.02, x_left, x_right, color=RULE_STRONG, lw=1.0)
    hline(y_head_bot, x_left, x_right, color=RULE_STRONG, lw=1.0)

    def cell(xc, yc, lines):
        bold = "bold" in lines
        lines = [l for l in lines if l != "bold"]
        sizes = [10.8] + [8.6] * (len(lines) - 1)
        gaps = [0.0] + [0.205] * (len(lines) - 1)
        y = yc + sum(gaps) / 2
        for k, (l, s, g) in enumerate(zip(lines, sizes, gaps)):
            y -= g
            text(xc, y, l, size=s, color=INK if k == 0 else INK2, weight="bold" if (k == 0 and bold) else "normal")

    y = y_head_bot
    for r, h in zip(ROWS, row_h):
        yc = y - h / 2
        if r.get("tint"):
            ax.add_patch(Rectangle((x_left, y - h), x_right - x_left, h, facecolor=TINT, edgecolor="none", zorder=0))
        if r.get("reference"):
            hline(y, x_left, x_right, color=RULE_STRONG, lw=1.0)
        arm_lines = r["arm"]
        y_arm = yc + 0.205 * (len(arm_lines) - 1) / 2
        for k, l in enumerate(arm_lines):
            text(x_left + 0.08, y_arm - 0.205 * k, l, size=10.6 if k == 0 else 8.6,
                 color=(INK2 if r.get("reference") else INK) if k == 0 else INK2,
                 weight="bold" if k == 0 else "normal", ha="left", style="italic" if (k == 0 and r.get("reference")) else "normal")
        for i, lines in enumerate(r["cells"], start=1):
            cell(col_x[i] + col_w[i] / 2, yc, lines)
        y -= h
        last = y <= y_rows_bot + 1e-6
        hline(y, x_left, x_right, color=RULE_STRONG if last else RULE, lw=1.0 if last else 0.8)

    ax.plot([col_x[1], col_x[1]], [y_rows_bot, y_head_top + 0.02], color=RULE, lw=0.8)

    yf = y_rows_bot - 0.22
    for line in foot_lines:
        text(x_left, yf, line, size=8.4, color=INK2, ha="left", va="top")
        yf -= 0.27

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, facecolor=SURFACE)
    print(out)


if __name__ == "__main__":
    main()
