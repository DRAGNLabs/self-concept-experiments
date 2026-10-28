"""Render the Muse negative-sign comparison table (generic-only axis vs generic + intent axis) as an image.

The numbers are copied from the analysis files named in SOURCES below (FINDINGS "Muse generic + intent dose pilot —
2026-10-28", "Muse answer-region 17-role pilot — 2026-10-23" and "Muse answer-region 17-role main run — 2026-10-26");
edit ROWS when the gi −0.16 arm lands.

    python experiments/schemer-axis/scripts/neg_axis_table.py --out <path>.png   (also .svg / .pdf by extension)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

SOURCES = {
    "generic-only pilot": "muse_glimmer/answer17_pilot/analysis.json, analysis_judged_as_judged.json (job 14706298)",
    "generic-only −0.16": "muse_glimmer/answer17_pilot/verdicts_main/analysis_main_proxy_as_judged.json (12-seed main run)",
    "generic + intent pilot": "muse_glimmer/gi/pilot/analysis.json, analysis_judged_neg_as_judged.json (job 14830281)",
}

# Design tokens (dataviz skill, light surface): text in text tokens, recessive rules, one quiet tint for the new axis.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
INK3 = "#8a8985"
RULE = "#d9d8d4"
RULE_STRONG = "#8a8985"
TINT = "#f1f0ec"

OLD = "generic only"
NEW = "generic + intent"

# Column groups: (title, width of each half in inches). The first column is the arm label.
ARM_W = 1.7
GROUPS = [
    ("Judged hack rate", 2.05),
    ("Proxy (shown pass, hidden fail)", 1.85),
    ("Hidden pass", 1.35),
    ("No-code / truncated turns", 1.45),
    ("Reasoning chars per turn", 1.35),
]

# Cell = list of lines; the first line is the value, later lines are secondary text. "bold" marks Holm p < 0.05.
# A row is {"arm": [lines], "old": [cells...], "new": [cells...] or "span": [cells...] (shared across both axes)}.
ROWS = [
    {
        "arm": ["unsteered", "12 seeds, 540 episodes", "shared reference arm"],
        "span": [
            ["27.8%", "150 of 193 dossiers"],
            ["35.7%"],
            ["48.0%"],
            ["0.0% / 0"],
            ["9.8k"],
        ],
    },
    {
        "arm": ["α −0.04", "4 seeds, 180 episodes"],
        "old": [
            ["26.1%", "Δ −1.7  [−7.0, +4.1]", "Holm p 0.63"],
            ["37.2%", "Δ +1.5  [−5.4, +8.3]", "Holm p 0.72"],
            ["45.0%", "Δ −3.0"],
            ["0.0% / 0"],
            ["11.5k", "+18%"],
        ],
        "new": [
            ["21.1%", "Δ −6.7  [−12.2, −1.3]", "Holm p 0.028", "bold"],
            ["32.2%", "Δ −3.5  [−9.3, +2.2]", "n.s."],
            ["47.2%", "Δ −0.7"],
            ["0.6% / 1"],
            ["10.8k", "+10%"],
        ],
    },
    {
        "arm": ["α −0.08", "4 seeds, 180 episodes"],
        "old": [
            ["16.1%", "Δ −11.7  [−17.8, −6.1]", "Holm p 0.0006", "bold"],
            ["25.0%", "Δ −10.7  [−17.4, −4.3]", "Holm p 0.005", "bold"],
            ["50.6%", "Δ +2.6"],
            ["1.1% / 2"],
            ["13.0k", "+34%"],
        ],
        "new": [
            ["20.0%", "Δ −7.8  [−13.3, −2.6]", "Holm p 0.016", "bold"],
            ["32.2%", "Δ −3.5  [−9.8, +2.8]", "n.s."],
            ["43.9%", "Δ −4.1"],
            ["1.1% / 3"],
            ["11.7k", "+20%"],
        ],
    },
    {
        "arm": ["α −0.16", "generic only: 12 seeds,", "540 episodes (main run)*"],
        "old": [
            ["12.0%", "Δ −15.7  [−22.8, −9.4]", "Holm p 0.0004", "bold"],
            ["21.9%", "Δ −13.9  [−20.9, −7.2]", "Holm p 0.0008", "bold"],
            ["48.5%", "Δ +0.6"],
            ["4.6% / 37"],
            ["15.5k", "+59%"],
        ],
        "new": "running",
        "new_note": ["running", "job 14853869, seeds 0–3"],
    },
]

TITLE = "Muse-Glimmer-30B, layer 26, negative-sign steering: generic-only axis vs generic + intent axis"
SUBTITLE = (
    "Generic only = 40 extraction questions, 17 fiction roles, answer region (pilot 2026-10-23).  "
    "Generic + intent = 64 train questions, same roles and region (pilot 2026-10-28).  "
    "Pilot arms: 4 seeds × 45 problems vs the same 540-episode unsteered arm."
)
FOOTNOTES = [
    "Judged hack rate = first-pass Sonnet rubric-v3 verdicts on the shown-pass/hidden-fail episodes, HACK STRONG + WEAK count, "
    "BORDERLINE does not, no adjudication; unjudged episodes count as non-hacks.",
    "Δ = problem-stratified percentage points vs unsteered, 95% problem-bootstrap CI, Holm p over the arms of each run; "
    "bold = Holm p < 0.05.  Hidden pass Δ never exceeds the ±10-point capability rule.",
    "* The generic-only −0.16 arm has no 4-seed pilot; the row is the 12-seed main run (proxy-set judged rate; "
    "with the hidden-pass signature set it is 15.6%).  Reasoning chars: % change vs unsteered.",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(Path.home() / "nobackup/autodelete/self-concept-experiments/schemer-axis/muse_glimmer/gi/pilot/neg_axis_comparison.png"))
    ap.add_argument("--dpi", type=int, default=220)
    args = ap.parse_args()

    # Geometry in inches; the axes spans the whole figure with 1 data unit = 1 inch.
    margin = 0.45
    col_x = [margin]  # left edges: arm, then old/new per group
    col_w = [ARM_W]
    for _title, half in GROUPS:
        for _ in range(2):
            col_x.append(col_x[-1] + col_w[-1])
            col_w.append(half)
    width = col_x[-1] + col_w[-1] + margin

    title_h, header_h, foot_h = 1.05, 0.78, 1.15
    row_h = [0.78 if "span" in r else 0.98 for r in ROWS]
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

    # Title block
    text(x_left, top - 0.12, TITLE, size=14, weight="bold", ha="left", va="top")
    text(x_left, top - 0.55, SUBTITLE, size=9.2, color=INK2, ha="left", va="top")

    # Header: group titles spanning each old/new pair, axis labels under them.
    y_head_top = top - title_h
    y_head_bot = y_head_top - header_h
    # tint behind the generic + intent columns (header through last row)
    y_rows_bot = y_head_bot - sum(row_h)
    for g in range(len(GROUPS)):
        i_new = 2 + 2 * g
        ax.add_patch(Rectangle((col_x[i_new], y_rows_bot), col_w[i_new], y_head_top - y_rows_bot, facecolor=TINT, edgecolor="none", zorder=0))

    text(x_left + 0.08, y_head_top - 0.42, "steering arm", size=9.5, color=INK2, ha="left")
    for g, (title, _half) in enumerate(GROUPS):
        i_old = 1 + 2 * g
        x0 = col_x[i_old]
        x1 = col_x[i_old + 1] + col_w[i_old + 1]
        text((x0 + x1) / 2, y_head_top - 0.2, title, size=10.3, weight="bold")
        hline(y_head_top - 0.38, x0 + 0.08, x1 - 0.08, color=RULE)
        text(x0 + col_w[i_old] / 2, y_head_top - 0.58, OLD, size=9.2, color=INK2)
        text(col_x[i_old + 1] + col_w[i_old + 1] / 2, y_head_top - 0.58, NEW, size=9.2, color=INK2)
    hline(y_head_bot, x_left, x_right, color=RULE_STRONG, lw=1.0)
    hline(y_head_top + 0.02, x_left, x_right, color=RULE_STRONG, lw=1.0)

    def cell(xc, yc, lines, h):
        """Value line in primary ink, secondary lines below; vertically centred as a block."""
        bold = "bold" in lines
        lines = [l for l in lines if l != "bold"]
        sizes = [10.8] + [8.6] * (len(lines) - 1)
        gaps = [0.0] + [0.205] * (len(lines) - 1)
        block = sum(gaps)
        y = yc + block / 2
        for k, (l, s, g) in enumerate(zip(lines, sizes, gaps)):
            y -= g
            text(xc, y, l, size=s, color=INK if k == 0 else INK2, weight="bold" if (k == 0 and bold) else "normal")

    # Body rows
    y = y_head_bot
    for r, h in zip(ROWS, row_h):
        yc = y - h / 2
        arm_lines = r["arm"]
        y_arm = yc + 0.205 * (len(arm_lines) - 1) / 2
        for k, l in enumerate(arm_lines):
            text(x_left + 0.08, y_arm - 0.205 * k, l, size=10.6 if k == 0 else 8.6, color=INK if k == 0 else INK2,
                 weight="bold" if k == 0 else "normal", ha="left")
        for g in range(len(GROUPS)):
            i_old = 1 + 2 * g
            i_new = i_old + 1
            if "span" in r:
                xc = (col_x[i_old] + col_x[i_new] + col_w[i_new]) / 2
                cell(xc, yc, r["span"][g], h)
            else:
                cell(col_x[i_old] + col_w[i_old] / 2, yc, r["old"][g], h)
                if r["new"] == "running":
                    if g == 0:
                        xc = col_x[i_new] + col_w[i_new] / 2
                        lines = r["new_note"]
                        text(xc, yc + 0.1, lines[0], size=10.2, color=INK3, style="italic")
                        text(xc, yc - 0.12, lines[1], size=8.6, color=INK3)
                    else:
                        text(col_x[i_new] + col_w[i_new] / 2, yc, "—", size=10.5, color=INK3)
                else:
                    cell(col_x[i_new] + col_w[i_new] / 2, yc, r["new"][g], h)
        y -= h
        hline(y, x_left, x_right, color=RULE if y > y_rows_bot + 1e-6 else RULE_STRONG, lw=0.8 if y > y_rows_bot + 1e-6 else 1.0)

    # Light vertical separators between metric groups
    for g in range(1, len(GROUPS)):
        xg = col_x[1 + 2 * g]
        ax.plot([xg, xg], [y_rows_bot, y_head_top + 0.02], color=RULE, lw=0.8)
    ax.plot([col_x[1], col_x[1]], [y_rows_bot, y_head_top + 0.02], color=RULE, lw=0.8)

    # Footnotes
    yf = y_rows_bot - 0.22
    for line in FOOTNOTES:
        text(x_left, yf, line, size=8.4, color=INK2, ha="left", va="top")
        yf -= 0.3

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, facecolor=SURFACE)
    print(out)


if __name__ == "__main__":
    main()
