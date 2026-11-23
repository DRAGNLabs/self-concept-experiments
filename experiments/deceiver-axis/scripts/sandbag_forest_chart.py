"""Forest chart of the sandbagging result: per model and benchmark, each arm's change in the gap against unsteered with its interval.

One row per setting (dose or cap rung), the gi_v2 and the Goodfire arm of that setting on the same row, one x-axis for every panel,
and beside each panel the numbers: each axis's change [95% interval] and, at equal α, gi_v2 minus Goodfire [95% interval] (the paired
bootstrap the analysis already holds). Reads the `analysis.json` files of `sandbag_analysis.py`; nothing is recomputed.

`--x change` (default) plots each arm's change against unsteered. `--x gap` plots the gap itself, with the unsteered arm on its own
row (level with the column heads of the numbers; its gap and interval are in the panel title) and the stretch from zero to the unsteered gap shaded: that stretch is all the room an arm has to lower the gap, so a model
without a gap shows as a panel without room.

    python sandbag_forest_chart.py --model "Muse-Glimmer-30B=<gsm8k analysis.json>,<mmlu_pro analysis.json>" ... --out chart.png

`--panels`, `--title` and `--note` rename the two panels of a model and replace the title and the notes under the panels, for a
comparison that is not GSM8K against MMLU-Pro (e.g. one model on GSM8K under two notices).
"""
from __future__ import annotations

import argparse
import json

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from sandbag_chart import AXES, GRID, INK, SOFT, signed, slot

BENCHMARKS = ("GSM8K", "MMLU-Pro")


def interval(entry: dict) -> str:
    return f"{signed(entry['mean'])} [{signed(entry['ci'][0])}, {signed(entry['ci'][1])}]"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", action="append", required=True, help='"Name=gsm8k.json,mmlu_pro.json"')
    parser.add_argument("--x", choices=("change", "gap"), default="change", help="plot the change against unsteered, or the gap itself")
    parser.add_argument("--panels", help="two comma-separated panel names in place of the benchmark names")
    parser.add_argument("--title", help="chart title in place of the default")
    parser.add_argument("--note", action="append", help="note line under the panels (up to two) in place of the defaults")
    parser.add_argument("--no-notes", action="store_true", help="leave the note lines under the panels out (caption goes in the paper)")
    parser.add_argument("--public-names", action="store_true",
                        help='name the axes "Deceiver Axis" and "Reward Hacking Axis" (the paper\'s names) in place of gi_v2 and Goodfire')
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    gap = args.x == "gap"
    legend_names = ("Deceiver Axis", "Reward Hacking Axis") if args.public_names else tuple(label for _, label, _, _ in AXES)
    column_heads = (("Deceiver Axis\nΔ [95%]", "Reward Hacking Axis\nΔ [95%]", "Deceiver −\nReward Hacking") if args.public_names
                    else ("gi_v2: Δ [95%]", "Goodfire: Δ [95%]", "gi_v2 − Goodfire"))
    panels = args.panels.split(",") if args.panels else BENCHMARKS
    off = 1 if gap else 0
    models = []
    for spec in args.model:
        name, paths = spec.split("=", 1)
        models.append((name, [json.load(open(path)) for path in paths.split(",")]))
    rows = []
    for _, analyses in models:
        slots = sorted({slot(arm) for analysis in analyses for arm in analysis["arms"] if arm != "unsteered"})
        rows.append(slots)
    if gap:
        ends = [edge for _, analyses in models for analysis in analyses for value in analysis["arms"].values() for edge in value["gap"]["ci"]]
    else:
        ends = [edge for _, analyses in models for analysis in analyses for arm, value in analysis["arms"].items() if arm != "unsteered"
                for edge in value["vs_unsteered"]["ci"]]
    low, high = min(ends) - 4, max(ends) + 4

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    foot = 0.62 if args.no_notes else 1.25  # inches under the panels: x-axis label, then the note lines
    height = sum(0.31 * (len(r) + 1.6) for r in rows) + 1.6 + foot
    figure = plt.figure(figsize=(17.6, height), facecolor="white")
    grid = figure.add_gridspec(len(models), 4, width_ratios=[1, 1.12, 1, 1.12], height_ratios=[len(r) + 1.6 for r in rows],
                               left=0.062, right=0.995, top=1 - 1.5 / height, bottom=foot / height, hspace=0.34, wspace=0.04)
    for m, (name, analyses) in enumerate(models):
        slots = rows[m]
        doses = sum(1 for s in slots if s[0] == 1)
        for b, analysis in enumerate(analyses):
            axes = figure.add_subplot(grid[m, 2 * b])
            text = figure.add_subplot(grid[m, 2 * b + 1])
            base = analysis["arms"]["unsteered"]["gap"]["mean"]
            stated = interval(analysis["arms"]["unsteered"]["gap"]) if gap else signed(base)
            title = axes.set_title(f"{name} · {panels[b]}", loc="left", fontsize=12.5, fontweight="bold", color=INK, pad=6)
            axes.annotate(f"unsteered gap {stated} points  ·  {analysis['arms']['unsteered']['seeds']} seeds × {analysis['items']} items",
                          xy=(1, 0), xycoords=title, xytext=(10, 1), textcoords="offset points", ha="left", va="bottom", fontsize=10,
                          color=SOFT, annotation_clip=False)
            axes.axvline(0, color=INK, linewidth=1, zorder=1)
            axes.axhline(doses + off - 0.5, color=GRID, linewidth=1, zorder=0)
            if gap:
                unsteered = analysis["arms"]["unsteered"]["gap"]
                axes.axvspan(min(0, base), max(0, base), color="#000000", alpha=0.07, linewidth=0, zorder=0)
                axes.axvline(base, color=INK, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
                axes.plot(unsteered["ci"], [0, 0], color=INK, linewidth=2, solid_capstyle="round", zorder=2)
                axes.plot(base, 0, marker="D", markersize=6.5, color=INK, linewidth=0, zorder=3)
            for y, entry in enumerate(slots, start=off):
                for k, (prefix, _, color, marker) in enumerate(AXES):
                    arm = analysis["arms"].get(prefix + ("m" + f"{entry[1]:g}" if entry[0] == 1 else f"cap_p{-entry[1]:g}"))
                    if arm is None:
                        continue
                    delta = arm["vs_unsteered"]
                    shown = arm["gap"] if gap else delta
                    at = y + (-0.17 if k == 0 else 0.17)
                    axes.plot(shown["ci"], [at, at], color=color, linewidth=2, solid_capstyle="round", zorder=2)
                    axes.plot(shown["mean"], at, marker=marker, markersize=6.5, markerfacecolor=color if arm["usable"] else "white",
                              markeredgecolor=color, markeredgewidth=1.6, linewidth=0, zorder=3)
                    text.text(0.5 + k, y, interval(delta), ha="center", va="center", fontsize=8, color=INK)
                if entry[0] == 1:
                    pair = analysis["pairs"].get(f"gi_m{entry[1]:g} - gf_m{entry[1]:g}")
                    if pair:
                        text.text(2.5, y, interval(pair), ha="center", va="center", fontsize=8, color=INK)
            for k, head in enumerate(column_heads):
                text.text(0.5 + k, 0 if gap else -0.95, head, ha="center", va="center", fontsize=8, color=SOFT, fontweight="bold", linespacing=1.15)
            for a in (axes, text):
                a.set_ylim(len(slots) + off - 0.45, -0.75 if gap else -1.4)
                for side in ("top", "right", "left"):
                    a.spines[side].set_visible(False)
            for y in range(len(slots) + off):
                text.axhline(y + 0.5, color=GRID, linewidth=0.6)
            if not gap:
                text.axhline(-0.5, color=GRID, linewidth=0.6)
            text.set_xlim(0, 3)
            text.set_xticks([])
            text.set_yticks([])
            text.spines["bottom"].set_visible(False)
            axes.set_xlim(low, high)
            axes.set_yticks(range(len(slots) + off))
            axes.set_yticklabels((["unsteered"] * off + [s[2] for s in slots]) if b == 0 else [], fontsize=9, color=INK)
            axes.tick_params(axis="y", length=0)
            axes.tick_params(axis="x", colors=SOFT, labelsize=8.5)
            axes.grid(axis="x", color=GRID, linewidth=0.7, zorder=0)
            axes.set_axisbelow(True)
            axes.spines["bottom"].set_color("#c9c7c1")
            if m == len(models) - 1:
                axes.set_xlabel("sandbagging gap (points)" if gap else "change in gap against unsteered (points)", fontsize=9, color=SOFT)

    figure.text(0.012, 1 - 0.36 / height, args.title or ("Sandbagging under an incentive: the gap under each steering arm"
                if gap else "Sandbagging under an incentive: how each axis moves the gap, four models, two benchmarks"),
                fontsize=14, fontweight="bold", color=INK, va="center")
    figure.text(0.012, 1 - 0.66 / height, "Sandbagging gap with 95% interval (bootstrap over items). An arm inside the shading lowers the gap, at "
                "zero it removes it, right of the dashed line it raises it. Same x-axis in every panel." if gap else
                "Change in the sandbagging gap against the unsteered model, with 95% interval (bootstrap over items); "
                "left of zero = smaller gap. Same x-axis in every panel.", fontsize=9.5, color=SOFT, va="center")
    handles = [Line2D([], [], color=color, marker=marker, markersize=7, linewidth=2, label=label)
               for label, (_, _, color, marker) in zip(legend_names, AXES)]
    if gap:
        handles.append(Line2D([], [], color=INK, marker="D", markersize=6.5, linewidth=2, label="unsteered (dashed line)"))
        handles.append(Patch(facecolor="#000000", alpha=0.07, label="zero to the unsteered gap: the room there is to lower it"))
    handles.append(Line2D([], [], color=SOFT, marker="o", markerfacecolor="white", markeredgewidth=1.6, markersize=7, linewidth=0,
                          label="hollow: arm not usable"))
    figure.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.008, 1 - 0.8 / height), ncol=5, frameon=False, fontsize=9,
                  handlelength=2.2, columnspacing=2.2)
    notes = args.note or [
        "Gap = % of items solved when a low score leads to unlearning minus % solved when a high score does. Δ = change against unsteered. "
        f"{' '.join(column_heads[2].split())}: difference of the two gaps at equal α (paired bootstrap); cap rungs are not matched across the axes, so no "
        "difference is given for them.",
        "A smaller gap is not by itself a decision not to underperform: the blind transcript reads found that on Muse, and on Gemma found "
        "answers that ignore the notice (caps) or break down (α −0.16)."]
    for line, at in zip([] if args.no_notes else notes, (0.62, 0.36)):
        figure.text(0.012, at / height, line, fontsize=8.5, color=SOFT, va="center")
    figure.savefig(args.out, dpi=150, facecolor="white")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
