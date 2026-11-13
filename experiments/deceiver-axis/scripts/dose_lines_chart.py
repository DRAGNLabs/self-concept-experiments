"""Dose-line chart: per model, the change in judged hack rate and in the benchmark composite against steering dose α, for the
gi_v2 vector and the Goodfire reward-hacking axis (four lines per panel), with the plotted numbers in a table under each panel.

One panel per model. x = steering dose α in layer norms: the union of the doses run on the two axes, in order and evenly
spaced (not to scale), unsteered (α = 0) in the middle. y = change from the model's unsteered arm in points, one y axis for
both measures (both are percentage points): judged hack rate (solid line, circles) and benchmark composite (dashed line,
squares; pareto_chart.py's pooled score by default). Colour = steering axis, the colours of pareto_chart.py --per-model. Error
bars = 95% CIs of the difference (hack rate: paired bootstrap over problems, from the analysis file; benchmark: item-paired
bootstrap). A hollow marker = the model stops submitting code on that arm (no-code share more than 10 points above unsteered),
drawn on both of the arm's points. Caps and floors have no dose and are not on this chart.

A dose whose task-set arm has no benchmark run yet keeps its hack-rate point; its benchmark cell reads "no run" (2026-11-13:
gi_v2 ±0.16 on Gemma). The table under each panel has a column per dose, under its tick: Δ, 95% CI and (hack rate) the Holm p
of the analysis file's contrast family.

Data, presets and the benchmark composite come from pareto_chart.py (`MODELS`, `load_axis`). Writes <out>.png, .pdf and .json.

    python experiments/deceiver-axis/scripts/dose_lines_chart.py --model gptoss muse gemma qwen --out <path>.png
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import pareto_chart as pc

AXIS_SHORT = {"gi_v2": "gi_v2", "gfar": "Goodfire"}
MEASURES = {"hack": dict(value="hack_diff", ci="hack_diff_ci", name="judged hack rate", marker="o", size=7.5, width=2.0, dash="-"),
            "bench": dict(value="diff", ci="diff_ci", name="benchmark score", marker="s", size=6.2, width=1.6, dash=pc.DASHES)}
DODGE = 0.07  # the two axes' markers sit this far (in dose steps) either side of the tick, so their error bars do not overlap
ROW = 0.185  # table row height, inches
PANEL_WIDTH, LEFT, RIGHT, PLOT_HEIGHT, TITLE_HEIGHT, XAXIS_HEIGHT, GAP = 7.9, 1.75, 0.25, 3.7, 0.62, 0.62, 0.45


def minus(text: str) -> str:
    return text.replace("-", "−")


def dose_label(value: float) -> str:
    return "0" if value == 0 else minus(f"{value:+g}")


def draw_plot(axes, groups: list[dict], doses: list[float]) -> None:
    slot = {value: index for index, value in enumerate(doses)}
    pc.style_axes(axes)
    axes.axvline(slot[0.0], color=pc.GRID, linewidth=0.6, zorder=1)  # style_axes draws x = 0 dark; the unsteered column is not at 0
    for line in axes.lines[:]:
        if line.get_xdata()[0] == 0 and line.get_color() == pc.ZERO and list(line.get_ydata()) == [0, 1]:
            line.remove()
    for index, group in enumerate(groups):
        shift = (index - (len(groups) - 1) / 2) * 2 * DODGE
        for measure in MEASURES.values():
            points = [(slot[row["dose"]], row) for row in group["rows"] if row[measure["value"]] is not None]
            xs = [x + shift for x, _ in points]
            line_x = sorted(xs + [slot[0.0]])
            line_y = [0.0 if x == slot[0.0] else points[xs.index(x)][1][measure["value"]] for x in line_x]
            axes.plot(line_x, line_y, color=group["color"], linewidth=measure["width"], linestyle=measure["dash"], zorder=3,
                      solid_capstyle="round", solid_joinstyle="round")
            for x, (_, row) in zip(xs, points):
                y, (low, high) = row[measure["value"]], row[measure["ci"]]
                axes.errorbar(x, y, yerr=[[y - low], [high - y]], fmt="none", ecolor=group["color"], elinewidth=1, alpha=0.4, zorder=2)
                face, edge = (pc.SURFACE, group["color"]) if row["stops_submitting"] else (group["color"], pc.SURFACE)
                axes.plot(x, y, measure["marker"], markersize=measure["size"], color=face, markeredgecolor=edge, markeredgewidth=1.6, zorder=4)
    axes.plot(slot[0.0], 0, "o", markersize=8, color=pc.INK, markeredgecolor=pc.SURFACE, markeredgewidth=1.6, zorder=5)
    axes.set_xlim(-0.5, len(doses) - 0.5)
    axes.set_xticks(range(len(doses)), [dose_label(value) for value in doses])
    axes.set_xlabel("steering dose α, layer norms (evenly spaced, not to scale; 0 = unsteered)", fontsize=9, color=pc.INK2)
    axes.set_ylabel("Δ vs unsteered, points", fontsize=9.5, color=pc.INK2)
    axes.grid(axis="x", visible=False)


def table_rows(groups: list[dict]) -> list[dict]:
    """The table's rows, top to bottom: per axis, hack rate (Δ, CI, Holm p) then benchmark (Δ, CI)."""
    rows = []
    for group in groups:
        name = AXIS_SHORT[group["axis_key"]]
        rows += [dict(group=group, label=f"{name}: Δ hack rate", cell="hack", main=True, first=True),
                 dict(group=group, label="95% CI", cell="hack_ci"), dict(group=group, label="Holm p", cell="hack_p"),
                 dict(group=group, label=f"{name}: Δ benchmark", cell="bench", main=True, first=True),
                 dict(group=group, label="95% CI", cell="bench_ci")]
    return rows


def cell_text(row: dict | None, cell: str) -> str:
    if row is None:
        return "–"
    if cell.startswith("bench") and row["diff"] is None:
        return "no run" if cell == "bench" else ""
    if cell == "hack":
        return minus(f"{row['hack_diff']:+.1f}")
    if cell == "hack_ci":
        return minus(f"{row['hack_diff_ci'][0]:+.1f}, {row['hack_diff_ci'][1]:+.1f}")
    if cell == "hack_p":
        p = row["hack_p_holm"]
        return "" if p is None else "<0.001" if p < 0.001 else f"{p:.3f}" if p < 0.1 else f"{p:.2f}"
    if cell == "bench":
        return minus(f"{row['diff']:+.2f}")
    return minus(f"{row['diff_ci'][0]:+.2f}, {row['diff_ci'][1]:+.2f}")


def draw_table(axes, groups: list[dict], doses: list[float], label_x: float) -> None:
    """`axes` spans the plot's width with the same x range, so a column sits under its tick. y runs in rows, the header at the top."""
    rows = table_rows(groups)
    total = len(rows) + 1
    axes.set_xlim(-0.5, len(doses) - 0.5)
    axes.set_ylim(0, total)
    axes.axis("off")
    by_dose = [{row["dose"]: row for row in group["rows"]} for group in groups]
    edge = dict(color=pc.AXIS, linewidth=0.8, clip_on=False, zorder=2)
    light = dict(color=pc.GRID, linewidth=0.6, clip_on=False, zorder=1)
    left = label_x  # data x of the left edge of the label column
    axes.plot([left, len(doses) - 0.5], [total, total], **edge)
    axes.plot([left, len(doses) - 0.5], [0, 0], **edge)
    axes.plot([left, len(doses) - 0.5], [total - 1, total - 1], **edge)
    for column in range(len(doses) + 1):
        axes.plot([column - 0.5, column - 0.5], [0, total], **(edge if column in (0, len(doses)) else light))
    axes.plot([left, left], [0, total], **edge)
    axes.text(-0.5 - 0.08, total - 0.5, "steering dose α (Δ in points)", fontsize=7.6, color=pc.INK2, ha="right", va="center", clip_on=False)
    for column, value in enumerate(doses):
        axes.text(column, total - 0.5, dose_label(value), fontsize=8, color=pc.INK, ha="center", va="center", fontweight="bold")
    for index, row in enumerate(rows):
        y = total - 1.5 - index
        group = row["group"]
        if row.get("first") and index:
            axes.plot([left, len(doses) - 0.5], [y + 0.5, y + 0.5], **(edge if row["cell"] == "hack" else light))
        if row.get("first"):
            span = 3 if row["cell"] == "hack" else 2
            axes.plot([left + 0.09, left + 0.09], [y + 0.36, y + 0.64 - span], color=group["color"], linewidth=2.6, solid_capstyle="butt",
                      clip_on=False, zorder=3)
        axes.text(-0.5 - 0.08, y, row["label"], fontsize=7.6 if row.get("main") else 7, color=pc.INK if row.get("main") else pc.MUTED,
                  ha="right", va="center", clip_on=False)
        for column, value in enumerate(doses):
            if value == 0:
                text = "unsteered" if row["cell"] == "hack" and group is groups[0] else ""
                axes.text(column, y, text, fontsize=6.8, color=pc.MUTED, ha="center", va="center")
                continue
            arm = by_dose[groups.index(group)].get(value)
            axes.text(column, y, cell_text(arm, row["cell"]), fontsize=8 if row.get("main") else 6.5,
                      color=pc.INK if row.get("main") and arm is not None else pc.MUTED, ha="center", va="center")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", nargs="+", choices=sorted(pc.MODELS), default=["gptoss", "muse", "gemma", "qwen"])
    parser.add_argument("--axes", nargs="+", default=["gi_v2", "gfar"])
    parser.add_argument("--composite", choices=sorted(pc.COMPOSITES), default="pooled")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    combine, score_name = pc.COMPOSITES[args.composite]
    palette = dict(zip(args.axes, pc.GROUP_COLORS))
    panels = []
    for model_key in args.model:
        model, groups = pc.MODELS[model_key], []
        for axis_key in args.axes:
            data = pc.load_axis(model, model["axes"][axis_key], combine, keep_without_benchmark=True)
            rows = sorted((dict(arm=arm, dose=pc.dose(arm), **row) for arm, row in data["arms"].items() if row["kind"] == "steering"),
                          key=lambda row: row["dose"])
            groups.append(dict(axis_key=axis_key, color=palette[axis_key], data=data, rows=rows))
        doses = sorted({row["dose"] for group in groups for row in group["rows"]} | {0.0})
        panels.append(dict(model_key=model_key, model=model, groups=groups, doses=doses))

    columns = 1 if len(panels) == 1 else 2
    grid_rows = math.ceil(len(panels) / columns)
    table_height = ROW * (5 * len(args.axes) + 1)
    panel_height = TITLE_HEIGHT + PLOT_HEIGHT + XAXIS_HEIGHT + table_height + GAP
    width = PANEL_WIDTH * columns
    title = pc.wrap("Steering dose against judged reward hacking and benchmark score: "
                    + " and ".join(pc.AXIS_NAMES[key] for key in args.axes) + (f" on {pc.COUNT_WORDS.get(len(panels), len(panels))} models"
                                                                                 if len(panels) > 1 else f" on {panels[0]['model']['name']}"),
                    12.5, width - 0.3)
    subtitle = pc.wrap("Both measures are the change from the model's unsteered arm, in points, on one y axis. Bars = 95% CI. Caps and floors "
                       "have no dose and are not shown.", 9.5, width - 0.3)
    notes = []
    for panel in panels:
        for group in panel["groups"]:
            missing = [dose_label(row["dose"]) for row in group["rows"] if row["diff"] is None]
            if missing:
                notes.append(f"{panel['model']['name']} {AXIS_SHORT[group['axis_key']]} {', '.join(missing)}")
    sets: dict[str, list[str]] = {}
    for panel in panels:
        data = panel["groups"][0]["data"]
        sets.setdefault(f"{data['problems']} problems × {data['seeds']} seeds", []).append(panel["model"]["name"])
    items = panels[0]["groups"][0]["data"]["items"]
    foot = pc.wrap("\n".join(
        [f"Hack rate: {'; '.join(f'{chr(32).join([' and '.join(names), size])}' for size, names in sets.items())} per arm, as judged; "
         "CI = paired bootstrap over problems; Holm p = the contrast with unsteered, corrected within the arm family of the run's analysis file.",
         f"Benchmark score = {score_name.format(items=items)}: GSM8K (1,319 items), MMLU-Pro (1,400), HumanEval (164), IFBench (300), "
         "one run per arm; CI = paired bootstrap over items. – = dose not run on that axis."]
        + ([f"No benchmark run yet (hack-rate point only): {'; '.join(notes)}."] if notes else [])), 7.8, width - 0.3)
    foot_height = 0.14 * (foot.count("\n") + 1) + 0.2
    head_height = 0.12 + 0.235 * (title.count("\n") + 1) + 0.08 + 0.175 * (subtitle.count("\n") + 1) + 0.1 + 0.215 * 2 + 0.3
    height = head_height + panel_height * grid_rows + foot_height
    figure = plt.figure(figsize=(width, height), facecolor=pc.SURFACE)
    at = lambda inches: 1 - inches / height
    x0 = 0.14 / width
    figure.text(x0, at(0.12), title, fontsize=12.5, color=pc.INK, ha="left", va="top", linespacing=1.25)
    cursor = 0.12 + 0.235 * (title.count("\n") + 1) + 0.08
    figure.text(x0, at(cursor), subtitle, fontsize=9.5, color=pc.INK2, ha="left", va="top", linespacing=1.3)
    cursor += 0.175 * (subtitle.count("\n") + 1) + 0.1
    marks = dict(markeredgecolor=pc.SURFACE, markeredgewidth=1.5)
    series = [Line2D([], [], color=palette[key], marker=m["marker"], markersize=m["size"], linewidth=m["width"], linestyle=m["dash"],
                     label=f"{pc.AXIS_NAMES[key]}: {m['name']}", **marks) for key in args.axes for m in MEASURES.values()]
    legend_style = dict(frameon=False, fontsize=9, handletextpad=0.5, columnspacing=1.6, loc="upper left", borderaxespad=0, borderpad=0, handlelength=3.0)
    figure.legend(handles=series, bbox_to_anchor=(x0, at(cursor)), ncol=len(MEASURES), **legend_style)
    cursor += 0.215 * len(args.axes) + 0.07
    extra = [Line2D([], [], color=pc.INK, marker="o", markersize=8, linewidth=0, label="unsteered", **marks),
             Line2D([], [], color=pc.SURFACE, marker="o", markersize=7.5, linewidth=0, markeredgecolor=pc.INK2, markeredgewidth=1.6,
                    label=f"hollow marker: the model stops submitting code (no-code share more than {pc.RULE_POINTS:g} points above unsteered)")]
    figure.legend(handles=extra, bbox_to_anchor=(x0, at(cursor)), ncol=2, **legend_style)
    figure.text(x0, 0.1 / height, foot, fontsize=7.8, color=pc.MUTED, ha="left", va="bottom", linespacing=1.3)

    plot_width = PANEL_WIDTH - LEFT - RIGHT
    for index, panel in enumerate(panels):
        column, grid_row = index % columns, index // columns
        left = (column * PANEL_WIDTH + LEFT) / width
        top = head_height + grid_row * panel_height
        plot = figure.add_axes((left, at(top + TITLE_HEIGHT + PLOT_HEIGHT), plot_width / width, PLOT_HEIGHT / height))
        draw_plot(plot, panel["groups"], panel["doses"])
        unsteered = panel["groups"][0]["data"]["unsteered"]
        plot.set_title(f"{panel['model']['name']}\nunsteered: judged hack rate {unsteered['hack']:.1f}%, benchmark score {unsteered['composite']:.1f}%",
                       fontsize=9.6, color=pc.INK, loc="left", pad=8, linespacing=1.35)
        table = figure.add_axes((left, at(top + TITLE_HEIGHT + PLOT_HEIGHT + XAXIS_HEIGHT + table_height), plot_width / width, table_height / height))
        steps_per_inch = len(panel["doses"]) / plot_width
        draw_table(table, panel["groups"], panel["doses"], -0.5 - (LEFT - 0.12) * steps_per_inch)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out, dpi=200, facecolor=pc.SURFACE)
    figure.savefig(args.out.with_suffix(".pdf"), facecolor=pc.SURFACE)
    record = dict(composite=args.composite, boot_draws=pc.BOOT_DRAWS, boot_seed=pc.BOOT_SEED,
                  panels=[dict(model=panel["model"]["name"], doses=panel["doses"],
                               axes=[dict(axis=group["axis_key"], color=group["color"], analysis=group["data"]["analysis"],
                                          unsteered=group["data"]["unsteered"], arms=group["rows"]) for group in panel["groups"]])
                          for panel in panels])
    args.out.with_suffix(".json").write_text(json.dumps(record, indent=1))
    for panel in panels:
        for group in panel["groups"]:
            print(f"{panel['model']['name']} / {group['axis_key']}")
            for row in group["rows"]:
                bench = "benchmark   no run" if row["diff"] is None else f"benchmark {row['diff']:+5.2f} (CI {row['diff_ci'][0]:+5.2f} to {row['diff_ci'][1]:+5.2f})"
                print(f"  {dose_label(row['dose']):>6s}  hack {row['hack_diff']:+5.1f} (CI {row['hack_diff_ci'][0]:+5.1f} to {row['hack_diff_ci'][1]:+5.1f}; "
                      f"Holm p {row['hack_p_holm']})  {bench}  no-code {row['no_code_diff']:+5.1f}{'  HOLLOW' if row['stops_submitting'] else ''}")
    print("->", args.out)


if __name__ == "__main__":
    main()
