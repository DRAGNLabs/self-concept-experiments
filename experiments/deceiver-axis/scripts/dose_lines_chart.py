"""Dose-line chart: per model, the change in judged hack rate and in the benchmark composite for every intervention arm of the
gi_v2 vector and the Goodfire reward-hacking axis, laid out along the x axis (four lines per panel).

One panel per model. The x axis has three sections, left to right: caps in order of their threshold percentile (lowest first),
steering doses α in layer norms with unsteered (α = 0) in the middle, and floors in order of their threshold percentile (lowest
first). Read left to right, the intervention goes from holding the projection down to pushing it up. Each section holds the
union of the rungs run on the two axes, evenly spaced (not to scale); the clamp sections are tinted. y = change from the model's
unsteered arm in points, one y axis for both measures (both are percentage points): judged hack rate (solid line, circles) and
benchmark composite (dashed line, squares; pareto_chart.py's pooled score by default). Colour = steering axis, the colours of
pareto_chart.py --per-model. The steering lines pass through the unsteered point; the cap and floor lines join their own rungs
only. Error bars = 95% CIs of the difference (hack rate: paired bootstrap over problems, from the analysis file; benchmark:
item-paired bootstrap). A hollow marker = the model stops submitting code on that arm (no-code share more than 10 points above
unsteered), drawn on both of the arm's points.

An arm with no benchmark run yet keeps its hack-rate point and is named in the footnote (2026-11-13: gi_v2 ±0.16 and the three
gi_v2 floors on Gemma, the Goodfire p25 cap on Muse and on Qwen). No table under the panels (user 2026-11-13: tables of the data
are handled separately); every plotted number is in the .json.

Data, presets and the benchmark composite come from pareto_chart.py (`MODELS`, `load_axis`). Writes <out>.png, .pdf and .json.

    python experiments/deceiver-axis/scripts/dose_lines_chart.py --model gptoss muse gemma qwen --out <path>.png
    python experiments/deceiver-axis/scripts/dose_lines_chart.py --kinds steering --out <path>.png      # steering doses only
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
SECTIONS = {"cap": "cap at percentile", "steering": "steering dose α (layer norms)", "floor": "floor at percentile"}
TINT = "#f3f2ec"  # background of the clamp sections
DODGE = 0.07  # the two axes' markers sit this far (in x steps) either side of the tick, so their error bars do not overlap
SECTION_GAP = 0.5  # extra x steps between two sections
PANEL_WIDTH, LEFT, RIGHT, PLOT_HEIGHT, TITLE_HEIGHT, XAXIS_HEIGHT, GAP = 8.4, 0.8, 0.25, 4.3, 0.62, 0.72, 0.3


def minus(text: str) -> str:
    return text.replace("-", "−")


def rung(arm: str) -> float:
    """Position of an arm within its section: the dose, or the threshold percentile."""
    return pc.dose(arm) if pc.kind(arm) == "steering" else float(pc.percentile(arm))


def rung_label(kind: str, value: float) -> str:
    if kind != "steering":
        return f"p{value:g}"
    return "0" if value == 0 else minus(f"{value:+g}")


def layout(groups: list[dict], kinds: list[str]) -> tuple[dict[tuple[str, float], float], list[dict]]:
    """x position of every (kind, rung) and the sections' extents, for the union of the rungs of the panel's axes."""
    slots, sections, x = {}, [], 0.0
    for kind in kinds:
        rungs = sorted({row["rung"] for group in groups for row in group["rows"] if row["kind"] == kind} | ({0.0} if kind == "steering" else set()))
        if not rungs:
            continue
        for index, value in enumerate(rungs):
            slots[(kind, value)] = x + index
        sections.append(dict(kind=kind, rungs=rungs, start=x - 0.5, end=x + len(rungs) - 0.5))
        x += len(rungs) + SECTION_GAP
    return slots, sections


def draw_plot(axes, groups: list[dict], kinds: list[str]) -> None:
    slots, sections = layout(groups, kinds)
    axes.set_facecolor(pc.SURFACE)
    for section in sections:
        if section["kind"] != "steering":
            axes.axvspan(section["start"], section["end"], color=TINT, linewidth=0, zorder=0)
    axes.axhline(0, color=pc.ZERO, linewidth=1.1, zorder=1.5)
    axes.grid(axis="y", color=pc.GRID, linewidth=0.6, zorder=0.5)
    axes.set_axisbelow(True)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(pc.AXIS)
    origin = slots.get(("steering", 0.0))
    if origin is not None:
        axes.axvline(origin, color=pc.GRID, linewidth=0.6, zorder=1)
    for index, group in enumerate(groups):
        shift = (index - (len(groups) - 1) / 2) * 2 * DODGE
        for section in sections:
            for measure in MEASURES.values():
                points = sorted(((slots[(row["kind"], row["rung"])] + shift, row) for row in group["rows"]
                                 if row["kind"] == section["kind"] and row[measure["value"]] is not None), key=lambda point: point[0])
                if not points:
                    continue
                line = [(x, row[measure["value"]]) for x, row in points] + ([(origin, 0.0)] if section["kind"] == "steering" else [])
                axes.plot(*zip(*sorted(line)), color=group["color"], linewidth=measure["width"], linestyle=measure["dash"], zorder=3,
                          solid_capstyle="round", solid_joinstyle="round")
                for x, row in points:
                    y, (low, high) = row[measure["value"]], row[measure["ci"]]
                    axes.errorbar(x, y, yerr=[[y - low], [high - y]], fmt="none", ecolor=group["color"], elinewidth=1, alpha=0.4, zorder=2)
                    face, edge = (pc.SURFACE, group["color"]) if row["stops_submitting"] else (group["color"], pc.SURFACE)
                    axes.plot(x, y, measure["marker"], markersize=measure["size"], color=face, markeredgecolor=edge, markeredgewidth=1.6, zorder=4)
    if origin is not None:
        axes.plot(origin, 0, "o", markersize=8, color=pc.INK, markeredgecolor=pc.SURFACE, markeredgewidth=1.6, zorder=5)
    axes.set_xlim(sections[0]["start"], sections[-1]["end"])
    axes.set_xticks([slots[(section["kind"], value)] for section in sections for value in section["rungs"]],
                    [rung_label(section["kind"], value) for section in sections for value in section["rungs"]])
    axes.tick_params(colors=pc.MUTED, labelsize=8.2)
    axes.set_ylabel("Δ vs unsteered, points", fontsize=9.5, color=pc.INK2)
    inches_per_step = axes.get_position().width * axes.figure.get_figwidth() / (sections[-1]["end"] - sections[0]["start"])
    for section in sections:
        width = (section["end"] - section["start"]) * inches_per_step
        axes.text((section["start"] + section["end"]) / 2, -0.075, pc.wrap(SECTIONS[section["kind"]], 9, max(1.25 * width, 0.9)), fontsize=9,
                  color=pc.INK2, ha="center", va="top", linespacing=1.2, transform=axes.get_xaxis_transform(), clip_on=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", nargs="+", choices=sorted(pc.MODELS), default=["gptoss", "muse", "gemma", "qwen"])
    parser.add_argument("--axes", nargs="+", default=["gi_v2", "gfar"])
    parser.add_argument("--kinds", nargs="+", choices=list(SECTIONS), default=list(SECTIONS), help="sections to draw (default: all three)")
    parser.add_argument("--composite", choices=sorted(pc.COMPOSITES), default="pooled")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    combine, score_name = pc.COMPOSITES[args.composite]
    kinds = [kind for kind in SECTIONS if kind in args.kinds]
    palette = dict(zip(args.axes, pc.GROUP_COLORS))
    panels = []
    for model_key in args.model:
        model, groups = pc.MODELS[model_key], []
        for axis_key in args.axes:
            data = pc.load_axis(model, model["axes"][axis_key], combine, keep_without_benchmark=True)
            rows = [dict(arm=arm, rung=rung(arm), **row) for arm, row in data["arms"].items() if row["kind"] in kinds]
            groups.append(dict(axis_key=axis_key, color=palette[axis_key], data=data, rows=rows))
        panels.append(dict(model_key=model_key, model=model, groups=groups))

    columns = 1 if len(panels) == 1 else 2
    grid_rows = math.ceil(len(panels) / columns)
    panel_height = TITLE_HEIGHT + PLOT_HEIGHT + XAXIS_HEIGHT + GAP
    width = PANEL_WIDTH * columns
    interventions = {("steering",): "Steering dose", ("cap", "steering", "floor"): "Capping, steering and flooring"}.get(
        tuple(kinds), ", ".join(kinds).capitalize())
    title = pc.wrap(f"{interventions} against judged reward hacking and benchmark score: "
                    + " and ".join(pc.AXIS_NAMES[key] for key in args.axes) + (f" on {pc.COUNT_WORDS.get(len(panels), len(panels))} models"
                                                                                 if len(panels) > 1 else f" on {panels[0]['model']['name']}"),
                    12.5, width - 0.3)
    order = {"cap": "caps by threshold", "steering": "steering doses", "floor": "floors by threshold"}
    subtitle = pc.wrap("Both measures are the change from the model's unsteered arm, in points, on one y axis. Bars = 95% CI. "
                       + (f"Left to right: {', '.join(order[kind] for kind in kinds)}. Rungs" if len(kinds) > 1 else "Doses")
                       + " are evenly spaced, not to scale.", 9.5, width - 0.1)
    notes = []
    for panel in panels:
        for group in panel["groups"]:
            missing = [rung_label(row["kind"], row["rung"]) if row["kind"] == "steering" else f"{row['kind']} {rung_label(row['kind'], row['rung'])}"
                       for row in sorted(group["rows"], key=lambda row: (list(SECTIONS).index(row["kind"]), row["rung"])) if row["diff"] is None]
            if missing:
                notes.append(f"{panel['model']['name']} {AXIS_SHORT[group['axis_key']]} {', '.join(missing)}")
    sets: dict[str, list[str]] = {}
    for panel in panels:
        data = panel["groups"][0]["data"]
        sets.setdefault(f"{data['problems']} problems × {data['seeds']} seeds", []).append(panel["model"]["name"])
    items = panels[0]["groups"][0]["data"]["items"]
    foot = pc.wrap("\n".join(
        [f"Hack rate: {'; '.join(' and '.join(names) + ' ' + size for size, names in sets.items())} per arm, as judged; "
         "CI = paired bootstrap over problems. Clamp thresholds are percentiles of the unsteered projection.",
         f"Benchmark score = {score_name.format(items=items)}: GSM8K (1,319 items), MMLU-Pro (1,400), HumanEval (164), IFBench (300), "
         "one run per arm; CI = paired bootstrap over items."]
        + ([f"No benchmark run yet (hack-rate point only): {'; '.join(notes)}."] if notes else [])), 7.8, width - 0.3)
    foot_height = 0.14 * (foot.count("\n") + 1) + 0.2
    head_height = 0.12 + 0.235 * (title.count("\n") + 1) + 0.08 + 0.175 * (subtitle.count("\n") + 1) + 0.1 + 0.215 * len(args.axes) + 0.07 + 0.3
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
        top = head_height + grid_row * panel_height
        plot = figure.add_axes(((column * PANEL_WIDTH + LEFT) / width, at(top + TITLE_HEIGHT + PLOT_HEIGHT), plot_width / width, PLOT_HEIGHT / height))
        draw_plot(plot, panel["groups"], kinds)
        unsteered = panel["groups"][0]["data"]["unsteered"]
        plot.set_title(f"{panel['model']['name']}\nunsteered: judged hack rate {unsteered['hack']:.1f}%, benchmark score {unsteered['composite']:.1f}%",
                       fontsize=9.6, color=pc.INK, loc="left", pad=8, linespacing=1.35)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out, dpi=200, facecolor=pc.SURFACE)
    figure.savefig(args.out.with_suffix(".pdf"), facecolor=pc.SURFACE)
    record = dict(composite=args.composite, kinds=kinds, boot_draws=pc.BOOT_DRAWS, boot_seed=pc.BOOT_SEED,
                  panels=[dict(model=panel["model"]["name"],
                               axes=[dict(axis=group["axis_key"], color=group["color"], analysis=group["data"]["analysis"],
                                          unsteered=group["data"]["unsteered"], arms=group["rows"]) for group in panel["groups"]])
                          for panel in panels])
    args.out.with_suffix(".json").write_text(json.dumps(record, indent=1))
    for panel in panels:
        for group in panel["groups"]:
            print(f"{panel['model']['name']} / {group['axis_key']}")
            for row in sorted(group["rows"], key=lambda row: (list(SECTIONS).index(row["kind"]), row["rung"])):
                bench = "benchmark   no run" if row["diff"] is None else f"benchmark {row['diff']:+5.2f} (CI {row['diff_ci'][0]:+5.2f} to {row['diff_ci'][1]:+5.2f})"
                print(f"  {row['kind']:8s} {rung_label(row['kind'], row['rung']):>6s}  hack {row['hack_diff']:+5.1f} (CI {row['hack_diff_ci'][0]:+5.1f} to "
                      f"{row['hack_diff_ci'][1]:+5.1f})  {bench}  no-code {row['no_code_diff']:+5.1f}{'  HOLLOW' if row['stops_submitting'] else ''}")
    print("->", args.out)


if __name__ == "__main__":
    main()
