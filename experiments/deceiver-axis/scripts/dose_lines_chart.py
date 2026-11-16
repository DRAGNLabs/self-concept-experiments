"""Dose-line chart: per model, the change in judged hack rate and in the benchmark composite for every intervention arm of the
gi_v2 vector and the Goodfire reward-hacking axis, laid out along the x axis (four lines per panel).

One panel per model. The x axis has three sections, left to right: caps in order of their threshold percentile (lowest first),
steering doses α in layer norms with unsteered (α = 0) in the middle, and floors in order of their threshold percentile (lowest
first). Read left to right, the intervention goes from holding the projection down to pushing it up. Each section holds the
union of the rungs run on the two axes, evenly spaced (not to scale); the clamp sections are tinted. y = change from the model's
unsteered arm in points, one y axis for both measures (both are percentage points): judged hack rate (solid line, circles) and
benchmark composite (dashed line, squares; pareto_chart.py's pooled score by default). Colour = steering axis, the colours of
pareto_chart.py --per-model. The steering lines pass through the unsteered point; the cap and floor lines join their own rungs
only. --split (user 2026-11-13) puts steering and clamping side by side instead, each around its own unsteered point: the doses
with α = 0 in the middle, then the caps in order of threshold percentile, a second unsteered point, and the floors in order of
threshold percentile; the cap and floor lines pass through that second point. Error bars = 95% CIs of the difference (hack rate: paired bootstrap over problems, from the analysis file; benchmark:
item-paired bootstrap). A hollow marker = the model stops submitting code on that arm (no-code share more than 10 points above
unsteered), drawn on both of the arm's points; the hack-rate line into and out of such an arm is short-dashed (user 2026-11-16:
a hack rate there is not to be trusted like the others). Two axes' markers at one rung are dodged either side of its tick; a
rung only one axis has sits on its tick.

--rule-ring (draft, 2026-11-16) draws a ring around the hack-rate marker of every arm outside the task-set dose rule (hidden-test
pass over all episodes or no-code more than 10 points from unsteered; `inside_dose_rule` of pareto_chart.load_axis), so the end of
a ladder is visible. It is a second mark, not the hollow one: an arm outside on hidden pass alone still submits code and its hack
rate stays comparable (hidden pass falls when hacking rises), so its marker stays filled and its line solid.

--clamp-scale (with --split; user 2026-11-16, concern that the cap and floor rungs were not to scale) places caps and floors by
threshold percentile instead of evenly: a cap at p sits (100 − p) points left of the clamp part's unsteered point and a floor at
p sits p points right of it, so the distance from that point is the share of unsteered positions beyond the threshold (a cap at
p100 or a floor at p0 would move nothing). The labelled ticks are the same reference percentiles on every panel; small unlabelled
ticks mark the rungs that were run. The nominal share tracks the measured open-loop share of GENERATED positions moved
(clamp_shares.json, 49 arms: mean gap 1.9 points, largest 11.1) but not of PROMPT positions (mean gap 17.6, largest 45.3).

--bench-strip (user 2026-11-16) takes the benchmark score out of the plot and into a strip underneath it, as in
outcome_bars_chart.py: the plot keeps the two hack-rate lines, the strip holds the two axes' benchmark points (squares in the axis
colour, with CIs, no line) on its own y axis, and the x labels sit under the strip.
--hidden-strip (with --bench-strip; draft, user 2026-11-16) adds a second strip under it: the change in hidden-test pass over all
episodes (diamonds; CI where the analysis file stores one), with dashed lines at the dose-rule limit of ±10 points. Hidden pass
falls when hacking rises (a hack that fails the hidden tests is a failure), so the strip shows the rule's measure, not capability.

An arm with no benchmark run yet keeps its hack-rate point and is named in the footnote. Rungs past the least extreme failing rung
of their ladder are left out by the presets (`omit` in pareto_chart.py: the Goodfire p25 caps on Muse and Qwen, user 2026-11-16).
No table under the panels (user 2026-11-13: tables of the data
are handled separately); every plotted number is in the .json.

Data, presets and the benchmark composite come from pareto_chart.py (`MODELS`, `load_axis`). Writes <out>.png, .pdf and .json.

    python experiments/deceiver-axis/scripts/dose_lines_chart.py --model gptoss muse gemma qwen --out <path>.png
    python experiments/deceiver-axis/scripts/dose_lines_chart.py --kinds steering --out <path>.png      # steering doses only
    python experiments/deceiver-axis/scripts/dose_lines_chart.py --split --out <path>.png               # steering | caps, unsteered, floors
    python experiments/deceiver-axis/scripts/dose_lines_chart.py --split --bench-strip --out <path>.png # benchmark score in a strip underneath
    python experiments/deceiver-axis/scripts/dose_lines_chart.py --split --bench-strip --clamp-scale --out <path>.png  # caps / floors to scale
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
from matplotlib.ticker import MaxNLocator

import pareto_chart as pc

AXIS_SHORT = {"gi_v2": "gi_v2", "gfar": "Goodfire"}
MEASURES = {"hack": dict(value="hack_diff", ci="hack_diff_ci", name="judged hack rate", marker="o", size=7.5, width=2.0, dash="-"),
            "bench": dict(value="diff", ci="diff_ci", name="benchmark score", marker="s", size=6.2, width=1.6, dash=pc.DASHES)}
# --hidden-strip: hidden-test pass over all episodes, the measure of the task-set dose rule; strip only, never a line in the plot
STRIP_ONLY = {"hidden": dict(value="hidden_pass_diff", ci="hidden_pass_diff_ci", name="hidden-test pass", marker="D", size=5.2, strip_size=4.9, width=0, dash="-")}
DRAWN = {**MEASURES, **STRIP_ONLY}
STRIP_LABELS = {"bench": "Δ benchmark,\npoints", "hidden": "Δ hidden pass,\npoints"}
SECTIONS = {"cap": "cap at percentile", "steering": "steering dose α (layer norms)", "floor": "floor at percentile"}
TINT = "#f3f2ec"  # background of the clamp sections
DODGE = 0.07  # where two axes have a point at one rung, their markers sit this far (in x steps) either side of the tick
SECTION_GAP = 0.5  # extra x steps between two sections
SPLIT_GAP = 1.0  # --split: extra x steps between the steering part and the clamp part
ORIGIN_PAD = 0.4  # --split: extra x steps either side of the clamp part's unsteered point (room for its tick label)
PANEL_WIDTH, LEFT, RIGHT, PLOT_HEIGHT, TITLE_HEIGHT, XAXIS_HEIGHT, GAP = 8.4, 0.8, 0.25, 4.3, 0.62, 0.72, 0.3
STRIP_GAP, STRIP_HEIGHT, STRIP_MARKER = 0.14, 1.0, 5.8  # --bench-strip: the benchmark strip under each plot (inches, marker size)
RULE_RING = dict(extra=5.5, width=1.0)  # --rule-ring: ring around the hack-rate marker of an arm outside the dose rule (points)
DISTRUST_DASH = (0, (1.1, 1.9))  # hack-rate line into or out of an arm where the model stops submitting code (shorter dashes than pc.DASHES)
# --clamp-scale: x steps per 100 percentile points, padding at the two ends, the reference ticks either side of the unsteered
# point, and the extra panel width (inches) that keeps rungs 5 percentile points apart from touching
CLAMP_STEPS, CLAMP_PAD, CLAMP_EXTRA_WIDTH = 6.0, 0.3, 1.0
CLAMP_TICKS = {"cap": (5, 25, 50, 75), "floor": (25, 50, 75, 95)}


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


def plan(groups: list[dict], kinds: list[str], split: bool = False,
         clamp_scale: bool = False) -> tuple[dict[tuple[str, float], float], list[dict], list[float]]:
    """What draw_plot draws: the x position of every (kind, rung), the parts of the x axis and the unsteered positions. A part has
    its extent, tick labels, captions (centre, text, width in steps) and chains; a chain is the kinds of arm one line joins and the
    unsteered position that line passes through (None: it joins its own rungs only). Default: layout()'s three sections, with one
    unsteered point in the steering section. `split`: a steering part, then a clamp part with its own unsteered point between the
    caps and the floors (caps in order of threshold percentile, unsteered, floors in order of threshold percentile).
    `clamp_scale` (with `split`): the clamp part is to scale in the threshold percentile. A cap at p sits (100 − p) points left of
    the unsteered point and a floor at p sits p points right of it, so the distance from the unsteered point is the share of
    unsteered positions beyond the threshold (a cap at p100 or a floor at p0 moves nothing). Labelled ticks are the same reference
    percentiles on every panel (CLAMP_TICKS); a part's `minor` ticks mark the rungs that were run and have no label."""
    if not split:
        slots, sections = layout(groups, kinds)
        origin = slots.get(("steering", 0.0))
        parts = [dict(start=section["start"], end=section["end"], tinted=section["kind"] != "steering",
                      ticks=[(slots[(section["kind"], value)], rung_label(section["kind"], value)) for value in section["rungs"]],
                      captions=[((section["start"] + section["end"]) / 2, SECTIONS[section["kind"]], section["end"] - section["start"])],
                      chains=[dict(kinds=(section["kind"],), origin=origin if section["kind"] == "steering" else None)]) for section in sections]
        return slots, parts, [] if origin is None else [origin]
    rungs = lambda kind: sorted({row["rung"] for group in groups for row in group["rows"] if row["kind"] == kind}) if kind in kinds else []
    slots, parts, origins, x = {}, [], [], 0.0
    if "steering" in kinds:
        doses = sorted(set(rungs("steering")) | {0.0})
        slots.update({("steering", value): x + index for index, value in enumerate(doses)})
        origins.append(slots[("steering", 0.0)])
        parts.append(dict(start=x - 0.5, end=x + len(doses) - 0.5, tinted=False, ticks=[(slots[("steering", value)], rung_label("steering", value)) for value in doses],
                          captions=[(x + (len(doses) - 1) / 2, SECTIONS["steering"], len(doses))], chains=[dict(kinds=("steering",), origin=origins[-1])]))
        x += len(doses) + SPLIT_GAP
    caps, floors = rungs("cap"), rungs("floor")
    if (caps or floors) and clamp_scale:
        start = x - 0.5
        origin = start + (CLAMP_PAD + CLAMP_STEPS if caps else 0.5)
        place = {"cap": lambda value: origin - CLAMP_STEPS * (100 - value) / 100, "floor": lambda value: origin + CLAMP_STEPS * value / 100}
        run = [(kind, value) for kind, values in (("cap", caps), ("floor", floors)) for value in values]
        slots.update({key: place[key[0]](key[1]) for key in run})
        origins.append(origin)
        labelled = [(kind, value) for kind, values in (("cap", caps), ("floor", floors)) if values for value in CLAMP_TICKS[kind]]
        parts.append(dict(start=start, end=origin + (CLAMP_STEPS + CLAMP_PAD if floors else 0.5), tinted=True,
                          ticks=sorted([(place[kind](value), f"p{value:g}") for kind, value in labelled] + [(origin, "unsteered")]),
                          minor=[slots[key] for key in run if key not in labelled],
                          captions=([(origin - CLAMP_STEPS / 2, SECTIONS["cap"], CLAMP_STEPS)] if caps else [])
                          + ([(origin + CLAMP_STEPS / 2, SECTIONS["floor"], CLAMP_STEPS)] if floors else []),
                          chains=[dict(kinds=("cap", "floor"), origin=origin)]))
    elif caps or floors:
        origin = x + len(caps) + (ORIGIN_PAD if caps else 0.0)
        first_floor = origin + 1 + ORIGIN_PAD
        slots.update({("cap", value): x + index for index, value in enumerate(caps)})
        slots.update({("floor", value): first_floor + index for index, value in enumerate(floors)})
        origins.append(origin)
        captions = ([(x + (len(caps) - 1) / 2, SECTIONS["cap"], len(caps))] if caps else []) + (
            [(first_floor + (len(floors) - 1) / 2, SECTIONS["floor"], len(floors))] if floors else [])
        parts.append(dict(start=x - 0.5, end=(first_floor + len(floors) - 0.5) if floors else origin + 0.5, tinted=True,
                          ticks=[(slots[("cap", value)], rung_label("cap", value)) for value in caps] + [(origin, "unsteered")]
                          + [(slots[("floor", value)], rung_label("floor", value)) for value in floors],
                          captions=captions, chains=[dict(kinds=("cap", "floor"), origin=origin)]))
    return slots, parts, origins


def draw_plot(axes, groups: list[dict], kinds: list[str], split: bool = False, measures: tuple[str, ...] = tuple(MEASURES),
              strip: bool = False, x_labels: bool = True, clamp_scale: bool = False, rule_ring: bool = False) -> None:
    """`measures`: which of DRAWN to draw. `strip`: the axes is a strip for one measure (points and CIs only, no lines, smaller
    markers, its own y label; the hidden-pass strip also carries the dose-rule lines at ±pc.RULE_POINTS). `x_labels` False: no tick labels or section captions (a plot with a strip underneath).
    A hack-rate line segment that starts or ends at an arm where the model stops submitting code is drawn in DISTRUST_DASH."""
    slots, parts, origins = plan(groups, kinds, split, clamp_scale)
    axes.set_facecolor(pc.SURFACE)
    for part in parts:
        if part["tinted"]:
            axes.axvspan(part["start"], part["end"], color=TINT, linewidth=0, zorder=0)
    axes.axhline(0, color=pc.ZERO, linewidth=1.1, zorder=1.5)
    axes.grid(axis="y", color=pc.GRID, linewidth=0.6, zorder=0.5)
    axes.set_axisbelow(True)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(pc.AXIS)
    for origin in origins:
        axes.axvline(origin, color=pc.GRID, linewidth=0.6, zorder=1)
    for index, group in enumerate(groups):
        shift = (index - (len(groups) - 1) / 2) * 2 * DODGE
        for part in parts:
            for chain in part["chains"]:
                for measure in (DRAWN[key] for key in measures):
                    shared = {(row["kind"], row["rung"]) for other in groups if other is not group for row in other["rows"]
                              if row[measure["value"]] is not None}  # rungs where another axis has a point too: only these are dodged
                    points = sorted(((slots[(row["kind"], row["rung"])] + (shift if (row["kind"], row["rung"]) in shared else 0.0), row)
                                     for row in group["rows"] if row["kind"] in chain["kinds"] and row[measure["value"]] is not None),
                                    key=lambda point: point[0])
                    if not points:
                        continue
                    line = sorted([(x, row[measure["value"]], row["stops_submitting"]) for x, row in points]
                                  + ([(chain["origin"], 0.0, False)] if chain["origin"] is not None else []))
                    for (x_a, y_a, hollow_a), (x_b, y_b, hollow_b) in zip(line, line[1:]) if not strip else []:
                        distrusted = measure is MEASURES["hack"] and (hollow_a or hollow_b)
                        axes.plot([x_a, x_b], [y_a, y_b], color=group["color"], linewidth=measure["width"],
                                  linestyle=DISTRUST_DASH if distrusted else measure["dash"], zorder=3, solid_capstyle="round", dash_capstyle="round")
                    for x, row in points:
                        y, interval = row[measure["value"]], row[measure["ci"]]
                        if interval is not None:  # hidden pass read from a run summary has no stored CI
                            axes.errorbar(x, y, yerr=[[y - interval[0]], [interval[1] - y]], fmt="none", ecolor=group["color"], elinewidth=1,
                                          alpha=0.55 if strip else 0.4, zorder=2)
                        face, edge = (pc.SURFACE, group["color"]) if row["stops_submitting"] else (group["color"], pc.SURFACE)
                        if rule_ring and not strip and measure is MEASURES["hack"] and row["inside_dose_rule"] is False:
                            axes.plot(x, y, "o", markersize=measure["size"] + RULE_RING["extra"], markerfacecolor="none",
                                      markeredgecolor=group["color"], markeredgewidth=RULE_RING["width"], zorder=3.5)
                        axes.plot(x, y, measure["marker"], markersize=measure.get("strip_size", STRIP_MARKER) if strip else measure["size"],
                                  color=face, markeredgecolor=edge,
                                  markeredgewidth=1.3 if strip else 1.6, zorder=4)
    for origin in origins:
        axes.plot(origin, 0, DRAWN[measures[0]]["marker"] if strip else "o",
                  markersize=DRAWN[measures[0]].get("strip_size", STRIP_MARKER) if strip else 8, color=pc.INK, markeredgecolor=pc.SURFACE,
                  markeredgewidth=1.3 if strip else 1.6, zorder=5)
    axes.set_xlim(parts[0]["start"], parts[-1]["end"])
    axes.set_xticks([x for part in parts for x, _ in part["ticks"]], [text for part in parts for _, text in part["ticks"]])
    axes.set_xticks([x for part in parts for x in part.get("minor", [])], minor=True)
    axes.tick_params(colors=pc.MUTED, labelsize=8.2)
    axes.tick_params(axis="x", which="minor", length=2.6, color=pc.MUTED)
    if strip:
        axes.margins(y=0.18)
        axes.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
        axes.set_ylabel(STRIP_LABELS[measures[0]], fontsize=8.6, color=pc.INK2, linespacing=1.15)
        if measures[0] == "hidden":  # the dose-rule lines; the lower one is always inside the y range
            low, high = axes.get_ylim()
            axes.set_ylim(min(low, -pc.RULE_POINTS - 2.5), max(high, 2.5))
            for limit in (-pc.RULE_POINTS, pc.RULE_POINTS):
                axes.axhline(limit, color=pc.INK2, linewidth=0.9, linestyle=(0, (4, 3)), zorder=1.6)
    else:
        axes.set_ylabel("Δ vs unsteered, points" if len(measures) > 1 else f"Δ {MEASURES[measures[0]]['name']}, points", fontsize=9.5, color=pc.INK2)
    if not x_labels:
        axes.tick_params(axis="x", which="both", labelbottom=False, length=0)
        return
    inches_per_step = axes.get_position().width * axes.figure.get_figwidth() / (parts[-1]["end"] - parts[0]["start"])
    for part in parts:
        for centre, text, steps in part["captions"]:
            axes.text(centre, -0.075 * PLOT_HEIGHT / STRIP_HEIGHT if strip else -0.075, pc.wrap(text, 9, max(1.25 * steps * inches_per_step, 1.35)), fontsize=9,
                      color=pc.INK2, ha="center", va="top", linespacing=1.2, transform=axes.get_xaxis_transform(), clip_on=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", nargs="+", choices=sorted(pc.MODELS), default=["gptoss", "muse", "gemma", "qwen"])
    parser.add_argument("--axes", nargs="+", default=["gi_v2", "gfar"])
    parser.add_argument("--kinds", nargs="+", choices=list(SECTIONS), default=list(SECTIONS), help="sections to draw (default: all three)")
    parser.add_argument("--split", action="store_true",
                        help="steering and clamping side by side, each around its own unsteered point: doses, then caps / unsteered / floors")
    parser.add_argument("--bench-strip", action="store_true",
                        help="benchmark score in a strip under each plot (own y axis) instead of on the plot's y axis")
    parser.add_argument("--clamp-scale", action="store_true",
                        help="with --split: caps and floors placed to scale by threshold percentile (distance from the unsteered point = "
                             "share of unsteered positions beyond the threshold) instead of evenly spaced")
    parser.add_argument("--hidden-strip", action="store_true",
                        help="with --bench-strip: a second strip under the benchmark strip with the change in hidden-test pass over all "
                             "episodes and the dose-rule line at -10 points")
    parser.add_argument("--rule-ring", action="store_true",
                        help="ring the hack-rate marker of every arm outside the task-set dose rule (hidden pass or no-code more than "
                             "10 points from unsteered)")
    parser.add_argument("--composite", choices=sorted(pc.COMPOSITES), default="pooled")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.clamp_scale and not args.split:
        parser.error("--clamp-scale needs --split (the clamp part's own unsteered point is the zero of its scale)")
    if args.hidden_strip and not args.bench_strip:
        parser.error("--hidden-strip needs --bench-strip (it sits under the benchmark strip)")
    legend_rows = len(MEASURES) + (len(STRIP_ONLY) if args.hidden_strip else 0)
    strips = (["bench"] if args.bench_strip else []) + (["hidden"] if args.hidden_strip else [])
    panel_width = PANEL_WIDTH + (CLAMP_EXTRA_WIDTH if args.clamp_scale else 0.0)
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
    panel_height = TITLE_HEIGHT + PLOT_HEIGHT + len(strips) * (STRIP_GAP + STRIP_HEIGHT) + XAXIS_HEIGHT + GAP
    width = panel_width * columns
    interventions = {("steering",): "Steering dose", ("cap", "steering", "floor"): "Capping, steering and flooring"}.get(
        tuple(kinds), ", ".join(kinds).capitalize())
    title = pc.wrap(f"{interventions} against judged reward hacking and benchmark score: "
                    + " and ".join(pc.AXIS_NAMES[key] for key in args.axes) + (f" on {pc.COUNT_WORDS.get(len(panels), len(panels))} models"
                                                                                 if len(panels) > 1 else f" on {panels[0]['model']['name']}"),
                    12.5, width - 0.3)
    order = {"cap": "caps by threshold", "steering": "steering doses", "floor": "floors by threshold"}
    subtitle = pc.wrap(("Change from the model's unsteered arm, in points: judged hack rate in the plot, benchmark score "
                        + ("and hidden-test pass in the two strips underneath (each its own y axis). " if args.hidden_strip else
                           "in the strip underneath (its own y axis). ") + "Bars = 95% CI. " if args.bench_strip else
                        "Both measures are the change from the model's unsteered arm, in points, on one y axis. Bars = 95% CI. ")
                       + ("Left: steering doses. Right (tinted): caps and floors by threshold, either side of their own unsteered point. Rungs"
                          if args.split and "steering" in kinds and len(kinds) > 1
                          else f"Left to right: {', '.join(order[kind] for kind in kinds)}. Rungs" if len(kinds) > 1 else "Doses")
                       + (" are evenly spaced, not to scale." if not args.clamp_scale else
                          " of the steering dose are evenly spaced, not to scale. Caps and floors are to scale in the threshold percentile: "
                          "the distance from the unsteered point is the share of unsteered positions beyond the threshold (cap at p: 100 − p; "
                          "floor at p: p). Small ticks mark the rungs that were run."), 9.5, width - 0.1)
    notes = []
    no_ci = [f"{panel['model']['name']} {AXIS_SHORT[group['axis_key']]}" for panel in panels for group in panel["groups"]
             if any(row["hidden_pass_diff_ci"] is None for row in group["rows"])]
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
        + ([f"Hidden-test pass = share of all episodes whose final code passes the hidden tests; dashed line = the dose-rule limit, "
            f"{pc.RULE_POINTS:g} points from unsteered. A hack that fails the hidden tests counts as a failure, so this measure falls when hacking "
            "rises and is not read alone as capability." + (f" No stored CI: {', '.join(no_ci)}." if no_ci else "")] if args.hidden_strip else [])
        + ([f"No benchmark run yet (hack-rate point only): {'; '.join(notes)}."] if notes else [])), 7.8, width - 0.3)
    foot_height = 0.14 * (foot.count("\n") + 1) + 0.2
    head_height = 0.12 + 0.235 * (title.count("\n") + 1) + 0.08 + 0.175 * (subtitle.count("\n") + 1) + 0.1 + 0.215 * legend_rows + 0.07 + 0.3 + (0.245 if args.rule_ring or args.hidden_strip else 0.0)
    height = head_height + panel_height * grid_rows + foot_height
    figure = plt.figure(figsize=(width, height), facecolor=pc.SURFACE)
    at = lambda inches: 1 - inches / height
    x0 = 0.14 / width
    figure.text(x0, at(0.12), title, fontsize=12.5, color=pc.INK, ha="left", va="top", linespacing=1.25)
    cursor = 0.12 + 0.235 * (title.count("\n") + 1) + 0.08
    figure.text(x0, at(cursor), subtitle, fontsize=9.5, color=pc.INK2, ha="left", va="top", linespacing=1.3)
    cursor += 0.175 * (subtitle.count("\n") + 1) + 0.1
    marks = dict(markeredgecolor=pc.SURFACE, markeredgewidth=1.5)
    listed = {**MEASURES, **(STRIP_ONLY if args.hidden_strip else {})}
    series = [Line2D([], [], color=palette[key], marker=m["marker"], markersize=m.get("strip_size", STRIP_MARKER) if name in strips else m["size"],
                     linewidth=0 if name in strips else m["width"], linestyle=m["dash"],
                     label=f"{pc.AXIS_NAMES[key]}: {m['name']}" + (" (strip)" if name in strips else ""), **marks)
              for key in args.axes for name, m in listed.items()]
    legend_style = dict(frameon=False, fontsize=9, handletextpad=0.5, columnspacing=1.6, loc="upper left", borderaxespad=0, borderpad=0, handlelength=3.0)
    figure.legend(handles=series, bbox_to_anchor=(x0, at(cursor)), ncol=len(args.axes), **legend_style)  # one column per axis
    cursor += 0.215 * legend_rows + 0.07
    extra = [Line2D([], [], color=pc.INK, marker="o", markersize=8, linewidth=0, label="unsteered", **marks),
             Line2D([], [], color=pc.INK2, marker="o", markersize=7.5, linewidth=MEASURES["hack"]["width"], linestyle=DISTRUST_DASH,
                    dash_capstyle="round", markerfacecolor=pc.SURFACE, markeredgecolor=pc.INK2, markeredgewidth=1.6,
                    label=f"hollow marker, dashed hack-rate line: the model stops submitting code (no-code share more than {pc.RULE_POINTS:g} "
                          "points above unsteered), so the hack rate there is not comparable")]
    second = []
    if args.rule_ring:
        second.append(Line2D([], [], color=pc.INK2, marker="o", markersize=7.5 + RULE_RING["extra"], linewidth=0, markerfacecolor="none",
                             markeredgecolor=pc.INK2, markeredgewidth=RULE_RING["width"],
                             label=f"ring: the arm is outside the dose rule (hidden-test pass or no-code more than {pc.RULE_POINTS:g} points "
                                   "from unsteered)"))
    if args.hidden_strip:
        second.append(Line2D([], [], color=pc.INK2, linewidth=0.9, linestyle=(0, (4, 3)),
                             label=f"dose-rule limit on hidden-test pass ({pc.RULE_POINTS:g} points from unsteered)"))
    if second:
        figure.legend(handles=extra[:1] + second, bbox_to_anchor=(x0, at(cursor)), ncol=1 + len(second), **legend_style)
        cursor += 0.215 + 0.03
        extra = extra[1:]
    figure.legend(handles=extra, bbox_to_anchor=(x0, at(cursor)), ncol=2, **legend_style)
    figure.text(x0, 0.1 / height, foot, fontsize=7.8, color=pc.MUTED, ha="left", va="bottom", linespacing=1.3)

    plot_width = panel_width - LEFT - RIGHT
    for index, panel in enumerate(panels):
        column, grid_row = index % columns, index // columns
        top = head_height + grid_row * panel_height
        plot = figure.add_axes(((column * panel_width + LEFT) / width, at(top + TITLE_HEIGHT + PLOT_HEIGHT), plot_width / width, PLOT_HEIGHT / height))
        if args.bench_strip:
            draw_plot(plot, panel["groups"], kinds, args.split, measures=("hack",), x_labels=False, clamp_scale=args.clamp_scale, rule_ring=args.rule_ring)
            for position, name in enumerate(strips, start=1):
                strip = figure.add_axes(((column * panel_width + LEFT) / width, at(top + TITLE_HEIGHT + PLOT_HEIGHT + position * (STRIP_GAP + STRIP_HEIGHT)),
                                         plot_width / width, STRIP_HEIGHT / height))
                draw_plot(strip, panel["groups"], kinds, args.split, measures=(name,), strip=True, x_labels=name == strips[-1],
                          clamp_scale=args.clamp_scale)
        else:
            draw_plot(plot, panel["groups"], kinds, args.split, clamp_scale=args.clamp_scale, rule_ring=args.rule_ring)
        unsteered = panel["groups"][0]["data"]["unsteered"]
        plot.set_title(f"{panel['model']['name']}\nunsteered: judged hack rate {unsteered['hack']:.1f}%, benchmark score {unsteered['composite']:.1f}%"
                       + (f", hidden-test pass {unsteered['hidden_pass']:.1f}%" if args.hidden_strip else ""),
                       fontsize=9.6, color=pc.INK, loc="left", pad=8, linespacing=1.35)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out, dpi=200, facecolor=pc.SURFACE)
    figure.savefig(args.out.with_suffix(".pdf"), facecolor=pc.SURFACE)
    record = dict(composite=args.composite, kinds=kinds, split=args.split, bench_strip=args.bench_strip, clamp_scale=args.clamp_scale, rule_ring=args.rule_ring, hidden_strip=args.hidden_strip,
                  omitted_by_preset={f"{panel['model']['name']} / {group['axis_key']}": group["data"]["omitted_by_preset"]
                                     for panel in panels for group in panel["groups"] if group["data"]["omitted_by_preset"]}, boot_draws=pc.BOOT_DRAWS, boot_seed=pc.BOOT_SEED,
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
