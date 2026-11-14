"""Dose-path chart: per model, the route each intervention takes away from the unsteered point on the trade-off plane as it gets
stronger, for the gi_v2 vector and the Goodfire reward-hacking axis.

One panel per model. x = change in the judged hack rate vs unsteered in points, y = change in the benchmark composite vs
unsteered in points (pareto_chart.py's pooled score by default), so a path that stays level changes hacking at no benchmark
cost and a path that drops pays for it. The unsteered arm is the origin. Each steering axis (colour) has up to three paths:

    steering   one solid line from the most negative dose through unsteered to the most positive dose (circles)
    caps       dashed, from unsteered through the caps from the highest threshold percentile to the lowest (down triangles)
    floors     dashed, from unsteered through the floors from the lowest threshold percentile to the highest (up triangles)

The markers grow along a path, so the larger marker is the stronger intervention (steering: in proportion to |α|, so one dose
has one size on both axes; clamps: by position along the path); only the last arm of each path is labelled. The benchmark axis
has one range for all panels (benchmark points mean the same on every model; --own-y gives each panel its own), the hack-rate
axis has the panel's own.
A hollow marker = the model stops submitting code on that arm (no-code share more than 10 points above unsteered). No error
bars: the grey band around zero is the typical 95% CI of a benchmark change in the panel (the median half-width over its arms);
the CIs of every arm are in the .json and on dose_lines_chart.py. An arm with no benchmark run has no position here and is
named in the footnote. This is pareto_chart.py --per-model with the axes swapped, the per-arm error bars and labels taken off
and the dose order carried by marker size.

    python experiments/deceiver-axis/scripts/dose_paths_chart.py --out <path>.png
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import pareto_chart as pc

BAND = "#ecebe5"
PANEL_WIDTH, LEFT, RIGHT, PLOT_HEIGHT, TITLE_HEIGHT, XAXIS_HEIGHT, GAP = 7.6, 0.85, 0.3, 4.4, 0.62, 0.55, 0.3
SMALL, LARGE = 5.2, 10.5  # marker sizes: towards unsteered, and at the panel's largest steering dose / the end of its longest clamp path


def paths(rows: dict) -> list[dict]:
    """The polylines of one axis: kind, arms in drawing order and each arm's rank from unsteered (1 = next to it)."""
    order = pc.series_order(rows)
    negative, positive = order["steering"]
    result = []
    if negative or positive:
        result.append(dict(kind="steering", arms=list(reversed(negative)) + [None] + positive,
                           rank={**{arm: index + 1 for index, arm in enumerate(negative)}, **{arm: index + 1 for index, arm in enumerate(positive)}},
                           ends=[arms[-1] for arms in (negative, positive) if arms]))
    for kind in ("cap", "floor"):
        for arms in order[kind]:
            if arms:
                result.append(dict(kind=kind, arms=[None] + arms, rank={arm: index + 1 for index, arm in enumerate(arms)}, ends=[arms[-1]]))
    return result


def panel_band(groups: list[dict]) -> float:
    return float(np.median([(row["diff_ci"][1] - row["diff_ci"][0]) / 2 for group in groups for row in group["data"]["arms"].values()]))


def benchmark_range(groups: list[dict]) -> tuple[float, float]:
    band = panel_band(groups)
    ys = [row["diff"] for group in groups for row in group["data"]["arms"].values()] + [-band, band]
    pad = 0.12 * (max(ys) - min(ys)) + 0.1
    return min(ys) - pad, max(ys) + pad


def draw_panel(axes, groups: list[dict], y_range: tuple[float, float]) -> None:
    pc.style_axes(axes)
    band = panel_band(groups)
    axes.axhspan(-band, band, color=BAND, linewidth=0, zorder=0.2)
    largest_dose = max((abs(pc.dose(arm)) for group in groups for path in group["paths"] if path["kind"] == "steering" for arm in path["rank"]), default=1.0)
    longest_clamp = max((len(path["rank"]) for group in groups for path in group["paths"] if path["kind"] != "steering"), default=1)
    labels, polylines = [], []
    for group in groups:
        rows, color = group["data"]["arms"], group["color"]
        for path in group["paths"]:
            style = pc.SERIES[path["kind"]]
            points = [(0.0, 0.0) if arm is None else (rows[arm]["hack_diff"], rows[arm]["diff"]) for arm in path["arms"]]
            polylines.append(points)
            axes.plot(*zip(*points), color=color, linewidth=2.0 if path["kind"] == "steering" else 1.6, zorder=3, solid_capstyle="round",
                      solid_joinstyle="round", linestyle="-" if path["kind"] == "steering" else pc.DASHES)
            for arm in path["arms"]:
                if arm is None:
                    continue
                row = rows[arm]
                strength = abs(pc.dose(arm)) / largest_dose if path["kind"] == "steering" else path["rank"][arm] / longest_clamp
                size = SMALL + (LARGE - SMALL) * strength + (1.0 if style["marker"] != "o" else 0.0)
                face, edge = (pc.SURFACE, color) if row["stops_submitting"] else (color, pc.SURFACE)
                axes.plot(row["hack_diff"], row["diff"], style["marker"], markersize=size, color=face, markeredgecolor=edge, markeredgewidth=1.6, zorder=4)
                if arm in path["ends"]:
                    labels.append(dict(text=row["label"], x=row["hack_diff"], y=row["diff"], leader=color, color=pc.shade(color), offset=None))
    xs = [x for line in polylines for x, _ in line]
    pad_x = 0.09 * (max(xs) - min(xs)) + 0.3
    axes.set_xlim(min(xs) - pad_x, max(xs) + pad_x)
    axes.set_ylim(*y_range)
    axes.set_xlabel("Δ judged hack rate vs unsteered, points (left = fewer hacks)", fontsize=9.5, color=pc.INK2)
    axes.set_ylabel("Δ benchmark score vs unsteered, points (down = worse)", fontsize=9.5, color=pc.INK2)
    pc.finish_panel(axes, labels, polylines, None, None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", nargs="+", choices=sorted(pc.MODELS), default=["gptoss", "muse", "gemma", "qwen"])
    parser.add_argument("--axes", nargs="+", default=["gi_v2", "gfar"])
    parser.add_argument("--composite", choices=sorted(pc.COMPOSITES), default="pooled")
    parser.add_argument("--own-y", action="store_true", help="each panel gets its own benchmark range (default: one range for all panels)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    combine, score_name = pc.COMPOSITES[args.composite]
    palette = dict(zip(args.axes, pc.GROUP_COLORS))
    panels = []
    for model_key in args.model:
        model, groups = pc.MODELS[model_key], []
        for axis_key in args.axes:
            data = pc.load_axis(model, model["axes"][axis_key], combine)
            groups.append(dict(axis_key=axis_key, color=palette[axis_key], data=data, model=model, paths=paths(data["arms"])))
        panels.append(dict(model_key=model_key, model=model, groups=groups))

    columns = 1 if len(panels) == 1 else 2
    grid_rows = math.ceil(len(panels) / columns)
    panel_height = TITLE_HEIGHT + PLOT_HEIGHT + XAXIS_HEIGHT + GAP
    width = PANEL_WIDTH * columns
    title = pc.wrap("Dose paths on the trade-off plane: how far each intervention moves judged reward hacking, and what it costs in benchmark "
                    "score, as it gets stronger, for " + " and ".join(pc.AXIS_NAMES[key] for key in args.axes)
                    + (f" on {pc.COUNT_WORDS.get(len(panels), len(panels))} models" if len(panels) > 1 else f" on {panels[0]['model']['name']}"),
                    12.5, width - 0.3)
    subtitle = pc.wrap("Each model's unsteered arm is the origin. Every path starts there and the markers grow as the intervention gets stronger; "
                       "the label names the last arm of the path (steering dose α in layer norms, or clamp threshold percentile). A level path "
                       "changes hacking at no benchmark cost." + ("" if args.own_y else " The benchmark axis has the same range in every panel."),
                       9.5, width - 0.1)
    foot = pc.wrap(pc.footnote([group for panel in panels for group in panel["groups"]], score_name, False, len(panels) > 1)
                   .replace("; bars = 95% CI of the difference, paired bootstrap over problems", "")
                   .replace("; bars = 95% CI of the difference, paired bootstrap over items", ""), 7.8, width - 0.3)
    foot_height = 0.14 * (foot.count("\n") + 1) + 0.2
    head_height = 0.12 + 0.235 * (title.count("\n") + 1) + 0.08 + 0.175 * (subtitle.count("\n") + 1) + 0.1 + 0.215 * 3 + 0.14 + 0.1
    height = head_height + panel_height * grid_rows + foot_height
    figure = plt.figure(figsize=(width, height), facecolor=pc.SURFACE)
    at = lambda inches: 1 - inches / height
    x0 = 0.14 / width
    figure.text(x0, at(0.12), title, fontsize=12.5, color=pc.INK, ha="left", va="top", linespacing=1.25)
    cursor = 0.12 + 0.235 * (title.count("\n") + 1) + 0.08
    figure.text(x0, at(cursor), subtitle, fontsize=9.5, color=pc.INK2, ha="left", va="top", linespacing=1.3)
    cursor += 0.175 * (subtitle.count("\n") + 1) + 0.1
    marks = dict(markeredgecolor=pc.SURFACE, markeredgewidth=1.5)
    legend_style = dict(frameon=False, fontsize=9, handletextpad=0.5, columnspacing=1.6, loc="upper left", borderaxespad=0, borderpad=0, handlelength=3.0)
    figure.legend(handles=[Line2D([], [], color=palette[key], marker="o", markersize=7.5, linewidth=2, label=pc.AXIS_NAMES[key], **marks) for key in args.axes],
                  bbox_to_anchor=(x0, at(cursor)), ncol=len(args.axes), **legend_style)
    cursor += 0.215 + 0.07
    kinds = [Line2D([], [], color=pc.INK2, marker=pc.SERIES[kind]["marker"], markersize=7.5, linewidth=2.0 if kind == "steering" else 1.6,
                    linestyle="-" if kind == "steering" else pc.DASHES, label=pc.OVERLAY_KIND_NAMES[kind], **marks) for kind in ("steering", "cap", "floor")]
    kinds.append(Line2D([], [], color=pc.INK, marker="o", markersize=8, linewidth=0, label="unsteered (origin)", **marks))
    figure.legend(handles=kinds, bbox_to_anchor=(x0, at(cursor)), ncol=4, **legend_style)
    cursor += 0.215 + 0.07
    extra = [Line2D([], [], color=pc.SURFACE, marker="o", markersize=7.5, linewidth=0, markeredgecolor=pc.INK2, markeredgewidth=1.6,
                    label=f"hollow marker: the model stops submitting code (no-code share more than {pc.RULE_POINTS:g} points above unsteered)"),
             Patch(facecolor=BAND, edgecolor="none", label="band: typical 95% CI of a benchmark change (median half-width over the panel's arms)")]
    figure.legend(handles=extra, bbox_to_anchor=(x0, at(cursor)), ncol=2, **legend_style)
    figure.text(x0, 0.1 / height, foot, fontsize=7.8, color=pc.MUTED, ha="left", va="bottom", linespacing=1.3)

    plot_width = PANEL_WIDTH - LEFT - RIGHT
    ranges = [benchmark_range(panel["groups"]) for panel in panels]
    shared = (min(low for low, _ in ranges), max(high for _, high in ranges))
    for index, panel in enumerate(panels):
        column, grid_row = index % columns, index // columns
        top = head_height + grid_row * panel_height
        plot = figure.add_axes(((column * PANEL_WIDTH + LEFT) / width, at(top + TITLE_HEIGHT + PLOT_HEIGHT), plot_width / width, PLOT_HEIGHT / height))
        draw_panel(plot, panel["groups"], ranges[index] if args.own_y else shared)
        unsteered = panel["groups"][0]["data"]["unsteered"]
        plot.set_title(f"{panel['model']['name']}\nunsteered: judged hack rate {unsteered['hack']:.1f}%, benchmark score {unsteered['composite']:.1f}%",
                       fontsize=9.6, color=pc.INK, loc="left", pad=8, linespacing=1.35)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out, dpi=200, facecolor=pc.SURFACE)
    figure.savefig(args.out.with_suffix(".pdf"), facecolor=pc.SURFACE)
    record = dict(composite=args.composite, benchmark_range="per panel" if args.own_y else list(shared), boot_draws=pc.BOOT_DRAWS, boot_seed=pc.BOOT_SEED,
                  panels=[dict(model=panel["model"]["name"],
                               axes=[dict(axis=group["axis_key"], color=group["color"], analysis=group["data"]["analysis"],
                                          unsteered=group["data"]["unsteered"], arms=group["data"]["arms"],
                                          paths=[dict(kind=path["kind"], arms=["unsteered" if arm is None else arm for arm in path["arms"]]) for path in group["paths"]],
                                          not_drawn_no_benchmark_run=group["data"]["omitted_no_benchmark_run"]) for group in panel["groups"]])
                          for panel in panels])
    args.out.with_suffix(".json").write_text(json.dumps(record, indent=1))
    print("->", args.out)


if __name__ == "__main__":
    main()
