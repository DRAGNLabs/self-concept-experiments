"""Effect heatmap: every intervention arm of the gi_v2 vector and the Goodfire reward-hacking axis on every model as a grid of
cells, coloured by the relative change in the judged hack rate and by the benchmark points lost.

One block per model, stacked. Columns are dose_lines_chart.py's x axis: caps in order of their threshold percentile, steering
doses α with unsteered (α = 0) in the middle, floors in order of their threshold percentile (the union of the rungs of the two
axes; a rung an axis did not run is left empty). Each steering axis has two rows of cells:

    hack rate    colour and number = change in the judged hack rate as % of the model's unsteered rate (−100 = no judged hack
                 left), so the models share one scale (user 2026-11-13). Diverging: blue = fewer hacks, red = more, pale = no
                 change; the scale runs from −100% to +200% and is clipped above (the number is not).
    benchmark    colour and number = change in the benchmark composite vs unsteered in points (pareto_chart.py's pooled score
                 by default). One grey ramp from no loss (pale) to 5 points lost (dark, clipped); a gain is pale.

A model whose unsteered hack rate is zero has no relative change (Qwen): its hack cells carry the change in points ("pt") and no
colour. The unsteered column holds the unsteered rate and score. A bold number = the 95% CI excludes zero (hack rate: paired
bootstrap over problems of the relative change, of the difference for Qwen; benchmark: item-paired bootstrap). A ring in a hack
cell = the model stops submitting code on that arm (no-code share more than 10 points above unsteered). "–" in a benchmark
cell = no benchmark run yet. Every number is in the .json.

    python experiments/deceiver-axis/scripts/effect_heatmap_chart.py --out <path>.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm, to_rgb
from matplotlib.patches import Rectangle

import dose_lines_chart as dl
import pareto_chart as pc

# The palette's diverging pair (blue <-> red) around its neutral midpoint, and a neutral ramp for the benchmark loss.
NEUTRAL = "#f0efec"
HACK_MAP = LinearSegmentedColormap.from_list("hack", ["#2a78d6", NEUTRAL, "#e34948"])
HACK_NORM = TwoSlopeNorm(vmin=-100, vcenter=0, vmax=200)
LOSS_MAP = LinearSegmentedColormap.from_list("loss", [NEUTRAL, "#3d3c3a"])
LOSS_NORM = Normalize(vmin=0, vmax=5)
AXIS_SHORT = {"gi_v2": "gi_v2 vector", "gfar": "Goodfire axis"}
CELL_WIDTH, HACK_HEIGHT, BENCH_HEIGHT, AXIS_GAP = 0.62, 0.44, 0.31, 0.13
LEFT, RIGHT, TITLE_HEIGHT, XAXIS_HEIGHT, GAP = 2.05, 0.3, 0.34, 0.62, 0.22
NUMBER_SIZE = 8.4


def minus(text: str) -> str:
    return text.replace("-", "−")


def signed(value: float, digits: int, unit: str = "") -> str:
    """Signed number with a true minus sign; a value that rounds to zero carries no sign."""
    text = f"{value:+.{digits}f}"
    return minus(text[1:] if float(text) == 0 else text) + unit


def ink_on(fill) -> str:
    red, green, blue = to_rgb(fill)
    return pc.INK if 0.2126 * red + 0.7152 * green + 0.0722 * blue > 0.42 else "#ffffff"


def cell(axes, x: float, y: float, height: float, fill, text: str, bold: bool = False, outline: str | None = None, color: str | None = None) -> None:
    axes.add_patch(Rectangle((x - 0.5, y), 1, height, facecolor=fill, edgecolor=pc.SURFACE, linewidth=2.2, zorder=2))
    if outline:
        inset_x, inset_y = 0.045, 0.045 * CELL_WIDTH
        axes.add_patch(Rectangle((x - 0.5 + inset_x, y + inset_y), 1 - 2 * inset_x, height - 2 * inset_y, facecolor="none", edgecolor=outline,
                                 linewidth=0.8, zorder=3))
    axes.text(x, y + height / 2, text, fontsize=NUMBER_SIZE, color=color or ink_on(fill), ha="center", va="center", zorder=4,
              fontweight="bold" if bold else "normal")


def excludes_zero(interval) -> bool:
    return interval is not None and (interval[0] > 0 or interval[1] < 0)


def draw_block(axes, panel: dict, slots: dict, sections: list[dict]) -> None:
    row_height = HACK_HEIGHT + BENCH_HEIGHT + AXIS_GAP
    total = row_height * len(panel["groups"]) - AXIS_GAP
    axes.set_xlim(sections[0]["start"], sections[-1]["end"])
    axes.set_ylim(total, 0)
    axes.set_facecolor(pc.SURFACE)
    for side in axes.spines.values():
        side.set_visible(False)
    axes.set_yticks([])
    origin = slots.get(("steering", 0.0))
    label_at = axes.get_yaxis_transform()
    for index, group in enumerate(panel["groups"]):
        top = index * row_height
        relative = group["data"]["unsteered"]["hack"] > 0
        inches = CELL_WIDTH * (sections[-1]["end"] - sections[0]["start"])
        axes.text(-1.02 / inches, top + (HACK_HEIGHT + BENCH_HEIGHT) / 2, AXIS_SHORT[group["axis_key"]], fontsize=9.6, color=pc.INK, ha="right",
                  va="center", transform=label_at, clip_on=False)
        axes.text(-0.1 / inches, top + HACK_HEIGHT / 2, "hack rate", fontsize=8.6, color=pc.INK2, ha="right", va="center", transform=label_at, clip_on=False)
        axes.text(-0.1 / inches, top + HACK_HEIGHT + BENCH_HEIGHT / 2, "benchmark", fontsize=8.6, color=pc.INK2, ha="right", va="center",
                  transform=label_at, clip_on=False)
        if origin is not None:
            unsteered = group["data"]["unsteered"]
            cell(axes, origin, top, HACK_HEIGHT, pc.SURFACE, f"{unsteered['hack']:.1f}%", outline=pc.INK2)
            cell(axes, origin, top + HACK_HEIGHT, BENCH_HEIGHT, pc.SURFACE, f"{unsteered['composite']:.1f}%", outline=pc.INK2)
        for row in group["rows"]:
            x = slots[(row["kind"], row["rung"])]
            if relative:
                cell(axes, x, top, HACK_HEIGHT, HACK_MAP(HACK_NORM(min(row["hack_rel"], HACK_NORM.vmax))), signed(row["hack_rel"], 0, "%"),
                     bold=excludes_zero(row["hack_rel_ci"]))
            else:
                cell(axes, x, top, HACK_HEIGHT, pc.SURFACE, signed(row["hack_diff"], 1, " pt"), bold=excludes_zero(row["hack_diff_ci"]), outline=pc.AXIS)
            if row["stops_submitting"]:
                fill = HACK_MAP(HACK_NORM(min(row["hack_rel"], HACK_NORM.vmax))) if relative else pc.SURFACE
                axes.plot(x + 0.36, top + 0.085, "o", markersize=4.2, markerfacecolor="none", markeredgecolor=ink_on(fill), markeredgewidth=1.0, zorder=5)
            if row["diff"] is None:
                cell(axes, x, top + HACK_HEIGHT, BENCH_HEIGHT, pc.SURFACE, "–", outline=pc.GRID, color=pc.MUTED)
            else:
                cell(axes, x, top + HACK_HEIGHT, BENCH_HEIGHT, LOSS_MAP(LOSS_NORM(min(max(-row["diff"], 0), LOSS_NORM.vmax))), signed(row["diff"], 1),
                     bold=excludes_zero(row["diff_ci"]))
    axes.set_xticks([slots[(section["kind"], value)] for section in sections for value in section["rungs"]],
                    [dl.rung_label(section["kind"], value) for section in sections for value in section["rungs"]])
    axes.tick_params(colors=pc.MUTED, labelsize=8.2, length=0, pad=4)
    for section in sections:
        width = (section["end"] - section["start"]) * CELL_WIDTH
        axes.text((section["start"] + section["end"]) / 2, total + 0.3, pc.wrap(dl.SECTIONS[section["kind"]], 9, max(1.25 * width, 0.9)), fontsize=9,
                  color=pc.INK2, ha="center", va="top", linespacing=1.2, clip_on=False)


def colorbar(figure, box: tuple[float, float, float, float], cmap, ticks: list[tuple[float, str]], label: str) -> None:
    axes = figure.add_axes(box)
    axes.imshow(np.linspace(0, 1, 256)[None, :], aspect="auto", cmap=cmap, extent=(0, 1, 0, 1))
    axes.set_yticks([])
    axes.set_xticks([position for position, _ in ticks], [text for _, text in ticks])
    axes.tick_params(colors=pc.MUTED, labelsize=8, length=2.5, pad=2)
    for side in axes.spines.values():
        side.set_visible(False)
    axes.set_title(label, fontsize=8.8, color=pc.INK2, loc="left", pad=4)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", nargs="+", choices=sorted(pc.MODELS), default=["gptoss", "muse", "gemma", "qwen"])
    parser.add_argument("--axes", nargs="+", default=["gi_v2", "gfar"])
    parser.add_argument("--kinds", nargs="+", choices=list(dl.SECTIONS), default=list(dl.SECTIONS))
    parser.add_argument("--composite", choices=sorted(pc.COMPOSITES), default="pooled")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    combine, score_name = pc.COMPOSITES[args.composite]
    kinds = [kind for kind in dl.SECTIONS if kind in args.kinds]
    panels = []
    for model_key in args.model:
        model, groups = pc.MODELS[model_key], []
        for axis_key in args.axes:
            data = pc.load_axis(model, model["axes"][axis_key], combine, keep_without_benchmark=True)
            rows = [dict(arm=arm, rung=dl.rung(arm), **row) for arm, row in data["arms"].items() if row["kind"] in kinds]
            groups.append(dict(axis_key=axis_key, data=data, rows=rows))
        slots, sections = dl.layout(groups, kinds)
        panels.append(dict(model_key=model_key, model=model, groups=groups, slots=slots, sections=sections))

    block_height = (HACK_HEIGHT + BENCH_HEIGHT + AXIS_GAP) * len(args.axes) - AXIS_GAP
    panel_height = TITLE_HEIGHT + block_height + XAXIS_HEIGHT + GAP
    widest = max(panel["sections"][-1]["end"] - panel["sections"][0]["start"] for panel in panels)
    width = LEFT + widest * CELL_WIDTH + RIGHT
    title = pc.wrap("Capping, steering and flooring on " + (f"{pc.COUNT_WORDS.get(len(panels), len(panels))} models" if len(panels) > 1 else panels[0]["model"]["name"])
                    + ": relative change in judged reward hacking and change in benchmark score, "
                    + " and ".join(pc.AXIS_NAMES[key] for key in args.axes), 12.5, width - 0.3)
    undefined = [panel["model"]["name"] for panel in panels if panel["groups"][0]["data"]["unsteered"]["hack"] == 0]
    subtitle = pc.wrap("Hack rate cells: change as % of the model's unsteered rate (−100% = no judged hack left). Benchmark cells: change in "
                       "points. The outlined column is the unsteered arm (rate and score). Bold = 95% CI excludes zero."
                       + (f" {' and '.join(undefined)}: unsteered hack rate is 0%, so the change is in points (pt) and has no colour." if undefined else ""),
                       9.5, width - 0.1)
    sets: dict[str, list[str]] = {}
    for panel in panels:
        data = panel["groups"][0]["data"]
        sets.setdefault(f"{data['problems']} problems × {data['seeds']} seeds", []).append(panel["model"]["name"])
    foot = pc.wrap("\n".join(
        [f"Hack rate: {'; '.join(' and '.join(names) + ' ' + size for size, names in sets.items())} per arm, as judged; CI = paired bootstrap "
         "over problems. Clamp thresholds are percentiles of the unsteered projection. Columns are evenly spaced, not to scale; an empty slot is "
         "a rung the other axis ran.",
         f"Benchmark score = {score_name.format(items=panels[0]['groups'][0]['data']['items'])}: GSM8K (1,319 items), MMLU-Pro (1,400), "
         "HumanEval (164), IFBench (300), one run per arm; CI = paired bootstrap over items. – = no benchmark run yet."]), 7.8, width - 0.3)
    foot_height = 0.14 * (foot.count("\n") + 1) + 0.2
    key_height = 0.62
    head_height = 0.12 + 0.235 * (title.count("\n") + 1) + 0.08 + 0.175 * (subtitle.count("\n") + 1) + 0.16 + key_height + 0.22
    height = head_height + panel_height * len(panels) + foot_height
    figure = plt.figure(figsize=(width, height), facecolor=pc.SURFACE)
    at = lambda inches: 1 - inches / height
    x0 = 0.14 / width
    figure.text(x0, at(0.12), title, fontsize=12.5, color=pc.INK, ha="left", va="top", linespacing=1.25)
    cursor = 0.12 + 0.235 * (title.count("\n") + 1) + 0.08
    figure.text(x0, at(cursor), subtitle, fontsize=9.5, color=pc.INK2, ha="left", va="top", linespacing=1.3)
    cursor += 0.175 * (subtitle.count("\n") + 1) + 0.16
    bar_top = cursor + 0.24
    bar_x = x0 + 0.2 / width
    colorbar(figure, (bar_x, at(bar_top + 0.15), 3.6 / width, 0.15 / height), HACK_MAP,
             [(0, "−100%"), (0.25, "−50%"), (0.5, "0"), (0.75, "+100%"), (1, "+200% or more")], "hack rate: change as % of the unsteered rate")
    colorbar(figure, (bar_x + 4.5 / width, at(bar_top + 0.15), 2.2 / width, 0.15 / height), LOSS_MAP,
             [(0, "0 or a gain"), (0.5, "−2.5"), (1, "−5 or more")], "benchmark: points lost")
    ring_x = bar_x + 7.45 / width
    ring = figure.add_axes((ring_x, at(bar_top + 0.15), 0.15 / width, 0.15 / height))
    ring.plot(0, 0, "o", markersize=4.6, markerfacecolor="none", markeredgecolor=pc.INK, markeredgewidth=1.0)
    ring.axis("off")
    figure.text(ring_x + 0.22 / width, at(bar_top + 0.075), f"the model stops submitting code (no-code share\nmore than {pc.RULE_POINTS:g} points above unsteered)",
                fontsize=8.8, color=pc.INK2, ha="left", va="center", linespacing=1.25)
    figure.text(x0, 0.1 / height, foot, fontsize=7.8, color=pc.MUTED, ha="left", va="bottom", linespacing=1.3)

    for index, panel in enumerate(panels):
        sections = panel["sections"]
        top = head_height + index * panel_height
        block_width = (sections[-1]["end"] - sections[0]["start"]) * CELL_WIDTH
        axes = figure.add_axes((LEFT / width, at(top + TITLE_HEIGHT + block_height), block_width / width, block_height / height))
        draw_block(axes, panel, panel["slots"], sections)
        unsteered = panel["groups"][0]["data"]["unsteered"]
        name = figure.text(x0, at(top + 0.02), f"{panel['model']['name']}", fontsize=10.4, color=pc.INK, ha="left", va="top", fontweight="bold")
        name_end = name.get_window_extent(figure.canvas.get_renderer()).x1 / (figure.dpi * width)
        figure.text(name_end + 0.14 / width, at(top + 0.035),
                    f"unsteered: judged hack rate {unsteered['hack']:.1f}%, benchmark score {unsteered['composite']:.1f}%", fontsize=9.2, color=pc.INK2,
                    ha="left", va="top")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out, dpi=200, facecolor=pc.SURFACE)
    figure.savefig(args.out.with_suffix(".pdf"), facecolor=pc.SURFACE)
    record = dict(composite=args.composite, kinds=kinds, hack_scale=dict(vmin=-100, center=0, vmax=200, unit="% of the unsteered rate"),
                  benchmark_scale=dict(no_loss=0, max_loss=5, unit="points"), boot_draws=pc.BOOT_DRAWS, boot_seed=pc.BOOT_SEED,
                  panels=[dict(model=panel["model"]["name"],
                               axes=[dict(axis=group["axis_key"], analysis=group["data"]["analysis"], unsteered=group["data"]["unsteered"],
                                          arms=group["rows"]) for group in panel["groups"]])
                          for panel in panels])
    args.out.with_suffix(".json").write_text(json.dumps(record, indent=1))
    for panel in panels:
        for group in panel["groups"]:
            print(f"{panel['model']['name']} / {group['axis_key']}")
            for row in sorted(group["rows"], key=lambda row: (list(dl.SECTIONS).index(row["kind"]), row["rung"])):
                relative = "   n/a" if row["hack_rel"] is None else f"{row['hack_rel']:+6.0f}% (CI {row['hack_rel_ci'][0]:+.0f} to {row['hack_rel_ci'][1]:+.0f})"
                bench = "no run" if row["diff"] is None else f"{row['diff']:+5.2f}"
                print(f"  {row['kind']:8s} {dl.rung_label(row['kind'], row['rung']):>6s}  hack {row['hack_diff']:+5.1f} pt  relative {relative}  benchmark {bench}"
                      f"{'  RING' if row['stops_submitting'] else ''}")
    print("->", args.out)


if __name__ == "__main__":
    main()
