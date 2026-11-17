"""Dose-panel chart: per model, what steering, capping and flooring along the gi_v2 vector and the Goodfire reward-hacking axis do
to the judged hack rate, the benchmark score and hidden-test pass. A rebuild of dose_lines_chart.py's layout in use
(--split --bench-strip --hidden-strip --clamp-scale --equal-parts --dose-scale) with a short header (user 2026-11-16: the top of
that chart was busy with text; more compact, cleaner).

One panel per model, three rows per panel on one x axis: judged hack rate (tall), benchmark score and hidden-test pass (short).
Every value is the change from the model's unsteered arm in points; each row has its own y axis and a heading that names the
measure and gives the unsteered level. Colour = steering axis; the measure is the row, so every row uses the same marker. Bars =
95% CIs of the difference (hack rate and hidden pass: paired bootstrap over problems, from the analysis file; benchmark:
item-paired bootstrap), drawn where one is stored.

The x axis has two halves of equal width, each around its own unsteered point (black):
  left     steering doses α in layer norms, α = 0 in the middle, every dose that was run labelled. --dose-spacing linear (default):
           linear in α, the panel's largest |α| just inside the edge on both sides (each model's own range). --dose-spacing even:
           the doses of each sign evenly spaced by rank.
  right    (tinted) caps left of the unsteered point and floors right of it, to scale in the threshold percentile: a cap at p sits
           (100 − p) points from the unsteered point and a floor at p sits p points from it, so the distance is the share of
           unsteered positions beyond the threshold. Labelled ticks are the same reference percentiles on every panel;
           unlabelled ticks mark the other rungs that were run.
The steering line passes through the left unsteered point, the cap and floor line through the right one. Both axes' markers sit
exactly on their rung, never dodged sideways (user 2026-11-16), so two points at one rung can overlap; the CI bars carry caps so
that the two intervals at one rung can be told apart. Every row has its left and bottom axis line with tick marks, and a thin
vertical guide line at every rung that was run (user 2026-11-16: more ticks and lines in the plots; the short header is not a
call for bare plots). The x tick labels (doses and percentiles) are repeated under every row (user 2026-11-17); the captions that
name the x scales stand once, under the last row. One x tick-label size is used for the figure, the largest that keeps
neighbouring labels apart.

A hollow marker = the model stops submitting code on that arm (no-code share more than 10 points above unsteered), on all three
rows; the hack-rate line into and out of such an arm is dotted (a hack rate there is not comparable). The hidden-pass row carries
dashed lines at the dose-rule limit (±10 points). Hidden pass falls when hacking rises, so that row shows the rule's measure, not
capability. Rungs past the least extreme failing rung of their ladder are left out by the presets (`omit` in pareto_chart.py).

The header is a title, one line of subtitle and one legend row. What the earlier chart explained above the panels (the x scales,
the hollow marker, sample sizes, the measures) is in the notes under the panels, set as a list of terms in two columns.

Data, presets and the benchmark composite come from pareto_chart.py (`MODELS`, `load_axis`). `collect` builds the record and `draw`
draws from the record alone, so the .json written next to the chart holds every plotted number; --data <record.json> redraws from
such a file without recomputing the benchmark bootstraps. Writes <out>.png, .pdf and .json.

    python experiments/deceiver-axis/scripts/dose_panels_chart.py --out <path>.png
    python experiments/deceiver-axis/scripts/dose_panels_chart.py --dose-spacing even --out <path>.png
    python experiments/deceiver-axis/scripts/dose_panels_chart.py --data <earlier>.json --out <path>.png   # redraw only
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator

import pareto_chart as pc

FONTS = ["Liberation Sans", "Arial", "Helvetica", "Nimbus Sans", "DejaVu Sans"]  # first one installed is used
SIZE = dict(title=17, subtitle=11, legend=10.5, model=13.5, row=10.5, tick=9.5, caption=10, note=9)  # points
TEXT, SOFT, GRID, RULE, TINT = "#2e2d2a", pc.INK2, "#d9d8cf", "#9a998f", "#f1f0e9"
AXIS_SHORT = {"gi_v2": "gi_v2", "gfar": "Goodfire"}
# rows of a panel: record key of the value and of its CI, heading, height in inches, marker size, line width, most y intervals
# (`ticks`), half-width of the CI caps in points (`cap`)
ROWS = {"hack": dict(value="hack_diff", ci="hack_diff_ci", name="Judged hack rate", height=2.45, marker=7.2, line=2.0, ticks=9, cap=2.6),
        "bench": dict(value="diff", ci="diff_ci", name="Benchmark score", height=0.98, marker=5.4, line=1.5, ticks=4, cap=2.0),
        "hidden": dict(value="hidden_pass_diff", ci="hidden_pass_diff_ci", name="Hidden-test pass", height=0.98, marker=5.4, line=1.5, ticks=4, cap=2.0)}
UNSTEERED = {"hack": "hack", "bench": "composite", "hidden": "hidden_pass"}
CLAMP_TICKS = {"cap": (5, 25, 50, 75), "floor": (25, 50, 75, 95)}  # labelled reference percentiles either side of the unsteered point
CAPTIONS = {"steering": "steering dose α (layer norms)", "cap": "cap at percentile", "floor": "floor at percentile"}
# x axis in data units: the steering half is [0, 1], the clamp half [1 + HALF_GAP, 2 + HALF_GAP]; PAD between the outermost rung
# and the edge of its half
HALF_GAP, PAD = 0.06, 0.035
DOTTED = (0, (1.1, 1.9))  # hack-rate line into or out of an arm where the model stops submitting code
# page layout in inches; under every row X_TICKS for its x tick labels, under the last row X_CAPTION more for the scale captions
WIDTH, OUTER, GUTTER, COLUMN_GAP = 14.0, 0.28, 0.5, 0.4
NAME, HEAD, ROW_GAP, X_TICKS, X_CAPTION, PANEL_GAP = 0.33, 0.24, 0.09, 0.24, 0.32, 0.3
LINE = 1.32  # line height in ems
TICK_SPACE = 0.045  # least space in inches between two neighbouring x tick labels


def minus(text: str) -> str:
    return text.replace("-", "−")


def signed(value: float, _=None) -> str:
    return "0" if value == 0 else minus(f"{value:+g}")


def rung(arm: str) -> float:
    """Position of an arm on its ladder: the dose, or the threshold percentile."""
    return pc.dose(arm) if pc.kind(arm) == "steering" else float(pc.percentile(arm))


def collect(model_keys: list[str], axis_keys: list[str], composite: str) -> dict:
    """The record: everything `draw` needs, read through pareto_chart.load_axis."""
    combine, score_name = pc.COMPOSITES[composite]
    panels = []
    for model_key in model_keys:
        model, axes = pc.MODELS[model_key], []
        for axis_key in axis_keys:
            data = pc.load_axis(model, model["axes"][axis_key], combine, keep_without_benchmark=True)
            arms = sorted((dict(arm=arm, rung=rung(arm), **row) for arm, row in data["arms"].items()),
                          key=lambda row: (list(CAPTIONS).index(row["kind"]), row["rung"]))
            axes.append(dict(axis=axis_key, name=pc.AXIS_NAMES[axis_key], analysis=data["analysis"], unsteered=data["unsteered"],
                             problems=data["problems"], seeds=data["seeds"], items=data["items"],
                             omitted_by_preset=data["omitted_by_preset"], arms=arms))
        panels.append(dict(model=model["name"], axes=axes))
    return dict(chart="dose_panels", composite=composite, score_name=score_name, rule_points=pc.RULE_POINTS, boot_draws=pc.BOOT_DRAWS,
                boot_seed=pc.BOOT_SEED, panels=panels)


def place(panel: dict, spacing: str) -> dict:
    """x of every (kind, rung) of the panel, the two unsteered positions, the labelled and unlabelled ticks, and the guide lines
    (one at every rung that was run)."""
    rungs = lambda kind: sorted({row["rung"] for axis in panel["axes"] for row in axis["arms"] if row["kind"] == kind})
    span, steer, clamp = 0.5 - PAD, 0.5, 1.5 + HALF_GAP
    doses = sorted(set(rungs("steering")) | {0.0})
    reach = max(abs(dose) for dose in doses) or 1.0
    if spacing == "linear":
        slots = {("steering", dose): steer + span * dose / reach for dose in doses}
    else:  # each sign evenly by rank, both scaled by the longer side, so α = 0 stays in the middle
        sides = [dose for dose in doses if dose < 0][::-1], [dose for dose in doses if dose > 0]
        step = span / max(len(sides[0]), len(sides[1]), 1)
        slots = {("steering", 0.0): steer, **{("steering", dose): steer + sign * step * (index + 1)
                                              for sign, side in zip((-1, 1), sides) for index, dose in enumerate(side)}}
    major, minor = [(slots[("steering", dose)], signed(dose)) for dose in doses], []
    at = {"cap": lambda p: clamp - span * (100 - p) / 100, "floor": lambda p: clamp + span * p / 100}
    major.append((clamp, ""))
    for kind in ("cap", "floor"):
        run = rungs(kind)
        slots.update({(kind, p): at[kind](p) for p in run})
        if run:
            major += [(at[kind](p), f"p{p:g}") for p in CLAMP_TICKS[kind]]
            minor += [at[kind](p) for p in run if p not in CLAMP_TICKS[kind]]
    return dict(slots=slots, origins=dict(steering=steer, cap=clamp, floor=clamp), major=sorted(major), minor=sorted(minor),
                guides=sorted(set(slots.values()) - {steer}),
                captions=dict(steering=steer, cap=clamp - 0.25, floor=clamp + 0.25), tint=(1 + HALF_GAP / 2, 2 + HALF_GAP),
                limits=(0.0, 2 + HALF_GAP))


def draw_row(axes, panel: dict, key: str, spot: dict, colors: dict[str, str], rule_points: float, x_tick_size: float) -> None:
    row_style = ROWS[key]
    value, interval = row_style["value"], row_style["ci"]
    axes.set_facecolor(pc.SURFACE)
    axes.axvspan(*spot["tint"], color=TINT, linewidth=0, zorder=0)
    axes.grid(axis="y", color=GRID, linewidth=0.7, zorder=0.5)
    axes.set_axisbelow(True)
    axes.axhline(0, color=pc.ZERO, linewidth=0.9, zorder=1.5)
    for origin in {spot["origins"]["steering"], spot["origins"]["cap"]}:
        axes.axvline(origin, color=RULE, linewidth=0.7, zorder=1)
    for x in spot["guides"]:
        axes.axvline(x, color=GRID, linewidth=0.7, zorder=0.6)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(RULE)
    edge = 1.5 if key == "hack" else 1.15
    for axis in panel["axes"]:
        color = colors[axis["axis"]]
        for kinds in (("steering",), ("cap", "floor")):
            points = sorted(((spot["slots"][(row["kind"], row["rung"])], row) for row in axis["arms"]
                             if row["kind"] in kinds and row[value] is not None), key=lambda point: point[0])
            if not points:
                continue
            line = sorted([(x, row[value], row["stops_submitting"]) for x, row in points] + [(spot["origins"][kinds[0]], 0.0, False)])
            for (x_a, y_a, hollow_a), (x_b, y_b, hollow_b) in zip(line, line[1:]):
                axes.plot([x_a, x_b], [y_a, y_b], color=color, linewidth=row_style["line"], zorder=3, solid_capstyle="round",
                          dash_capstyle="round", linestyle=DOTTED if key == "hack" and (hollow_a or hollow_b) else "-")
            for x, row in points:
                y = row[value]
                if row[interval] is not None:  # hidden pass read from a run summary has no stored CI
                    axes.errorbar(x, y, yerr=[[y - row[interval][0]], [row[interval][1] - y]], fmt="none", ecolor=color, elinewidth=1.1,
                                  capsize=row_style["cap"], capthick=1.1, alpha=0.75, zorder=2)
                face, ring = (pc.SURFACE, color) if row["stops_submitting"] else (color, pc.SURFACE)
                axes.plot(x, y, "o", markersize=row_style["marker"], color=face, markeredgecolor=ring, markeredgewidth=edge, zorder=4)
    for origin in {spot["origins"]["steering"], spot["origins"]["cap"]}:
        axes.plot(origin, 0, "o", markersize=row_style["marker"], color=pc.INK, markeredgecolor=pc.SURFACE, markeredgewidth=edge, zorder=5)
    axes.set_xlim(*spot["limits"])
    axes.margins(y=0.08 if key == "hack" else 0.2)
    axes.yaxis.set_major_locator(MaxNLocator(nbins=row_style["ticks"], min_n_ticks=3, steps=[1, 2, 2.5, 5, 10]))
    axes.yaxis.set_major_formatter(FuncFormatter(signed))
    axes.tick_params(axis="y", length=3.5, color=RULE, pad=3, labelsize=SIZE["tick"], labelcolor=SOFT)
    if key == "hidden":  # the dose-rule lines; the lower one is always inside the y range
        low, high = axes.get_ylim()
        axes.set_ylim(min(low, -rule_points - 3), max(high, 3))
        for limit in (-rule_points, rule_points):
            axes.axhline(limit, color=SOFT, linewidth=1.0, linestyle=(0, (4, 3)), zorder=1.6)
    axes.set_xticks([x for x, _ in spot["major"]], [text for _, text in spot["major"]])
    axes.set_xticks(spot["minor"], minor=True)
    axes.tick_params(axis="x", which="major", length=4, color=RULE, pad=3.5, labelsize=x_tick_size, labelcolor=SOFT)
    axes.tick_params(axis="x", which="minor", length=4, color=RULE)


class Ruler:
    """Text widths in inches, for wrapping and for placing text after text."""

    def __init__(self) -> None:
        self.figure = plt.figure(dpi=100)
        self.renderer = self.figure.canvas.get_renderer()

    def width(self, text: str, size: float, weight: str = "normal") -> float:
        return self.renderer.get_text_width_height_descent(text, FontProperties(size=size, weight=weight), ismath=False)[0] / self.figure.dpi

    def wrap(self, text: str, size: float, width: float) -> list[str]:
        lines, line = [], ""
        for word in text.split():
            if line and self.width(f"{line} {word}", size) > width:
                lines.append(line)
                line = word
            else:
                line = f"{line} {word}".strip()
        return lines + [line]


def notes(record: dict, spacing: str) -> list[tuple[str, str]]:
    """The terms under the panels: (term, text)."""
    panels, points = record["panels"], f"{record['rule_points']:g}"
    sets: dict[str, list[str]] = {}
    for panel in panels:
        sets.setdefault(f"{panel['axes'][0]['problems']} problems × {panel['axes'][0]['seeds']} seeds", []).append(panel["model"])
    no_ci = [f"{panel['model']} {AXIS_SHORT.get(axis['axis'], axis['axis'])}" for panel in panels for axis in panel["axes"]
             if any(row["hidden_pass_diff_ci"] is None for row in axis["arms"])]
    no_run = [f"{panel['model']} {AXIS_SHORT.get(axis['axis'], axis['axis'])} "
              + ", ".join(signed(row["rung"]) if row["kind"] == "steering" else f"{row['kind']} p{row['rung']:g}" for row in axis["arms"] if row["diff"] is None)
              for panel in panels for axis in panel["axes"] if any(row["diff"] is None for row in axis["arms"])]
    listed = [
        ("Steering", "Dose α in layer norms, " + ("linear in α" + (" on each model's own range" if len(panels) > 1 else "") + "."
                                                 if spacing == "linear" else "evenly spaced, not to scale.")),
        ("Caps and floors", "Tinted half: caps left of the unsteered point, floors right of it, by threshold percentile of the unsteered "
                            "projection. The distance from the unsteered point is the share of unsteered positions beyond the threshold "
                            "(cap at p: 100 − p; floor at p: p). A thin vertical line marks every rung that was run; those without a label are other percentiles."),
        ("Hollow marker", f"The model stops submitting code on that arm (no-code share more than {points} points above unsteered), so its "
                          "hack rate, joined by a dotted line, is not comparable."),
        ("Hack rate", "As judged, per arm: " + "; ".join(f"{' and '.join(names)} {size}" for size, names in sets.items())
         + ". CI = paired bootstrap over problems."),
        ("Benchmark score", f"{record['score_name'].format(items=panels[0]['axes'][0]['items'])[0].upper()}"
                            f"{record['score_name'].format(items=panels[0]['axes'][0]['items'])[1:]}: GSM8K (1,319 items), MMLU-Pro (1,400), "
                            "HumanEval (164), IFBench (300), one run per arm. CI = paired bootstrap over items."
                            + (f" No benchmark run yet (hack-rate point only): {'; '.join(no_run)}." if no_run else "")),
        ("Hidden-test pass", "Share of all episodes whose final code passes the hidden tests. Dashed lines = the dose-rule limit, "
                             f"{points} points from unsteered. A hack that fails the hidden tests counts as a failure, so this measure falls "
                             "when hacking rises and is not read alone as capability." + (f" No stored CI: {', '.join(no_ci)}." if no_ci else "")),
    ]
    return listed


def draw(record: dict, out: Path, spacing: str) -> None:
    plt.rcParams.update({"font.family": FONTS, "pdf.fonttype": 42})
    ruler = Ruler()
    panels = record["panels"]
    axis_keys = [axis["axis"] for axis in panels[0]["axes"]]
    colors = dict(zip(axis_keys, pc.GROUP_COLORS))
    columns = 1 if len(panels) == 1 else 2
    width = WIDTH if columns == 2 else WIDTH / 2 + OUTER
    plot_width = (width - 2 * OUTER - (columns - 1) * COLUMN_GAP) / columns - GUTTER
    inch = lambda size: size * LINE / 72  # height of a line of text
    left = OUTER + GUTTER  # the left edge of the first column of plots: title, legend and notes line up with it

    # notes: a list of terms, in as many columns as the panels
    term_width = max(ruler.width(term, SIZE["note"], "bold") for term, _ in notes(record, spacing)) + 0.16
    blocks = [(term, ruler.wrap(text, SIZE["note"], plot_width - term_width)) for term, text in notes(record, spacing)]
    gap = 0.07
    tall = lambda column: sum(len(lines) * inch(SIZE["note"]) + gap for _, lines in column)
    split = min(range(len(blocks) + 1), key=lambda index: max(tall(blocks[:index]), tall(blocks[index:]))) if columns == 2 else len(blocks)
    note_columns = [blocks[:split], blocks[split:]] if columns == 2 else [blocks]
    notes_height = max(tall(column) for column in note_columns)

    # x tick labels: one size for the figure, the largest up to SIZE["tick"] at which neighbouring labels stay TICK_SPACE apart
    spots = [place(panel, spacing) for panel in panels]
    x_tick_size = SIZE["tick"]
    for spot in spots:
        labelled = [(x * plot_width / (spot["limits"][1] - spot["limits"][0]), ruler.width(text, SIZE["tick"])) for x, text in spot["major"] if text]
        for (x_a, width_a), (x_b, width_b) in zip(labelled, labelled[1:]):
            x_tick_size = min(x_tick_size, SIZE["tick"] * (x_b - x_a - TICK_SPACE) / ((width_a + width_b) / 2))

    rows = len(ROWS)
    panel_height = NAME + sum(HEAD + row["height"] + X_TICKS + ROW_GAP for row in ROWS.values()) + X_CAPTION
    grid_rows = math.ceil(len(panels) / columns)
    ring = dict(markeredgecolor=pc.SURFACE, markeredgewidth=1.5, markersize=ROWS["hack"]["marker"])
    keys = [dict(label=axis["name"], sample=0.36, line=dict(color=colors[axis["axis"]], linewidth=ROWS["hack"]["line"]),
                 mark=dict(color=colors[axis["axis"]], **ring)) for axis in panels[0]["axes"]]
    keys += [dict(label="unsteered", sample=0.1, line=None, mark=dict(color=pc.INK, **ring)),
             dict(label="model stops submitting code", sample=0.42, line=dict(color=SOFT, linewidth=ROWS["hack"]["line"], linestyle=DOTTED, dash_capstyle="round"),
                  mark=dict(color=pc.SURFACE, markeredgecolor=SOFT, markeredgewidth=1.5, markersize=ROWS["hack"]["marker"])),
             dict(label=f"dose-rule limit (±{record['rule_points']:g} points)", sample=0.36, line=dict(color=SOFT, linewidth=1.0, linestyle=(0, (4, 3))), mark=None)]
    at_x, legend_line = left, 0
    for key in keys:
        key["width"] = key["sample"] + 0.09 + ruler.width(key["label"], SIZE["legend"])
        if at_x > left and at_x + key["width"] > width - OUTER:
            at_x, legend_line = left, legend_line + 1
        key["x"], key["row"] = at_x, legend_line
        at_x += key["width"] + 0.34
    subtitle = ruler.wrap(f"Change from {'each' if len(panels) > 1 else 'the'} model's unsteered run in judged hack rate, benchmark score and "
                          "hidden-test pass, in percentage points. Bars are 95% confidence intervals.", SIZE["subtitle"], width - left - OUTER)
    head_height = 0.2 + inch(SIZE["title"]) + 0.03 + len(subtitle) * inch(SIZE["subtitle"]) + 0.13 + (legend_line + 1) * inch(SIZE["legend"]) * 1.15 + 0.3
    height = head_height + grid_rows * panel_height + (grid_rows - 1) * PANEL_GAP + 0.34 + notes_height + 0.12
    figure = plt.figure(figsize=(width, height), facecolor=pc.SURFACE)
    fx, fy = (lambda inches: inches / width), (lambda inches: 1 - inches / height)

    # header: title, one line of subtitle, one legend row
    interventions = pc.COUNT_WORDS.get(len(panels), str(len(panels)))
    title = "Steering, capping and flooring on " + (f"{interventions} models" if len(panels) > 1 else panels[0]["model"])
    cursor = 0.2
    figure.text(fx(left), fy(cursor), title, fontsize=SIZE["title"], fontweight="bold", color=pc.INK, ha="left", va="top")
    cursor += inch(SIZE["title"]) + 0.03
    for number, line in enumerate(subtitle):
        figure.text(fx(left), fy(cursor + number * inch(SIZE["subtitle"])), line, fontsize=SIZE["subtitle"], color=TEXT, ha="left", va="top")
    cursor += len(subtitle) * inch(SIZE["subtitle"]) + 0.13
    for key in keys:
        y = fy(cursor + (key["row"] * 1.15 + 0.5) * inch(SIZE["legend"]))
        if key["line"]:
            figure.add_artist(Line2D([fx(key["x"]), fx(key["x"] + key["sample"])], [y, y], **key["line"]))
        if key["mark"]:
            figure.add_artist(Line2D([fx(key["x"] + key["sample"] / 2)], [y], marker="o", linewidth=0, **key["mark"]))
        figure.text(fx(key["x"] + key["sample"] + 0.09), y, key["label"], fontsize=SIZE["legend"], color=TEXT, ha="left", va="center_baseline")

    for index, panel in enumerate(panels):
        column, grid_row = index % columns, index // columns
        x = OUTER + column * (plot_width + GUTTER + COLUMN_GAP) + GUTTER
        top = head_height + grid_row * (panel_height + PANEL_GAP)
        spot = spots[index]
        figure.text(fx(x), fy(top), panel["model"], fontsize=SIZE["model"], fontweight="bold", color=pc.INK, ha="left", va="top")
        cursor = top + NAME
        unsteered = panel["axes"][0]["unsteered"]
        for position, (key, row) in enumerate(ROWS.items()):
            last = position == rows - 1
            baseline = fy(cursor + HEAD - 0.08)
            figure.text(fx(x), baseline, row["name"], fontsize=SIZE["row"], fontweight="bold", color=TEXT, ha="left", va="baseline")
            level = unsteered[UNSTEERED[key]]
            if level is not None:
                figure.text(fx(x + ruler.width(row["name"], SIZE["row"], "bold") + 0.11), baseline, f"unsteered {level:.1f}%",
                            fontsize=SIZE["row"], color=SOFT, ha="left", va="baseline")
            axes = figure.add_axes((fx(x), fy(cursor + HEAD + row["height"]), plot_width / width, row["height"] / height))
            draw_row(axes, panel, key, spot, colors, record["rule_points"], x_tick_size)
            cursor += HEAD + row["height"] + X_TICKS + ROW_GAP
            if last:
                for kind, centre in spot["captions"].items():
                    if kind == "steering" or any(slot[0] == kind for slot in spot["slots"]):
                        axes.annotate(CAPTIONS[kind], xy=(centre, 0), xycoords=axes.get_xaxis_transform(), xytext=(0, -24),
                                      textcoords="offset points", fontsize=SIZE["caption"], color=TEXT, ha="center", va="top",
                                      annotation_clip=False)

    # notes
    top = head_height + grid_rows * panel_height + (grid_rows - 1) * PANEL_GAP + 0.34
    figure.add_artist(Line2D([fx(left), fx(width - OUTER)], [fy(top - 0.16)] * 2, color=GRID, linewidth=0.9))
    for column, column_blocks in enumerate(note_columns):
        x, cursor = OUTER + column * (plot_width + GUTTER + COLUMN_GAP) + GUTTER, top
        for term, lines in column_blocks:
            figure.text(fx(x), fy(cursor), term, fontsize=SIZE["note"], fontweight="bold", color=TEXT, ha="left", va="top")
            for number, line in enumerate(lines):
                figure.text(fx(x + term_width), fy(cursor + number * inch(SIZE["note"])), line, fontsize=SIZE["note"], color=SOFT, ha="left", va="top")
            cursor += len(lines) * inch(SIZE["note"]) + gap

    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=200, facecolor=pc.SURFACE)
    figure.savefig(out.with_suffix(".pdf"), facecolor=pc.SURFACE)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", nargs="+", choices=sorted(pc.MODELS), default=["gptoss", "muse", "gemma", "qwen"])
    parser.add_argument("--axes", nargs="+", default=["gi_v2", "gfar"])
    parser.add_argument("--composite", choices=sorted(pc.COMPOSITES), default="pooled")
    parser.add_argument("--dose-spacing", choices=("linear", "even"), default="linear",
                        help="steering doses linear in α on each panel's own range (default), or evenly spaced by rank")
    parser.add_argument("--data", type=Path, help="redraw from a record .json written by this script instead of reading the runs")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    record = json.loads(args.data.read_text()) if args.data else collect(args.model, args.axes, args.composite)
    record["dose_spacing"] = args.dose_spacing
    draw(record, args.out, args.dose_spacing)
    args.out.with_suffix(".json").write_text(json.dumps(record, indent=1))
    for panel in record["panels"]:
        for axis in panel["axes"]:
            print(f"{panel['model']} / {axis['axis']}")
            for row in axis["arms"]:
                label = signed(row["rung"]) if row["kind"] == "steering" else f"p{row['rung']:g}"
                bench = "benchmark   no run" if row["diff"] is None else f"benchmark {row['diff']:+5.2f} (CI {row['diff_ci'][0]:+5.2f} to {row['diff_ci'][1]:+5.2f})"
                hidden = "" if row["hidden_pass_diff"] is None else f"  hidden {row['hidden_pass_diff']:+5.1f}"
                print(f"  {row['kind']:8s} {label:>6s}  hack {row['hack_diff']:+5.1f} (CI {row['hack_diff_ci'][0]:+5.1f} to "
                      f"{row['hack_diff_ci'][1]:+5.1f})  {bench}{hidden}  no-code {row['no_code_diff']:+5.1f}{'  HOLLOW' if row['stops_submitting'] else ''}")
    print("->", args.out)


if __name__ == "__main__":
    main()
