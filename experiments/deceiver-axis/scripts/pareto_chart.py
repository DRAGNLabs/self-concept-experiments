"""Trade-off ("Pareto") chart: judged hack rate on the task set against a benchmark composite, per intervention arm.

Default layout: one panel per (model, steering axis) pair (gi_v2 vector, Goodfire RH axis), colour = kind of intervention.
--overlay draws every pair on one set of axes instead: colour = the pair (the axis when one model is drawn, the model when
several are), marker = kind of intervention (circle steering, triangle down cap, triangle up floor), steering lines solid and
clamp lines dashed. Every arm is one point:
  x = benchmark composite of the arm's benchmark run (capability_eval_vllm.py output, <capeval root>/<arm>/<task>.jsonl) over
      GSM8K, MMLU-Pro, HumanEval and IFBench, in %. --composite pooled (default): the share of all benchmark items answered
      correctly (3,183 items; each benchmark weighted by its item count). --composite mean: the plain mean of the four
      accuracies. --composite geometric: their geometric mean. On the gpt-oss arms IFBench (300 items, 53% unsteered) carries
      83% of the sampling variance of the geometric mean and 63% of the plain mean's, against 27% of the pooled score's
      (figures/pareto/composite_options.py, 2026-11-13), and the pooled CI is less than half as wide;
  y = judged hack rate (as judged; main_analysis.py output, `judged_hack` of the analysis file named in the preset), in %.
Both axes are drawn as the arm's difference from unsteered, in points, so the unsteered arm is the origin (its levels are in
the subtitle); the two zero lines are drawn darker than the grid. Lines run from the origin outwards in dose order: steering
(most negative dose ... unsteered ... most positive dose) in one colour; capping (weakest rung first: highest percentile
threshold) and flooring (weakest rung first: lowest percentile) share the second colour and differ by marker (cap = triangle
pointing down, floor = triangle pointing up) and by their labels, as the two signs of steering share theirs.

--y relative draws the change in the judged hack rate as % of the model's unsteered rate instead of in points (-100 = no judged
hack left), for charts that put models with different unsteered rates on one y axis; its CI is a paired bootstrap over problems
of the ratio (both rates redrawn). A model whose unsteered rate is zero has no relative change and is left out (Qwen).
--y-max sets the top of the y axis: an arm above it is not drawn as a marker, its line leaves through the top of the frame and
its label at the top edge carries the value.

Error bars are 95% CIs of the difference: vertical = the problem-level paired bootstrap CI of the analysis file
(`contrasts[arm].ci95`); horizontal = an item-paired bootstrap of the composite (items resampled within each benchmark, the
same draw for arm and unsteered, 10,000 draws, fixed seed). An arm on which the model stops submitting code (share of episodes
with no code submitted more than 10 points above unsteered) is drawn hollow. The marker reads no-code alone (user 2026-11-13:
hidden-test pass is confounded with hacking); the task-set dose rule itself is unchanged and its status per arm
(`inside_dose_rule`, `outside_on`) stays in the JSON.
Label offsets (points) per arm come from the preset's `offsets` or --offsets; every other label is placed by a greedy search
that avoids markers, lines and the labels already placed (`place_labels`; always in --overlay). Every panel has its own x and
y range (user 2026-11-13); --shared-axes shares both across the panels.

An arm of the analysis file with no benchmark run is left out, printed and named in the footnote (2026-11-13: the Goodfire p25
cap on Muse and on Qwen; those runs are queued by Koby). Analysis arm names of the Gemma / Qwen sets (`gfar_L30_a-0.04`,
`gfar_cap_p75_L20-38`) are mapped to the benchmark-run names (`gf_m0.04`, `gf_cap_p75`).

Writes <out>.png, <out>.pdf and <out>.json (every plotted number).

    python experiments/deceiver-axis/scripts/pareto_chart.py --model gptoss --out <path>.png [--axes gi_v2 gfar] [--overlay]
    python experiments/deceiver-axis/scripts/pareto_chart.py --model gptoss muse gemma qwen --axes gfar --overlay --out <path>.png
"""
from __future__ import annotations

import argparse
import json
import math
import re
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import to_rgb
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D

N = Path.home() / "nobackup/autodelete/self-concept-experiments/schemer-axis"
TASKS = ("gsm8k", "mmlu_pro", "humaneval", "ifbench")
BOOT_DRAWS, BOOT_SEED = 10_000, 0
RULE_POINTS = 10.0  # task-set dose rule: |Δ hidden pass| and |Δ no-code| within 10 points of unsteered; hollow marker: Δ no-code above it

# categorical slots 1-2 of the chart palette (validated). Caps and floors share the clamp colour (user 2026-11-13: the pair is
# to clamping what the two signs are to steering); marker shape and the direct label tell them apart.
STEER, CLAMP = "#2a78d6", "#eb6834"
ZERO = "#52514e"  # the two zero lines (the unsteered level), darker than the grid
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
# --overlay: one colour per (model, axis) pair. The palette's first three slots plus the dark violet step; the four pass the
# validator on all pairs (scatter): worst CVD ΔE 9.2, worst normal-vision ΔE 16.3 (validate_palette.js --pairs all, 2026-11-13).
# The palette's own fourth slot (yellow) fails all-pairs next to orange.
GROUP_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7")
SERIES = {"steering": dict(color=STEER, marker="o", name="steering (add α · layer norm)"),
          "cap": dict(color=CLAMP, marker="v", name="cap (clamp projection from above)"),
          "floor": dict(color=CLAMP, marker="^", name="floor (clamp projection from below)")}

MODELS = {
    "gptoss": dict(
        name="gpt-oss-120b", task_set="45 problems × 12 seeds", capeval=N / "gi_v2/capeval",
        axes={"gi_v2": dict(title="gi_v2 persona-contrast vector (layer 17; clamps on layers 12–23)", prefix="gi_",
                            analysis=N / "gi_v2/pilot/analysis_floorladder_all_as_judged.json",
                            offsets={"gi_m0.1": (12, -5), "gi_m0.2": (-10, -10), "gi_p0.1": (10, 6), "gi_cap_p50": (-10, 5),
                                     "gi_cap_p25": (-12, 1), "gi_cap_p5": (13, -9), "gi_floor_p50": (-10, 7), "gi_floor_p95": (-10, -7),
                                     "unsteered": (12, 7)}),
              "gfar": dict(title="Goodfire reward-hacking axis (layer 17; clamps on layers 12–23)", prefix="gf_",
                           analysis=N / "gi_v2/pilot/analysis_gfx01_all_as_judged.json", summary=N / "gi_v2/pilot/summary_gfar.json",
                           offsets={"gf_cap_p75": (13, 4), "gf_cap_p50": (12, -10), "gf_cap_p25": (0, -14), "gf_cap_p95": (12, -8),
                                    "gf_m0.1": (-10, 3)})}),
    "muse": dict(
        name="Muse-Glimmer-30B", task_set="45 problems × 12 seeds", capeval=N / "muse_glimmer/gi_v2/capeval",
        axes={"gfar": dict(title="Goodfire reward-hacking axis (layer 26; clamps on layers 17–33)", prefix="gf_",
                           analysis=N / "muse_glimmer/gi_v2/pilot/analysis_ctl_m01_s0-11_all_as_judged.json")}),
    "gemma": dict(
        name="Gemma 4 31B", task_set="108 problems × 16 seeds", capeval=N / "gemma4_31b_gi_v2/capeval",
        axes={"gfar": dict(title="Goodfire reward-hacking axis (layer 30; clamps on layers 20–38)", prefix="gf_",
                           analysis=N / "gemma4_31b_gi_v2/rh/hard108/analysis_gfar6/analysis_all_as_judged.json")}),
    "qwen": dict(
        name="Qwen3.8-27B", task_set="61 problems × 12 seeds", capeval=N / "qwen38_27b_gi_v2/capeval",
        axes={"gfar": dict(title="Goodfire reward-hacking axis (layer 32; clamps on layers 22–42)", prefix="gf_",
                           analysis=N / "qwen38_27b_gi_v2/rh/qwen_hard61/analysis_gfar_gfdose/analysis_all_as_judged.json")}),
}
AXIS_NAMES = {"gi_v2": "gi_v2 persona-contrast vector", "gfar": "Goodfire reward-hacking axis"}


def canonical(arm: str) -> str:
    """Benchmark-run name of an analysis arm: gfar_L30_a-0.04 -> gf_m0.04, gfar_cap_p75_L20-38 -> gf_cap_p75; others unchanged."""
    if match := re.fullmatch(r"gfar_L\d+_a(-?)(\d*\.\d+)", arm):
        return f"gf_{'m' if match.group(1) else 'p'}{match.group(2)}"
    if match := re.fullmatch(r"gfar_(cap|floor)_p(\d+)_L\d+-\d+", arm):
        return f"gf_{match.group(1)}_p{match.group(2)}"
    return arm


def kind(arm: str) -> str:
    return "cap" if "_cap_" in arm else "floor" if "_floor_" in arm else "steering"


def dose(arm: str) -> float:
    sign, value = re.fullmatch(r"g[if]_([pm])(\d*\.\d+)", arm).groups()
    return float(value) * (1 if sign == "p" else -1)


def percentile(arm: str) -> int:
    return int(re.search(r"_p(\d+)$", arm).group(1))


def label(arm: str) -> str:
    if kind(arm) == "steering":
        return f"{'+' if dose(arm) > 0 else '−'}{abs(dose(arm)):g}"
    return f"{kind(arm)} p{percentile(arm)}"


def correct_by_item(root: Path, arm: str, task: str) -> dict[str, int]:
    lines = (root / arm / f"{task}.jsonl").read_text().splitlines()
    return {record["id"]: int(record["correct"]) for record in map(json.loads, (line for line in lines if line.strip()))}


def geometric_mean(accuracies: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Geometric mean over the last axis (0 if any accuracy is 0)."""
    return np.exp(np.log(np.clip(accuracies, 1e-12, None)).mean(axis=-1)) * (accuracies > 0).all(axis=-1)


def plain_mean(accuracies: np.ndarray, counts: np.ndarray) -> np.ndarray:
    return accuracies.mean(axis=-1)


def pooled(accuracies: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Share of all items answered correctly: the accuracies weighted by item count."""
    return (accuracies * counts).sum(axis=-1) / counts.sum()


COMPOSITES = {"pooled": (pooled, "share of all {items:,} benchmark items answered correctly"),
              "mean": (plain_mean, "mean accuracy over the four benchmarks"),
              "geometric": (geometric_mean, "geometric mean of the four benchmark accuracies")}


def composite(root: Path, arm: str, combine) -> dict:
    """Benchmark composite of the arm and of unsteered on the shared items, and the paired bootstrap CI of the difference."""
    rng = np.random.default_rng(BOOT_SEED)
    accuracy, arm_draws, base_draws = {}, [], []
    for task in TASKS:
        base, steered = correct_by_item(root, "unsteered", task), correct_by_item(root, arm, task)
        ids = sorted(set(base) & set(steered))
        if len(ids) != len(base) or len(ids) != len(steered):
            raise ValueError(f"{arm} / {task}: {len(steered)} items against {len(base)} unsteered, {len(ids)} shared")
        b, a = np.array([base[i] for i in ids]), np.array([steered[i] for i in ids])
        draw = rng.integers(0, len(ids), size=(BOOT_DRAWS, len(ids)))
        accuracy[task] = dict(n=len(ids), unsteered=float(b.mean()), arm=float(a.mean()))
        arm_draws.append(a[draw].mean(axis=1))
        base_draws.append(b[draw].mean(axis=1))
    counts = np.array([accuracy[t]["n"] for t in TASKS])
    gm_arm = float(combine(np.array([accuracy[t]["arm"] for t in TASKS]), counts))
    gm_base = float(combine(np.array([accuracy[t]["unsteered"] for t in TASKS]), counts))
    diffs = combine(np.stack(arm_draws, axis=1), counts) - combine(np.stack(base_draws, axis=1), counts)
    low, high = np.percentile(diffs, [2.5, 97.5])
    return dict(accuracy=accuracy, composite=100 * gm_arm, composite_unsteered=100 * gm_base, diff=100 * (gm_arm - gm_base),
                diff_ci=[100 * float(low), 100 * float(high)])


def hidden_pass_diffs(analysis: dict, axis: dict, arms: list[str]) -> dict[str, float | None]:
    """Δ hidden-test pass over all episodes vs unsteered, in points. From the analysis file's `hidden_pass`; an analysis file written
    with the 2026-11-07 default has none, and the rate then comes from the run summary (pilot_summary.py): hidden pass among non-proxy
    episodes × share of non-proxy episodes (every hidden pass is a non-proxy episode; pooled = problem mean on a balanced arm)."""
    if "hidden_pass" in analysis:
        return {arm: 100 * analysis["hidden_pass"]["arms"][arm]["diff_vs_unsteered"] for arm in arms}
    if "summary" not in axis:  # neither source: the dose-rule status of these arms is not computed here (None in the JSON)
        return {arm: None for arm in arms}
    summary = json.loads(axis["summary"].read_text())
    rate = {arm: 100 * summary[arm]["honest_hidden_pass_rate"] * (1 - summary[arm]["shown_pass_hidden_fail_rate"]) for arm in arms + ["unsteered"]}
    return {arm: rate[arm] - rate["unsteered"] for arm in arms}


def relative_change(judged: dict, arm: str) -> dict:
    """Change in the judged hack rate as % of the unsteered rate (-100 = no judged hack left), with a paired bootstrap over
    problems of the ratio (both rates recomputed in every draw). Not defined when the unsteered rate is zero."""
    problems = sorted(judged["arms"]["unsteered"]["per_problem_rate"])
    base = np.array([judged["arms"]["unsteered"]["per_problem_rate"][p] for p in problems])
    rate = np.array([judged["arms"][arm]["per_problem_rate"][p] for p in problems])
    if base.mean() == 0:
        return dict(hack_rel=None, hack_rel_ci=None)
    draw = np.random.default_rng(BOOT_SEED).integers(0, len(problems), size=(BOOT_DRAWS, len(problems)))
    base_draws, rate_draws = base[draw].mean(axis=1), rate[draw].mean(axis=1)
    defined = base_draws > 0
    low, high = np.percentile(100 * (rate_draws[defined] / base_draws[defined] - 1), [2.5, 97.5])
    return dict(hack_rel=100 * float(rate.mean() / base.mean() - 1), hack_rel_ci=[float(low), float(high)],
                hack_rel_draws_undefined=int((~defined).sum()))


def load_axis(model: dict, axis: dict, combine) -> dict:
    analysis = json.loads(axis["analysis"].read_text())
    judged, no_code = analysis["judged_hack"], analysis["no_code"]
    source_arms = [arm for arm in judged["arms"] if canonical(arm).startswith(axis["prefix"])]
    rows, omitted, hidden_diffs = {}, [], hidden_pass_diffs(analysis, axis, source_arms)
    for source in source_arms:
        arm = canonical(source)
        if not (model["capeval"] / arm).is_dir():
            omitted.append(arm)
            continue
        bench = composite(model["capeval"], arm, combine)
        contrast = judged["contrasts"][source]
        hidden_diff = hidden_diffs[source]
        no_code_diff = 100 * no_code["arms"][source]["diff_vs_unsteered"]
        measures = [("no_code", no_code_diff)] + ([("hidden_pass", hidden_diff)] if hidden_diff is not None else [])
        rows[arm] = dict(kind=kind(arm), label=label(arm), analysis_arm=source, hack=100 * judged["arms"][source]["rate"],
                         hack_diff=100 * contrast["diff"], hack_diff_ci=[100 * contrast["ci95"][0], 100 * contrast["ci95"][1]],
                         episodes=judged["arms"][source]["episodes"], hidden_pass_diff=hidden_diff, no_code_diff=no_code_diff,
                         inside_dose_rule=None if hidden_diff is None else all(abs(value) <= RULE_POINTS for _, value in measures),
                         outside_on=[name for name, value in measures if abs(value) > RULE_POINTS],
                         stops_submitting=no_code_diff > RULE_POINTS, **relative_change(judged, source), **bench)
    unsteered_composite = {round(row["composite_unsteered"], 9) for row in rows.values()}
    assert len(unsteered_composite) == 1, unsteered_composite
    unsteered = dict(hack=100 * judged["arms"]["unsteered"]["rate"], episodes=judged["arms"]["unsteered"]["episodes"],
                     composite=next(iter(rows.values()))["composite_unsteered"],
                     accuracy={t: next(iter(rows.values()))["accuracy"][t]["unsteered"] for t in TASKS})
    items = sum(row["n"] for row in next(iter(rows.values()))["accuracy"].values())
    return dict(unsteered=unsteered, arms=rows, omitted_no_benchmark_run=omitted, analysis=str(axis["analysis"]), problems=judged["problems"],
                seeds=judged["seeds"], items=items)


def series_order(rows: dict) -> dict[str, list[list[str]]]:
    """Per series, the polylines to draw, each a list of arms starting next to the unsteered point."""
    steering = [arm for arm, row in rows.items() if row["kind"] == "steering"]
    negative = sorted((arm for arm in steering if dose(arm) < 0), key=dose, reverse=True)
    positive = sorted((arm for arm in steering if dose(arm) > 0), key=dose)
    caps = sorted((arm for arm, row in rows.items() if row["kind"] == "cap"), key=percentile, reverse=True)
    floors = sorted((arm for arm, row in rows.items() if row["kind"] == "floor"), key=percentile)
    return {"steering": [negative, positive], "cap": [caps], "floor": [floors]}


LABEL_SIZE = 8.5
# Preset label offsets for --overlay, keyed by the tuple of drawn pair keys, then pair key, then arm (points, to the near text edge).
OVERLAY_OFFSETS: dict[tuple[str, ...], dict[str, dict[str, tuple[int, int]]]] = {}
DEFAULT_OFFSETS = {"steering": (-9, 6), "cap": (10, -5), "floor": (10, 5)}
Y_KEYS = {"points": ("hack_diff", "hack_diff_ci"), "relative": ("hack_rel", "hack_rel_ci")}


def label_box(text: str) -> tuple[float, float]:
    """Approximate width and height of a label in points."""
    return 0.5 * LABEL_SIZE * len(text) + 4, LABEL_SIZE + 3.5


def draw_label(axes: Axes, text: str, x: float, y: float, offset: tuple[float, float], centered: bool, leader_color: str = AXIS,
               color: str = INK2) -> None:
    """`offset` in points: to the near edge of the text (centered=False, preset offsets) or to its centre (placed labels)."""
    dx, dy = offset
    width, height = label_box(text)
    gap = math.hypot(dx, dy) if not centered else math.hypot(max(abs(dx) - width / 2, 0), max(abs(dy) - height / 2, 0))
    leader = dict(arrowstyle="-", color=leader_color, linewidth=0.8, shrinkA=0, shrinkB=5) if gap > (16 if leader_color == AXIS else 11) else None
    ha = "center" if centered or dx == 0 else "left" if dx > 0 else "right"
    axes.annotate(text, (x, y), textcoords="offset points", xytext=offset, fontsize=LABEL_SIZE, color=color, zorder=6, ha=ha, va="center",
                  arrowprops=leader, bbox={"boxstyle": "round,pad=0.12", "facecolor": SURFACE, "edgecolor": "none", "alpha": 0.85})


def overlap(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def place_labels(axes: Axes, labels: list[dict], polylines: list[list[tuple[float, float]]]) -> None:
    """Draw the labels. One with a preset `offset` goes there; the others are placed greedily, most crowded point first, at the
    candidate position (16 directions x 7 distances) that least overlaps markers, lines and the labels already placed."""
    figure = axes.figure
    figure.canvas.draw()
    scale = 72 / figure.dpi
    def points(x: float, y: float) -> np.ndarray:
        return axes.transData.transform((x, y)) * scale
    left, bottom, right, top = (value * scale for value in axes.bbox.extents)
    position = [points(item["x"], item["y"]) for item in labels]
    obstacles = [((px - 5.5, py - 5.5, px + 5.5, py + 5.5), 4.0) for (px, py), item in zip(position, labels) if not item.get("off_scale")]
    for polyline in polylines:
        vertices = [points(x, y) for x, y in polyline]
        for start, end in zip(vertices, vertices[1:]):
            for t in np.linspace(0, 1, max(int(np.hypot(*(end - start)) / 4), 2)):
                px, py = start + t * (end - start)
                obstacles.append(((px - 1.5, py - 1.5, px + 1.5, py + 1.5), 1.0))
    placed: list[tuple[float, float, float, float]] = []

    def box(index: int, centre: np.ndarray) -> tuple[float, float, float, float]:
        width, height = label_box(labels[index]["text"])
        return (centre[0] - width / 2, centre[1] - height / 2, centre[0] + width / 2, centre[1] + height / 2)

    for index, item in enumerate(labels):
        if item.get("offset") is not None:
            dx, dy = item["offset"]
            width, _ = label_box(item["text"])
            placed.append(box(index, position[index] + np.array([dx + (width / 2) * np.sign(dx), dy])))
            draw_label(axes, item["text"], item["x"], item["y"], item["offset"], centered=False, color=item.get("color", INK2))
    crowding = [sum(np.hypot(*(position[i] - other)) < 40 for other in position) for i in range(len(labels))]
    for index in sorted((i for i, item in enumerate(labels) if item.get("offset") is None), key=lambda i: -crowding[i]):
        width, height = label_box(labels[index]["text"])
        best = None
        for radius in (7, 11, 16, 22, 30, 40, 52):
            for step in range(16):
                angle = 2 * math.pi * step / 16
                centre = position[index] + np.array([math.cos(angle) * (radius + width / 2), math.sin(angle) * (radius + height / 2)])
                candidate = box(index, centre)
                outside = (width * height) - overlap(candidate, (left + 2, bottom + 2, right - 2, top - 2))
                gaps = [math.hypot(max(candidate[0] - px, 0, px - candidate[2]), max(candidate[1] - py, 0, py - candidate[3])) for px, py in position]
                stray = min(gap for other, gap in enumerate(gaps) if other != index) < gaps[index] + 4 if len(gaps) > 1 else False
                cost = (sum(weight * overlap(candidate, other) for other, weight in obstacles) + 4.0 * sum(overlap(candidate, other) for other in placed)
                        + 50.0 * outside + 0.6 * radius + (3.0 if radius > 16 else 0.0) + (60.0 if stray else 0.0))
                if best is None or cost < best[0]:
                    best = (cost, candidate, centre - position[index])
        placed.append(best[1])
        draw_label(axes, labels[index]["text"], labels[index]["x"], labels[index]["y"], tuple(best[2]), centered=True,
                   leader_color=labels[index].get("leader", AXIS), color=labels[index].get("color", INK2))


def shade(color: str, toward: str = INK, share: float = 0.45) -> tuple[float, ...]:
    return tuple((1 - share) * a + share * b for a, b in zip(to_rgb(color), to_rgb(toward)))


def draw_arms(axes: Axes, data: dict, y_keys: tuple[str, str], offsets: dict, group_color: str | None, auto: bool,
              y_max: float | None = None) -> tuple[list[dict], list]:
    """The lines, error bars and markers of one (model, axis) pair; returns its label requests and polylines. `group_color` None:
    colour by kind of intervention, all lines solid. Otherwise the pair's colour, steering solid and clamps dashed, and the labels
    in a dark shade of the pair's colour (a dense overlay cannot keep every label next to its marker). An arm above `y_max` is off
    the scale: its line runs out of the top of the frame and its label, with the value, sits at the top edge."""
    rows, y_key, ci_key = data["arms"], *y_keys
    labels, polylines = [], []
    text_color = shade(group_color) if group_color else INK2
    for series, groups in series_order(rows).items():
        style = SERIES[series]
        color = group_color or style["color"]
        for arms in groups:
            if not arms:
                continue
            polyline = [(0.0, 0.0)] + [(rows[a]["diff"], rows[a][y_key]) for a in arms]
            polylines.append(polyline)
            dashed = group_color is not None and series != "steering"
            axes.plot(*zip(*polyline), color=color, linewidth=1.6 if dashed else 2, zorder=3, solid_capstyle="round", solid_joinstyle="round",
                      linestyle=(0, (3.2, 2.2)) if dashed else "-")
            for arm in arms:
                row = rows[arm]
                x, y = row["diff"], row[y_key]
                axes.errorbar(x, y, xerr=[[x - row["diff_ci"][0]], [row["diff_ci"][1] - x]], yerr=[[y - row[ci_key][0]], [row[ci_key][1] - y]],
                              fmt="none", ecolor=color, elinewidth=1, alpha=0.4 if group_color is None else 0.32, zorder=2)
                face, edge = (SURFACE, color) if row["stops_submitting"] else (color, SURFACE)
                axes.plot(x, y, style["marker"], markersize=9 if style["marker"] != "o" else 8, color=face, markeredgecolor=edge,
                          markeredgewidth=1.8, zorder=4)
                if y_max is not None and y > y_max:
                    labels.append(dict(text=f"↑ {row['label']} at {y:+.0f}%", x=x, y=y_max, leader=color, color=text_color, offset=None, off_scale=True))
                    continue
                labels.append(dict(text=row["label"], x=x, y=y, leader=color if group_color else AXIS, color=text_color,
                                   offset=offsets.get(arm, None if auto else DEFAULT_OFFSETS[series])))
    return labels, polylines


def style_axes(axes: Axes) -> None:
    axes.set_facecolor(SURFACE)
    axes.axhline(0, color=ZERO, linewidth=1.1, zorder=1.5)
    axes.axvline(0, color=ZERO, linewidth=1.1, zorder=1.5)
    axes.grid(color=GRID, linewidth=0.6, zorder=0)
    axes.set_axisbelow(True)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(AXIS)
    axes.tick_params(colors=MUTED, labelsize=8.5, labelleft=True)


def set_x_range(axes: Axes, rows: list[dict]) -> None:
    axes.set_xlim(min(min(row["diff_ci"][0] for row in rows), 0) - 0.4, max(max(row["diff_ci"][1] for row in rows), 0) + 0.4)


def finish_panel(axes: Axes, labels: list[dict], polylines: list, unsteered_offset: tuple[int, int] | None, y_max: float | None) -> None:
    axes.plot(0, 0, "o", markersize=8.5, color=INK, markeredgecolor=SURFACE, markeredgewidth=1.8, zorder=5)
    low, high = axes.get_ylim()
    axes.set_ylim(low, high if y_max is None else min(high, y_max))
    place_labels(axes, labels + [dict(text="unsteered", x=0.0, y=0.0, offset=unsteered_offset, color=INK)], polylines)


def wrap(text: str, fontsize: float, width: float) -> str:
    """Wrap each line of `text` to `width` inches at about 0.58 em per character."""
    chars = int(width * 72 / (0.58 * fontsize))
    return "\n".join(textwrap.fill(line, chars) for line in text.split("\n"))


def footnote(groups: list[dict], score_name: str, relative: bool, multi_model: bool) -> str:
    sets: dict[str, list[str]] = {}
    for group in groups:
        names = sets.setdefault(f"{group['data']['problems']} problems × {group['data']['seeds']} seeds", [])
        if group["model"]["name"] not in names:
            names.append(group["model"]["name"])
    task_sets = "; ".join(f"{' and '.join(names)} {size}" if multi_model else size for size, names in sets.items())
    interval = ("95% CI of the relative change, paired bootstrap over problems (both rates redrawn)" if relative
                else "95% CI of the difference, paired bootstrap over problems")
    lines = [f"Hack rate: {task_sets} per arm, as judged; bars = {interval}.",
             f"Benchmark score = {score_name.format(items=groups[0]['data']['items'])}: GSM8K (1,319 items), MMLU-Pro (1,400), HumanEval (164), "
             f"IFBench (300), one run per arm; bars = 95% CI of the difference, paired bootstrap over items."]
    left_out = [f"{group['model']['name']} {label(arm)}" for group in groups for arm in group["data"]["omitted_no_benchmark_run"]]
    undefined = [group["model"]["name"] for group in groups if relative and group["data"]["unsteered"]["hack"] == 0]
    notes = ([f"Not drawn, no benchmark run yet: {', '.join(left_out)}."] if left_out else []) + (
        [f"Not drawn, unsteered judged hack rate is 0% so there is no relative change: {', '.join(undefined)}."] if undefined else [])
    return "\n".join(lines + ([" ".join(notes)] if notes else []))


OVERLAY_KIND_NAMES = {"steering": "steering (add α · layer norm)", "cap": "cap (clamp from above)", "floor": "floor (clamp from below)"}
DASHES = (0, (3.2, 2.2))


def shape_handles(groups: list[dict], color: str | None) -> list[Line2D]:
    """Legend entries for the kinds of intervention present; `color` None = the kind's own colour (panel layout)."""
    kinds = {row["kind"] for group in groups for row in group["data"]["arms"].values()}
    marks = dict(markersize=8, markeredgecolor=SURFACE, markeredgewidth=1.5)
    handles = [Line2D([], [], color=color or SERIES[k]["color"], marker=SERIES[k]["marker"], linewidth=2 if color is None or k == "steering" else 1.6,
                      linestyle="-" if color is None or k == "steering" else DASHES, label=SERIES[k]["name"] if color is None else OVERLAY_KIND_NAMES[k],
                      **marks) for k in SERIES if k in kinds]
    return handles + [Line2D([], [], color=INK, marker="o", linewidth=0, label="unsteered (origin)", **marks)]


def header(figure, width: float, height: float, title: str, subtitle: str, group_handles: list[Line2D], shapes: list[Line2D],
           hollow_color: str | None) -> float:
    """Title, subtitle and legends from the top of the figure down; returns the inches used."""
    at = lambda inches: 1 - inches / height
    x = 0.14 / width
    title, subtitle = wrap(title, 12.5, width - 0.3), wrap(subtitle, 9.5, width - 0.3)
    figure.text(x, at(0.12), title, fontsize=12.5, color=INK, ha="left", va="top", linespacing=1.25)
    cursor = 0.12 + 0.235 * (title.count("\n") + 1) + 0.08
    figure.text(x, at(cursor), subtitle, fontsize=9.5, color=INK2, ha="left", va="top", linespacing=1.3)
    cursor += 0.175 * (subtitle.count("\n") + 1) + 0.1
    legend_style = dict(frameon=False, fontsize=9, handletextpad=0.5, columnspacing=1.6, loc="upper left", borderaxespad=0, borderpad=0)
    if group_handles:
        columns = 1 if len(group_handles) > 2 else len(group_handles)
        figure.legend(handles=group_handles, bbox_to_anchor=(x, at(cursor)), ncol=columns, **legend_style)
        cursor += 0.215 * math.ceil(len(group_handles) / columns) + 0.07
    figure.legend(handles=shapes, bbox_to_anchor=(x, at(cursor)), ncol=len(shapes), handlelength=3.0, **legend_style)
    cursor += 0.235
    hollow = [Line2D([], [], color=SURFACE, marker=SERIES[k]["marker"], markersize=8, markeredgecolor=hollow_color or SERIES[k]["color"],
                     markeredgewidth=1.8, linewidth=0) for k in ("steering", "cap")]
    figure.legend(handles=[tuple(hollow)], labels=[f"hollow marker: the model stops submitting code (share of episodes with no code submitted "
                                                   f"more than {RULE_POINTS:g} points above unsteered)"],
                  handler_map={tuple: HandlerTuple(ndivide=None, pad=0.6)}, bbox_to_anchor=(x, at(cursor)), handlelength=2.4, **legend_style)
    return cursor + 0.22


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", nargs="+", choices=sorted(MODELS), default=["gptoss"])
    parser.add_argument("--axes", nargs="+", default=["gi_v2", "gfar"])
    parser.add_argument("--composite", choices=sorted(COMPOSITES), default="pooled")
    parser.add_argument("--y", choices=sorted(Y_KEYS), default="points", help="judged hack rate: change in points, or as %% of the unsteered rate")
    parser.add_argument("--y-max", type=float, help="top of the y axis; an arm above it is named at the top edge with its value")
    parser.add_argument("--overlay", action="store_true", help="all (model, axis) pairs on one set of axes, colour = pair")
    parser.add_argument("--shared-axes", action="store_true", help="panel layout: one x and y range for all panels (default: own range per panel)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--offsets", type=Path, help="JSON {pair key: {arm: [dx, dy]}} of label offsets in points (key: axis, or model/axis)")
    args = parser.parse_args()
    combine, score_name = COMPOSITES[args.composite]
    relative, y_keys = args.y == "relative", Y_KEYS[args.y]
    multi_model = len(args.model) > 1
    groups = []
    for model_key in args.model:
        for axis_key in args.axes:
            if axis_key not in MODELS[model_key]["axes"]:
                raise SystemExit(f"{model_key} has no preset for axis {axis_key}")
            groups.append(dict(key=f"{model_key}/{axis_key}" if multi_model else axis_key, model=MODELS[model_key], axis_key=axis_key,
                               preset=MODELS[model_key]["axes"][axis_key], data=load_axis(MODELS[model_key], MODELS[model_key]["axes"][axis_key], combine)))
    drawn = [group for group in groups if group["data"]["arms"] and not (relative and group["data"]["unsteered"]["hack"] == 0)]
    file_offsets = json.loads(args.offsets.read_text()) if args.offsets else {}
    every = [row for group in drawn for row in group["data"]["arms"].values()]
    y_label = ("Change in judged hack rate, % of the unsteered rate (−100 = none left; lower = better)" if relative
               else "Δ judged hack rate vs unsteered, points (lower = better)")
    x_label = "Δ benchmark score vs unsteered, points (higher = better)"
    label_text = "Labels: steering dose α (in layer norms) or clamp threshold (percentile of the unsteered projection)."
    what = "relative change in judged reward hacking" if relative else "change in judged reward hacking"
    levels = lambda data: f"judged hack rate {data['unsteered']['hack']:.1f}%, benchmark score {data['unsteered']['composite']:.1f}%"
    models = ", ".join(dict.fromkeys(g["model"]["name"] for g in drawn))
    axis_names = " and ".join(dict.fromkeys(AXIS_NAMES[g["axis_key"]] for g in drawn))
    subtitle = ("Each model's unsteered arm is the origin. " if multi_model else f"Unsteered: {levels(drawn[0]['data'])}. ") + label_text

    if args.overlay:
        if len(drawn) > len(GROUP_COLORS):
            raise SystemExit(f"--overlay draws at most {len(GROUP_COLORS)} pairs (validated colours); use the panel layout")
        width, plot_height, columns, rows = 10.0, 7.6, 1, 1
        title = (f"{axis_names} on {len(drawn)} models: {what} against change in benchmark score" if multi_model
                 else f"{models}, {len(drawn)} steering directions: {what} against change in benchmark score")
        for group, color in zip(drawn, GROUP_COLORS):
            group["color"] = color
        names = [f"{g['model']['name']} (unsteered: {levels(g['data'])})" if multi_model else AXIS_NAMES[g["axis_key"]] for g in drawn]
        group_handles = [Line2D([], [], color=g["color"], marker="o", linewidth=2, markersize=8, markeredgecolor=SURFACE, markeredgewidth=1.5, label=name)
                         for g, name in zip(drawn, names)]
    else:
        columns = len(drawn) if len(drawn) <= 3 else 2
        rows = math.ceil(len(drawn) / columns)
        width, plot_height = 6.5 * columns, 5.9 * rows
        title, group_handles = f"{models}: {what} against change in benchmark score, per intervention", []
    foot = wrap(footnote(groups, score_name, relative, multi_model), 7.8, width - 0.3)
    foot_height = 0.14 * (foot.count("\n") + 1) + 0.2
    # the header height depends only on line counts, so measure it on a scratch figure first
    scratch = plt.figure(figsize=(width, 10))
    head_height = header(scratch, width, 10, title, subtitle, group_handles, shape_handles(drawn, INK2 if args.overlay else None), None)
    plt.close(scratch)
    height = head_height + plot_height + foot_height
    shared = args.shared_axes and not args.overlay
    figure, grid = plt.subplots(rows, columns, figsize=(width, height), facecolor=SURFACE, sharex=shared, sharey=shared, squeeze=False)
    header(figure, width, height, title, subtitle, group_handles, shape_handles(drawn, INK2 if args.overlay else None), INK2 if args.overlay else None)
    figure.text(0.14 / width, 0.1 / height, foot, fontsize=7.8, color=MUTED, ha="left", va="bottom", linespacing=1.3)
    panels = grid.ravel()
    pending = []
    if args.overlay:
        axes = panels[0]
        style_axes(axes)
        labels, polylines = [], []
        for group in drawn:
            offsets = {**OVERLAY_OFFSETS.get(tuple(g["key"] for g in drawn), {}).get(group["key"], {}),
                       **{k: tuple(v) for k, v in file_offsets.get(group["key"], {}).items()}}
            more_labels, more_lines = draw_arms(axes, group["data"], y_keys, offsets, group["color"], auto=True, y_max=args.y_max)
            labels += more_labels
            polylines += more_lines
        set_x_range(axes, every)
        axes.set_xlabel(x_label, fontsize=9.5, color=INK2)
        axes.set_ylabel(y_label, fontsize=9.5, color=INK2)
        pending.append((axes, labels, polylines, None))
    else:
        for axes in panels[len(drawn):]:
            axes.set_visible(False)
        for axes, group in zip(panels, drawn):
            style_axes(axes)
            preset = group["preset"]
            offsets = {**preset.get("offsets", {}), **{k: tuple(v) for k, v in file_offsets.get(group["key"], {}).items()}}
            labels, polylines = draw_arms(axes, group["data"], y_keys, offsets, None, auto="offsets" not in preset, y_max=args.y_max)
            panel_title = f"{group['model']['name']}, {preset['title']}\nunsteered: {levels(group['data'])}" if multi_model else preset["title"]
            axes.set_title(panel_title, fontsize=9.4 if multi_model else 10, color=INK, loc="left", pad=8, linespacing=1.35)
            axes.set_xlabel(x_label, fontsize=9.5, color=INK2)
            if not shared or axes in grid[:, 0]:
                axes.set_ylabel(y_label, fontsize=9.5, color=INK2)
            set_x_range(axes, every if shared else list(group["data"]["arms"].values()))
            pending.append((axes, labels, polylines, offsets.get("unsteered", (10, 9) if "offsets" in preset else None)))
    figure.tight_layout(rect=(0, foot_height / height, 1, 1 - (head_height + 0.05) / height), w_pad=2.5 if not shared else 1.0, h_pad=2.0)
    for axes, labels, polylines, unsteered_offset in pending:
        finish_panel(axes, labels, polylines, unsteered_offset, args.y_max)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out, dpi=200, facecolor=SURFACE)
    figure.savefig(args.out.with_suffix(".pdf"), facecolor=SURFACE)
    record = dict(composite=args.composite, y=args.y, y_max=args.y_max, layout="overlay" if args.overlay else "panels", shared_axes=bool(shared),
                  boot_draws=BOOT_DRAWS, boot_seed=BOOT_SEED,
                  groups=[dict(key=g["key"], model=g["model"]["name"], axis=g["axis_key"], title=g["preset"]["title"], drawn=g in drawn,
                               color=g.get("color"), **g["data"]) for g in groups])
    args.out.with_suffix(".json").write_text(json.dumps(record, indent=1))
    for group in groups:
        d, u = group["data"], group["data"]["unsteered"]
        print(f"{group['model']['name']} / {group['axis_key']}: unsteered hack {u['hack']:.1f}%, composite {u['composite']:.2f}"
              + ("" if group in drawn else "  (NOT DRAWN)") + (f"  omitted, no benchmark run: {d['omitted_no_benchmark_run']}" if d["omitted_no_benchmark_run"] else ""))
        for arm, row in d["arms"].items():
            hidden = "   n/a" if row["hidden_pass_diff"] is None else f"{row['hidden_pass_diff']:+6.1f}"
            rel = "undefined" if row["hack_rel"] is None else f"{row['hack_rel']:+6.1f}% (CI {row['hack_rel_ci'][0]:+6.1f} to {row['hack_rel_ci'][1]:+6.1f})"
            print(f"  {arm:14s} hack {row['hack']:5.1f} ({row['hack_diff']:+5.1f}, CI {row['hack_diff_ci'][0]:+5.1f} to {row['hack_diff_ci'][1]:+5.1f}; rel {rel})  "
                  f"composite {row['composite']:6.2f} ({row['diff']:+5.2f}, CI {row['diff_ci'][0]:+5.2f} to {row['diff_ci'][1]:+5.2f})  "
                  f"hidden {hidden} no-code {row['no_code_diff']:+5.1f} {'OUTSIDE' if row['inside_dose_rule'] is False else ''}")
    print("->", args.out)


if __name__ == "__main__":
    main()
