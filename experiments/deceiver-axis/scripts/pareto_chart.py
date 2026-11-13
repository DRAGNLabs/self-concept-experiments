"""Trade-off ("Pareto") chart: judged hack rate on the task set against a benchmark composite, per intervention arm.

One panel per steering axis of a model (gi_v2 vector, Goodfire RH axis). Every arm is one point:
  x = benchmark composite of the arm's benchmark run (capability_eval_vllm.py output, <capeval root>/<arm>/<task>.jsonl) over
      GSM8K, MMLU-Pro, HumanEval and IFBench, in %. --composite pooled (default): the share of all benchmark items answered
      correctly (3,183 items; each benchmark weighted by its item count). --composite mean: the plain mean of the four
      accuracies. --composite geometric: their geometric mean. On the gpt-oss arms IFBench (300 items, 53% unsteered) carries
      83% of the sampling variance of the geometric mean and 63% of the plain mean's, against 27% of the pooled score's
      (figures/pareto/composite_options.py, 2026-11-13), and the pooled CI is less than half as wide;
  y = judged hack rate (as judged; main_analysis.py output, `judged_hack` of the analysis file named in the preset), in %.
Both axes are drawn as the arm's difference from unsteered, in points, so the unsteered arm is the origin (its levels are in
the subtitle). Three series, each a line from the origin outwards in dose order: steering (most negative dose ... unsteered ...
most positive dose), capping (weakest rung first: highest percentile threshold) and flooring (weakest rung first: lowest
percentile).

Error bars are 95% CIs of the difference: vertical = the problem-level paired bootstrap CI of the analysis file
(`contrasts[arm].ci95`); horizontal = an item-paired bootstrap of the composite (items resampled within each benchmark, the
same draw for arm and unsteered, 10,000 draws, fixed seed). An arm outside the task-set dose rule (hidden-test pass or no-code
more than 10 points from unsteered) is drawn hollow. Label offsets (points) per arm come from the preset's `offsets` or --offsets.

Writes <out>.png, <out>.pdf and <out>.json (every plotted number).

    python experiments/deceiver-axis/scripts/pareto_chart.py --model gptoss --out <path>.png [--axes gi_v2 gfar]
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

N = Path.home() / "nobackup/autodelete/self-concept-experiments/schemer-axis"
TASKS = ("gsm8k", "mmlu_pro", "humaneval", "ifbench")
BOOT_DRAWS, BOOT_SEED = 10_000, 0
RULE_POINTS = 10.0  # task-set dose rule: |Δ hidden pass| and |Δ no-code| within 10 points of unsteered

# categorical slots 1-3 of the chart palette (validated: CVD ΔE 9.2 worst pair; aqua is under 3:1 on the surface, so every
# point is direct-labelled and each series has its own marker shape)
STEER, CAP, FLOOR = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SERIES = {"steering": dict(color=STEER, marker="o", name="steering (add α · layer norm)"),
          "cap": dict(color=CAP, marker="v", name="capping (clamp projection from above)"),
          "floor": dict(color=FLOOR, marker="^", name="flooring (clamp projection from below)")}

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
}


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


COMPOSITES = {"pooled": (pooled, "Δ benchmark score vs unsteered (points; share of all {items:,} benchmark items correct)"),
              "mean": (plain_mean, "Δ benchmark score vs unsteered (points; mean accuracy of the four benchmarks)"),
              "geometric": (geometric_mean, "Δ benchmark score vs unsteered (points; geometric mean of the four accuracies)")}


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


def hidden_pass_diffs(analysis: dict, axis: dict, arms: list[str]) -> dict[str, float]:
    """Δ hidden-test pass over all episodes vs unsteered, in points. From the analysis file's `hidden_pass`; an analysis file written
    with the 2026-11-07 default has none, and the rate then comes from the run summary (pilot_summary.py): hidden pass among non-proxy
    episodes × share of non-proxy episodes (every hidden pass is a non-proxy episode; pooled = problem mean on a balanced arm)."""
    if "hidden_pass" in analysis:
        return {arm: 100 * analysis["hidden_pass"]["arms"][arm]["diff_vs_unsteered"] for arm in arms}
    summary = json.loads(axis["summary"].read_text())
    rate = {arm: 100 * summary[arm]["honest_hidden_pass_rate"] * (1 - summary[arm]["shown_pass_hidden_fail_rate"]) for arm in arms + ["unsteered"]}
    return {arm: rate[arm] - rate["unsteered"] for arm in arms}


def load_axis(model: dict, axis: dict, combine) -> dict:
    analysis = json.loads(axis["analysis"].read_text())
    judged, no_code = analysis["judged_hack"], analysis["no_code"]
    arms = [arm for arm in judged["arms"] if arm.startswith(axis["prefix"])]
    rows, hidden_diffs = {}, hidden_pass_diffs(analysis, axis, arms)
    for arm in arms:
        if not (model["capeval"] / arm).is_dir():
            raise FileNotFoundError(f"no benchmark run for {arm} in {model['capeval']}")
        bench = composite(model["capeval"], arm, combine)
        contrast = judged["contrasts"][arm]
        hidden_diff = hidden_diffs[arm]
        no_code_diff = 100 * no_code["arms"][arm]["diff_vs_unsteered"]
        rows[arm] = dict(kind=kind(arm), label=label(arm), hack=100 * judged["arms"][arm]["rate"], hack_diff=100 * contrast["diff"],
                         hack_diff_ci=[100 * contrast["ci95"][0], 100 * contrast["ci95"][1]], episodes=judged["arms"][arm]["episodes"],
                         hidden_pass_diff=hidden_diff, no_code_diff=no_code_diff,
                         inside_dose_rule=abs(hidden_diff) <= RULE_POINTS and abs(no_code_diff) <= RULE_POINTS, **bench)
    unsteered_composite = {round(row["composite_unsteered"], 9) for row in rows.values()}
    assert len(unsteered_composite) == 1, unsteered_composite
    unsteered = dict(hack=100 * judged["arms"]["unsteered"]["rate"], episodes=judged["arms"]["unsteered"]["episodes"],
                     composite=next(iter(rows.values()))["composite_unsteered"],
                     accuracy={t: next(iter(rows.values()))["accuracy"][t]["unsteered"] for t in TASKS})
    items = sum(row["n"] for row in next(iter(rows.values()))["accuracy"].values())
    return dict(unsteered=unsteered, arms=rows, analysis=str(axis["analysis"]), problems=judged["problems"], seeds=judged["seeds"], items=items)


def series_order(rows: dict) -> dict[str, list[list[str]]]:
    """Per series, the polylines to draw, each a list of arms starting next to the unsteered point."""
    steering = [arm for arm, row in rows.items() if row["kind"] == "steering"]
    negative = sorted((arm for arm in steering if dose(arm) < 0), key=dose, reverse=True)
    positive = sorted((arm for arm in steering if dose(arm) > 0), key=dose)
    caps = sorted((arm for arm, row in rows.items() if row["kind"] == "cap"), key=percentile, reverse=True)
    floors = sorted((arm for arm, row in rows.items() if row["kind"] == "floor"), key=percentile)
    return {"steering": [negative, positive], "cap": [caps], "floor": [floors]}


def annotate(axes: Axes, text: str, x: float, y: float, offset: tuple[int, int], color: str = INK2) -> None:
    dx, dy = offset
    leader = dict(arrowstyle="-", color=AXIS, linewidth=0.7, shrinkA=0, shrinkB=5) if math.hypot(dx, dy) > 18 else None
    axes.annotate(text, (x, y), textcoords="offset points", xytext=offset, fontsize=8.5, color=color, zorder=6,
                  ha="left" if dx > 0 else "right" if dx < 0 else "center", va="center", arrowprops=leader,
                  bbox={"boxstyle": "round,pad=0.12", "facecolor": SURFACE, "edgecolor": "none", "alpha": 0.85})


DEFAULT_OFFSETS = {"steering": (-9, 6), "cap": (10, -5), "floor": (10, 5)}


def draw_panel(axes: Axes, data: dict, title: str, offsets: dict[str, tuple[int, int]]) -> None:
    """Both axes are differences from unsteered, in points; the unsteered arm is the origin."""
    rows = data["arms"]
    axes.axhline(0, color=AXIS, linewidth=1, zorder=1)
    axes.axvline(0, color=AXIS, linewidth=1, zorder=1)
    for series, polylines in series_order(rows).items():
        style = SERIES[series]
        for arms in polylines:
            if not arms:
                continue
            axes.plot([0.0] + [rows[a]["diff"] for a in arms], [0.0] + [rows[a]["hack_diff"] for a in arms], color=style["color"],
                      linewidth=2, zorder=3, solid_capstyle="round", solid_joinstyle="round")
            for arm in arms:
                row = rows[arm]
                x, y = row["diff"], row["hack_diff"]
                axes.errorbar(x, y, xerr=[[x - row["diff_ci"][0]], [row["diff_ci"][1] - x]],
                              yerr=[[y - row["hack_diff_ci"][0]], [row["hack_diff_ci"][1] - y]],
                              fmt="none", ecolor=style["color"], elinewidth=1, alpha=0.4, zorder=2)
                face = style["color"] if row["inside_dose_rule"] else SURFACE
                edge = SURFACE if row["inside_dose_rule"] else style["color"]
                axes.plot(x, y, style["marker"], markersize=9 if style["marker"] != "o" else 8, color=face, markeredgecolor=edge,
                          markeredgewidth=1.8, zorder=4)
                annotate(axes, row["label"], x, y, offsets.get(arm, DEFAULT_OFFSETS[series]))
    axes.plot(0, 0, "o", markersize=8.5, color=INK, markeredgecolor=SURFACE, markeredgewidth=1.8, zorder=5)
    annotate(axes, "unsteered", 0, 0, offsets.get("unsteered", (10, 9)), color=INK)
    axes.set_title(title, fontsize=10, color=INK, loc="left", pad=8)
    axes.grid(color=GRID, linewidth=0.6, zorder=0)
    axes.set_axisbelow(True)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(AXIS)
    axes.tick_params(colors=MUTED, labelsize=8.5)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=sorted(MODELS), default="gptoss")
    parser.add_argument("--axes", nargs="+", default=["gi_v2", "gfar"])
    parser.add_argument("--composite", choices=sorted(COMPOSITES), default="pooled")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--offsets", type=Path, help="JSON {axis: {arm: [dx, dy]}} of label offsets in points")
    args = parser.parse_args()
    model = MODELS[args.model]
    combine, x_label = COMPOSITES[args.composite]
    data = {axis: load_axis(model, model["axes"][axis], combine) for axis in args.axes}
    offsets = json.loads(args.offsets.read_text()) if args.offsets else {}

    figure, axes_row = plt.subplots(1, len(args.axes), figsize=(6.5 * len(args.axes), 7.0), facecolor=SURFACE, sharex=True, sharey=True,
                                    squeeze=False)
    for axes, axis in zip(axes_row[0], args.axes):
        axes.set_facecolor(SURFACE)
        preset = model["axes"][axis]
        draw_panel(axes, data[axis], preset["title"], {**preset.get("offsets", {}), **{k: tuple(v) for k, v in offsets.get(axis, {}).items()}})
        axes.set_xlabel(x_label.format(items=data[axis]["items"]) + "  →  better", fontsize=9, color=INK2)
        axes.tick_params(labelleft=True)
    axes_row[0][0].set_ylabel("Δ judged hack rate on the task set vs unsteered (points)  ↓  better", fontsize=9.5, color=INK2)
    every = [row for d in data.values() for row in d["arms"].values()]
    axes_row[0][0].set_xlim(min(row["diff_ci"][0] for row in every) - 0.4, max(row["diff_ci"][1] for row in every) + 0.4)
    handles = [Line2D([], [], color=s["color"], marker=s["marker"], markersize=8, markeredgecolor=SURFACE, markeredgewidth=1.5, linewidth=2,
                      label=s["name"]) for s in SERIES.values()]
    handles.append(Line2D([], [], color=INK, marker="o", markersize=8, markeredgecolor=SURFACE, markeredgewidth=1.5, linewidth=0, label="unsteered"))
    handles.append(Line2D([], [], color=SURFACE, marker="o", markersize=8, markeredgecolor=INK2, markeredgewidth=1.8, linewidth=0,
                          label="hollow: outside the task-set dose rule"))
    figure.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.012, 0.915), ncol=len(handles), frameon=False, fontsize=9,
                  handletextpad=0.5, columnspacing=1.6)
    first = next(iter(data.values()))
    figure.suptitle(f"{model['name']}: change in judged reward hacking against change in benchmark score, per intervention", x=0.015, y=0.985,
                    ha="left", fontsize=12.5, color=INK)
    figure.text(0.015, 0.932, f"Unsteered: judged hack rate {first['unsteered']['hack']:.1f}%, benchmark score {first['unsteered']['composite']:.1f}%. "
                "Labels: steering dose α (in layer norms) or clamp threshold (percentile of the unsteered projection).", fontsize=9.5,
                color=INK2, ha="left", va="bottom")
    figure.text(0.015, 0.012, f"Hack rate: {first['problems']} problems × {first['seeds']} seeds per arm, as judged; bars = 95% CI of the difference, "
                f"paired bootstrap over problems.\nBenchmark score: one run per arm on GSM8K (1,319 items), MMLU-Pro (1,400), HumanEval (164) and "
                f"IFBench (300); bars = 95% CI of the difference, paired bootstrap over items.", fontsize=7.8, color=MUTED, ha="left", va="bottom")
    figure.tight_layout(rect=(0, 0.05, 1, 0.885))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out, dpi=200, facecolor=SURFACE)
    figure.savefig(args.out.with_suffix(".pdf"), facecolor=SURFACE)
    args.out.with_suffix(".json").write_text(json.dumps(dict(model=model["name"], composite=args.composite, boot_draws=BOOT_DRAWS, boot_seed=BOOT_SEED, axes=data), indent=1))
    for axis, d in data.items():
        u = d["unsteered"]
        print(f"{axis}: unsteered hack {u['hack']:.1f}%, composite {u['composite']:.2f}")
        for arm, row in d["arms"].items():
            print(f"  {arm:14s} hack {row['hack']:5.1f} ({row['hack_diff']:+5.1f}, CI {row['hack_diff_ci'][0]:+5.1f} to {row['hack_diff_ci'][1]:+5.1f})  "
                  f"composite {row['composite']:6.2f} ({row['diff']:+5.2f}, CI {row['diff_ci'][0]:+5.2f} to {row['diff_ci'][1]:+5.2f})  "
                  f"hidden {row['hidden_pass_diff']:+5.1f} no-code {row['no_code_diff']:+5.1f} {'' if row['inside_dose_rule'] else 'OUTSIDE'}")
    print("->", args.out)


if __name__ == "__main__":
    main()
