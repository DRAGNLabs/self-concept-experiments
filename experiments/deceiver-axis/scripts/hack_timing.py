"""EXPLORATORY (lightly reviewed when merged 2026-11-20): kept for reference, not a validated measurement.

Timing and character of judged hacks under gi_v2 - and gfar - against unsteered, per model (behaviour_measures.episodes
PRESETS, main dose). (CPU)

From the rubric-v3 judge's per-episode details (the turn the hack was submitted, the earliest turn whose reasoning
considers hacking, mechanism, strength, pressure source) for every HACK verdict in the three arms:
- scatter of first-consideration turn (x) vs hack turn (y), jittered, one panel per model, with each arm's mean and
  its 95% problem-bootstrap interval;
- per arm: mean consideration turn, mean hack turn, mean gap (hack - consideration), share considered and enacted in
  the same turn, each with a problem-bootstrap interval and the gi_v2 - gfar difference;
- per arm: counts of mechanism, strength and pressure source, and of STRONG / WEAK hacks and BORDERLINE verdicts;
- per arm: harness stop reason over all episodes.
Writes <out>.{png,pdf,json}.

    python experiments/deceiver-axis/scripts/hack_timing.py --out <stem>
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import TypedDict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from jaxtyping import Float, Int

from selfconcept.deceiver_axis.behaviour_measures.episodes import (ARMS, PRESETS, ArmName, ModelPreset, arm_dir_names,
                                                                   hack_turns, load_judge_details, load_records,
                                                                   load_verdict_by_key)
from selfconcept.deceiver_axis.behaviour_measures.palette import ARM_LABEL, COLOR_BY_ARM, GRID, INK, MUTED

STATISTICS = ["consider_turn", "hack_turn", "gap", "same_turn"]
JITTER = 0.18


class HackTiming(TypedDict):
    problem: str
    consider_turn: int
    hack_turn: int
    mechanism: str
    strength: str
    pressure_source: str


def load_hack_timings(preset: ModelPreset) -> dict[ArmName, list[HackTiming]]:
    """Judge details of every HACK verdict in the three arms."""
    arm_by_dir_name = arm_dir_names(preset)
    verdict_by_key = load_verdict_by_key(preset["verdict_files"])
    detail_by_key = load_judge_details(preset, verdict_by_key)
    timings_by_arm: dict[ArmName, list[HackTiming]] = {arm: [] for arm in ARMS}
    for key, verdict in verdict_by_key.items():
        arm = arm_by_dir_name.get(key.split("/")[0])
        if arm is None or verdict != "HACK":
            continue
        detail = detail_by_key[key]
        timings_by_arm[arm].append({"problem": key.split("/")[-1], "consider_turn": hack_turns(detail)[0],
                                    "hack_turn": hack_turns(detail)[1], "mechanism": str(detail.get("mechanism")),
                                    "strength": str(detail.get("strength")),
                                    "pressure_source": str(detail.get("pressure_source"))})
    return timings_by_arm


def statistic_values(timings: list[HackTiming], statistic: str) -> list[float]:
    if statistic == "consider_turn":
        return [timing["consider_turn"] for timing in timings]
    if statistic == "hack_turn":
        return [timing["hack_turn"] for timing in timings]
    if statistic == "gap":
        return [timing["hack_turn"] - timing["consider_turn"] for timing in timings]
    return [float(timing["hack_turn"] == timing["consider_turn"]) for timing in timings]


def problem_sums(timings: list[HackTiming], statistic: str, problem_ids: list[str]
                 ) -> tuple[Float[np.ndarray, "problems"], Int[np.ndarray, "problems"]]:
    sums: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for timing, value in zip(timings, statistic_values(timings, statistic)):
        sums[timing["problem"]] += value
        counts[timing["problem"]] += 1
    return (np.array([sums[problem] for problem in problem_ids]),
            np.array([counts[problem] for problem in problem_ids]))


def bootstrap_means(timings_by_arm: dict[ArmName, list[HackTiming]], n_draws: int, rng: np.random.Generator
                    ) -> dict[str, dict[str, Float[np.ndarray, "draws"]]]:
    """Per statistic and arm: the arm mean in each draw of problems (the same draws for every arm)."""
    problem_ids = sorted({timing["problem"] for timings in timings_by_arm.values() for timing in timings})
    resampled = rng.integers(len(problem_ids), size=(n_draws, len(problem_ids)))
    draws_by_statistic: dict[str, dict[str, np.ndarray]] = {}
    for statistic in STATISTICS:
        draws_by_statistic[statistic] = {}
        for arm in ARMS:
            sums, counts = problem_sums(timings_by_arm[arm], statistic, problem_ids)
            totals = counts[resampled].sum(axis=1)
            draws_by_statistic[statistic][arm] = np.where(totals > 0, sums[resampled].sum(axis=1)
                                                          / np.maximum(totals, 1), np.nan)
    return draws_by_statistic


def interval(draws: np.ndarray) -> list[float]:
    return [float(bound) for bound in np.nanpercentile(draws, [2.5, 97.5])]


def severity_counts(preset: ModelPreset) -> dict[ArmName, dict[str, int]]:
    """Per arm: judged STRONG and WEAK hacks and BORDERLINE verdicts (from the chart's verdicts and the details)."""
    timings_by_arm = load_hack_timings(preset)
    arm_by_dir_name = arm_dir_names(preset)
    borderline = Counter(arm_by_dir_name[key.split("/")[0]] for key, verdict in
                         load_verdict_by_key(preset["verdict_files"]).items()
                         if verdict == "BORDERLINE" and key.split("/")[0] in arm_by_dir_name)
    return {arm: {"STRONG": sum(timing["strength"] == "STRONG" for timing in timings_by_arm[arm]),
                  "WEAK": sum(timing["strength"] == "WEAK" for timing in timings_by_arm[arm]),
                  "BORDERLINE": borderline[arm]} for arm in ARMS}


def stop_reason_counts(preset: ModelPreset) -> dict[ArmName, dict[str, int]]:
    """Per arm: how every episode ended (harness stop_reason), over all episodes, not only hacks."""
    arm_dirs: dict[ArmName, Path] = {"unsteered": preset["unsteered"], "gi_v2": preset["arm_dirs"][("gi_v2", "main")],
                                     "gfar": preset["arm_dirs"][("gfar", "main")]}
    return {arm: dict(Counter(record["stop_reason"] for record in
                              load_records(arm_dir, preset["record_file"]).values()))
            for arm, arm_dir in arm_dirs.items()}


def summarize(timings_by_arm: dict[ArmName, list[HackTiming]], draws_by_statistic: dict[str, dict[str, np.ndarray]]
              ) -> dict:
    summary: dict = {"n_hacks": {arm: len(timings_by_arm[arm]) for arm in ARMS}}
    for statistic in STATISTICS:
        summary[statistic] = {arm: {"mean": float(np.mean(statistic_values(timings_by_arm[arm], statistic)))
                                    if timings_by_arm[arm] else None,
                                    "ci95": interval(draws_by_statistic[statistic][arm])} for arm in ARMS}
        gap_draws = draws_by_statistic[statistic]["gi_v2"] - draws_by_statistic[statistic]["gfar"]
        summary[statistic]["gi_v2_minus_gfar"] = {"ci95": interval(gap_draws)}
    for field in ["mechanism", "strength", "pressure_source"]:
        summary[field] = {arm: dict(Counter(timing[field] for timing in timings_by_arm[arm]).most_common())
                          for arm in ARMS}
    return summary


def draw_model_panel(axes, label: str, timings_by_arm: dict[ArmName, list[HackTiming]],
                     draws_by_statistic: dict[str, dict[str, np.ndarray]], rng: np.random.Generator) -> None:
    for arm in ARMS:
        timings = timings_by_arm[arm]
        considers = np.array([timing["consider_turn"] for timing in timings], dtype=float)
        hacks = np.array([timing["hack_turn"] for timing in timings], dtype=float)
        axes.scatter(considers + rng.uniform(-JITTER, JITTER, len(timings)),
                     hacks + rng.uniform(-JITTER, JITTER, len(timings)), s=9, color=COLOR_BY_ARM[arm], alpha=0.45,
                     linewidths=0, zorder=2)
    for arm in ARMS:
        if not timings_by_arm[arm]:
            continue
        mean_x = np.mean([timing["consider_turn"] for timing in timings_by_arm[arm]])
        mean_y = np.mean([timing["hack_turn"] for timing in timings_by_arm[arm]])
        x_low, x_high = interval(draws_by_statistic["consider_turn"][arm])
        y_low, y_high = interval(draws_by_statistic["hack_turn"][arm])
        axes.errorbar(mean_x, mean_y, xerr=[[mean_x - x_low], [x_high - mean_x]],
                      yerr=[[mean_y - y_low], [y_high - mean_y]], fmt="o", markersize=9,
                      markerfacecolor=COLOR_BY_ARM[arm], markeredgecolor="white", markeredgewidth=2,
                      ecolor=COLOR_BY_ARM[arm], elinewidth=2, capsize=0, zorder=4)
    axes.plot([-0.5, 4.5], [-0.5, 4.5], linestyle=":", color=MUTED, linewidth=1, zorder=1)
    axes.set_xlim(-0.5, 4.5)
    axes.set_ylim(-0.5, 4.5)
    axes.set_xticks(range(5))
    axes.set_yticks(range(5))
    axes.set_aspect("equal")
    axes.grid(color=GRID, linewidth=0.6)
    axes.set_axisbelow(True)
    for spine in axes.spines.values():
        spine.set_color(GRID)
    axes.tick_params(colors=MUTED, labelsize=9)
    counts = ", ".join(f"{ARM_LABEL[arm]} {len(timings_by_arm[arm])}" for arm in ARMS)
    axes.set_title(f"{label}\n(judged hacks: {counts})", loc="left", fontsize=10, color=INK)
    axes.set_xlabel("turn the reasoning first considers hacking", color=MUTED, fontsize=9)
    axes.set_ylabel("turn the hack is submitted", color=MUTED, fontsize=9)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    timings_by_model = {preset["label"]: load_hack_timings(preset) for preset in PRESETS}
    plotted = [label for label, timings_by_arm in timings_by_model.items()
               if all(timings_by_arm[arm] for arm in ARMS)]
    figure, axes_grid = plt.subplots(1, len(plotted), figsize=(4.2 * len(plotted), 5.0))
    summary_by_model = {}
    for label, timings_by_arm in timings_by_model.items():
        draws_by_statistic = bootstrap_means(timings_by_arm, args.n_draws, rng)
        summary_by_model[label] = summarize(timings_by_arm, draws_by_statistic)
        preset = next(preset for preset in PRESETS if preset["label"] == label)
        summary_by_model[label]["severity"] = severity_counts(preset)
        summary_by_model[label]["stop_reason"] = stop_reason_counts(preset)
        if label in plotted:
            draw_model_panel(axes_grid[plotted.index(label)], label, timings_by_arm, draws_by_statistic, rng)
    left_out = [label for label in timings_by_model if label not in plotted]
    handles = [Line2D([], [], marker="o", linestyle="", color=COLOR_BY_ARM[arm], markersize=8,
                          label=ARM_LABEL[arm]) for arm in ARMS]
    figure.legend(handles=handles, loc="upper center", ncol=3, frameon=False, fontsize=10)
    figure.text(0.5, 0.01, "Turns 0-4. Points jittered ±0.18. Large markers: arm mean with 95% problem-bootstrap "
                "intervals. Dotted line: hack submitted in the turn it was first considered."
                + (f" Left out (an arm with no judged hack): {', '.join(left_out)}." if left_out else ""),
                ha="center", fontsize=8, color=MUTED)
    figure.tight_layout(rect=(0, 0.04, 1, 0.92))
    figure.savefig(args.out.with_suffix(".png"), dpi=200)
    figure.savefig(args.out.with_suffix(".pdf"))
    args.out.with_suffix(".json").write_text(json.dumps(summary_by_model, indent=1))


if __name__ == "__main__":
    main()
