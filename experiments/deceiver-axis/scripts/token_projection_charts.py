"""EXPLORATORY (lightly reviewed when merged 2026-11-20): kept for reference, not a validated measurement.

Charts from token_projection_analysis.py output (<analysis>/<model slug>.pt). (CPU)

- roc/<slug>__<direction>__arm-<arm>.png: token-level ROC curve per layer (small multiples), reasoning and answer tokens,
  pooled over problems, with the pooled AUROC;
- auroc/<slug>_<direction>.png: problem-stratified AUROC against layer, rows = segment, columns = token level and
  episode mean / median / max, one line per arm with its problem-bootstrap 95% band;
- scatter/<slug>__arm-<arm>__<segment>__<statistic>.png: per layer, each episode's gi_v2 statistic (x) against its gfar
  statistic (y) over the whole episode, judged hacks against NOT_HACK + unjudged (BORDERLINE left out), for the
  50th ... 100th percentile.

    python experiments/deceiver-axis/scripts/token_projection_charts.py --analysis <dir> --out <dir>
"""

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D

from selfconcept.deceiver_axis.behaviour_measures.episodes import ARMS
from selfconcept.deceiver_axis.behaviour_measures.palette import ARM_LABEL, COLOR_BY_ARM, GRID, INK, MUTED

SEGMENT_COLORS = {"reasoning": "#2a78d6", "answer": "#eb6834"}
HACK_COLOR, NON_HACK_COLOR = "#eb6834", "#8a8984"
DIRECTION_LABEL = {"gi_v2": "gi_v2", "gf_all": "gfar"}
COLUMNS = 10
MEASURES = ["token", "mean", "median", "max"]
MEASURE_LABEL = {"token": "token level", "mean": "episode mean", "median": "episode median", "max": "episode max"}


def layer_grid(n_layers: int, panel_size: float = 1.6):
    rows = math.ceil(n_layers / COLUMNS)
    figure, axes_grid = plt.subplots(rows, COLUMNS, figsize=(COLUMNS * panel_size, rows * panel_size + 1.0),
                                     squeeze=False)
    for axes in axes_grid.ravel()[n_layers:]:
        axes.set_visible(False)
    return figure, axes_grid.ravel()[:n_layers]


def style_small(axes, title: str) -> None:
    axes.set_title(title, fontsize=7, color=INK, pad=2)
    axes.tick_params(labelsize=5, colors=MUTED, length=2)
    for spine in axes.spines.values():
        spine.set_color(GRID)


def roc_curve(histogram: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(false positive rates, true positive rates) from a (bins, 2) histogram, thresholds from the top bin down."""
    from_top = np.cumsum(histogram[::-1], axis=0)
    totals = np.maximum(histogram.sum(0), 1)
    return np.concatenate([[0], from_top[:, 0] / totals[0]]), np.concatenate([[0], from_top[:, 1] / totals[1]])


def save_roc_figures(result: dict, out: Path) -> None:
    for direction_index, direction in enumerate(result["directions"]):
        for arm in ARMS:
            arm_result = result["arms"][arm]
            figure, axes_list = layer_grid(len(result["record_layers"]))
            for layer_index, (axes, layer) in enumerate(zip(axes_list, result["record_layers"])):
                labels = []
                for segment_index, segment in enumerate(result["segments"]):
                    histogram = arm_result["histogram"][segment_index, layer_index, direction_index]
                    if histogram[:, 1].sum() == 0 or histogram[:, 0].sum() == 0:
                        continue
                    false_positive, true_positive = roc_curve(histogram)
                    axes.plot(false_positive, true_positive, color=SEGMENT_COLORS[segment], linewidth=1)
                    labels.append(f"{segment[0]} {arm_result['token_auroc_pooled'][segment_index, layer_index, direction_index]:.2f}")
                axes.plot([0, 1], [0, 1], linestyle=":", color=MUTED, linewidth=0.6)
                axes.set_xlim(0, 1)
                axes.set_ylim(0, 1)
                axes.set_aspect("equal")
                axes.text(0.97, 0.03, "\n".join(labels), ha="right", va="bottom", fontsize=5.5, color=INK,
                          transform=axes.transAxes)
                style_small(axes, f"L{layer}")
            counts = arm_result["token_counts"]
            figure.suptitle(f"{result['model']} · {DIRECTION_LABEL.get(direction, direction)} · {ARM_LABEL[arm]}: "
                            f"token-level ROC per layer (x false-positive rate, y true-positive rate; pooled AUROC, "
                            f"r reasoning / a answer)\nhack tokens (first-considered and hack turns of HACK episodes): "
                            f"reasoning {counts[0, 0, 0, 1]:,}, answer {counts[1, 0, 0, 1]:,}; non-hack tokens (NOT_HACK "
                            f"and unjudged episodes): reasoning {counts[0, 0, 0, 0]:,}, answer {counts[1, 0, 0, 0]:,}",
                            x=0.01, ha="left", fontsize=9, color=INK)
            handles = [Line2D([], [], color=color, label=segment) for segment, color in SEGMENT_COLORS.items()]
            figure.legend(handles=handles, loc="upper right", frameon=False, fontsize=8)
            figure.tight_layout(rect=(0, 0, 1, 0.94))
            figure.savefig(out / f"{result['slug']}__{DIRECTION_LABEL.get(direction, direction)}__arm-{arm}.png", dpi=150)
            plt.close(figure)


def measure_values(arm_result: dict, measure: str) -> tuple[np.ndarray, np.ndarray]:
    if measure == "token":
        return arm_result["token_auroc_stratified"], arm_result["token_auroc_stratified_ci"]
    return arm_result["episode_auroc_stratified"][measure]


def save_auroc_figures(result: dict, out: Path) -> None:
    layers = result["record_layers"]
    for direction_index, direction in enumerate(result["directions"]):
        figure, axes_grid = plt.subplots(len(result["segments"]), len(MEASURES), figsize=(4 * len(MEASURES), 6.4),
                                         sharex=True, sharey=True, squeeze=False)
        for segment_index, segment in enumerate(result["segments"]):
            for measure_index, measure in enumerate(MEASURES):
                axes = axes_grid[segment_index, measure_index]
                for arm in ARMS:
                    point, interval = measure_values(result["arms"][arm], measure)
                    axes.fill_between(layers, interval[0, segment_index, :, direction_index],
                                      interval[1, segment_index, :, direction_index], color=COLOR_BY_ARM[arm],
                                      alpha=0.15, linewidth=0)
                    axes.plot(layers, point[segment_index, :, direction_index], color=COLOR_BY_ARM[arm], linewidth=1.6)
                axes.axhline(0.5, linestyle=":", color=MUTED, linewidth=0.8)
                axes.set_ylim(0, 1)
                axes.grid(color=GRID, linewidth=0.5)
                axes.set_axisbelow(True)
                axes.tick_params(labelsize=8, colors=MUTED)
                for spine in axes.spines.values():
                    spine.set_color(GRID)
                axes.set_title(f"{segment} · {MEASURE_LABEL[measure]}", loc="left", fontsize=9, color=INK)
                if segment_index == len(result["segments"]) - 1:
                    axes.set_xlabel("layer", fontsize=8, color=MUTED)
            axes_grid[segment_index, 0].set_ylabel("problem-stratified AUROC", fontsize=8, color=MUTED)
        handles = [Line2D([], [], color=COLOR_BY_ARM[arm], linewidth=2, label=ARM_LABEL[arm]) for arm in ARMS]
        figure.legend(handles=handles, loc="upper right", ncol=3, frameon=False, fontsize=9)
        figure.suptitle(f"{result['model']} · {DIRECTION_LABEL.get(direction, direction)} projection: problem-stratified "
                        "AUROC of judged hacks by layer (bands: 95% problem bootstrap)", x=0.01, ha="left",
                        fontsize=10, color=INK)
        figure.text(0.01, 0.005, "Token level: hack = tokens of the first-considered and hack turns of HACK episodes; "
                    "non-hack = tokens of NOT_HACK and unjudged episodes. Episode level: whole-episode statistic, HACK vs "
                    "NOT_HACK + unjudged. BORDERLINE left out.", fontsize=7, color=MUTED)
        figure.tight_layout(rect=(0, 0.03, 1, 0.93))
        figure.savefig(out / f"{result['slug']}_{DIRECTION_LABEL.get(direction, direction)}.png", dpi=170)
        plt.close(figure)


def save_scatter_figures(result: dict, out: Path) -> None:
    gi_index = result["directions"].index("gi_v2")
    gf_index = 1 - gi_index
    for arm in ARMS:
        arm_result = result["arms"][arm]
        labels = np.array(arm_result["episode_labels"])
        hack, non_hack = labels == "hack", labels == "not_hack"
        for segment_index, segment in enumerate(result["segments"]):
            for statistic_index, statistic in enumerate(result["episode_statistics"]):
                if statistic == "mean":
                    continue
                figure, axes_list = layer_grid(len(result["record_layers"]), panel_size=1.7)
                values = arm_result["episode_stats"][:, segment_index, statistic_index]
                for layer_index, (axes, layer) in enumerate(zip(axes_list, result["record_layers"])):
                    x, y = values[:, layer_index, gi_index], values[:, layer_index, gf_index]
                    axes.scatter(x[non_hack], y[non_hack], s=2, color=NON_HACK_COLOR, alpha=0.35, linewidths=0)
                    axes.scatter(x[hack], y[hack], s=4, color=HACK_COLOR, alpha=0.8, linewidths=0)
                    style_small(axes, f"L{layer}")
                handles = [Line2D([], [], marker="o", linestyle="", color=HACK_COLOR, label=f"HACK ({hack.sum()})"),
                           Line2D([], [], marker="o", linestyle="", color=NON_HACK_COLOR,
                                  label=f"NOT_HACK + unjudged ({non_hack.sum()})")]
                figure.legend(handles=handles, loc="upper right", ncol=2, frameon=False, fontsize=8)
                label = "max" if statistic == "p100" else ("median" if statistic == "p50" else f"{statistic[1:]}th percentile")
                figure.suptitle(f"{result['model']} · {ARM_LABEL[arm]} · {segment} tokens · whole-episode {label}: "
                                "gi_v2 projection (x) vs gfar projection (y), per layer", x=0.01, ha="left", fontsize=9,
                                color=INK)
                figure.tight_layout(rect=(0, 0, 1, 0.95))
                figure.savefig(out / f"{result['slug']}__arm-{arm}__{segment}__{statistic}.png", dpi=130)
                plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    for kind in ("roc", "auroc", "scatter"):
        (args.out / kind).mkdir(parents=True, exist_ok=True)
    for path in sorted(args.analysis.glob("*.pt")):
        result = torch.load(path, weights_only=False)
        save_roc_figures(result, args.out / "roc")
        save_auroc_figures(result, args.out / "auroc")
        save_scatter_figures(result, args.out / "scatter")


if __name__ == "__main__":
    main()
