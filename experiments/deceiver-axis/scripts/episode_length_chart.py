"""Bar chart of mean reasoning tokens per episode for unsteered, gi_v2 - and gfar - per model, from token_entropy.py's
JSON (measure reasoning_tokens: mean over problems of the per-problem mean). Steered bars carry the 95% problem-
bootstrap interval of their change from unsteered, drawn around the steered value. One panel per model (scales
differ by an order of magnitude). (CPU)

    python experiments/deceiver-axis/scripts/episode_length_chart.py --entropy-json <token_entropy.json> --out <stem>
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from selfconcept.deceiver_axis.behaviour_measures.episodes import ARMS
from selfconcept.deceiver_axis.behaviour_measures.palette import ARM_LABEL, COLOR_BY_ARM, GRID, INK, MUTED
from selfconcept.deceiver_axis.behaviour_measures.turn_lengths import MeasureSummary, SummaryByModel


def draw_length_panel(axes: Axes, model: str, lengths: MeasureSummary) -> None:
    """Bars of unsteered / gi_v2 / gfar mean reasoning tokens, steered bars with their change interval."""
    values = [lengths["unsteered"], lengths["gi_v2"]["value"], lengths["gfar"]["value"]]
    lower_errors = [0.0] + [lengths[arm]["change"] - lengths[arm]["ci95"][0] for arm in ("gi_v2", "gfar")]
    upper_errors = [0.0] + [lengths[arm]["ci95"][1] - lengths[arm]["change"] for arm in ("gi_v2", "gfar")]
    positions = range(len(ARMS))
    axes.bar(positions, values, width=0.6, color=[COLOR_BY_ARM[arm] for arm in ARMS], zorder=2)
    axes.errorbar(positions[1:], values[1:], yerr=[lower_errors[1:], upper_errors[1:]], fmt="none",
                  ecolor=INK, elinewidth=1.2, capsize=3, zorder=3)
    for position, value, upper in zip(positions, values, upper_errors):
        axes.text(position, value + upper, f"{value / 1000:.1f}k", ha="center", va="bottom", fontsize=9, color=INK)
    axes.set_xticks(list(positions), [ARM_LABEL[arm] for arm in ARMS], fontsize=9, color=MUTED)
    axes.set_title(model, loc="left", fontsize=11, color=INK)
    style_axes(axes)


def style_axes(axes: Axes) -> None:
    axes.grid(axis="y", color=GRID, linewidth=0.6)
    axes.set_axisbelow(True)
    axes.tick_params(colors=MUTED, labelsize=9)
    axes.margins(y=0.12)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(GRID)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--entropy-json", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    summary_by_model: SummaryByModel = json.loads(args.entropy_json.read_text())
    figure, axes_grid = plt.subplots(1, len(summary_by_model), figsize=(3.2 * len(summary_by_model), 4.2))
    for axes, (model, summary) in zip(axes_grid, summary_by_model.items()):
        draw_length_panel(axes, model, summary["reasoning_tokens"])
    axes_grid[0].set_ylabel("reasoning tokens per episode (mean)", color=MUTED, fontsize=9)
    figure.text(0.5, 0.01, "Mean over problems of the per-problem mean, all turns' reasoning, the model's own tokenizer. "
                "Error bars: 95% problem-bootstrap interval of the change from unsteered. Main dose.",
                ha="center", fontsize=8, color=MUTED)
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    figure.savefig(args.out.with_suffix(".png"), dpi=200)
    figure.savefig(args.out.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
