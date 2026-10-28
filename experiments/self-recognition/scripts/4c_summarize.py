"""Summarize one model's stage-4 validation (CPU only): the ValidationSummary json plus two charts, AUROC by layer
and the shift in P(own) against alpha with the random-direction range.

With --run.pilot true it instead logs the pilot calibration table (per layer, condition and alpha), from which
alpha_max is read by hand: the largest |alpha| whose cells keep answer mass >= 0.8x and are not degenerate.

Both read every 4b shard of the model together.

Usage (from experiments/self-recognition):
    python scripts/4c_summarize.py --run.model Qwen/Qwen3-32B
"""

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.ticker import MaxNLocator

from selfconcept.self_recognition.reading import reading_summary, transcript_reading_summary
from selfconcept.self_recognition.schemas import (
    DEFAULT_RUN_DIR,
    ReadingSummary,
    SteeringCell,
    TranscriptReadingSummary,
    ValidationSummary,
    judgements_path,
    layer_norms_path,
    model_slug,
    read_judgements,
    read_span_projections,
    read_transcript_projections,
    reading_path,
    transcript_reading_path,
    validation_summary_path,
)
from selfconcept.self_recognition.steering_validation import (
    STEERING_CONDITIONS,
    baseline_consistency,
    pilot_rows,
    read_model_continuations,
    read_model_steering_records,
    steering_cells,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a")
RANDOM_COLOR = "#a3a29c"


@dataclass(frozen=True)
class RunConfig:
    model: str
    run_dir: Path = DEFAULT_RUN_DIR
    steering_layer: int | None = None
    max_degenerate_fraction: float = 0.1
    pilot: bool = False
    pilot_name: str = "pilot"


def log_pilot_table(run: RunConfig) -> None:
    pilot_dir = run.run_dir / run.pilot_name
    rows = pilot_rows(read_model_steering_records(pilot_dir, run.model), read_model_continuations(pilot_dir, run.model))
    logger.info("layer  condition      alpha   n   dP(own)  mass_ratio  degenerate")
    for row in rows:
        logger.info(
            "%5d  %-13s %6.2f %4d %8.3f %10.2f  %s",
            row.layer,
            row.condition,
            row.alpha,
            row.record_count,
            row.mean_own_choice_probability_shift,
            row.answer_mass_ratio,
            "-" if row.degenerate_fraction is None else f"{row.degenerate_fraction:.2f}",
        )


def style_axes(axes: Axes, title: str, x_label: str, y_label: str) -> None:
    axes.set_facecolor(SURFACE)
    axes.set_title(title, color=TEXT_PRIMARY, loc="left", fontsize=11)
    axes.set_xlabel(x_label, color=TEXT_SECONDARY)
    axes.set_ylabel(y_label, color=TEXT_SECONDARY)
    axes.tick_params(colors=TEXT_SECONDARY)
    axes.grid(color=GRID, linewidth=0.8)
    axes.set_axisbelow(True)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(GRID)


def plot_auroc_by_layer(
    reading: ReadingSummary,
    transcript_reading: TranscriptReadingSummary,
    model: str,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(figsize=(8, 4.5), facecolor=SURFACE)
    series = [
        ("Judge prompt, holdout", reading["holdout_auroc_by_layer"]),
        ("Judge prompt, train", reading["train_auroc_by_layer"]),
        ("Plain transcript, holdout", transcript_reading["holdout_auroc_by_layer"]),
    ]
    for (label, auroc_by_layer), color in zip(series, SERIES_COLORS, strict=True):
        axes.plot(range(len(auroc_by_layer)), auroc_by_layer, color=color, linewidth=2, label=label)
    axes.axhline(0.5, color=RANDOM_COLOR, linewidth=1, linestyle="--")
    axes.axvline(reading["best_layer"], color=RANDOM_COLOR, linewidth=1, linestyle=":")
    axes.annotate(f"best layer {reading['best_layer']}", (reading["best_layer"], 0.5), xytext=(4, 4),
                  textcoords="offset points", color=TEXT_SECONDARY, fontsize=8)
    style_axes(axes, f"{model}: own vs other text separation", "Decoder layer", "AUROC (own > other)")
    axes.xaxis.set_major_locator(MaxNLocator(integer=True))
    axes.legend(frameon=False, labelcolor=TEXT_SECONDARY, fontsize=8, loc="upper right")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, facecolor=SURFACE)
    plt.close(figure)


def plot_steering_shift(cells: Sequence[SteeringCell], model: str, layer: int, output_path: Path) -> None:
    """One panel per condition: the self-recognition shift with its CI, and each random direction as a gray dot."""
    figure, panels = plt.subplots(1, len(STEERING_CONDITIONS), figsize=(12, 4), sharey=True, facecolor=SURFACE)
    for axes, condition, color in zip(panels, STEERING_CONDITIONS, SERIES_COLORS, strict=True):
        self_cells = sorted(
            (cell for cell in cells if cell["condition"] == condition and cell["direction"] == "self_recognition"),
            key=lambda cell: cell["alpha"],
        )
        random_cells = [cell for cell in cells if cell["condition"] == condition and cell["direction"] == "random"]
        alphas = [cell["alpha"] for cell in self_cells]
        estimates = [cell["own_choice_probability_shift"][0] for cell in self_cells]
        axes.fill_between(
            alphas,
            [cell["own_choice_probability_shift"][1] for cell in self_cells],
            [cell["own_choice_probability_shift"][2] for cell in self_cells],
            color=color,
            alpha=0.18,
            linewidth=0,
        )
        axes.plot(alphas, estimates, color=color, linewidth=2, marker="o", markersize=5, label="Self-recognition vector")
        incoherent = [cell for cell in self_cells if not cell["coherent"]]
        axes.scatter([cell["alpha"] for cell in incoherent], [cell["own_choice_probability_shift"][0] for cell in incoherent],
                     s=80, facecolors="none", edgecolors=TEXT_PRIMARY, linewidths=1.2, label="Fails coherence", zorder=3)
        axes.scatter([cell["alpha"] for cell in random_cells], [cell["own_choice_probability_shift"][0] for cell in random_cells],
                     s=30, color=RANDOM_COLOR, label="Random directions", zorder=2)
        axes.axhline(0, color=GRID, linewidth=1)
        style_axes(axes, condition.replace("_", " "), "alpha (x layer norm)", "Shift in P(own)" if condition == STEERING_CONDITIONS[0] else "")
    panels[0].legend(frameon=False, labelcolor=TEXT_SECONDARY, fontsize=8, loc="best")
    figure.suptitle(f"{model}: steering at layer {layer}, paired against alpha = 0", color=TEXT_PRIMARY, x=0.01, ha="left")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, facecolor=SURFACE)
    plt.close(figure)


def summarize(run: RunConfig) -> ValidationSummary:
    span_records = read_span_projections(reading_path(run.run_dir, run.model))
    holdout_span_records = [record for record in span_records if record["split"] == "holdout"]
    reading = reading_summary(
        [record for record in span_records if record["split"] == "train"],
        holdout_span_records,
        json.loads(layer_norms_path(run.run_dir, run.model).read_text()),
    )
    transcript_records = read_transcript_projections(transcript_reading_path(run.run_dir, run.model))
    judgements = read_judgements(judgements_path(run.run_dir, run.model))
    transcript_reading = transcript_reading_summary(
        [record for record in transcript_records if record["split"] == "train"],
        [record for record in transcript_records if record["split"] == "holdout"],
        reading["best_layer"],
        judgements,
    )
    steering_layer = reading["best_layer"] if run.steering_layer is None else run.steering_layer
    steering_records = read_model_steering_records(run.run_dir, run.model)
    baseline_max_logprob_difference, baseline_argmax_agreement = baseline_consistency(steering_records, judgements, steering_layer)
    return {
        "model": run.model,
        "holdout_prompt_count": len({record["prompt_id"] for record in holdout_span_records}),
        "holdout_record_count": len(holdout_span_records),
        "steering_layer": steering_layer,
        "baseline_max_logprob_difference": baseline_max_logprob_difference,
        "baseline_argmax_agreement": baseline_argmax_agreement,
        "reading": reading,
        "transcript_reading": transcript_reading,
        "steering": steering_cells(
            steering_records,
            read_model_continuations(run.run_dir, run.model),
            steering_layer,
            run.max_degenerate_fraction,
        ),
    }


def log_headline(summary: ValidationSummary) -> None:
    reading = summary["reading"]
    transcript_reading = summary["transcript_reading"]
    logger.info(
        "reading A @ layer %d: holdout AUROC %s (random 95th pct %.3f), probe paired accuracy %s vs model %.3f",
        reading["best_layer"], reading["holdout_auroc"], reading["random_auroc_95th_percentile"],
        reading["probe_paired_accuracy"], reading["model_paired_accuracy"],
    )
    logger.info(
        "reading B @ layer %d: holdout AUROC %s (random 95th pct %.3f), paired difference %s, length-adjusted %s",
        transcript_reading["best_layer"], transcript_reading["holdout_auroc"], transcript_reading["random_auroc_95th_percentile"],
        transcript_reading["paired_difference"], transcript_reading["length_adjusted_own_coefficient"],
    )
    logger.info(
        "alpha=0 vs stage 2: max |dlogprob| %.3f, argmax agreement %.3f",
        summary["baseline_max_logprob_difference"], summary["baseline_argmax_agreement"],
    )
    for cell in summary["steering"]:
        if cell["direction"] == "self_recognition":
            logger.info(
                "%-13s alpha %+.2f: dP(own) %s, self-attribution %s, mass ratio %.2f, coherent %s, wilcoxon p %.2g",
                cell["condition"], cell["alpha"], cell["own_choice_probability_shift"], cell["self_attribution_rate"],
                cell["answer_mass_ratio"], cell["coherent"], cell["wilcoxon_p_value"],
            )


def main(run: RunConfig) -> None:
    logger.info("%s", run)
    if run.pilot:
        log_pilot_table(run)
        return
    summary = summarize(run)
    output_path = validation_summary_path(run.run_dir, run.model)
    output_path.write_text(json.dumps(summary, indent=2))
    logger.info("wrote %s", output_path)
    log_headline(summary)
    chart_prefix = output_path.parent / model_slug(run.model)
    plot_auroc_by_layer(summary["reading"], summary["transcript_reading"], run.model, chart_prefix.with_name(f"{chart_prefix.name}_auroc_by_layer.png"))
    plot_steering_shift(summary["steering"], run.model, summary["steering_layer"], chart_prefix.with_name(f"{chart_prefix.name}_steering.png"))


if __name__ == "__main__":
    from jsonargparse import auto_cli

    auto_cli(main)
