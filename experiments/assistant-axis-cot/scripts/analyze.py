"""CoT-vs-final statistics for one run's token projections: mean and variance differences per condition, and how
much a persona prompt lowers each region.

Usage (from experiments/assistant-axis-cot):
    python scripts/analyze.py --config configs/analyze.yaml --run.run olmo32b-heldout
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import numpy as np
import torch

from selfconcept.assistant_axis_cot.records import AxisName, Condition, TokenProjectionRecord
from selfconcept.assistant_axis_cot.statistics import (
    CONTENT_REGIONS,
    MEASURES,
    ContentRegion,
    ConversationSummary,
    Measure,
    PairedTest,
    PersonaDrop,
    VarianceDecomposition,
    cot_minus_final_mean,
    cot_over_final_variance_ratio,
    persona_drop,
    summarize_conversations,
    variance_decomposition,
)
from selfconcept.common.paths import scratch_dir

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CONDITIONS: tuple[Condition, ...] = ("unprompted", "persona")


@dataclass(frozen=True)
class RunConfig:
    run: str = "olmo32b-heldout"  # reads scratch assistant-axis-cot/<run>/token_projections.pt
    output_dir: Path = Path("results")
    min_region_tokens: int = 16
    resample_count: int = 2000
    seed: int = 0


class ConditionStatistics(TypedDict):
    conversation_count: int
    cot_minus_final_mean: PairedTest
    cot_over_final_variance_ratio: PairedTest
    cot_over_final_robust_variance_ratio: PairedTest
    variance_decomposition_by_region: dict[ContentRegion, VarianceDecomposition]


class AxisMeasureStatistics(TypedDict):
    statistics_by_condition: dict[Condition, ConditionStatistics]
    persona_drop: PersonaDrop


def condition_statistics(
    summaries: list[ConversationSummary], rng: np.random.Generator, resample_count: int
) -> ConditionStatistics:
    return {
        "conversation_count": len(summaries),
        "cot_minus_final_mean": cot_minus_final_mean(summaries, rng, resample_count),
        "cot_over_final_variance_ratio": cot_over_final_variance_ratio(summaries, rng, resample_count, "variance"),
        "cot_over_final_robust_variance_ratio": cot_over_final_variance_ratio(
            summaries, rng, resample_count, "robust_variance"
        ),
        "variance_decomposition_by_region": {
            region: variance_decomposition(summaries, region) for region in CONTENT_REGIONS
        },
    }


def axis_measure_statistics(
    records: list[TokenProjectionRecord],
    axis: AxisName,
    measure: Measure,
    run: RunConfig,
    rng: np.random.Generator,
) -> AxisMeasureStatistics:
    summaries = summarize_conversations(records, axis, measure, run.min_region_tokens)
    return {
        "statistics_by_condition": {
            condition: condition_statistics(
                [summary for summary in summaries if summary["condition"] == condition], rng, run.resample_count
            )
            for condition in CONDITIONS
        },
        "persona_drop": persona_drop(summaries, rng, run.resample_count),
    }


def format_test(test: PairedTest) -> str:
    return f"{test['estimate']:.3g} [{test['ci_low']:.3g}, {test['ci_high']:.3g}] (p={test['wilcoxon_p']:.2g}, n={test['count']})"


def markdown_summary(
    statistics_by_measure_by_axis: dict[AxisName, dict[Measure, AxisMeasureStatistics]],
    run_name: str,
    record_count: int,
) -> str:
    lines = [
        f"# {run_name}",
        "",
        f"{record_count} conversations. Estimates are means over conversations (Q1, Q2) or questions (Q3), with "
        "95% question-bootstrap CIs and two-sided Wilcoxon signed-rank p.",
        "",
    ]
    for axis, statistics_by_measure in statistics_by_measure_by_axis.items():
        for measure, statistics in statistics_by_measure.items():
            lines += [
                f"## axis={axis}, measure={measure}",
                "",
                "| condition | CoT − final mean | CoT/final variance | CoT/final robust variance | "
                "within / between variance (CoT) | within / between variance (final) |",
                "|---|---|---|---|---|---|",
            ]
            for condition, condition_stats in statistics["statistics_by_condition"].items():
                decomposition_by_region = condition_stats["variance_decomposition_by_region"]
                lines.append(
                    f"| {condition} | {format_test(condition_stats['cot_minus_final_mean'])} "
                    f"| {format_test(condition_stats['cot_over_final_variance_ratio'])} "
                    f"| {format_test(condition_stats['cot_over_final_robust_variance_ratio'])} "
                    + "".join(
                        f"| {decomposition_by_region[region]['within']:.3g} / {decomposition_by_region[region]['between']:.3g} "
                        for region in CONTENT_REGIONS
                    )
                    + "|"
                )
            drop = statistics["persona_drop"]
            lines += [
                "",
                f"Persona drop (unprompted − persona), {drop['question_count']} questions:",
                f"- final: {format_test(drop['drop_by_region']['final'])}",
                f"- CoT: {format_test(drop['drop_by_region']['cot'])}",
                f"- final − CoT: {format_test(drop['final_minus_cot'])}",
                f"- final − CoT, each in units of its unprompted within-conversation SD: "
                f"{format_test(drop['standardized_final_minus_cot'])}",
                "",
            ]
    return "\n".join(lines)


def main(run: RunConfig = RunConfig()) -> None:
    saved = torch.load(scratch_dir("assistant-axis-cot") / run.run / "token_projections.pt", weights_only=False)
    records: list[TokenProjectionRecord] = saved["records"]
    axes: list[AxisName] = sorted(records[0]["projections_by_axis"])
    rng = np.random.default_rng(run.seed)

    statistics_by_measure_by_axis: dict[AxisName, dict[Measure, AxisMeasureStatistics]] = {
        axis: {measure: axis_measure_statistics(records, axis, measure, run, rng) for measure in MEASURES}
        for axis in axes
    }

    output_dir = run.output_dir / run.run
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "stats.json").write_text(
        json.dumps({"model": saved["model"], "layer": saved["layer"], "statistics": statistics_by_measure_by_axis}, indent=2)
    )
    (output_dir / "stats.md").write_text(markdown_summary(statistics_by_measure_by_axis, run.run, len(records)))
    logger.info(f"Wrote {output_dir}/stats.json and stats.md")


if __name__ == "__main__":
    from jsonargparse import auto_cli

    auto_cli(main)
