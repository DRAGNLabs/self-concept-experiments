"""Region mean ablation statistics for one run: per condition, the ablated x measured region grid of next-token KL for
the assistant axis and the random baseline, and how much of the axis ablation's projection shift survives later layers.

Usage (from experiments/assistant-axis-cot, after slurm/ablate.sbatch):
    python scripts/analyze_ablation.py --config configs/analyze_ablation.yaml --run.run olmo32b-heldout
Writes results/<run>/ablation_<condition>.{json,md}.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from selfconcept.assistant_axis_cot.ablation import AblationRun
from selfconcept.assistant_axis_cot.ablation_statistics import (
    RANDOM_MEAN_NAME,
    AblationAnalysis,
    CellEstimate,
    analyze_ablation_run,
)
from selfconcept.assistant_axis_cot.records import LABELLED_SEQUENCE_REGIONS, Condition
from selfconcept.common.paths import scratch_dir

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CONDITIONS: tuple[Condition, ...] = ("unprompted", "persona")


@dataclass(frozen=True)
class RunConfig:
    run: str = "olmo32b-heldout"  # reads scratch assistant-axis-cot/<run>/ablation_<condition>.pt
    output_dir: Path = Path("results")
    min_region_tokens: int = 16
    resample_count: int = 2000
    seed: int = 0
    robustness_layer_offsets: list[int] = field(default_factory=lambda: [0, 1, 2, 4, 8, 16])


def format_cell(cell: CellEstimate) -> str:
    return f"{cell['mean']:.3g} [{cell['ci_low']:.3g}, {cell['ci_high']:.3g}] (n={cell['conversation_count']})"


def kl_grid_lines(analysis: AblationAnalysis, name: str) -> list[str]:
    lines = [
        f"### {name}",
        "",
        "| ablated \\ measured | " + " | ".join(LABELLED_SEQUENCE_REGIONS) + " |",
        "|---" * (len(LABELLED_SEQUENCE_REGIONS) + 1) + "|",
    ]
    for ablated_index, ablated in enumerate(LABELLED_SEQUENCE_REGIONS):
        cells = [
            ("noise floor: " if measured_index < ablated_index else "")
            + format_cell(analysis["kl_grid_by_name"][name][ablated][measured])
            for measured_index, measured in enumerate(LABELLED_SEQUENCE_REGIONS)
        ]
        lines.append(f"| {ablated} | " + " | ".join(cells) + " |")
    return lines + [""]


def markdown_summary(analysis: AblationAnalysis, run_name: str, robustness_layer_offsets: list[int]) -> str:
    random_names = [
        name for name in analysis["kl_grid_by_name"] if name not in (analysis["axis_name"], RANDOM_MEAN_NAME)
    ]
    lines = [
        f"# {run_name}: region mean ablation, {analysis['condition']}",
        "",
        f"{analysis['model']}, layer {analysis['layer']}, {analysis['conversation_count']} conversations. Each "
        f"region's projection onto a direction is set to its unprompted mean at layer {analysis['layer']}, "
        "teacher-forced on the unablated transcript.",
        "",
        "## KL(clean ‖ ablated) per token",
        "",
        "Mean over conversations of each conversation's mean per-token KL of the next-token distribution read at "
        "the measured region's tokens, with 95% question-bootstrap CIs. Rows: ablated region; columns: measured "
        "region. Cells left of the diagonal are causally unaffected, so measure only numerical noise.",
        "",
    ]
    for name in [analysis["axis_name"], RANDOM_MEAN_NAME, *random_names]:
        if name in analysis["kl_grid_by_name"]:
            lines += kl_grid_lines(analysis, name)

    std_by_region_by_direction = analysis["projection_std_by_region_by_direction"]
    lines += [
        "## Per-token projection SD over the unprompted transcripts",
        "",
        "How much each direction varies within a region, so how much its ablation moves.",
        "",
        "| direction | " + " | ".join(LABELLED_SEQUENCE_REGIONS) + " |",
        "|---" * (len(LABELLED_SEQUENCE_REGIONS) + 1) + "|",
    ]
    for direction_name, std_by_region in std_by_region_by_direction.items():
        region_stds = " | ".join(f"{std_by_region[region]:.3g}" for region in LABELLED_SEQUENCE_REGIONS)
        lines.append(f"| {direction_name} | {region_stds} |")

    layers = analysis["layers"]
    column_indices = sorted(
        {offset for offset in robustness_layer_offsets if offset < len(layers)} | {len(layers) - 1}
    )
    lines += [
        "",
        f"## Robustness of the {analysis['axis_name']} ablation",
        "",
        "RMS shift in each layer's own axis projection, as a fraction of the injected shift (the ablated region's at "
        f"the ablation layer, {analysis['layer']}).",
        "",
        "| ablated | measured | " + " | ".join(f"layer {layers[index]}" for index in column_indices) + " |",
        "|---|---" + "|---" * len(column_indices) + "|",
    ]
    for ablated, fraction_by_measured in analysis["surviving_shift_fraction_by_measured_by_ablated"].items():
        for measured, fractions in fraction_by_measured.items():
            layer_fractions = " | ".join(f"{fractions[index]:.3g}" for index in column_indices)
            lines.append(f"| {ablated} | {measured} | {layer_fractions} |")
    return "\n".join(lines) + "\n"


def main(run: RunConfig = RunConfig()) -> None:
    output_dir = run.output_dir / run.run
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(run.seed)
    for condition in CONDITIONS:
        ablation_path = scratch_dir("assistant-axis-cot") / run.run / f"ablation_{condition}.pt"
        if not ablation_path.exists():
            logger.info(f"No {ablation_path}; skipping {condition}")
            continue
        ablation_run: AblationRun = torch.load(ablation_path, weights_only=False)
        analysis = analyze_ablation_run(ablation_run, run.min_region_tokens, rng, run.resample_count)
        (output_dir / f"ablation_{condition}.json").write_text(json.dumps(analysis, indent=2))
        (output_dir / f"ablation_{condition}.md").write_text(
            markdown_summary(analysis, run.run, run.robustness_layer_offsets)
        )
        logger.info(f"Wrote {output_dir}/ablation_{condition}.json and .md")


if __name__ == "__main__":
    from jsonargparse import auto_cli

    auto_cli(main)
