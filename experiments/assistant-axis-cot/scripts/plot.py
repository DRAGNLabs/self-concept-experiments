"""Charts of one run's token projections across the CoT/final boundary.

Usage (from experiments/assistant-axis-cot):
    python scripts/plot.py --config configs/plot.yaml --run.run olmo32b-heldout
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import torch

from selfconcept.assistant_axis_cot.plotting import plot_absolute_position, plot_percent_of_response
from selfconcept.assistant_axis_cot.records import AxisName, Condition, TokenProjectionRecord
from selfconcept.assistant_axis_cot.statistics import Measure, has_both_regions
from selfconcept.common.paths import scratch_dir

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunConfig:
    run: str = "olmo32b-heldout"  # reads scratch assistant-axis-cot/<run>/token_projections.pt
    output_dir: Path = Path("results")
    axis: AxisName = "response_only"
    measure: Measure = "projection"
    min_region_tokens: int = 16
    examples_per_condition: int = 4
    seed: int = 0


def sample_examples(
    records: list[TokenProjectionRecord], condition: Condition, count: int, rng: np.random.Generator
) -> list[TokenProjectionRecord]:
    """count random records of the condition, each from a different role where there are enough roles."""
    condition_records = [record for record in records if record["condition"] == condition]
    shuffled = [condition_records[index] for index in rng.permutation(len(condition_records))]
    first_of_each_role = list({record["role"]: record for record in reversed(shuffled)}.values())
    return (first_of_each_role if len(first_of_each_role) >= count else shuffled)[:count]


def main(run: RunConfig = RunConfig()) -> None:
    saved = torch.load(scratch_dir("assistant-axis-cot") / run.run / "token_projections.pt", weights_only=False)
    records = [record for record in saved["records"] if has_both_regions(record, run.min_region_tokens)]
    title = f"{saved['model']}, layer {saved['layer']}, {run.axis} axis"
    output_dir = run.output_dir / run.run
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(run.seed)

    figures_by_name = {
        "percent_of_response": plot_percent_of_response(records, run.axis, run.measure, title),
        "absolute_position": plot_absolute_position(
            {
                condition: sample_examples(records, condition, run.examples_per_condition, rng)
                for condition in ("unprompted", "persona")
            },
            run.axis,
            run.measure,
            title,
        ),
    }
    for name, figure in figures_by_name.items():
        path = output_dir / f"{name}_{run.axis}_{run.measure}.png"
        figure.savefig(path, dpi=150)
        logger.info(f"Wrote {path}")


if __name__ == "__main__":
    from jsonargparse import auto_cli

    auto_cli(main)
