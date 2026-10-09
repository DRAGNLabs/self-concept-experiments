"""Charts of one run's region mean ablation statistics: KL grids of the assistant axis beside the random baseline, and
how much of the axis ablation's projection shift survives later layers.

Usage (from experiments/assistant-axis-cot, after scripts/analyze_ablation.py):
    python scripts/plot_ablation.py --config configs/plot_ablation.yaml --run.run olmo32b-heldout
Writes results/<run>/{ablation_kl,ablation_surviving_shift}_<condition>.png.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from selfconcept.assistant_axis_cot.ablation_statistics import AblationAnalysis
from selfconcept.assistant_axis_cot.plotting import plot_ablation_kl_grids, plot_surviving_shift
from selfconcept.assistant_axis_cot.records import Condition

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CONDITIONS: tuple[Condition, ...] = ("unprompted", "persona")


@dataclass(frozen=True)
class RunConfig:
    run: str = "olmo32b-heldout"  # reads results/<run>/ablation_<condition>.json
    output_dir: Path = Path("results")


def main(run: RunConfig = RunConfig()) -> None:
    output_dir = run.output_dir / run.run
    for condition in CONDITIONS:
        analysis_path = output_dir / f"ablation_{condition}.json"
        if not analysis_path.exists():
            logger.info(f"No {analysis_path}; skipping {condition}")
            continue
        analysis: AblationAnalysis = json.loads(analysis_path.read_text())
        title = (
            f"{analysis['model']}, layer {analysis['layer']}, {condition} "
            f"({analysis['conversation_count']} conversations)"
        )
        figures_by_name = {
            "ablation_kl": plot_ablation_kl_grids(analysis, title),
            "ablation_surviving_shift": plot_surviving_shift(analysis, title),
        }
        for name, figure in figures_by_name.items():
            path = output_dir / f"{name}_{condition}.png"
            figure.savefig(path, dpi=150, bbox_inches="tight")
            logger.info(f"Wrote {path}")


if __name__ == "__main__":
    from jsonargparse import auto_cli

    auto_cli(main)
