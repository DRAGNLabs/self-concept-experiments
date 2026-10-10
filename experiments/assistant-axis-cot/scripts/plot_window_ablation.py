"""Charts of one run's window mean ablation statistics: the ablated x measured region grid of window position x
distance KL heatmaps, for the assistant axis and the random baseline on one shared log color scale.

Usage (from experiments/assistant-axis-cot, after scripts/analyze_window_ablation.py):
    python scripts/plot_window_ablation.py --config configs/plot_window_ablation.yaml --run.run olmo32b-heldout
Writes results/<run>/window_ablation_kl_<condition>_<direction>.png.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from selfconcept.assistant_axis_cot.ablation_statistics import RANDOM_MEAN_NAME
from selfconcept.assistant_axis_cot.plotting import plot_window_ablation_kl, window_kl_norm
from selfconcept.assistant_axis_cot.records import Condition
from selfconcept.assistant_axis_cot.window_ablation_statistics import WindowAblationAnalysis

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CONDITIONS: tuple[Condition, ...] = ("unprompted", "persona")


@dataclass(frozen=True)
class RunConfig:
    run: str = "olmo32b-heldout"  # reads results/<run>/window_ablation_<condition>.json
    output_dir: Path = Path("results")


def main(run: RunConfig = RunConfig()) -> None:
    output_dir = run.output_dir / run.run
    for condition in CONDITIONS:
        analysis_path = output_dir / f"window_ablation_{condition}.json"
        if not analysis_path.exists():
            logger.info(f"No {analysis_path}; skipping {condition}")
            continue
        analysis: WindowAblationAnalysis = json.loads(analysis_path.read_text())
        title = (
            f"{analysis['model']}, layer {analysis['layer']}, {condition} ({analysis['conversation_count']} "
            f"conversations, {analysis['window_size']}-token windows)"
        )
        names = [analysis["axis_name"], RANDOM_MEAN_NAME]
        norm = window_kl_norm(analysis, names)
        for name in names:
            path = output_dir / f"window_ablation_kl_{condition}_{name}.png"
            plot_window_ablation_kl(analysis, name, norm, title).savefig(path, dpi=150, bbox_inches="tight")
            logger.info(f"Wrote {path}")


if __name__ == "__main__":
    from jsonargparse import auto_cli

    auto_cli(main)
