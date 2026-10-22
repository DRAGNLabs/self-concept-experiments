"""Charts of one run's window mean ablation statistics: the ablated x measured region grid of window position x
distance heatmaps of the assistant axis's KL and the random baseline's, on one shared log color scale, and of the
axis's difference from and ratio to the random baseline.

Usage (from experiments/assistant-axis-cot, after scripts/analyze_window_ablation.py):
    python scripts/plot_window_ablation.py --config configs/plot_window_ablation.yaml --run.run olmo32b-heldout
Writes results/<run>/window_ablation_<view>_<condition>.png.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
from jaxtyping import Float

matplotlib.use("Agg")

from selfconcept.assistant_axis_cot.ablation_statistics import RANDOM_MEAN_NAME
from selfconcept.assistant_axis_cot.plotting import (
    HeatmapScale,
    WindowView,
    difference_scale,
    kl_scale,
    plot_window_ablation,
    ratio_scale,
    window_cell_means,
    window_view_values,
)
from selfconcept.assistant_axis_cot.records import Condition
from selfconcept.assistant_axis_cot.window_ablation_statistics import WindowAblationAnalysis, WindowKLCells

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CONDITIONS: tuple[Condition, ...] = ("unprompted", "persona")


@dataclass(frozen=True)
class RunConfig:
    run: str = "olmo32b-heldout"  # reads results/<run>/window_ablation_<condition>.json
    output_dir: Path = Path("results")
    difference_linear_threshold: float = 1e-5  # the symmetric log difference scale is linear within this of 0


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
        axis_name = analysis["axis_name"]

        def axis_kl(cells: WindowKLCells) -> Float[np.ndarray, "window_offset distance"]:
            return window_cell_means(cells, axis_name)

        def random_kl(cells: WindowKLCells) -> Float[np.ndarray, "window_offset distance"]:
            return window_cell_means(cells, RANDOM_MEAN_NAME)

        def axis_minus_random(cells: WindowKLCells) -> Float[np.ndarray, "window_offset distance"]:
            return axis_kl(cells) - random_kl(cells)

        def axis_over_random(cells: WindowKLCells) -> Float[np.ndarray, "window_offset distance"]:
            return axis_kl(cells) / random_kl(cells)

        shared_kl_scale = kl_scale(
            np.concatenate([window_view_values(analysis, axis_kl), window_view_values(analysis, random_kl)])
        )
        view_and_scale_by_name: dict[str, tuple[WindowView, HeatmapScale]] = {
            f"kl_{axis_name}": (axis_kl, shared_kl_scale),
            f"kl_{RANDOM_MEAN_NAME}": (random_kl, shared_kl_scale),
            "difference": (
                axis_minus_random,
                difference_scale(
                    window_view_values(analysis, axis_minus_random), axis_name, run.difference_linear_threshold
                ),
            ),
            "ratio": (axis_over_random, ratio_scale(window_view_values(analysis, axis_over_random), axis_name)),
        }
        for view_name, (view, scale) in view_and_scale_by_name.items():
            path = output_dir / f"window_ablation_{view_name}_{condition}.png"
            figure = plot_window_ablation(analysis, view, scale, f"{title}: {view_name.replace('_', ' ')}")
            figure.savefig(path, dpi=150, bbox_inches="tight")
            logger.info(f"Wrote {path}")


if __name__ == "__main__":
    from jsonargparse import auto_cli

    auto_cli(main)
