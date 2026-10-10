"""Window mean ablation statistics for one run: per condition, ablated x measured region, the mean next-token KL by the
window's position before its region's end and the measured tokens' distance after the window, for the assistant axis
and the random baseline.

Usage (from experiments/assistant-axis-cot, after slurm/ablate_windows.sbatch):
    python scripts/analyze_window_ablation.py --config configs/analyze_window_ablation.yaml --run.run olmo32b-heldout
Writes results/<run>/window_ablation_<condition>.json.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import torch

from selfconcept.assistant_axis_cot.records import Condition
from selfconcept.assistant_axis_cot.window_ablation import WindowAblationRun
from selfconcept.assistant_axis_cot.window_ablation_statistics import analyze_window_ablation_run
from selfconcept.common.paths import scratch_dir

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CONDITIONS: tuple[Condition, ...] = ("unprompted", "persona")


@dataclass(frozen=True)
class RunConfig:
    run: str = "olmo32b-heldout"  # reads scratch assistant-axis-cot/<run>/window_ablation_<condition>.pt
    output_dir: Path = Path("results")
    min_conversation_count: int = 20


def main(run: RunConfig = RunConfig()) -> None:
    output_dir = run.output_dir / run.run
    output_dir.mkdir(parents=True, exist_ok=True)
    for condition in CONDITIONS:
        window_ablation_path = scratch_dir("assistant-axis-cot") / run.run / f"window_ablation_{condition}.pt"
        if not window_ablation_path.exists():
            logger.info(f"No {window_ablation_path}; skipping {condition}")
            continue
        window_ablation_run: WindowAblationRun = torch.load(window_ablation_path, weights_only=False)
        analysis = analyze_window_ablation_run(window_ablation_run, run.min_conversation_count)
        analysis_path = output_dir / f"window_ablation_{condition}.json"
        analysis_path.write_text(json.dumps(analysis))
        logger.info(f"Wrote {analysis_path}")


if __name__ == "__main__":
    from jsonargparse import auto_cli

    auto_cli(main)
