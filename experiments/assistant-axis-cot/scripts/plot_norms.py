"""Residual norms of a run's unprompted transcripts by token position, and the tokens whose norm stands out from their
transcript's, to find positions (like an attention sink) that would dominate a region's mean projection.

Usage (from experiments/assistant-axis-cot):
    python scripts/plot_norms.py --config configs/plot_norms.yaml --run.run olmo32b-heldout
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import torch
from transformers import AutoTokenizer

from selfconcept.assistant_axis_cot.ablation import NormOutlierToken, RegionMeans, norm_outlier_tokens, norm_ratios
from selfconcept.assistant_axis_cot.plotting import plot_residual_norm_by_position
from selfconcept.common.hf_strong_types import HFTokenizer
from selfconcept.common.paths import scratch_dir

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunConfig:
    run: str = "olmo32b-heldout"  # reads scratch assistant-axis-cot/<run>/region_means.pt
    output_dir: Path = Path("results")
    min_transcripts: int = 10
    norm_ratio_threshold: float = 3.0
    example_positions: int = 5


def outlier_report(
    region_means: RegionMeans,
    outlier_tokens: list[NormOutlierToken],
    tokenizer: HFTokenizer,
    norm_ratio_threshold: float,
) -> str:
    first_token_ratios = np.array([norm_ratios(record)[0] for record in region_means["records"]])
    lines = [
        f"# Residual norm outliers: {region_means['model']}, layer {region_means['layer']}",
        "",
        f"{len(region_means['records'])} unprompted transcripts. Norm ratio = a token's residual norm over its "
        "transcript's median.",
        "",
        f"Position 0: norm ratio median {np.median(first_token_ratios):.3g} "
        f"(range {first_token_ratios.min():.3g}–{first_token_ratios.max():.3g}).",
        "",
        f"Tokens with any occurrence above ratio {norm_ratio_threshold:g} or below {1 / norm_ratio_threshold:.3g}:",
        "",
        "| token | id | outlying / occurrences | outlying ratios | example positions |",
        "|---|---|---|---|---|",
    ]
    for outlier_token in outlier_tokens:
        token_text = repr(tokenizer.decode([outlier_token["token_id"]])).replace("|", "\\|")
        lines.append(
            f"| `{token_text}` | {outlier_token['token_id']} "
            f"| {outlier_token['outlier_count']} / {outlier_token['occurrence_count']} "
            f"| {outlier_token['min_outlying_norm_ratio']:.3g}–{outlier_token['max_outlying_norm_ratio']:.3g} "
            f"| {outlier_token['example_positions']} |"
        )
    if not outlier_tokens:
        lines.append("| none | | | | |")
    return "\n".join(lines) + "\n"


def main(run: RunConfig = RunConfig()) -> None:
    region_means: RegionMeans = torch.load(
        scratch_dir("assistant-axis-cot") / run.run / "region_means.pt", weights_only=False
    )
    output_dir = run.output_dir / run.run
    output_dir.mkdir(parents=True, exist_ok=True)

    figure = plot_residual_norm_by_position(
        region_means["records"], f"{region_means['model']}, layer {region_means['layer']}", run.min_transcripts
    )
    figure_path = output_dir / "residual_norm_by_position.png"
    figure.savefig(figure_path, dpi=150)
    logger.info(f"Wrote {figure_path}")

    tokenizer: HFTokenizer = AutoTokenizer.from_pretrained(region_means["model"])
    outlier_tokens = norm_outlier_tokens(region_means["records"], run.norm_ratio_threshold, run.example_positions)
    report_path = output_dir / "norm_outliers.md"
    report_path.write_text(outlier_report(region_means, outlier_tokens, tokenizer, run.norm_ratio_threshold))
    logger.info(f"Wrote {report_path}")


if __name__ == "__main__":
    from jsonargparse import auto_cli

    auto_cli(main)
