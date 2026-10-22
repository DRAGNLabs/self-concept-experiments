"""Plot per-token axis projections vs. position, split by axis and by thinking/response block.

Reads the per-conversation projections from ``token_axis_projections.py`` and draws a 2x2 grid
(rows = thinking/response block, cols = all-tokens/response-only axis). Each panel carries two
series -- the default role and all non-default roles -- as a mean trace with a 10th-90th
percentile envelope, the individual conversations faint in the background, and a rug along the
bottom marking where each conversation's block ended. No GPU needed.

Usage (from experiments/assistant-axis):
    python scripts/projection/plot_token_axis_projections.py --config configs/projection/plot_token_axis_projections.yaml
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from jaxtyping import Float
from matplotlib.axes import Axes

from selfconcept.common.paths import scratch_dir

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BlockName = Literal["thinking", "response"]
AxisName = Literal["all_tokens", "response_only"]
BLOCK_NAMES: tuple[BlockName, ...] = ("thinking", "response")
AXIS_NAMES: tuple[AxisName, ...] = ("all_tokens", "response_only")

GROUP_COLOR_BY_LABEL: dict[str, str] = {"default role": "#0072B2", "non-default roles": "#D55E00"}


@dataclass(frozen=True)
class RunConfig:
    """Projections input, figure output, and how many faint background lines to draw per group."""

    subdir: str = "olmo32b-thinking"  # scratch subtree holding this run's projections
    output: Path = Path("results/projection/token_axis_projections.png")
    max_background_lines: int = 120


class ProjectionTrace(NamedTuple):
    """Aggregate statistics over one group's per-token projections, indexed by token position."""

    mean: Float[np.ndarray, " position"]
    low_percentile: Float[np.ndarray, " position"]
    high_percentile: Float[np.ndarray, " position"]


def pad_to_matrix(arrays: list[Float[np.ndarray, " n_tokens"]]) -> Float[np.ndarray, "conversation position"]:
    """Stack ragged per-conversation projections into a NaN-padded (conversation, position) matrix."""
    max_length = max(len(array) for array in arrays)
    matrix = np.full((len(arrays), max_length), np.nan, dtype=np.float32)
    for row, array in enumerate(arrays):
        matrix[row, : len(array)] = array
    return matrix


def aggregate_trace(arrays: list[Float[np.ndarray, " n_tokens"]]) -> ProjectionTrace:
    """Mean and 10th/90th-percentile envelope at each token position over available conversations."""
    matrix = pad_to_matrix(arrays)
    return ProjectionTrace(
        mean=np.nanmean(matrix, axis=0),
        low_percentile=np.nanpercentile(matrix, 10, axis=0),
        high_percentile=np.nanpercentile(matrix, 90, axis=0),
    )


def evenly_spaced_subsample(
    arrays: list[Float[np.ndarray, " n_tokens"]], limit: int
) -> list[Float[np.ndarray, " n_tokens"]]:
    """Return at most ``limit`` arrays, evenly spaced through the list."""
    if len(arrays) <= limit:
        return arrays
    indices = np.linspace(0, len(arrays) - 1, limit).round().astype(int)
    return [arrays[index] for index in indices]


def draw_group(
    ax: Axes,
    arrays: list[Float[np.ndarray, " n_tokens"]],
    color: str,
    label: str,
    rug_axes_fraction: float,
    max_background_lines: int,
) -> None:
    """Draw one group's faint traces, mean+envelope, and block-end rug onto a panel."""
    for array in evenly_spaced_subsample(arrays, max_background_lines):
        ax.plot(np.arange(len(array)), array, color=color, alpha=0.06, linewidth=0.5, zorder=1)

    trace = aggregate_trace(arrays)
    positions = np.arange(len(trace.mean))
    ax.fill_between(positions, trace.low_percentile, trace.high_percentile, color=color, alpha=0.2, linewidth=0, zorder=2)
    ax.plot(positions, trace.mean, color=color, linewidth=2.0, label=label, zorder=3)

    block_end_positions = [len(array) - 1 for array in arrays]
    ax.plot(
        block_end_positions,
        np.full(len(block_end_positions), rug_axes_fraction),
        marker="|",
        linestyle="none",
        color=color,
        alpha=0.5,
        markersize=6,
        transform=ax.get_xaxis_transform(),
        zorder=2,
    )


def arrays_for(
    records: list[dict], is_default: bool, block: BlockName, axis: AxisName
) -> list[Float[np.ndarray, " n_tokens"]]:
    """Collect the non-empty per-conversation projection arrays for one group/block/axis."""
    return [
        record["projections_by_block_by_axis"][block][axis]
        for record in records
        if record["is_default"] == is_default and block in record["projections_by_block_by_axis"]
        and len(record["projections_by_block_by_axis"][block][axis]) > 0
    ]


def main(run: RunConfig = RunConfig()) -> None:
    """Render the 2x2 projection-vs-position grid to ``run.output``."""
    output_path = run.output.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    projections_path = scratch_dir("assistant-axis") / run.subdir / "token_axis_projections.pt"
    saved = torch.load(projections_path, map_location="cpu", weights_only=False)
    records = saved["records"]
    logger.info(f"Loaded {len(records)} conversation projections (layer {saved['target_layer']})")

    fig, axes_grid = plt.subplots(len(BLOCK_NAMES), len(AXIS_NAMES), figsize=(15, 10))
    for row, block in enumerate(BLOCK_NAMES):
        for column, axis_name in enumerate(AXIS_NAMES):
            ax = axes_grid[row][column]
            for is_default, label, rug_fraction in (
                (True, "default role", 0.05),
                (False, "non-default roles", 0.02),
            ):
                arrays = arrays_for(records, is_default, block, axis_name)
                if arrays:
                    draw_group(ax, arrays, GROUP_COLOR_BY_LABEL[label], label, rug_fraction, run.max_background_lines)

            ax.set_title(f"{axis_name} axis · {block} tokens")
            ax.set_xlabel("Token position within block")
            ax.set_ylabel("Projection onto unit axis")
            ax.grid(True, alpha=0.3, linewidth=0.5)
            ax.margins(x=0)

    handles, labels = axes_grid[0][0].get_legend_handles_labels()
    fig.suptitle(f"Per-token axis projection vs. position (layer {saved['target_layer']})", y=0.99)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved figure to {output_path}")


if __name__ == "__main__":
    from jsonargparse import auto_cli

    auto_cli(main)
