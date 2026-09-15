"""Compare persona axes by cosine similarity.

Each axis is a bare tensor of shape (n_layers, hidden_dim) as written by
pipeline/5_axis.py. Given several labelled axes, reports the flattened
whole-axis cosine and the per-layer cosine distribution for every pair, and
writes the summary to JSON.

Usage (from experiments/assistant-axis):
    python compare_axes.py --axis ported=a.pt --axis original=b.pt --output cmp.json
"""

from __future__ import annotations

import argparse
import json
import logging
from itertools import combinations
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_axis(path: Path) -> Tensor:
    """Load an axis tensor of shape (n_layers, hidden), coerced to float32 for stable cosine math."""
    axis = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(axis, dict):
        axis = axis["axis"]
    return axis.float()


def whole_axis_cosine(left: Tensor, right: Tensor) -> float:
    """Cosine similarity between the two (n_layers, hidden) axes flattened to single vectors."""
    return F.cosine_similarity(left.flatten(), right.flatten(), dim=0).item()


def per_layer_cosine(left: Tensor, right: Tensor) -> Tensor:
    """Cosine similarity between the two (n_layers, hidden) axes at each layer, shape (n_layers,)."""
    return F.cosine_similarity(left, right, dim=1)


def compare_pair(left: Tensor, right: Tensor) -> dict[str, object]:
    """Whole-axis and per-layer cosine summary for one pair of axes."""
    layer_cosines = per_layer_cosine(left, right)
    return {
        "whole_axis_cosine": whole_axis_cosine(left, right),
        "per_layer_min": layer_cosines.min().item(),
        "per_layer_mean": layer_cosines.mean().item(),
        "per_layer_max": layer_cosines.max().item(),
        "per_layer_cosine": [round(c, 4) for c in layer_cosines.tolist()],
    }


def parse_labelled_axis(spec: str) -> tuple[str, Path]:
    """Split a ``label=path`` CLI argument."""
    label, _, path = spec.partition("=")
    if not label or not path:
        raise argparse.ArgumentTypeError(f"expected label=path, got {spec!r}")
    return label, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare persona axes by cosine similarity")
    parser.add_argument("--axis", type=parse_labelled_axis, action="append", required=True,
                        metavar="LABEL=PATH", help="a labelled axis file; pass two or more")
    parser.add_argument("--output", type=Path, required=True, help="where to write the JSON summary")
    args = parser.parse_args()

    axis_by_label = {label: load_axis(path) for label, path in args.axis}
    for label, axis in axis_by_label.items():
        logger.info(f"{label}: shape {tuple(axis.shape)}")

    comparison_by_pair = {
        f"{left_label} vs {right_label}": compare_pair(axis_by_label[left_label], axis_by_label[right_label])
        for left_label, right_label in combinations(axis_by_label, 2)
    }

    for pair, summary in comparison_by_pair.items():
        logger.info(
            f"{pair}: whole-axis cosine={summary['whole_axis_cosine']:.4f} | "
            f"per-layer min/mean/max="
            f"{summary['per_layer_min']:.4f}/{summary['per_layer_mean']:.4f}/{summary['per_layer_max']:.4f}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(comparison_by_pair, indent=2))
    logger.info(f"Wrote comparison to {args.output}")


if __name__ == "__main__":
    main()
