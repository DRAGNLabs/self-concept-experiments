# Derived from safety-research/assistant-axis (https://github.com/safety-research/assistant-axis),
# MIT licensed. See the NOTICE file at the repository root for the full license text.
"""Compute a persona axis from the per-role vectors.

Formula: axis = mean(anchor) - mean(other roles). The axis points FROM the other roles
TOWARD the anchor persona. With the default anchor ("default") this is the Assistant Axis;
pass a different ``anchor`` to build another persona's axis.

Usage (from experiments/assistant-axis):
    python scripts/pipeline/5_axis.py --config configs/pipeline/5_axis.yaml
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from tqdm import tqdm

from selfconcept.common.paths import scratch_dir

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunConfig:
    """Vectors location, output path, and which role anchors the axis."""

    vectors_dir: Path = scratch_dir("assistant-axis") / "vectors"
    output: Path = scratch_dir("assistant-axis") / "axis.pt"
    anchor: str = "default"


def load_vector(vector_file: Path) -> dict:
    """Load vector data from a .pt file."""
    return torch.load(vector_file, map_location="cpu", weights_only=False)


def main(run: RunConfig = RunConfig()) -> None:
    """Compute the anchor-vs-rest axis and save it to ``run.output``."""
    run.output.parent.mkdir(parents=True, exist_ok=True)

    vector_files = sorted(run.vectors_dir.glob("*.pt"))
    logger.info(f"Found {len(vector_files)} vector files")

    # Separate the anchor vector from the contrast set (all other roles).
    anchor_vectors = []
    contrast_vectors = []
    for vec_file in tqdm(vector_files, desc="Loading vectors"):
        data = load_vector(vec_file)
        role = data.get("role", vec_file.stem)
        (anchor_vectors if role == run.anchor else contrast_vectors).append(data["vector"])

    logger.info(f"Loaded {len(anchor_vectors)} anchor vectors, {len(contrast_vectors)} contrast vectors")

    if not anchor_vectors:
        logger.error(f"anchor '{run.anchor}' not found in {run.vectors_dir} — was it generated and "
                     f"did it clear --min_count score=3 samples in step 4?")
        sys.exit(1)
    if not contrast_vectors:
        logger.error("No contrast vectors found")
        sys.exit(1)

    anchor_mean = torch.stack(anchor_vectors).mean(dim=0)      # (n_layers, hidden_dim)
    contrast_mean = torch.stack(contrast_vectors).mean(dim=0)  # (n_layers, hidden_dim)

    # Points from the other roles toward the anchor persona.
    axis = anchor_mean - contrast_mean

    norms = axis.norm(dim=1)
    logger.info(f"Axis shape: {tuple(axis.shape)}")
    logger.info(f"Per-layer norms (first 10): {[round(n.item(), 4) for n in norms[:10]]}")
    logger.info(f"Mean norm: {norms.mean():.4f} | Max norm: {norms.max():.4f} (layer {norms.argmax().item()})")

    torch.save(axis, run.output)
    logger.info(f"Saved axis to {run.output}")


if __name__ == "__main__":
    from jsonargparse import auto_cli

    auto_cli(main)
