# Derived from safety-research/assistant-axis (https://github.com/safety-research/assistant-axis),
# MIT licensed. See the NOTICE file at the repository root for the full license text.
"""Compute per-role vectors from activations and judge scores.

For each role, reduces its per-response activations (from step 2) to a single vector:
  - "default" roles: mean over ALL activations (no score filtering).
  - other roles: mean over activations whose judge score (step 3) is 3 (fully in role),
    requiring at least ``min_count`` such samples.
Writes one ``{role}.pt`` file per role holding {"vector", "type", "role"}.

Usage (from experiments/assistant-axis):
    python scripts/pipeline/4_vectors.py --config configs/pipeline/4_vectors.yaml
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import torch
from tqdm import tqdm

from selfconcept.common.paths import scratch_dir

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunConfig:
    """Input/output locations and the score=3 sample threshold."""

    activations_dir: Path = scratch_dir("assistant-axis") / "activations"
    scores_dir: Path = scratch_dir("assistant-axis") / "scores"
    output_dir: Path = scratch_dir("assistant-axis") / "vectors"
    min_count: int = 50
    overwrite: bool = False


def load_scores(scores_file: Path) -> dict[str, int]:
    """Load scores from a JSON file."""
    with open(scores_file, 'r') as f:
        return json.load(f)


def load_activations(activations_file: Path) -> dict[str, torch.Tensor]:
    """Load activations from a .pt file."""
    return torch.load(activations_file, map_location="cpu", weights_only=False)


def compute_pos_3_vector(activations: dict[str, torch.Tensor], scores: dict[str, int], min_count: int) -> torch.Tensor:
    """Mean activation over responses with score=3; raises if fewer than min_count."""
    filtered_acts = [act for key, act in activations.items() if scores.get(key) == 3]

    if len(filtered_acts) < min_count:
        raise ValueError(f"Only {len(filtered_acts)} score=3 samples, need {min_count}")

    return torch.stack(filtered_acts).mean(dim=0)  # (n_layers, hidden_dim)


def compute_mean_vector(activations: dict[str, torch.Tensor]) -> torch.Tensor:
    """Mean activation over all responses (no filtering)."""
    return torch.stack(list(activations.values())).mean(dim=0)  # (n_layers, hidden_dim)


def main(run: RunConfig = RunConfig()) -> None:
    """Compute and save a per-role vector for every activation file."""
    run.output_dir.mkdir(parents=True, exist_ok=True)

    activation_files = sorted(run.activations_dir.glob("*.pt"))
    logger.info(f"Found {len(activation_files)} activation files")

    successful = skipped = failed = 0

    for act_file in tqdm(activation_files, desc="Computing vectors"):
        role = act_file.stem
        output_file = run.output_dir / f"{role}.pt"

        if output_file.exists() and not run.overwrite:
            skipped += 1
            continue

        activations = load_activations(act_file)
        if not activations:
            logger.warning(f"No activations for {role}")
            failed += 1
            continue

        try:
            if "default" in role:
                vector = compute_mean_vector(activations)
                vector_type = "mean"
            else:
                scores_file = run.scores_dir / f"{role}.json"
                if not scores_file.exists():
                    logger.warning(f"No scores file for {role}")
                    failed += 1
                    continue
                scores = load_scores(scores_file)
                vector = compute_pos_3_vector(activations, scores, run.min_count)
                vector_type = "pos_3"

            torch.save({"vector": vector, "type": vector_type, "role": role}, output_file)
            successful += 1
        except ValueError as e:
            logger.warning(f"{role}: {e}")
            failed += 1

    logger.info(f"Summary: {successful} successful, {skipped} skipped, {failed} failed")


if __name__ == "__main__":
    from jsonargparse import auto_cli

    auto_cli(main)
