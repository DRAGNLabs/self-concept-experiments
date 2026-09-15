# Derived from safety-research/assistant-axis (https://github.com/safety-research/assistant-axis),
# MIT licensed. See the NOTICE file at the repository root for the full license text.
"""Select the held-out steering roles for the role-susceptibility experiment (Section 3.2.1).

Computing per-role vectors for OLMo would require the full generate->activations->vectors
pipeline. Instead we reuse the published per-role vectors for three other models
(``lu-christina/assistant-axis-vectors``), project each role onto that model's assistant
axis at its middle layer, and rank roles most->least Assistant-like within each model. A
role's cross-model score is its mean rank across the three models (Borda), which is robust
to per-model scale differences. We drop the roles the OLMo axis was trained on (see
``OLMO_TRAIN_ROLES``) so the steering set is disjoint from the axis training set, then take
the ``top_k`` most Assistant-like of the rest.

For each selected role we emit every ``pos`` system-prompt variant from its instruction
file, one JSONL line per variant, in the ``6_steered_traces`` ``SystemPrompt`` schema.

Usage (from experiments/assistant-axis):
    python pipeline/select_steer_roles.py --config configs/select_steer_roles.yaml
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import jsonlines
import numpy as np
import torch
from huggingface_hub import snapshot_download
from jaxtyping import Float
from torch import Tensor

from selfconcept.assistant_axis.models import get_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# HuggingFace model name (keyed by its subfolder in the vectors repo) whose get_config
# supplies the middle-layer index to project at.
HF_MODEL_NAME_BY_VECTORS_FOLDER: dict[str, str] = {
    "gemma-2-27b": "google/gemma-2-27b-it",
    "qwen-3-32b": "Qwen/Qwen3-32B",
    "llama-3.3-70b": "meta-llama/Llama-3.3-70B-Instruct",
}

# Roles the OLMo-3.1-32B-Think axis was trained on (olmo32b_thinking.sbatch); excluded from
# the steering set to keep train and test disjoint.
OLMO_TRAIN_ROLES: frozenset[str] = frozenset({
    "default", "accountant", "comedian", "chef", "detective", "pirate", "poet", "scientist",
    "lawyer", "doctor", "teacher", "philosopher", "hacker", "therapist", "engineer",
    "musician", "historian",
})

# Leading imperative openings on instruction `pos` strings, longest first so the most
# specific match wins. Stripping stops before the role's article so "You are an accountant
# who..." becomes "an accountant who...".
_IMPERATIVE_OPENINGS: tuple[str, ...] = (
    "embody the persona of", "embody the role of", "take on the persona of",
    "take on the role of", "assume the persona of", "assume the role of",
    "play the role of", "please respond as", "please embody", "you are", "please be",
    "respond as", "serve as", "play the", "act as", "embody", "be",
)
_ROLE_PHRASE_PATTERN = re.compile(
    r"^(?:" + "|".join(_IMPERATIVE_OPENINGS) + r")\s+(?=(?:a|an|the)\b)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class RunConfig:
    """Vectors source, role instructions, and how many steering roles to select."""

    vectors_repo: str = "lu-christina/assistant-axis-vectors"
    roles_dir: Path = Path("data/roles/instructions")
    output: Path = Path("data/steer_prompts.jsonl")
    top_k: int = 50
    exclude_train_roles: bool = True


def load_projection_by_role(
    axis: Float[Tensor, "layers hidden"],
    role_vectors_dir: Path,
    layer: int,
) -> dict[str, float]:
    """Projection of each role vector onto the unit axis direction at ``layer``."""
    axis_direction = axis[layer] / (axis[layer].norm() + 1e-8)
    projection_by_role: dict[str, float] = {}
    for vector_file in sorted(role_vectors_dir.glob("*.pt")):
        vector: Float[Tensor, "layers hidden"] = torch.load(
            vector_file, map_location="cpu", weights_only=False
        ).float()
        projection_by_role[vector_file.stem] = float(vector[layer] @ axis_direction)
    return projection_by_role


def load_model_projections(vectors_repo: str) -> dict[str, dict[str, float]]:
    """For each model, the projection-onto-axis of every published role vector."""
    projection_by_role_by_model: dict[str, dict[str, float]] = {}
    for folder, model_name in HF_MODEL_NAME_BY_VECTORS_FOLDER.items():
        local_repo = Path(
            snapshot_download(
                vectors_repo,
                repo_type="dataset",
                allow_patterns=[f"{folder}/assistant_axis.pt", f"{folder}/role_vectors/*.pt"],
            )
        )
        axis: Float[Tensor, "layers hidden"] = torch.load(
            local_repo / folder / "assistant_axis.pt", map_location="cpu", weights_only=False
        ).float()
        layer = get_config(model_name)["target_layer"]
        projection_by_role = load_projection_by_role(
            axis, local_repo / folder / "role_vectors", layer
        )
        logger.info("%s: %d role projections at layer %d", folder, len(projection_by_role), layer)
        projection_by_role_by_model[folder] = projection_by_role
    return projection_by_role_by_model


def rank_by_role(projection_by_role: dict[str, float]) -> dict[str, int]:
    """0-indexed rank of each role, most Assistant-like (highest projection) first."""
    roles_most_assistant_first = sorted(
        projection_by_role, key=lambda role: projection_by_role[role], reverse=True
    )
    return {role: rank for rank, role in enumerate(roles_most_assistant_first)}


def log_ranking_agreement(rank_by_role_by_model: dict[str, dict[str, int]], roles: list[str]) -> None:
    """Spearman correlation of each model pair's ranking over the shared roles."""
    models = sorted(rank_by_role_by_model)
    for i, model_a in enumerate(models):
        for model_b in models[i + 1 :]:
            ranks_a = [rank_by_role_by_model[model_a][role] for role in roles]
            ranks_b = [rank_by_role_by_model[model_b][role] for role in roles]
            spearman = float(np.corrcoef(ranks_a, ranks_b)[0, 1])
            logger.info("Spearman(%s, %s) = %.3f", model_a, model_b, spearman)


def select_roles(
    rank_by_role_by_model: dict[str, dict[str, int]],
    shared_roles: list[str],
    excluded_roles: frozenset[str],
    top_k: int,
) -> list[tuple[str, float]]:
    """The ``top_k`` roles with the lowest mean rank (most Assistant-like), excluded set removed."""
    mean_rank_by_role = {
        role: float(np.mean([rank_by_role[role] for rank_by_role in rank_by_role_by_model.values()]))
        for role in shared_roles
        if role not in excluded_roles
    }
    roles_by_mean_rank = sorted(mean_rank_by_role, key=lambda role: mean_rank_by_role[role])
    return [(role, mean_rank_by_role[role]) for role in roles_by_mean_rank[:top_k]]


def role_phrase_from_pos(pos: str) -> str:
    """Strip the leading imperative from a `pos` system prompt, keeping the role's article."""
    return _ROLE_PHRASE_PATTERN.sub("", pos.strip()).rstrip(".")


def pos_variants(role: str, roles_dir: Path) -> list[str]:
    instruction = json.loads((roles_dir / f"{role}.json").read_text())["instruction"]
    return [item["pos"] for item in instruction]


def main(run: RunConfig = RunConfig()) -> None:
    """Select the steering roles and write their system-prompt variants to ``run.output``."""
    projection_by_role_by_model = load_model_projections(run.vectors_repo)

    shared_roles = sorted(
        set.intersection(*(set(p) for p in projection_by_role_by_model.values()))
    )
    logger.info("%d roles shared across all models", len(shared_roles))

    rank_by_role_by_model = {
        model: rank_by_role(projection_by_role)
        for model, projection_by_role in projection_by_role_by_model.items()
    }
    log_ranking_agreement(rank_by_role_by_model, shared_roles)

    excluded_roles = OLMO_TRAIN_ROLES if run.exclude_train_roles else frozenset()
    selected = select_roles(rank_by_role_by_model, shared_roles, excluded_roles, run.top_k)

    logger.info("Selected %d roles (role: mean_rank):", len(selected))
    for role, mean_rank in selected:
        logger.info("  %-24s %.1f", role, mean_rank)

    run.output.parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(run.output, "w") as writer:
        for role, _ in selected:
            for pos in pos_variants(role, run.roles_dir):
                writer.write({"role_id": role, "role": role_phrase_from_pos(pos)})
    logger.info("Wrote %s", run.output)


if __name__ == "__main__":
    from jsonargparse import auto_cli

    auto_cli(main)
