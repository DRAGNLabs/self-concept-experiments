"""Per-region mean projections of a run's unprompted transcripts onto the assistant axis and random baseline
directions: the targets of region mean ablation. Keeps every token's projections and residual norm too.

Usage (from experiments/assistant-axis-cot):
    python scripts/region_means.py --config configs/region_means.yaml
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import torch
from jaxtyping import Float
from torch import Tensor

from selfconcept.assistant_axis.models import get_config
from selfconcept.assistant_axis_cot.ablation import RegionMeans, random_unit_directions, region_projection_moments
from selfconcept.assistant_axis_cot.projection import (
    load_tokenizer_and_model,
    load_unit_axis_by_name,
    project_sequences_in_batches,
)
from selfconcept.assistant_axis_cot.records import AxisName, SequenceProjectionRecord
from selfconcept.assistant_axis_cot.transcripts import load_labelled_conversations, thinking_prompt_completions
from selfconcept.common.activation_extraction.model_specifics import get_model_specifics
from selfconcept.common.paths import scratch_dir

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunConfig:
    model: str = "allenai/Olmo-3.1-32B-Think"
    responses_dir: Path = scratch_dir("assistant-axis-cot") / "olmo32b-heldout" / "responses"
    axis_name: AxisName = "response_only"
    axis_path: Path = scratch_dir("assistant-axis") / "olmo32b-thinking" / "axis_response_only.pt"
    output: Path = scratch_dir("assistant-axis-cot") / "olmo32b-heldout" / "region_means.pt"
    layer: int | None = None  # None -> get_config(model)'s target layer
    random_direction_count: int = 3
    seed: int = 0
    max_length: int = 40960
    max_batch_tokens: int = 32768
    attn_implementation: str | None = None
    conversations_per_role: int | None = None


def main(run: RunConfig = RunConfig()) -> None:
    tokenizer, model = load_tokenizer_and_model(run.model, run.attn_implementation)
    model_specifics = get_model_specifics(run.model, tokenizer)
    layer = run.layer if run.layer is not None else get_config(run.model)["target_layer"]
    hidden_size = model.config.hidden_size
    unit_axis = load_unit_axis_by_name(
        {run.axis_name: run.axis_path}, layer, model.config.num_hidden_layers, hidden_size
    )[run.axis_name]
    random_directions = random_unit_directions(run.random_direction_count, hidden_size, run.seed)
    unit_direction_by_name: dict[str, Float[Tensor, " hidden"]] = {
        run.axis_name: unit_axis,
        **{f"random_{index}": direction for index, direction in enumerate(random_directions)},
    }
    logger.info(f"Projecting at layer {layer} onto {list(unit_direction_by_name)}")

    unprompted_conversations = [
        (metadata, conversation)
        for metadata, conversation in load_labelled_conversations(run.responses_dir, run.conversations_per_role)
        if metadata["condition"] == "unprompted"
    ]
    examples = thinking_prompt_completions(
        tokenizer, run.model, [conversation for _, conversation in unprompted_conversations]
    )
    pad_token_id = tokenizer.eos_token_id
    assert pad_token_id is not None

    projections_by_example = project_sequences_in_batches(
        model,
        model_specifics,
        examples,
        unit_direction_by_name,
        layer,
        pad_token_id,
        run.max_length,
        run.max_batch_tokens,
    )
    records: list[SequenceProjectionRecord] = [
        {**metadata, **projections}
        for (metadata, _), projections in zip(unprompted_conversations, projections_by_example, strict=True)
    ]
    moments_by_region_by_direction = {
        direction_name: region_projection_moments(projections_by_example, direction_name)
        for direction_name in unit_direction_by_name
    }
    for direction_name, moments_by_region in moments_by_region_by_direction.items():
        logger.info(f"{direction_name}: {moments_by_region}")

    region_means: RegionMeans = {
        "model": run.model,
        "layer": layer,
        "axis_name": run.axis_name,
        "unit_direction_by_name": unit_direction_by_name,
        "moments_by_region_by_direction": moments_by_region_by_direction,
        "records": records,
    }
    run.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(region_means, run.output)
    logger.info(f"Saved region means over {len(records)} unprompted transcripts to {run.output}")


if __name__ == "__main__":
    from jsonargparse import auto_cli

    auto_cli(main)
