"""Project every completion token of a run's thinking transcripts onto the assistant axis or axes.

Usage (from experiments/assistant-axis-cot):
    python scripts/project.py --config configs/project.yaml
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import torch

from selfconcept.assistant_axis.models import get_config
from selfconcept.assistant_axis_cot.projection import (
    completion_projections,
    load_tokenizer_and_model,
    load_unit_axis_by_name,
    project_sequences_in_batches,
)
from selfconcept.assistant_axis_cot.records import AxisName, TokenProjectionRecord
from selfconcept.assistant_axis_cot.transcripts import load_labelled_conversations, thinking_prompt_completions
from selfconcept.common.activation_extraction.model_specifics import get_model_specifics
from selfconcept.common.paths import scratch_dir

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunConfig:
    model: str = "allenai/Olmo-3.1-32B-Think"
    responses_dir: Path = scratch_dir("assistant-axis-cot") / "olmo32b-heldout" / "responses"
    axis_path_by_name: dict[AxisName, Path] = field(default_factory=dict)
    output: Path = scratch_dir("assistant-axis-cot") / "olmo32b-heldout" / "token_projections.pt"
    layer: int | None = None  # None -> get_config(model)'s target layer
    max_length: int = 40960
    max_batch_tokens: int = 32768  # padded tokens per forward pass; the full-vocab logits dominate memory
    attn_implementation: str | None = None
    conversations_per_role: int | None = None


def main(run: RunConfig = RunConfig()) -> None:
    tokenizer, model = load_tokenizer_and_model(run.model, run.attn_implementation)
    model_specifics = get_model_specifics(run.model, tokenizer)
    layer = run.layer if run.layer is not None else get_config(run.model)["target_layer"]
    unit_axis_by_name = load_unit_axis_by_name(
        run.axis_path_by_name, layer, model.config.num_hidden_layers, model.config.hidden_size
    )
    logger.info(f"Projecting at layer {layer} onto {sorted(unit_axis_by_name)}")

    labelled_conversations = load_labelled_conversations(run.responses_dir, run.conversations_per_role)
    examples = thinking_prompt_completions(
        tokenizer, run.model, [conversation for _, conversation in labelled_conversations]
    )
    pad_token_id = tokenizer.eos_token_id
    assert pad_token_id is not None

    projections_by_example = project_sequences_in_batches(
        model, model_specifics, examples, unit_axis_by_name, layer, pad_token_id, run.max_length, run.max_batch_tokens
    )
    records: list[TokenProjectionRecord] = [
        {**metadata, **completion_projections(example, model_specifics.split_completion(example), projections)}
        for (metadata, _), example, projections in zip(
            labelled_conversations, examples, projections_by_example, strict=True
        )
    ]

    run.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": run.model, "layer": layer, "records": records}, run.output)
    logger.info(f"Saved {len(records)} records ({sum(record['truncated'] for record in records)} truncated) to {run.output}")


if __name__ == "__main__":
    from jsonargparse import auto_cli

    auto_cli(main)
