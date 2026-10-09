"""Region mean ablation of the assistant axis and random baseline directions over one condition's transcripts,
teacher-forced: next-token KL from the unablated model, and the shift in each later layer's axis projection.

Usage (from experiments/assistant-axis-cot, after scripts/region_means.py):
    python scripts/ablate.py --config configs/ablate.yaml
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import torch

from selfconcept.assistant_axis_cot.ablation import AblationRun, RegionMeans, ablate_regions
from selfconcept.assistant_axis_cot.projection import load_tokenizer_and_model, load_unit_axis_by_layer
from selfconcept.assistant_axis_cot.records import AblationRecord, Condition
from selfconcept.assistant_axis_cot.transcripts import (
    load_labelled_conversations,
    map_in_padded_batches,
    thinking_prompt_completions,
)
from selfconcept.common.activation_extraction.model_specifics import get_model_specifics
from selfconcept.common.paths import scratch_dir

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunConfig:
    model: str = "allenai/Olmo-3.1-32B-Think"
    responses_dir: Path = scratch_dir("assistant-axis-cot") / "olmo32b-heldout" / "responses"
    region_means_path: Path = scratch_dir("assistant-axis-cot") / "olmo32b-heldout" / "region_means.pt"
    axis_path: Path = scratch_dir("assistant-axis") / "olmo32b-thinking" / "axis_response_only.pt"
    condition: Condition = "unprompted"
    output: Path = scratch_dir("assistant-axis-cot") / "olmo32b-heldout" / "ablation_unprompted.pt"
    max_length: int = 40960
    max_batch_tokens: int = 32768
    kl_chunk_size: int = 1024
    attn_implementation: str | None = None
    conversations_per_role: int | None = None


def main(run: RunConfig = RunConfig()) -> None:
    region_means: RegionMeans = torch.load(run.region_means_path, weights_only=False)
    if region_means["model"] != run.model:
        raise ValueError(f"region means are for {region_means['model']}, not {run.model}")
    tokenizer, model = load_tokenizer_and_model(run.model, run.attn_implementation)
    model_specifics = get_model_specifics(run.model, tokenizer)
    layers = list(range(region_means["layer"], model.config.num_hidden_layers))
    unit_axis_by_layer = load_unit_axis_by_layer(
        run.axis_path, layers, model.config.num_hidden_layers, model.config.hidden_size
    )
    region_means_axis = region_means["unit_direction_by_name"][region_means["axis_name"]]
    if not torch.equal(unit_axis_by_layer[region_means["layer"]], region_means_axis):
        raise ValueError(f"{run.axis_path} is not the axis the region means were computed with")
    logger.info(f"Ablating at layer {region_means['layer']} along {list(region_means['unit_direction_by_name'])}")

    labelled_conversations = [
        (metadata, conversation)
        for metadata, conversation in load_labelled_conversations(run.responses_dir, run.conversations_per_role)
        if metadata["condition"] == run.condition
    ]
    examples = thinking_prompt_completions(
        tokenizer, run.model, [conversation for _, conversation in labelled_conversations]
    )
    pad_token_id = tokenizer.eos_token_id
    assert pad_token_id is not None

    results = map_in_padded_batches(
        examples,
        run.max_length,
        run.max_batch_tokens,
        lambda batch: ablate_regions(
            model,
            model_specifics,
            batch,
            region_means,
            unit_axis_by_layer,
            pad_token_id,
            run.max_length,
            run.kl_chunk_size,
        ),
    )
    records: list[AblationRecord] = [
        {**metadata, **result} for (metadata, _), result in zip(labelled_conversations, results, strict=True)
    ]

    ablation_run: AblationRun = {
        "model": run.model,
        "layer": region_means["layer"],
        "axis_name": region_means["axis_name"],
        "condition": run.condition,
        "direction_names": list(region_means["unit_direction_by_name"]),
        "layers": layers,
        "moments_by_region_by_direction": region_means["moments_by_region_by_direction"],
        "records": records,
    }
    run.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ablation_run, run.output)
    truncated_count = sum(record["truncated"] for record in records)
    logger.info(f"Saved {len(records)} records ({truncated_count} truncated) to {run.output}")


if __name__ == "__main__":
    from jsonargparse import auto_cli

    auto_cli(main)
