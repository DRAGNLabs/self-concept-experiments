"""Window mean ablation of the assistant axis and random baseline directions over one condition's transcripts,
teacher-forced: the next-token KL from the unablated model at every token from each window onward.

Usage (from experiments/assistant-axis-cot, after scripts/region_means.py):
    python scripts/ablate_windows.py --config configs/ablate_windows.yaml
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import torch
from tqdm import tqdm

from selfconcept.assistant_axis_cot.ablation import RegionMeans
from selfconcept.assistant_axis_cot.projection import load_tokenizer_and_model
from selfconcept.assistant_axis_cot.records import Condition, WindowAblationRecord
from selfconcept.assistant_axis_cot.transcripts import load_labelled_conversations, thinking_prompt_completions
from selfconcept.assistant_axis_cot.window_ablation import WindowAblationRun, ablate_windows
from selfconcept.common.activation_extraction.model_specifics import get_model_specifics
from selfconcept.common.paths import scratch_dir

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunConfig:
    model: str = "allenai/Olmo-3.1-32B-Think"
    responses_dir: Path = scratch_dir("assistant-axis-cot") / "olmo32b-heldout" / "responses"
    region_means_path: Path = scratch_dir("assistant-axis-cot") / "olmo32b-heldout" / "region_means.pt"
    condition: Condition = "unprompted"
    output: Path = scratch_dir("assistant-axis-cot") / "olmo32b-heldout" / "window_ablation_unprompted.pt"
    window_size: int = 50
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
    logger.info(f"Ablating at layer {region_means['layer']} along {list(region_means['unit_direction_by_name'])}")

    labelled_conversations = [
        (metadata, conversation)
        for metadata, conversation in load_labelled_conversations(run.responses_dir, run.conversations_per_role)
        if metadata["condition"] == run.condition
    ]
    examples = thinking_prompt_completions(
        tokenizer, run.model, [conversation for _, conversation in labelled_conversations], {}
    )
    pad_token_id = tokenizer.eos_token_id
    assert pad_token_id is not None

    records: list[WindowAblationRecord] = [
        {
            **metadata,
            **ablate_windows(
                model,
                model_specifics,
                example,
                region_means,
                pad_token_id,
                run.max_length,
                run.window_size,
                run.max_batch_tokens,
                run.kl_chunk_size,
            ),
        }
        for (metadata, _), example in tqdm(
            zip(labelled_conversations, examples, strict=True), total=len(examples), desc="Conversations"
        )
    ]

    window_ablation_run: WindowAblationRun = {
        "model": run.model,
        "layer": region_means["layer"],
        "axis_name": region_means["axis_name"],
        "condition": run.condition,
        "direction_names": list(region_means["unit_direction_by_name"]),
        "window_size": run.window_size,
        "records": records,
    }
    run.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(window_ablation_run, run.output)
    truncated_count = sum(record["truncated"] for record in records)
    window_count = sum(len(record["windows"]) for record in records)
    logger.info(f"Saved {len(records)} records ({truncated_count} truncated, {window_count} windows) to {run.output}")


if __name__ == "__main__":
    from jsonargparse import auto_cli

    auto_cli(main)
