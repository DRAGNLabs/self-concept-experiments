"""Project every completion token of a run's thinking transcripts onto the assistant axis or axes.

Usage (from experiments/assistant-axis-cot):
    python scripts/project.py --config configs/project.yaml
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedModel

from selfconcept.assistant_axis.models import get_config
from selfconcept.assistant_axis_cot.projection import load_unit_axis_by_name, project_completions
from selfconcept.assistant_axis_cot.records import AxisName, TokenProjectionRecord
from selfconcept.assistant_axis_cot.transcripts import load_labelled_conversations, padded_token_budget_batches
from selfconcept.common.activation_extraction.model_specifics import get_model_specifics
from selfconcept.common.hf_strong_types import HFTokenizer
from selfconcept.common.loading import load_causal_lm
from selfconcept.common.paths import scratch_dir
from selfconcept.common.span_targeting.types import PromptCompletion
from selfconcept.transcript_generation.vllm_generation import generated_prompt_completion

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
    tokenizer: HFTokenizer = AutoTokenizer.from_pretrained(run.model)
    load_kwargs = {} if run.attn_implementation is None else {"attn_implementation": run.attn_implementation}
    model: PreTrainedModel = load_causal_lm(run.model, dtype=torch.bfloat16, device_map="auto", **load_kwargs)
    model.eval()
    model_specifics = get_model_specifics(run.model, tokenizer)
    layer = run.layer if run.layer is not None else get_config(run.model)["target_layer"]
    unit_axis_by_name = load_unit_axis_by_name(
        run.axis_path_by_name, layer, model.config.num_hidden_layers, model.config.hidden_size
    )
    logger.info(f"Projecting at layer {layer} onto {sorted(unit_axis_by_name)}")

    labelled_conversations = load_labelled_conversations(run.responses_dir, run.conversations_per_role)
    examples: list[PromptCompletion] = [
        generated_prompt_completion(tokenizer, run.model, conversation, enable_thinking=True, chat_template_kwargs={})
        for _, conversation in labelled_conversations
    ]
    sequence_lengths = [
        min(len(example.prompt_token_ids) + len(example.completion_token_ids), run.max_length) for example in examples
    ]
    pad_token_id = tokenizer.eos_token_id
    assert pad_token_id is not None

    record_by_index: dict[int, TokenProjectionRecord] = {}
    for batch_indices in tqdm(padded_token_budget_batches(sequence_lengths, run.max_batch_tokens), desc="Batches"):
        batch_projections = project_completions(
            model,
            model_specifics,
            [examples[index] for index in batch_indices],
            unit_axis_by_name,
            layer,
            pad_token_id,
            run.max_length,
        )
        for index, completion_projections in zip(batch_indices, batch_projections, strict=True):
            record_by_index[index] = {**labelled_conversations[index][0], **completion_projections}
    records = [record_by_index[index] for index in range(len(examples))]

    run.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": run.model, "layer": layer, "records": records}, run.output)
    logger.info(f"Saved {len(records)} records ({sum(record['truncated'] for record in records)} truncated) to {run.output}")


if __name__ == "__main__":
    from jsonargparse import auto_cli

    auto_cli(main)
