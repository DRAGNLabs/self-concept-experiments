# Derived from safety-research/assistant-axis (https://github.com/safety-research/assistant-axis),
# MIT licensed. See the NOTICE file at the repository root for the full license text.
"""Extract mean assistant-response activations from per-role response JSONL files.

Loads each role's responses (from step 1), runs the model to capture per-layer hidden
states over the assistant turns, and saves one ``{role}.pt`` file mapping response keys to
mean-pooled activation tensors of shape (num_layers, hidden_size). A single model spans all
visible GPUs via device_map="auto"; roles are processed sequentially.

Usage (from experiments/assistant-axis):
    python pipeline/2_activations.py --config configs/2_activations.yaml
"""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jsonlines
import torch
from tqdm import tqdm

from selfconcept.assistant_axis.internals import (
    ActivationExtractor,
    ConversationEncoder,
    ProbingModel,
    SpanMapper,
)
from selfconcept.common.hf_strong_types import Conversation
from selfconcept.common.paths import scratch_dir

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DTYPE_MAP: dict[str, torch.dtype] = {
    "auto": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "half": torch.float16,
}


@dataclass(frozen=True)
class RunConfig:
    """Model, data, and extraction parameters for one activations run."""

    model: str = "google/gemma-2-27b-it"
    responses_dir: Path = scratch_dir("assistant-axis") / "responses"
    output_dir: Path = scratch_dir("assistant-axis") / "activations"
    layers: str = "all"  # "all" or comma-separated layer indices
    batch_size: int = 16
    max_length: int = 2048
    dtype: str = "bfloat16"  # one of DTYPE_MAP
    # PORT_ASSUMPTION[non-thinking]: defaults thinking OFF; only affects Qwen (see below).
    thinking: bool = False
    roles: list[str] | None = None


def load_responses(responses_file: Path) -> list[dict]:
    """Load responses from a JSONL file."""
    with jsonlines.open(responses_file, 'r') as reader:
        return list(reader)


def extract_activations_batch(
    pm: ProbingModel,
    conversations: list[Conversation],
    layers: list[int],
    batch_size: int = 16,
    max_length: int = 2048,
    enable_thinking: bool = False,
) -> list[torch.Tensor | None]:
    """Extract per-conversation mean assistant-turn activations."""
    assert pm.tokenizer is not None
    encoder = ConversationEncoder(pm.tokenizer, pm.model_name)
    extractor = ActivationExtractor(pm, encoder)
    span_mapper = SpanMapper()

    # PORT_ASSUMPTION[model-specific]: enable_thinking is only threaded through for Qwen;
    # PORT_ASSUMPTION[non-thinking]: other families ignore it and are treated as non-thinking.
    chat_kwargs: dict[str, Any] = {}
    if 'qwen' in pm.model_name.lower():
        chat_kwargs['enable_thinking'] = enable_thinking

    all_activations: list[torch.Tensor | None] = []

    for batch_start in range(0, len(conversations), batch_size):
        batch_conversations = conversations[batch_start:batch_start + batch_size]

        batch_activations, batch_metadata = extractor.batch_conversations(
            batch_conversations,
            layer=layers,
            max_length=max_length,
            **chat_kwargs,
        )

        _, batch_spans, _ = encoder.build_batch_turn_spans(batch_conversations, **chat_kwargs)

        # Per-turn mean activations: list of tensors, each (num_turns, num_layers, hidden_size)
        conv_activations_list = span_mapper.map_spans(batch_activations, batch_spans, batch_metadata)

        for conv_acts in conv_activations_list:
            if conv_acts.numel() == 0:
                all_activations.append(None)
                continue

            # conv_acts: (num_turns, num_layers, hidden_size). Assistant turns are the odd
            # indices (turn 0 = user, turn 1 = assistant, ...); mean over them.
            if conv_acts.shape[0] >= 2:
                assistant_act = conv_acts[1::2]
                if assistant_act.shape[0] > 0:
                    all_activations.append(assistant_act.mean(dim=0).cpu())
                else:
                    all_activations.append(None)
            else:
                all_activations.append(None)

        del batch_activations
        if (batch_start // batch_size) % 5 == 0:
            torch.cuda.empty_cache()

    return all_activations


def process_role(
    pm: ProbingModel,
    role_file: Path,
    output_dir: Path,
    layers: list[int],
    batch_size: int,
    max_length: int,
    enable_thinking: bool = False,
) -> None:
    """Extract and save activations for a single role file."""
    role = role_file.stem
    responses = load_responses(role_file)
    if not responses:
        return

    conversations = [resp["conversation"] for resp in responses]
    metadata = [
        {"prompt_index": resp["prompt_index"], "question_index": resp["question_index"], "label": resp["label"]}
        for resp in responses
    ]

    logger.info(f"Processing {role}: {len(conversations)} conversations")

    activations_list = extract_activations_batch(
        pm=pm,
        conversations=conversations,
        layers=layers,
        batch_size=batch_size,
        max_length=max_length,
        enable_thinking=enable_thinking,
    )

    activations_dict = {}
    for act, meta in zip(activations_list, metadata):
        if act is not None:
            key = f"{meta['label']}_p{meta['prompt_index']}_q{meta['question_index']}"
            activations_dict[key] = act

    if activations_dict:
        torch.save(activations_dict, output_dir / f"{role}.pt")
        logger.info(f"Saved {len(activations_dict)} activations for {role}")

    gc.collect()
    torch.cuda.empty_cache()


def main(run: RunConfig = RunConfig()) -> None:
    """Extract activations for every role's responses under the configured model."""
    run.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading model: {run.model}")
    pm = ProbingModel(run.model, dtype=DTYPE_MAP[run.dtype])

    n_layers = len(pm.get_layers())
    logger.info(f"Model has {n_layers} layers")
    if run.layers == "all":
        layers = list(range(n_layers))
    else:
        layers = [int(x.strip()) for x in run.layers.split(",")]
    logger.info(f"Extracting {len(layers)} layers")

    response_files = sorted(run.responses_dir.glob("*.jsonl"))
    if run.roles:
        response_files = [f for f in response_files if f.stem in run.roles]

    role_files = []
    for f in response_files:
        if (run.output_dir / f"{f.stem}.pt").exists():
            logger.info(f"Skipping {f.stem} (already exists)")
            continue
        role_files.append(f)

    logger.info(f"Processing {len(role_files)} roles")
    for role_file in tqdm(role_files, desc="Processing roles"):
        process_role(pm, role_file, run.output_dir, layers, run.batch_size, run.max_length, run.thinking)

    logger.info("Done!")


if __name__ == "__main__":
    from jsonargparse import auto_cli

    auto_cli(main)
