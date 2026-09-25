"""Measure per-layer mean post-MLP residual-stream norms for a model on lmsys-chat-1m.

Samples conversations from a chat dataset, runs the model under HF transformers (one model
spanning all visible GPUs via device_map="auto"), and averages the L2 norm of each decoder
layer's output hidden state over all non-padding tokens. Writes ``layer_norms.pt`` (a
LayerNormResult) plus a readable ``layer_norms.json`` sidecar. Re-running skips work already
on disk.

Usage (from experiments/assistant-axis):
    python scripts/steering/activation_norms.py --config configs/steering/activation_norms.yaml
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
from datasets import Dataset

from selfconcept.assistant_axis.internals import ProbingModel, measure_layer_norms
from selfconcept.common.hf_strong_types import Conversation
from selfconcept.common.paths import scratch_dir

from byutils import load_dataset

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
    """Model, data, and measurement parameters for one norm-measurement run."""

    model: str = "allenai/Olmo-3.1-32B-Think"
    dataset: str = "lmsys/lmsys-chat-1m"
    split: str = "train"
    num_conversations: int = 1000
    english_only: bool = True
    max_length: int = 2048
    batch_size: int = 8
    layers: str = "all"  # "all" or comma-separated layer indices
    dtype: str = "bfloat16"  # one of DTYPE_MAP
    seed: int = 0
    output: Path = scratch_dir("assistant-axis") / "layer_norms.pt"


def load_sampled_conversations(
    dataset_name: str,
    split: str,
    num_conversations: int,
    english_only: bool,
    seed: int,
) -> list[Conversation]:
    """Sample conversations from a chat dataset, mapping each to the {role, content} format."""
    dataset = cast(Dataset, load_dataset(dataset_name, split=split))
    if english_only:
        dataset = dataset.filter(lambda row: row["language"] == "English")
    dataset = dataset.shuffle(seed=seed).select(range(min(num_conversations, len(dataset))))
    conversation_column = cast(list[list[dict[str, str]]], dataset["conversation"])
    return cast(
        list[Conversation],
        [
            [{"role": turn["role"], "content": turn["content"]} for turn in conversation]
            for conversation in conversation_column
        ],
    )


def resolve_layers(layers: str, num_layers: int) -> list[int]:
    """Resolve the layers config ("all" or comma-separated indices) to a list of indices."""
    if layers == "all":
        return list(range(num_layers))
    return [int(index.strip()) for index in layers.split(",")]


def main(run: RunConfig = RunConfig()) -> None:
    """Measure and save per-layer mean activation norms for the configured model."""
    if run.output.exists():
        logger.info("Output %s already exists, skipping", run.output)
        return
    run.output.parent.mkdir(parents=True, exist_ok=True)

    conversations = load_sampled_conversations(
        run.dataset, run.split, run.num_conversations, run.english_only, run.seed
    )
    logger.info("Loaded %d conversations from %s", len(conversations), run.dataset)

    logger.info("Loading model: %s", run.model)
    probing_model = ProbingModel(run.model, dtype=DTYPE_MAP[run.dtype])
    layers = resolve_layers(run.layers, len(probing_model.get_layers()))
    logger.info("Measuring norms over %d layers", len(layers))

    result = measure_layer_norms(
        probing_model,
        conversations,
        layers,
        max_length=run.max_length,
        batch_size=run.batch_size,
    )
    logger.info(
        "Averaged over %d tokens from %d conversations (%d skipped)",
        result["num_tokens"],
        result["num_conversations"],
        result["num_skipped"],
    )

    torch.save(result, run.output)
    json_path = run.output.with_suffix(".json")
    json_path.write_text(
        json.dumps(
            {
                "model": run.model,
                "dataset": run.dataset,
                "layer_norms": result["layer_norms"].tolist(),
                "num_layers": result["num_layers"],
                "num_tokens": result["num_tokens"],
                "num_conversations": result["num_conversations"],
                "num_skipped": result["num_skipped"],
            },
            indent=2,
        )
    )
    logger.info("Saved layer norms to %s and %s", run.output, json_path)


if __name__ == "__main__":
    from jsonargparse import auto_cli

    auto_cli(main)
