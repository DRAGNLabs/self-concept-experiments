"""Project per-token assistant activations onto persona axes, split by thinking/response.

For every conversation in the per-role response JSONL files, runs the model to capture the
middle-layer hidden state at each assistant token, projects each token onto the (unit) all-tokens
and response-only axes, and splits the tokens into the thinking block (inside <think>...</think>)
and the response block (after </think>). One scalar projection per token is kept -- the heavy
hidden states never leave the batch -- so the output is a small per-conversation record that the
companion plotting script turns into projection-vs-position traces.

Usage (from experiments/assistant-axis):
    python scripts/projection/token_axis_projections.py --config configs/projection/token_axis_projections.yaml
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypedDict

import jsonlines
import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor
from tqdm import tqdm

from selfconcept.assistant_axis.internals import ActivationExtractor, ConversationEncoder, ProbingModel
from selfconcept.assistant_axis.internals.model_specifics._shared import (
    _iter_over_turns,
    _overlaps_any,
    _think_char_spans,
)
from selfconcept.assistant_axis.models import get_config
from selfconcept.common.hf_strong_types import Conversation, HFTokenizer, configure_call
from selfconcept.common.paths import scratch_dir

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BlockName = Literal["thinking", "response"]
AxisName = Literal["all_tokens", "response_only"]


class ConversationProjections(TypedDict):
    """Per-token axis projections for one conversation, indexed by block then axis."""

    role: str
    is_default: bool
    projections_by_block_by_axis: dict[BlockName, dict[AxisName, Float[np.ndarray, " n_tokens"]]]


@dataclass(frozen=True)
class RunConfig:
    """Model, data, axes, and the layer at which to project."""

    model: str = "allenai/Olmo-3.1-32B-Think"
    responses_dir: Path = scratch_dir("assistant-axis") / "olmo32b-thinking" / "responses"
    axis_all_tokens: Path = scratch_dir("assistant-axis") / "olmo32b-thinking" / "axis_all_tokens.pt"
    axis_response_only: Path = scratch_dir("assistant-axis") / "olmo32b-thinking" / "axis_response_only.pt"
    output: Path = scratch_dir("assistant-axis") / "olmo32b-thinking" / "token_axis_projections.pt"
    target_layer: int | None = None  # None -> get_config(model)'s middle layer
    batch_size: int = 8
    max_length: int = 40960
    roles: list[str] | None = None


def load_unit_axis(axis_path: Path, target_layer: int) -> Float[Tensor, " hidden"]:
    """Load an axis .pt and return the unit direction at the target layer."""
    data = torch.load(axis_path, map_location="cpu", weights_only=False)
    axis: Float[Tensor, "layers hidden"] = (data["axis"] if isinstance(data, dict) else data).float()
    direction = axis[target_layer]
    return direction / direction.norm()


def load_conversations(responses_file: Path) -> list[Conversation]:
    """Load the conversation field from every record in a role's response JSONL."""
    with jsonlines.open(responses_file, "r") as reader:
        return [record["conversation"] for record in reader]


def assistant_turn_positions_by_block(
    full_ids: list[int], tokenizer: HFTokenizer
) -> dict[BlockName, list[int]] | None:
    """Absolute token positions of the first assistant turn, split at </think>.

    Returns None when the turn's tokens cannot be re-aligned to character offsets (so the
    <think> block boundary is untrustworthy), matching the guard in the shared span logic.
    """
    assistant_ranges = [
        (content_start, content_end)
        for _, role, content_start, content_end in _iter_over_turns(tokenizer, full_ids)
        if role == "assistant"
    ]
    if not assistant_ranges:
        return None
    content_start, content_end = assistant_ranges[0]

    turn_ids = full_ids[content_start:content_end]
    turn_text = tokenizer.decode(turn_ids)
    think_char_spans = _think_char_spans(turn_text)
    encoded = configure_call(tokenizer).return_offsets_mapping(True)(turn_text, add_special_tokens=False)
    if encoded["input_ids"] != turn_ids:
        return None

    positions_by_block: dict[BlockName, list[int]] = {"thinking": [], "response": []}
    for local_index, (char_start, char_end) in enumerate(encoded["offset_mapping"]):
        block: BlockName = "thinking" if _overlaps_any(char_start, char_end, think_char_spans) else "response"
        positions_by_block[block].append(content_start + local_index)
    return positions_by_block


def project_batch(
    extractor: ActivationExtractor,
    encoder: ConversationEncoder,
    conversations: list[Conversation],
    unit_axis_by_name: dict[AxisName, Float[Tensor, " hidden"]],
    target_layer: int,
    max_length: int,
) -> list[dict[BlockName, dict[AxisName, Float[np.ndarray, " n_tokens"]]]]:
    """Extract, project, and block-split one batch of conversations."""
    assert encoder.tokenizer is not None
    tokenizer = encoder.tokenizer

    activations, metadata = extractor.batch_conversations(conversations, layer=[target_layer], max_length=max_length)
    layer_activations: Float[Tensor, "batch seq hidden"] = activations[0].float()
    projections_by_axis = {
        axis_name: (layer_activations @ unit.to(layer_activations.device)).cpu().numpy()
        for axis_name, unit in unit_axis_by_name.items()
    }

    batch_records: list[dict[BlockName, dict[AxisName, Float[np.ndarray, " n_tokens"]]]] = []
    for conversation_index, conversation in enumerate(conversations):
        full_ids = encoder.token_ids(conversation, add_generation_prompt=False)
        positions_by_block = assistant_turn_positions_by_block(full_ids, tokenizer)
        truncated_length = metadata["truncated_lengths"][conversation_index]

        projections_by_block_by_axis: dict[BlockName, dict[AxisName, Float[np.ndarray, " n_tokens"]]] = {}
        if positions_by_block is not None:
            for block, positions in positions_by_block.items():
                kept = np.array([position for position in positions if position < truncated_length], dtype=np.int64)
                projections_by_block_by_axis[block] = {
                    axis_name: projections_by_axis[axis_name][conversation_index, kept].astype(np.float32)
                    for axis_name in unit_axis_by_name
                }
        batch_records.append(projections_by_block_by_axis)
    return batch_records


def main(run: RunConfig = RunConfig()) -> None:
    """Project every role's assistant tokens onto both axes and save per-conversation records."""
    output_path = run.output.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading model: {run.model}")
    probing_model = ProbingModel(run.model)
    assert probing_model.tokenizer is not None
    encoder = ConversationEncoder(probing_model.tokenizer, probing_model.model_name)
    extractor = ActivationExtractor(probing_model, encoder)

    target_layer = run.target_layer if run.target_layer is not None else get_config(run.model)["target_layer"]
    logger.info(f"Projecting at layer {target_layer} of {len(probing_model.get_layers())}")

    unit_axis_by_name: dict[AxisName, Float[Tensor, " hidden"]] = {
        "all_tokens": load_unit_axis(run.axis_all_tokens.expanduser(), target_layer),
        "response_only": load_unit_axis(run.axis_response_only.expanduser(), target_layer),
    }

    response_files = sorted(run.responses_dir.expanduser().glob("*.jsonl"))
    if run.roles:
        response_files = [file for file in response_files if file.stem in run.roles]

    records: list[ConversationProjections] = []
    for response_file in tqdm(response_files, desc="Roles"):
        role = response_file.stem
        conversations = load_conversations(response_file)
        for batch_start in range(0, len(conversations), run.batch_size):
            batch = conversations[batch_start : batch_start + run.batch_size]
            for projections_by_block_by_axis in project_batch(
                extractor, encoder, batch, unit_axis_by_name, target_layer, run.max_length
            ):
                records.append(
                    ConversationProjections(
                        role=role,
                        is_default=role == "default",
                        projections_by_block_by_axis=projections_by_block_by_axis,
                    )
                )
        logger.info(f"Projected {role}: {len(conversations)} conversations")

    torch.save({"model": run.model, "target_layer": target_layer, "records": records}, output_path)
    logger.info(f"Saved {len(records)} conversation projections to {output_path}")


if __name__ == "__main__":
    from jsonargparse import auto_cli

    auto_cli(main)
