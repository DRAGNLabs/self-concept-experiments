"""Measure per-layer mean post-MLP residual-stream norms over a set of conversations.

A forward hook on each decoder layer captures that layer's output hidden state (the post-MLP
residual stream); we take its L2 norm at every non-padding token and average across all tokens
and conversations, yielding one scalar per layer. This is the activation-magnitude baseline used
to normalise steering-vector strength per layer.
"""

from __future__ import annotations

import logging
from typing import Any, TypedDict

import torch
import torch.nn as nn
from jaxtyping import Float, Int

from selfconcept.assistant_axis.internals.conversation import ConversationEncoder
from selfconcept.assistant_axis.internals.model import ProbingModel
from selfconcept.common.hf_strong_types import Conversation
from selfconcept.common.torch_utils import build_padded_batch

logger = logging.getLogger(__name__)


class LayerNormResult(TypedDict):
    layer_norms: Float[torch.Tensor, "num_layers"]  # mean L2 norm per requested layer
    num_layers: int
    num_tokens: int
    num_conversations: int
    num_skipped: int


def _tokenize_conversations(
    encoder: ConversationEncoder,
    conversations: list[Conversation],
    chat_kwargs: dict[str, Any],
) -> tuple[list[list[int]], int]:
    """Templatise + tokenise each conversation, skipping any that fail templating."""
    token_id_lists: list[list[int]] = []
    num_skipped = 0
    for conversation in conversations:
        try:
            token_id_lists.append(
                encoder.token_ids(conversation, add_generation_prompt=False, **chat_kwargs)
            )
        except Exception:
            num_skipped += 1
    if num_skipped:
        logger.warning("Skipped %d conversations that failed chat templating", num_skipped)
    return token_id_lists, num_skipped


def _capture_layer_activations(
    model: nn.Module,
    layer_modules: nn.ModuleList,
    layer_indices: list[int],
    input_ids: Int[torch.Tensor, "batch seq"],
    attention_mask: Int[torch.Tensor, "batch seq"],
) -> dict[int, Float[torch.Tensor, "batch seq hidden"]]:
    """Run one forward pass, returning each requested layer's output hidden state via hooks."""
    activations_by_layer: dict[int, Float[torch.Tensor, "batch seq hidden"]] = {}

    def make_hook(layer_index: int):
        def hook(_module: nn.Module, _inputs: Any, output: Any) -> None:
            activations_by_layer[layer_index] = output[0] if isinstance(output, tuple) else output

        return hook

    handles = [layer_modules[i].register_forward_hook(make_hook(i)) for i in layer_indices]
    try:
        with torch.inference_mode():
            model(input_ids=input_ids, attention_mask=attention_mask)
    finally:
        for handle in handles:
            handle.remove()
    return activations_by_layer


def _batch_summed_layer_norms(
    model: nn.Module,
    layer_modules: nn.ModuleList,
    layers: list[int],
    input_ids: Int[torch.Tensor, "batch seq"],
    attention_mask: Int[torch.Tensor, "batch seq"],
) -> Float[torch.Tensor, "num_layers"]:
    """Summed L2 norm of non-pad tokens per layer for one batch, accumulated in float64."""
    activations_by_layer = _capture_layer_activations(
        model, layer_modules, layers, input_ids, attention_mask
    )
    summed_norms = torch.zeros(len(layers), dtype=torch.float64)
    for layer_position, layer_index in enumerate(layers):
        token_norms = activations_by_layer[layer_index].float().norm(dim=-1)
        mask = attention_mask.to(device=token_norms.device, dtype=token_norms.dtype)
        summed_norms[layer_position] = float((token_norms * mask).sum().item())
    return summed_norms


def measure_layer_norms(
    probing_model: ProbingModel,
    conversations: list[Conversation],
    layers: list[int],
    max_length: int = 2048,
    batch_size: int = 8,
    **chat_kwargs: Any,
) -> LayerNormResult:
    """Mean L2 norm of the post-MLP residual stream per layer, over all non-pad tokens."""
    assert probing_model.model is not None and probing_model.tokenizer is not None
    pad_token_id = probing_model.tokenizer.pad_token_id
    assert pad_token_id is not None

    encoder = ConversationEncoder(probing_model.tokenizer, probing_model.model_name)
    layer_modules = probing_model.get_layers()
    device = probing_model.device

    token_id_lists, num_skipped = _tokenize_conversations(encoder, conversations, chat_kwargs)

    summed_norm_by_layer = torch.zeros(len(layers), dtype=torch.float64)
    token_count = 0
    for batch_start in range(0, len(token_id_lists), batch_size):
        batch = token_id_lists[batch_start : batch_start + batch_size]
        input_ids, attention_mask = build_padded_batch(batch, pad_token_id, max_length, device)
        summed_norm_by_layer += _batch_summed_layer_norms(
            probing_model.model, layer_modules, layers, input_ids, attention_mask
        )
        token_count += int(attention_mask.sum().item())
        if (batch_start // batch_size) % 5 == 0:
            torch.cuda.empty_cache()

    layer_norms = (summed_norm_by_layer / token_count).to(torch.float32)
    return LayerNormResult(
        layer_norms=layer_norms,
        num_layers=len(layers),
        num_tokens=token_count,
        num_conversations=len(token_id_lists),
        num_skipped=num_skipped,
    )
