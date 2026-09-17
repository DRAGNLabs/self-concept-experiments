"""Forward-hook utilities for capturing and patching activations at entity/attribute positions."""

from dataclasses import dataclass

import torch
from torch import nn

from .tasks import BindingRow, query_prompt


@dataclass
class EncodedPrompt:
    input_ids: torch.Tensor  # (1, seq)
    context_length: int  # leading tokens that lie entirely inside the context
    entity_positions: list[int]  # last token of each entity mention
    attribute_positions: list[int]  # last token of each attribute mention


def decoder_layers(hf_model: nn.Module) -> nn.ModuleList:
    for path in ("model.layers", "transformer.h"):  # Llama-style, GPT-2
        try:
            layers = hf_model.get_submodule(path)
        except AttributeError:
            continue
        if isinstance(layers, nn.ModuleList):
            return layers
    raise AttributeError(f"cannot find decoder layers on {type(hf_model).__name__}")


def encode_prompt(tokenizer, row: BindingRow, query_index: int, device: torch.device) -> EncodedPrompt:
    enc = tokenizer(query_prompt(row, query_index), return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    context_end = len(row["context"])

    context_length = 0
    while context_length < len(offsets) and offsets[context_length][1] <= context_end:
        context_length += 1

    return EncodedPrompt(
        input_ids=torch.tensor([enc["input_ids"]], device=device),
        context_length=context_length,
        entity_positions=[_token_covering(offsets, end - 1) for _, end in row["entity_spans"]],
        attribute_positions=[_token_covering(offsets, end - 1) for _, end in row["attribute_spans"]],
    )


def _token_covering(offsets: list[tuple[int, int]], char_index: int) -> int:
    for i, (start, end) in enumerate(offsets):
        if start <= char_index < end:
            return i
    raise ValueError(f"no token covers character {char_index}")


def _hidden(output) -> torch.Tensor:
    return output[0] if isinstance(output, tuple) else output


def run_with_cache(hf_model: nn.Module, layers: nn.ModuleList, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward pass returning (last-position logits, layer outputs of shape (n_layers, seq, hidden))."""
    outputs: list[torch.Tensor] = []
    handles = [
        layer.register_forward_hook(lambda _m, _i, out: outputs.append(_hidden(out)[0].detach().clone()))
        for layer in layers
    ]
    try:
        logits = hf_model(input_ids=input_ids).logits[0, -1]
    finally:
        for handle in handles:
            handle.remove()
    return logits, torch.stack(outputs)


def run_with_patched_context(
    hf_model: nn.Module,
    layers: nn.ModuleList,
    input_ids: torch.Tensor,
    clean: torch.Tensor,
    offsets: torch.Tensor,
    context_length: int,
) -> torch.Tensor:
    """Forward pass with every context position pinned to clean + offset at every layer.

    Pinning the whole context (not just adding offsets) matches the paper's setup of
    intervening on frozen context activations: an offset at one position can't
    propagate to later context positions, only to the query tokens that read them.
    """

    def make_hook(layer_index: int):
        def hook(_module, _inputs, output):
            _hidden(output)[0, :context_length] = clean[layer_index, :context_length] + offsets[layer_index]

        return hook

    handles = [layer.register_forward_hook(make_hook(i)) for i, layer in enumerate(layers)]
    try:
        return hf_model(input_ids=input_ids).logits[0, -1]
    finally:
        for handle in handles:
            handle.remove()
