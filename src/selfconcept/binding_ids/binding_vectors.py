"""Estimate binding vectors from activation differences and study their geometry."""

import torch
from torch import nn

from .activations import encode_prompt, run_with_cache
from .tasks import BindingRow


def estimate_binding_deltas(
    hf_model: nn.Module,
    layers: nn.ModuleList,
    tokenizer,
    rows: list[BindingRow],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean (second pair − first pair) activation difference at entity and attribute positions.

    Returns (delta_entity, delta_attribute), each (n_layers, positions_per_mention, hidden).
    Averaged over contexts this equals the paper's E[Z_1(c) − Z_0(c')] over independent
    c, c'. Under the binding-ID account these are the binding-vector offsets between pairs.
    """
    delta_entity = delta_attribute = torch.zeros(())
    for row in rows:
        # Context activations don't depend on the query, so either query works
        prompt = encode_prompt(tokenizer, row, query_index=0, device=device)
        _, clean = run_with_cache(hf_model, layers, prompt.input_ids)
        first_e, second_e = prompt.entity_positions
        first_a, second_a = prompt.attribute_positions
        # Accumulate in fp32: summing hundreds of bf16 differences loses precision
        clean = clean.float()
        delta_entity = delta_entity + clean[:, second_e] - clean[:, first_e]
        delta_attribute = delta_attribute + clean[:, second_a] - clean[:, first_a]
    return delta_entity / len(rows), delta_attribute / len(rows)
