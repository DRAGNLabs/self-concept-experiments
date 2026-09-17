"""Causal interventions: factorizability, position independence, and mean interventions."""

import torch

from .activations import EncodedPrompt

CONDITIONS = ("none", "swap_entities", "swap_attributes", "swap_both")


def mean_intervention_offsets(
    prompt: EncodedPrompt,
    condition: str,
    delta_entity: torch.Tensor,
    delta_attribute: torch.Tensor,
) -> torch.Tensor:
    """Additive offsets over the context that swap the two pairs' binding IDs.

    Shape (n_layers, context_length, hidden); zero except at the swapped positions.
    """
    n_layers, hidden = delta_entity.shape
    offsets = delta_entity.new_zeros(n_layers, prompt.context_length, hidden)
    if condition in ("swap_entities", "swap_both"):
        first, second = prompt.entity_positions
        offsets[:, first] += delta_entity
        offsets[:, second] -= delta_entity
    if condition in ("swap_attributes", "swap_both"):
        first, second = prompt.attribute_positions
        offsets[:, first] += delta_attribute
        offsets[:, second] -= delta_attribute
    return offsets


def expected_answer(condition: str, query_index: int) -> int:
    """Index of the attribute the binding-ID account predicts for the queried entity.

    Swapping one side's IDs rebinds each entity to the other attribute; swapping both
    sides leaves the pairs matched, so the original answer should come back.
    """
    flipped = condition in ("swap_entities", "swap_attributes")
    return 1 - query_index if flipped else query_index
