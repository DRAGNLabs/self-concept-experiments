"""Causal interventions: factorizability, position independence, and mean interventions."""

import torch

from .activations import EncodedPrompt

SWAP_CONDITIONS = ("swap_entities", "swap_attributes", "swap_both")
# Same offsets with each delta replaced by a random vector of matching norm (paper's control)
RANDOM_CONDITIONS = ("random_entities", "random_attributes", "random_both")
CONDITIONS = ("none", *SWAP_CONDITIONS, *RANDOM_CONDITIONS)


def random_like_norm(delta: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Random directions with the same norm as `delta` at every layer and position."""
    noise = torch.randn(delta.shape, generator=generator, dtype=torch.float32).to(delta)
    return noise * (delta.norm(dim=-1, keepdim=True) / noise.norm(dim=-1, keepdim=True))


def mean_intervention_offsets(
    prompt: EncodedPrompt,
    condition: str,
    delta_entity: torch.Tensor,
    delta_attribute: torch.Tensor,
) -> torch.Tensor:
    """Additive offsets over the context that swap the two pairs' binding IDs.

    Deltas are (n_layers, positions_per_mention, hidden). Returns (n_layers, context_length,
    hidden), zero except at the swapped positions. For random_* conditions, pass random
    deltas; the condition name only selects which side is offset.
    """
    n_layers, _, hidden = delta_entity.shape
    offsets = delta_entity.new_zeros(n_layers, prompt.context_length, hidden)
    if condition.endswith(("_entities", "_both")):
        first, second = prompt.entity_positions
        offsets[:, first] += delta_entity
        offsets[:, second] -= delta_entity
    if condition.endswith(("_attributes", "_both")):
        first, second = prompt.attribute_positions
        offsets[:, first] += delta_attribute
        offsets[:, second] -= delta_attribute
    return offsets


def expected_answer(condition: str, query_index: int) -> int:
    """Index of the attribute the binding-ID account predicts for the queried entity.

    Swapping one side's IDs rebinds each entity to the other attribute; swapping both
    sides leaves the pairs matched, so the original answer should come back. Random
    offsets shouldn't move any binding.
    """
    flipped = condition in ("swap_entities", "swap_attributes")
    return 1 - query_index if flipped else query_index
