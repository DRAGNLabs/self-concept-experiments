# Derived from safety-research/assistant-axis (https://github.com/safety-research/assistant-axis),
# MIT licensed. See the NOTICE file at the repository root for the full license text.
"""Additive activation steering along a direction at one decoder layer.

Reproduces the Section 3.2.1 setup: add ``coefficient * vector`` to a decoder layer's
output (the post-MLP residual stream) at every token position (``positions="all"``) or
only the final position (``positions="last"``), for the duration of a context, then
remove the hook. This is the same site the axis pipeline reads activations from, located
via ``ProbingModel.get_layers``.

Ported from the reference ActivationSteering (addition path only).
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from typing import Any, Literal

import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from selfconcept.assistant_axis.internals.model import ProbingModel

SteerPositions = Literal["all", "last"]


def _add_steering(
    hidden_states: Float[Tensor, "batch seq hidden"],
    steering: Float[Tensor, "hidden"],
    positions: SteerPositions,
) -> Float[Tensor, "batch seq hidden"]:
    if positions == "all":
        return hidden_states + steering
    updated = hidden_states.clone()
    updated[:, -1, :] += steering
    return updated


@contextmanager
def apply_steering(
    probing_model: ProbingModel,
    layer: int,
    vector: Float[Tensor, "hidden"],
    coefficient: float,
    *,
    positions: SteerPositions = "all",
) -> Generator[None, None, None]:
    """Additively steer one decoder layer's output for the duration of the context."""
    steering_vector = coefficient * vector

    def hook(_module: nn.Module, _inputs: Any, output: Any) -> Any:
        hidden_states = output[0] if isinstance(output, tuple) else output
        steering = steering_vector.to(hidden_states.device, hidden_states.dtype)
        steered = _add_steering(hidden_states, steering, positions)
        if isinstance(output, tuple):
            return (steered, *output[1:])
        return steered

    handle = probing_model.get_layers()[layer].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()
