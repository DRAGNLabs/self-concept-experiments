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
from typing import Literal

from jaxtyping import Float
from torch import Tensor

from selfconcept.assistant_axis.internals.model import ProbingModel
from selfconcept.common.steering import steer_decoder_layer

SteerPositions = Literal["all", "last"]


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
    with steer_decoder_layer(probing_model.get_layers()[layer], vector, coefficient, positions=positions):
        yield
