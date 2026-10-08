"""Additive activation steering along a direction at one decoder layer's output (the residual stream)."""

from collections.abc import Generator
from contextlib import contextmanager
from typing import Any, Literal

from jaxtyping import Bool, Float
from torch import Tensor, nn

type SteerPositions = Literal["all", "last"] | Bool[Tensor, "batch seq"]


def add_steering(
    residual_stream: Float[Tensor, "batch seq hidden"],
    steering: Float[Tensor, "hidden"],
    positions: SteerPositions,
) -> Float[Tensor, "batch seq hidden"]:
    if isinstance(positions, Tensor):
        if positions.shape != residual_stream.shape[:2]:
            raise ValueError(f"steering mask {tuple(positions.shape)} does not match residual {tuple(residual_stream.shape)}")
        mask = positions.to(residual_stream.device)
        return residual_stream + mask[..., None].to(residual_stream.dtype) * steering
    if positions == "all":
        return residual_stream + steering
    updated = residual_stream.clone()
    updated[:, -1, :] += steering
    return updated


@contextmanager
def steer_decoder_layer(
    decoder_layer: nn.Module,
    vector: Float[Tensor, "hidden"],
    coefficient: float,
    *,
    positions: SteerPositions = "all",
) -> Generator[None, None, None]:
    """Adds ``coefficient * vector`` to the layer's output for the duration of the context."""
    steering_vector = coefficient * vector

    def hook(_module: nn.Module, _inputs: Any, output: Any) -> Any:
        hidden_states = output[0] if isinstance(output, tuple) else output
        if isinstance(positions, Tensor) and hidden_states.shape[1] == 1 and positions.shape != hidden_states.shape[:2]:
            # A mask covers the prompt. During cached generation each later step feeds one new token, so its shape
            # no longer matches; that token is unmasked, and the masked positions' steering is already in the KV cache.
            return output
        steering = steering_vector.to(hidden_states.device, hidden_states.dtype)
        steered = add_steering(hidden_states, steering, positions)
        if isinstance(output, tuple):
            return (steered, *output[1:])
        return steered

    handle = decoder_layer.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()
