"""Mean ablation of a direction at one decoder layer's output (the residual stream)."""

from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

from jaxtyping import Bool, Float
from torch import Tensor, nn


def mean_ablate(
    residual_stream: Float[Tensor, "batch seq hidden"],
    unit_direction: Float[Tensor, " hidden"],
    target_projection: float,
    mask: Bool[Tensor, "batch seq"],
) -> Float[Tensor, "batch seq hidden"]:
    """Sets the projection onto unit_direction to target_projection at the masked positions, leaving every orthogonal
    component, and every unmasked position, unchanged."""
    if mask.shape != residual_stream.shape[:2]:
        raise ValueError(f"ablation mask {tuple(mask.shape)} does not match residual {tuple(residual_stream.shape)}")
    direction = unit_direction.to(residual_stream.device).float()
    projection = residual_stream.float() @ direction
    shift = mask.to(residual_stream.device) * (target_projection - projection)
    return (residual_stream.float() + shift[..., None] * direction).to(residual_stream.dtype)


@contextmanager
def mean_ablate_decoder_layer(
    decoder_layer: nn.Module,
    unit_direction: Float[Tensor, " hidden"],
    target_projection: float,
    mask: Bool[Tensor, "batch seq"],
) -> Generator[None, None, None]:
    """Mean-ablates the layer's output along unit_direction at the masked positions for the duration of the context."""

    def hook(_module: nn.Module, _inputs: Any, output: Any) -> Any:
        hidden_states = output[0] if isinstance(output, tuple) else output
        ablated = mean_ablate(hidden_states, unit_direction, target_projection, mask)
        if isinstance(output, tuple):
            return (ablated, *output[1:])
        return ablated

    handle = decoder_layer.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()
