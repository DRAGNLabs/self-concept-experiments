"""Residual-stream interventions at decoder-layer outputs: additive shift and projection capping.

Both are forward hooks on the same post-decoder-layer site that ``capture`` reads, applied at every position
of every forward pass for the duration of the context (prompt and generated tokens alike).

- ``add_direction``: h <- h + coefficient * direction. Callers pass an absolute coefficient; the convention
  used in the steering experiments is alpha * (mean activation norm at the layer) with a unit direction.
- ``cap_direction``: h <- h - max(0, <h, u> - threshold) * u, with u the unit direction, at each listed layer.
  The projection onto u is clipped at the threshold from above; nothing changes below it.
"""
from collections.abc import Generator
from contextlib import contextmanager

import torch
from torch import nn

from selfconcept.soo.activations import get_decoder_layers


def _apply(output, transform):
    hidden = output[0] if isinstance(output, tuple) else output
    steered = transform(hidden)
    return (steered, *output[1:]) if isinstance(output, tuple) else steered


@contextmanager
def _hooks(model: nn.Module, transforms_by_layer: dict[int, object]) -> Generator[None]:
    decoder_layers = get_decoder_layers(model)
    if any(layer < 0 or layer >= len(decoder_layers) for layer in transforms_by_layer):
        raise ValueError("Intervention layer is outside the model")
    handles = []
    for layer, transform in transforms_by_layer.items():
        def hook(_module, _inputs, output, transform=transform):
            return _apply(output, transform)
        handles.append(decoder_layers[layer].register_forward_hook(hook))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def unit(direction: torch.Tensor) -> torch.Tensor:
    norm = direction.float().norm()
    if not torch.isfinite(norm) or norm <= 0:
        raise ValueError("Direction must be finite and nonzero")
    return direction.float() / norm


@contextmanager
def add_direction(model: nn.Module, layer: int, direction: torch.Tensor, coefficient: float) -> Generator[None]:
    """Add ``coefficient * direction`` (direction used as given, not normalized) at one layer."""
    vector = direction.float() * coefficient

    def transform(hidden):
        return hidden + vector.to(hidden.device, hidden.dtype)

    with _hooks(model, {layer: transform}):
        yield


@contextmanager
def cap_direction(model: nn.Module, thresholds_by_layer: dict[int, float],
                  directions_by_layer: dict[int, torch.Tensor]) -> Generator[None]:
    """Clip the projection onto each layer's unit direction at that layer's threshold, from above."""
    if set(thresholds_by_layer) != set(directions_by_layer):
        raise ValueError("Cap thresholds and directions must cover the same layers")
    transforms = {}
    for layer, threshold in thresholds_by_layer.items():
        u = unit(directions_by_layer[layer])

        def transform(hidden, u=u, threshold=float(threshold)):
            u_local = u.to(hidden.device, torch.float32)
            projection = hidden.float() @ u_local
            excess = (projection - threshold).clamp_min(0)
            return (hidden.float() - excess.unsqueeze(-1) * u_local).to(hidden.dtype)

        transforms[layer] = transform
    with _hooks(model, transforms):
        yield
