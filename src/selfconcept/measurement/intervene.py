"""Residual-stream interventions at decoder-layer outputs: additive shift and projection capping.

Both are forward hooks on the same post-decoder-layer site that ``capture`` reads, applied at every position
of every forward pass for the duration of the context (prompt and generated tokens alike).

- ``add_direction``: h <- h + coefficient * direction. Callers pass an absolute coefficient; the convention
  used in the steering experiments is alpha * (mean activation norm at the layer) with a unit direction.
- ``cap_direction``: h <- h - max(0, <h, u> - threshold) * u, with u the unit direction, at each listed layer.
  The projection onto u is clipped at the threshold from above; nothing changes below it.

The HF path (``add_direction`` / ``cap_direction``) hooks the decoder layers' outputs. vLLM's gpt-oss block returns
``(mlp_output, residual)`` with the residual stream materialized only as their sum in the next layer's fused norm;
``register_vllm_transforms`` applies the same transforms to that sum, so a steered vLLM run and a steered HF run
intervene on the same quantity at the same site.
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


def add_transform(direction: torch.Tensor, coefficient: float):
    """h <- h + coefficient * direction (direction used as given)."""
    vector = direction.float() * coefficient

    def transform(hidden):
        return hidden + vector.to(hidden.device, hidden.dtype)

    return transform


def cap_transform(direction: torch.Tensor, threshold: float):
    """h <- h - max(0, <h, u> - threshold) * u with u = direction / |direction|."""
    u = unit(direction)
    threshold = float(threshold)

    def transform(hidden):
        u_local = u.to(hidden.device, torch.float32)
        projection = hidden.float() @ u_local
        excess = (projection - threshold).clamp_min(0)
        return (hidden.float() - excess.unsqueeze(-1) * u_local).to(hidden.dtype)

    return transform


def register_vllm_transforms(layers, transforms_by_layer: dict[int, object]) -> list:
    """Register the transforms as forward hooks on vLLM gpt-oss ``TransformerBlock`` modules (``model.model.layers``).

    Each block returns ``(mlp_output, residual)``; the residual stream after the block is their sum ``s``. The hook
    returns ``(mlp_output + (transform(s) - s), residual)`` so the next block sees ``transform(s)``. Handles are
    returned, not managed by a context, because the hooks must outlive the call that registers them inside the
    engine's worker (``LLM.apply_model``). Not for compiled or CUDA-graph-captured models: register before
    capture or run with ``enforce_eager=True``."""
    handles = []
    for layer, transform in transforms_by_layer.items():
        if layer < 0 or layer >= len(layers):
            raise ValueError("Intervention layer is outside the model")

        def hook(_module, _inputs, output, transform=transform):
            mlp_output, residual = output
            stream = mlp_output + residual
            return (mlp_output + (transform(stream) - stream).to(mlp_output.dtype), residual)

        handles.append(layers[layer].register_forward_hook(hook))
    return handles


def patch_vllm_blocks(block_cls, transforms_by_layer: dict[int, object]):
    """Patch a vLLM block class's ``forward`` so blocks whose ``layer_idx`` is in ``transforms_by_layer`` apply the
    transform to the residual stream ``mlp_output + residual`` (same arithmetic as ``register_vllm_transforms``).

    Unlike a hook registered after engine start, a class patch made before ``LLM(...)`` is traced into torch.compile
    and captured into CUDA graphs, so it stays in effect at full speed. Transform tensors must already live on the
    model's device (no host-to-device copies inside a captured graph). Returns a function that restores the
    original forward."""
    original = block_cls.forward
    transforms = dict(transforms_by_layer)

    def forward(self, hidden_states, positions, residual):
        mlp_output, residual = original(self, hidden_states, positions, residual)
        transform = transforms.get(self.layer_idx)
        if transform is not None:
            stream = mlp_output + residual
            mlp_output = mlp_output + (transform(stream) - stream).to(mlp_output.dtype)
        return mlp_output, residual

    block_cls.forward = forward

    def restore():
        block_cls.forward = original

    return restore


def unit(direction: torch.Tensor) -> torch.Tensor:
    norm = direction.float().norm()
    if not torch.isfinite(norm) or norm <= 0:
        raise ValueError("Direction must be finite and nonzero")
    return direction.float() / norm


@contextmanager
def add_direction(model: nn.Module, layer: int, direction: torch.Tensor, coefficient: float) -> Generator[None]:
    """Add ``coefficient * direction`` (direction used as given, not normalized) at one layer."""
    with _hooks(model, {layer: add_transform(direction, coefficient)}):
        yield


@contextmanager
def cap_direction(model: nn.Module, thresholds_by_layer: dict[int, float],
                  directions_by_layer: dict[int, torch.Tensor]) -> Generator[None]:
    """Clip the projection onto each layer's unit direction at that layer's threshold, from above."""
    if set(thresholds_by_layer) != set(directions_by_layer):
        raise ValueError("Cap thresholds and directions must cover the same layers")
    transforms = {layer: cap_transform(directions_by_layer[layer], threshold)
                  for layer, threshold in thresholds_by_layer.items()}
    with _hooks(model, transforms):
        yield
