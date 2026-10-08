"""Load layer-by-hidden axis tensors as per-layer unit directions and score residuals against them."""
from collections.abc import Sequence
from pathlib import Path

from jaxtyping import Float
import numpy as np
import torch
from torch import Tensor

from selfconcept.measurement.interface import TokenScorer

type UnitDirectionsByLayer = dict[int, dict[str, np.ndarray]]


def load_unit_axes(
    path: Path,
    layers: Sequence[int],
    num_layers: int,
    hidden_size: int,
    model_name: str | None = None,
) -> UnitDirectionsByLayer:
    """Return signed unit directions; never center activations or flip the axis.

    Legacy artifacts are bare tensors and cannot establish model identity. When
    a wrapped artifact includes a model name, enforce it as well as dimensions.
    """
    data = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(data, dict):
        if model_name is not None and data.get("model", model_name) != model_name:
            raise ValueError("Assistant axis belongs to a different model")
        data = data["axis"]
    if not isinstance(data, torch.Tensor) or data.shape != (num_layers, hidden_size):
        raise ValueError(f"Assistant axis must have shape ({num_layers}, {hidden_size})")
    if not layers or len(set(layers)) != len(layers) or any(l < 0 or l >= num_layers for l in layers):
        raise ValueError("Axis layers must be unique zero-based decoder layer indices")
    directions = {}
    for layer in layers:
        direction = data[layer].float()
        norm = direction.norm()
        if not torch.isfinite(direction).all() or not torch.isfinite(norm) or norm <= 0:
            raise ValueError(f"Assistant axis at layer {layer} must be finite and nonzero")
        directions[layer] = {"direction": (direction / norm).numpy()}
    return directions


def unit_axis_scorer(directions_by_layer: UnitDirectionsByLayer) -> TokenScorer:
    def score(*, layer: int, residual: Float[Tensor, "seq hidden"]) -> Float[Tensor, "seq"]:
        return residual @ torch.as_tensor(directions_by_layer[layer]["direction"], device=residual.device)
    return score
