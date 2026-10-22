"""Load the layer-by-hidden tensors produced by the assistant-axis pipeline."""
from pathlib import Path

from selfconcept.measurement.interface import Measurement
from selfconcept.measurement.unit_axes import load_unit_axes, unit_axis_scorer


def build_axis_measurement(path: Path, layers: list[int], num_layers: int, hidden_size: int,
                           model_name: str | None = None) -> Measurement:
    """The first of ``layers`` is the primary layer."""
    directions_by_layer = load_unit_axes(path, layers, num_layers, hidden_size, model_name)
    return Measurement("assistant-axis", list(layers), layers[0], unit_axis_scorer(directions_by_layer))
