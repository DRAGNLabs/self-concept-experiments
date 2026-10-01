from typing import Literal, NamedTuple, Protocol

from jaxtyping import Float
from torch import Tensor

MeasurementName = Literal["cotness", "assistant-axis"]


class TokenScorer(Protocol):
    def __call__(self, *, layer: int, residual: Float[Tensor, "seq hidden"]) -> Float[Tensor, "seq"]: ...


class Measurement(NamedTuple):
    name: MeasurementName
    layers: list[int]
    primary_layer: int
    score: TokenScorer
