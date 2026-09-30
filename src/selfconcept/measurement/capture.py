from collections.abc import Generator
from contextlib import AbstractContextManager, contextmanager
from typing import overload

from jaxtyping import Float
from torch import nn, Tensor
from torch.utils.hooks import RemovableHandle

from selfconcept.soo.activations import get_decoder_layers
from .interface import TokenScorer

type ResidualsByLayer = dict[int, list[Float[Tensor, "seq hidden"]]]
type ScoresByLayer = dict[int, list[Float[Tensor, "seq"]]]


@overload
def capture(model: nn.Module, layers: list[int], score: None = None) -> AbstractContextManager[ResidualsByLayer]: ...
@overload
def capture(model: nn.Module, layers: list[int], score: TokenScorer) -> AbstractContextManager[ScoresByLayer]: ...
@contextmanager
def capture(model: nn.Module, layers: list[int], score: TokenScorer | None = None) -> Generator[ResidualsByLayer | ScoresByLayer]:
    """Capture residual output (not attention o_proj), scored on-device when ``score`` is given.

    Generation consumes the prompt then one token at a time. Its final sampled
    token is unprocessed; callers explicitly exclude it from token scores.
    """
    values_by_layer: dict[int, list[Tensor]] = {layer: [] for layer in layers}
    decoder_layers: nn.ModuleList = get_decoder_layers(model)
    handles: list[RemovableHandle] = []
    for layer in layers:
        def hook(module: nn.Module, inputs: tuple[Tensor, ...], output: Tensor | tuple[Tensor, ...], layer: int = layer) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            hidden = hidden.detach().float()[0]
            if score is not None:
                hidden = score(layer=layer, residual=hidden)
            values_by_layer[layer].append(hidden.cpu())
        handles.append(decoder_layers[layer].register_forward_hook(hook))
    try:
        yield values_by_layer
    finally:
        for handle in handles:
            handle.remove()
