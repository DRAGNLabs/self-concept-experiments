from collections.abc import Iterator
from contextlib import contextmanager

from jaxtyping import Float
from torch import nn, Tensor

from selfconcept.soo.activations import get_decoder_layers
from .interface import TokenScorer


@contextmanager
def capture(
    model: nn.Module, layers: list[int], score: TokenScorer | None = None,
) -> Iterator[dict[int, list[Float[Tensor, "seq hidden"]] | list[Float[Tensor, "seq"]]]]:
    """Capture residual output (not attention o_proj), scored on-device when ``score`` is given.

    Generation consumes the prompt then one token at a time. Its final sampled
    token is unprocessed; callers explicitly exclude it from token scores.
    """
    values_by_layer = {layer: [] for layer in layers}
    handles = []
    for layer in layers:
        def hook(module, inputs, output, layer=layer):
            hidden = output[0] if isinstance(output, tuple) else output
            hidden = hidden.detach().float()[0]
            if score is not None:
                hidden = score(layer=layer, residual=hidden)
            values_by_layer[layer].append(hidden.cpu())
        handles.append(get_decoder_layers(model)[layer].register_forward_hook(hook))
    try:
        yield values_by_layer
    finally:
        for handle in handles:
            handle.remove()
