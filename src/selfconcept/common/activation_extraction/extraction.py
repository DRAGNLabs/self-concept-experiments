from collections.abc import Sequence

import torch
from jaxtyping import Float
from torch import Tensor, nn
from torch.utils.hooks import RemovableHandle
from transformers import PreTrainedModel

from selfconcept.common.activation_extraction.model_specifics import ModelSpecifics
from selfconcept.common.span_targeting.span_targeters import SpanTargeter, build_span_mask
from selfconcept.common.span_targeting.types import PromptCompletion, TokenSpan
from selfconcept.common.torch_utils import build_padded_batch


def extract_span_activations(
    model: PreTrainedModel,
    examples: Sequence[PromptCompletion],
    targeter: SpanTargeter,
    model_specifics: ModelSpecifics,
    pad_token_id: int,
    max_length: int,
    layers: Sequence[int] | None = None,
) -> list[Float[Tensor, "layer token hidden"]]:
    """Decoder-layer outputs at each example's targeted tokens; all layers when ``layers`` is None."""
    return extract_token_span_activations(
        model,
        [example.prompt_token_ids + example.completion_token_ids for example in examples],
        [targeter.target_spans(model_specifics.split_completion(example)) for example in examples],
        model_specifics.get_decoder_layers(model),
        pad_token_id,
        max_length,
        layers,
    )


def extract_token_span_activations(
    model: PreTrainedModel,
    token_ids_by_example: Sequence[list[int]],
    token_spans_by_example: Sequence[list[TokenSpan]],
    decoder_layers: nn.ModuleList,
    pad_token_id: int,
    max_length: int,
    layers: Sequence[int] | None = None,
) -> list[Float[Tensor, "layer token hidden"]]:
    """Decoder-layer outputs at each example's span tokens, concatenated in span order; spans past
    max_length are clipped. All layers when ``layers`` is None."""
    input_ids, attention_mask = build_padded_batch(list(token_ids_by_example), pad_token_id, max_length, model.device)
    target_mask = build_span_mask(token_spans_by_example, input_ids.shape[1], model.device)
    selected_layers = range(len(decoder_layers)) if layers is None else layers

    targeted_hidden_by_layer: dict[int, Float[Tensor, "total_token hidden"]] = {}

    def make_hook(layer: int):
        def hook(module: nn.Module, inputs: tuple[Tensor, ...], output: Tensor | tuple[Tensor, ...]) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            targeted_hidden_by_layer[layer] = hidden[target_mask.to(hidden.device)]

        return hook

    handles: list[RemovableHandle] = [
        decoder_layers[layer].register_forward_hook(make_hook(layer)) for layer in selected_layers
    ]
    try:
        with torch.inference_mode():
            model(input_ids=input_ids, attention_mask=attention_mask)
    finally:
        for handle in handles:
            handle.remove()

    output_device = targeted_hidden_by_layer[selected_layers[0]].device
    targeted_hidden = torch.stack([targeted_hidden_by_layer[layer].to(output_device) for layer in selected_layers])
    return list(targeted_hidden.split(target_mask.sum(dim=1).tolist(), dim=1))
