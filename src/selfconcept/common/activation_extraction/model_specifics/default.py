from dataclasses import dataclass

from torch import nn
from transformers import PreTrainedModel

from selfconcept.common.activation_extraction.model_specifics.types import ModelSpecifics
from selfconcept.common.span_targeting.model_specifics import ModelSpecifics as SpanModelSpecifics
from selfconcept.common.span_targeting.types import CompletionSpans, PromptCompletion


def find_decoder_layers(model: PreTrainedModel) -> nn.ModuleList:
    """The ModuleList, under the text decoder, of the layer classes transformers marks as unsplittable."""
    decoder_layer_class_names = set(model._no_split_modules or [])  # pyright: ignore[reportPrivateUsage]
    for module in model.get_decoder().modules():
        if isinstance(module, nn.ModuleList) and len(module) > 0 and type(module[0]).__name__ in decoder_layer_class_names:
            return module
    raise ValueError(f"no decoder layers found in {type(model).__name__}")


@dataclass(frozen=True)
class DefaultModelSpecifics(ModelSpecifics):
    span_model_specifics: SpanModelSpecifics

    def split_completion(self, example: PromptCompletion) -> CompletionSpans:
        return self.span_model_specifics.split_completion(example)

    def get_decoder_layers(self, model: PreTrainedModel) -> nn.ModuleList:
        return find_decoder_layers(model)
