from typing import Protocol

from torch import nn
from transformers import PreTrainedModel

from selfconcept.common.span_targeting.model_specifics import ModelSpecifics as SpanModelSpecifics


class ModelSpecifics(SpanModelSpecifics, Protocol):
    def get_decoder_layers(self, model: PreTrainedModel) -> nn.ModuleList: ...
