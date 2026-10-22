from collections.abc import Callable

from selfconcept.common.activation_extraction.model_specifics.default import DefaultModelSpecifics
from selfconcept.common.activation_extraction.model_specifics.types import ModelSpecifics
from selfconcept.common.hf_strong_types import HFTokenizer
from selfconcept.common.span_targeting.model_specifics import ModelSpecifics as SpanModelSpecifics
from selfconcept.common.span_targeting.model_specifics import get_model_specifics as get_span_model_specifics

MODEL_SPECIFICS_FACTORY_BY_NAME_SUBSTRING: dict[str, Callable[[SpanModelSpecifics], ModelSpecifics]] = {}


def get_model_specifics(model_name: str, tokenizer: HFTokenizer) -> ModelSpecifics:
    span_model_specifics = get_span_model_specifics(model_name, tokenizer)
    for name_substring, model_specifics_factory in MODEL_SPECIFICS_FACTORY_BY_NAME_SUBSTRING.items():
        if name_substring in model_name.lower():
            return model_specifics_factory(span_model_specifics)
    return DefaultModelSpecifics(span_model_specifics)
