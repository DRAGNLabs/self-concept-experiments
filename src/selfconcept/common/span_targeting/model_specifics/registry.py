from collections.abc import Callable

from selfconcept.common.hf_strong_types import HFTokenizer
from selfconcept.common.span_targeting.model_specifics.no_thinking import NoThinkingModelSpecifics
from selfconcept.common.span_targeting.model_specifics.think_tag import ThinkTagModelSpecifics
from selfconcept.common.span_targeting.model_specifics.types import ModelSpecifics

MODEL_SPECIFICS_FACTORY_BY_NAME_SUBSTRING: dict[str, Callable[[HFTokenizer], ModelSpecifics]] = {
    "gemma-2": NoThinkingModelSpecifics,
    "gemma-3": NoThinkingModelSpecifics,
    "llama": NoThinkingModelSpecifics,
    "qwen3": ThinkTagModelSpecifics,
    "olmo-3": ThinkTagModelSpecifics,
}


def get_model_specifics(model_name: str, tokenizer: HFTokenizer) -> ModelSpecifics:
    for name_substring, model_specifics_factory in MODEL_SPECIFICS_FACTORY_BY_NAME_SUBSTRING.items():
        if name_substring in model_name.lower():
            return model_specifics_factory(tokenizer)
    raise ValueError(f"no span-targeting model specifics registered for {model_name}")
