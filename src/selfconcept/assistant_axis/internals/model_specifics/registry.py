from selfconcept.assistant_axis.internals.model_specifics.gemma_llama import GemmaLlamaModelSpecifics
from selfconcept.assistant_axis.internals.model_specifics.olmo import OlmoModelSpecifics
from selfconcept.assistant_axis.internals.model_specifics.qwen import QwenModelSpecifics
from selfconcept.assistant_axis.internals.model_specifics.types import ModelSpecifics


MODEL_SPECIFICS_REGISTRY = {
    'qwen': QwenModelSpecifics(),
    'gemma': GemmaLlamaModelSpecifics(),
    'llama': GemmaLlamaModelSpecifics(),
    'olmo': OlmoModelSpecifics(),
}

def get_model_specifics_by_name(name: str) -> ModelSpecifics:
    for key, value in MODEL_SPECIFICS_REGISTRY.items():
        if key in name.lower():
            return value

    raise ValueError(f"could not find model specifics for {name}")
