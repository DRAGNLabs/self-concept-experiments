from selfconcept.assistant_axis.internals.model_specifics.gemma4 import Gemma4ModelSpecifics
from selfconcept.assistant_axis.internals.model_specifics.gemma_llama import GemmaLlamaModelSpecifics
from selfconcept.assistant_axis.internals.model_specifics.gpt_model_specifics import GptModelSpecifics
from selfconcept.assistant_axis.internals.model_specifics.muse_glimmer import MuseGlimmerModelSpecifics
from selfconcept.assistant_axis.internals.model_specifics.olmo import OlmoModelSpecifics
from selfconcept.assistant_axis.internals.model_specifics.qwen import Qwen35ModelSpecifics, QwenModelSpecifics
from selfconcept.assistant_axis.internals.model_specifics.types import ModelSpecifics
from selfconcept.assistant_axis.internals.model_specifics.unembedding import UnembeddingSpecifics


MODEL_SPECIFICS_REGISTRY = {
    'qwen': QwenModelSpecifics(),
    'gemma-4': Gemma4ModelSpecifics(),  # before 'gemma': first substring match wins
    'gemma4': Gemma4ModelSpecifics(),
    'gemma': GemmaLlamaModelSpecifics(),
    'llama': GemmaLlamaModelSpecifics(),
    'olmo': OlmoModelSpecifics(),
    'gpt-oss': GptModelSpecifics(),
    'muse-glimmer': MuseGlimmerModelSpecifics(),
    'muse_glimmer': MuseGlimmerModelSpecifics(),
    'glimmer': MuseGlimmerModelSpecifics(),
}

def get_model_specifics_by_name(name: str) -> ModelSpecifics:
    for key, value in MODEL_SPECIFICS_REGISTRY.items():
        if key in name.lower():
            return value

    raise ValueError(f"could not find model specifics for {name}")


UNEMBEDDING_SPECIFICS_BY_MODEL_SUBSTRING: dict[str, UnembeddingSpecifics] = {
    'gpt-oss': GptModelSpecifics(),
    'muse-glimmer': MuseGlimmerModelSpecifics(),
    'gemma-4': Gemma4ModelSpecifics(),
    'qwen3.8': Qwen35ModelSpecifics(),
}


def unembedding_specifics_for(model_name: str) -> UnembeddingSpecifics:
    for substring, specifics in UNEMBEDDING_SPECIFICS_BY_MODEL_SUBSTRING.items():
        if substring in model_name.lower():
            return specifics
    raise ValueError(f"no unembedding specifics for {model_name}: implement UnembeddingSpecifics.gain_scaled_unembedding "
                     "on its model specifics and add it to UNEMBEDDING_SPECIFICS_BY_MODEL_SUBSTRING")
