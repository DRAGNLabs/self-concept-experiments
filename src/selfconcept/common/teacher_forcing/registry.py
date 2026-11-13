from selfconcept.common.generation.model_specifics.gemma4 import Gemma4GenerationSpecifics
from selfconcept.common.generation.model_specifics.harmony import HarmonyGenerationSpecifics
from selfconcept.common.generation.model_specifics.muse_glimmer import MuseGlimmerGenerationSpecifics
from selfconcept.common.generation.model_specifics.think_tag import QwenThinkGenerationSpecifics
from selfconcept.common.teacher_forcing.completion_markup import CompletionMarkupSpecifics

COMPLETION_MARKUP_SPECIFICS_BY_NAME_SUBSTRING: dict[str, CompletionMarkupSpecifics] = {
    "gpt-oss": HarmonyGenerationSpecifics(),
    "gemma-4": Gemma4GenerationSpecifics(),
    "muse-glimmer": MuseGlimmerGenerationSpecifics(),
    "qwen": QwenThinkGenerationSpecifics(),
}


def completion_markup_specifics_for(model_name: str) -> CompletionMarkupSpecifics:
    for name_substring, specifics in COMPLETION_MARKUP_SPECIFICS_BY_NAME_SUBSTRING.items():
        if name_substring in model_name.lower():
            return specifics
    raise ValueError(f"no completion markup for {model_name}: implement CompletionMarkupSpecifics.completion_markup on its "
                     "generation specifics and add it to COMPLETION_MARKUP_SPECIFICS_BY_NAME_SUBSTRING")
