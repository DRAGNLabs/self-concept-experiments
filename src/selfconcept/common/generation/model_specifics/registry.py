from typing import Any

from selfconcept.common.generation.model_specifics.forced_think_close import ForcedThinkCloseGenerationSpecifics
from selfconcept.common.generation.model_specifics.gemma4 import Gemma4GenerationSpecifics
from selfconcept.common.generation.model_specifics.harmony import HarmonyGenerationSpecifics
from selfconcept.common.generation.model_specifics.muse_glimmer import MuseGlimmerGenerationSpecifics
from selfconcept.common.generation.model_specifics.no_thinking import NoThinkingGenerationSpecifics
from selfconcept.common.generation.model_specifics.think_flag import ThinkFlagGenerationSpecifics
from selfconcept.common.generation.model_specifics.types import GenerationSpecifics
from selfconcept.common.hf_strong_types import AllRoles

GENERATION_SPECIFICS_BY_NAME_SUBSTRING: dict[str, GenerationSpecifics[AllRoles, Any]] = {
    "qwen": ThinkFlagGenerationSpecifics(),
    "gemma-4": Gemma4GenerationSpecifics(),  # before "gemma": the first substring match wins
    "gemma4": Gemma4GenerationSpecifics(),
    "gemma": NoThinkingGenerationSpecifics(),
    "llama": NoThinkingGenerationSpecifics(),
    "olmo": ForcedThinkCloseGenerationSpecifics(),
    "gpt-oss": HarmonyGenerationSpecifics(),
    "glm": ThinkFlagGenerationSpecifics(),
    "muse-glimmer": MuseGlimmerGenerationSpecifics(),
    "muse_glimmer": MuseGlimmerGenerationSpecifics(),
    "glimmer": MuseGlimmerGenerationSpecifics(),
}


def get_generation_specifics(model_name: str) -> GenerationSpecifics[AllRoles, Any]:
    for name_substring, generation_specifics in GENERATION_SPECIFICS_BY_NAME_SUBSTRING.items():
        if name_substring in model_name.lower():
            return generation_specifics
    raise ValueError(f"could not find generation specifics for {model_name}")
