from selfconcept.common.generation.model_specifics.forced_think_close import ForcedThinkCloseGenerationSpecifics
from selfconcept.common.generation.model_specifics.harmony import HarmonyGenerationSpecifics
from selfconcept.common.generation.model_specifics.no_thinking import NoThinkingGenerationSpecifics
from selfconcept.common.generation.model_specifics.think_flag import ThinkFlagGenerationSpecifics
from selfconcept.common.generation.model_specifics.types import GenerationSpecifics

GENERATION_SPECIFICS_BY_NAME_SUBSTRING: dict[str, GenerationSpecifics] = {
    "qwen": ThinkFlagGenerationSpecifics(),
    "gemma": NoThinkingGenerationSpecifics(),
    "llama": NoThinkingGenerationSpecifics(),
    "olmo": ForcedThinkCloseGenerationSpecifics(),
    "gpt-oss": HarmonyGenerationSpecifics(),
    "glm": ThinkFlagGenerationSpecifics(),
}


def get_generation_specifics(model_name: str) -> GenerationSpecifics:
    for name_substring, generation_specifics in GENERATION_SPECIFICS_BY_NAME_SUBSTRING.items():
        if name_substring in model_name.lower():
            return generation_specifics
    raise ValueError(f"could not find generation specifics for {model_name}")
