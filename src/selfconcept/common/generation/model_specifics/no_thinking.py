from typing import Any

from selfconcept.common.generation.model_specifics.types import GenerationSpecifics, HFStyleAssistantTurn


class NoThinkingGenerationSpecifics(HFStyleAssistantTurn, GenerationSpecifics):
    def set_thinking_flag(self, old_chat_kwargs: dict[str, Any], enable_thinking: bool) -> dict[str, Any]:
        if enable_thinking:
            raise NotImplementedError("thinking currently not supported for Gemma type models")
        else:
            return old_chat_kwargs

