from typing import Any

from selfconcept.common.generation.model_specifics.types import GenerationSpecifics, HFStyleAssistantTurn


class ThinkFlagGenerationSpecifics(HFStyleAssistantTurn, GenerationSpecifics):
    def set_thinking_flag(self, old_chat_kwargs: dict[str, Any], enable_thinking: bool) -> dict[str, Any]:
        return { **old_chat_kwargs, "enable_thinking": enable_thinking }
