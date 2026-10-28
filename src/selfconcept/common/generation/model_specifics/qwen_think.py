"""Qwen 3.x thinking: with ``enable_thinking`` the generation prompt ends in ``<think>\\n``, so a completion is
``{reasoning}\\n</think>\\n\\n{content}``. The chat template renders an assistant turn's ``reasoning_content`` back
inside the think block and never splits ``content`` itself, so the reasoning must be stored in its own field."""
from typing import Any

from selfconcept.common.generation.model_specifics.types import GenerationSpecifics
from selfconcept.common.hf_strong_types import AllRoles, ConversationTurn

THINK_CLOSE = "</think>"


class QwenAssistantTurn(ConversationTurn, total=False):
    reasoning_content: str


class QwenThinkGenerationSpecifics(GenerationSpecifics[AllRoles, QwenAssistantTurn]):
    def set_thinking_flag(self, old_chat_kwargs: dict[str, Any], enable_thinking: bool) -> dict[str, Any]:
        return {**old_chat_kwargs, "enable_thinking": enable_thinking}

    def to_assistant_turn(self, completion: str) -> QwenAssistantTurn:
        reasoning, close, content = completion.partition(THINK_CLOSE)
        if not close:
            return {"role": "assistant", "content": completion}
        return {"role": "assistant", "content": content.strip(), "reasoning_content": reasoning.strip()}

    def to_completion(self, assistant_turn: QwenAssistantTurn) -> str:
        if "reasoning_content" not in assistant_turn:
            return assistant_turn["content"]
        return f"{assistant_turn['reasoning_content']}\n{THINK_CLOSE}\n\n{assistant_turn['content']}"
