"""GLM 4.5 thinking: the generation prompt ends in ``<|assistant|>`` and the model writes ``\\n<think>{reasoning}</think>\\n
{content}`` itself (the tags are ordinary tokens, so they survive decoding). The template re-renders a turn's
``reasoning_content`` as that think block, so the reasoning is stored there and ``content`` is the answer alone."""
from typing import Any

from selfconcept.common.generation.model_specifics.types import GenerationSpecifics, ReasoningContentAssistantTurn
from selfconcept.common.hf_strong_types import AllRoles

THINK_OPEN, THINK_CLOSE = "<think>", "</think>"


class GlmThinkGenerationSpecifics(GenerationSpecifics[AllRoles, ReasoningContentAssistantTurn]):
    def set_thinking_flag(self, old_chat_kwargs: dict[str, Any], enable_thinking: bool) -> dict[str, Any]:
        return {**old_chat_kwargs, "enable_thinking": enable_thinking}

    def to_assistant_turn(self, completion: str) -> ReasoningContentAssistantTurn:
        reasoning, close, content = completion.partition(THINK_CLOSE)
        if not close:
            return {"role": "assistant", "content": completion}
        return {"role": "assistant", "content": content.strip(), "reasoning_content": reasoning.strip().removeprefix(THINK_OPEN).strip()}

    def to_completion(self, assistant_turn: ReasoningContentAssistantTurn) -> str:
        if "reasoning_content" not in assistant_turn:
            return assistant_turn["content"]
        return f"\n{THINK_OPEN}{assistant_turn['reasoning_content']}{THINK_CLOSE}\n{assistant_turn['content']}"
