from typing import Any

from selfconcept.common.generation.model_specifics.types import GenerationSpecifics, HFStyleAssistantTurn
from selfconcept.common.hf_strong_types import ConversationTurn

THINK_OPEN, THINK_CLOSE = "<think>", "</think>"
END_OF_TURN_MARKERS = ("<|im_end|>", "<|endoftext|>")


def split_qwen_think_completion(raw_completion: str) -> tuple[str, str]:
    """-> (reasoning, content) of a Qwen3.x completion generated with thinking on (the generation prompt ends with
    ``<think>\\n``, so the completion is ``{reasoning}</think>{answer}``). content is "" when ``</think>`` never came
    (the model stopped or was cut off mid-thought). End-of-turn markers left by ``skip_special_tokens=False`` are dropped."""
    text = raw_completion
    for marker in END_OF_TURN_MARKERS:
        text = text.replace(marker, "")
    end = text.find(THINK_CLOSE)
    if end == -1:
        return text.removeprefix(THINK_OPEN).strip(), ""
    return text[:end].removeprefix(THINK_OPEN).strip(), text[end + len(THINK_CLOSE):].strip()


class ThinkFlagGenerationSpecifics(HFStyleAssistantTurn, GenerationSpecifics):
    def set_thinking_flag(self, old_chat_kwargs: dict[str, Any], enable_thinking: bool) -> dict[str, Any]:
        return { **old_chat_kwargs, "enable_thinking": enable_thinking }


class QwenThinkGenerationSpecifics(ThinkFlagGenerationSpecifics):
    """Qwen3.x with thinking on: the generation prompt ends with ``<think>\\n``, so a completion is
    ``{reasoning}\\n</think>\\n\\n{answer}``. Stored whole as ``content``, a later chat-template render prepends an
    empty think block to it (``<think>\\n\\n</think>\\n\\n{reasoning}</think>...``), which is not what the model saw
    and puts the reasoning in front of every judge. So the turn is split into ``reasoning_content`` (the template's
    field) and ``content``; a completion without ``</think>`` (thinking off) stays a bare answer."""

    def to_assistant_turn(self, completion: str) -> ConversationTurn:
        end = completion.find(THINK_CLOSE)
        if end == -1:
            return {"role": "assistant", "content": completion}
        reasoning = completion[:end].removeprefix(THINK_OPEN).strip()
        return {"role": "assistant", "content": completion[end + len(THINK_CLOSE):].strip(), "reasoning_content": reasoning}  # type: ignore[typeddict-unknown-key]

    def to_completion(self, assistant_turn: ConversationTurn) -> str:
        reasoning = assistant_turn.get("reasoning_content")  # type: ignore[misc]
        if reasoning is None:
            return assistant_turn["content"]
        return f"{reasoning}\n{THINK_CLOSE}\n\n{assistant_turn['content']}"
