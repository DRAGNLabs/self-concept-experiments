from typing import Any

from selfconcept.common.generation.model_specifics.types import GenerationSpecifics, HFStyleAssistantTurn

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

