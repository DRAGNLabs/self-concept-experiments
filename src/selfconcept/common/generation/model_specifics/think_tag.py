"""Models that think inside ``<think>...</think>`` (Qwen 3.x, GLM 4.5). Their chat templates render an assistant turn's
reasoning from ``reasoning_content``, so the reasoning is stored there and ``content`` is the answer alone. Only where
the opening tag comes from differs, which ``completion_format`` (the raw completion, for ``to_completion``) records:

- Qwen: the generation prompt ends in ``<think>\\n``, so the completion is ``{reasoning}\\n</think>\\n\\n{content}``.
- GLM: the prompt ends at ``<|assistant|>`` and the model writes ``\\n<think>{reasoning}</think>``, usually with nothing
  before the answer (sometimes a newline, which is not kept).
"""
from typing import Any

from selfconcept.common.generation.model_specifics.types import GenerationSpecifics, ReasoningContentAssistantTurn
from selfconcept.common.hf_strong_types import AllRoles

THINK_OPEN, THINK_CLOSE = "<think>", "</think>"


class ThinkTagGenerationSpecifics(GenerationSpecifics[AllRoles, ReasoningContentAssistantTurn]):
    completion_format: str

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
        return self.completion_format.format(reasoning=assistant_turn["reasoning_content"], content=assistant_turn["content"])


class QwenThinkGenerationSpecifics(ThinkTagGenerationSpecifics):
    completion_format = "{reasoning}\n</think>\n\n{content}"


class GlmThinkGenerationSpecifics(ThinkTagGenerationSpecifics):
    completion_format = "\n<think>{reasoning}</think>{content}"
