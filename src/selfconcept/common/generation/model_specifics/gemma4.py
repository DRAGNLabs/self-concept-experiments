"""Gemma 4 generation specifics: the thought channel.

The Gemma 4 chat template injects ``<|think|>`` into the system turn when ``enable_thinking`` is set and the model
then opens its thought channel itself; a completion is ``[<|channel>thought\\n{reasoning}\\n<channel|>]{content}``,
kept raw (``skip_special_tokens = False``) and split here into ``reasoning`` and ``content``. The template renders
an assistant turn's ``reasoning`` field back as the thought channel (for turns after the last user message), and
closes an empty thought channel itself when thinking is off, so no direct-answer prefix is needed.
"""
from typing import Any

from selfconcept.common.generation.model_specifics.types import GenerationSpecifics
from selfconcept.common.hf_strong_types import AllRoles, ConversationTurn

THOUGHT_OPEN = "<|channel>thought\n"
THOUGHT_CLOSE = "<channel|>"
TURN_OPEN = "<|turn>"
TURN_CLOSE = "<turn|>"


class Gemma4AssistantTurn(ConversationTurn, total=False):
    reasoning: str


def split_gemma4_completion(raw_completion: str) -> tuple[str | None, str]:
    """-> (reasoning, content): the thought-channel body (None when the model opened none) and the answer after it.
    content is "" when the thought channel was never closed."""
    if THOUGHT_OPEN.strip() not in raw_completion:
        return None, _strip_control(raw_completion).strip()
    _, _, after_open = raw_completion.partition("<|channel>")
    after_open = after_open.removeprefix("thought").lstrip("\n")
    if THOUGHT_CLOSE not in after_open:
        return _strip_control(after_open).strip(), ""
    reasoning, _, content = after_open.partition(THOUGHT_CLOSE)
    return reasoning.strip(), _strip_control(content).strip()


def join_gemma4_completion(reasoning: str | None, content: str) -> str:
    """Inverse of split_gemma4_completion (up to the stripped control tokens and whitespace)."""
    if reasoning is None:
        return content
    return f"{THOUGHT_OPEN}{reasoning}\n{THOUGHT_CLOSE}{content}"


def _strip_control(text: str) -> str:
    for token in (TURN_CLOSE, TURN_OPEN, "<eos>", "<end_of_turn>"):
        text = text.replace(token, "")
    return text


class Gemma4GenerationSpecifics(GenerationSpecifics[AllRoles, Gemma4AssistantTurn]):
    skip_special_tokens = False

    def set_thinking_flag(self, old_chat_kwargs: dict[str, Any], enable_thinking: bool) -> dict[str, Any]:
        return {**old_chat_kwargs, "enable_thinking": enable_thinking}

    def to_assistant_turn(self, completion: str) -> Gemma4AssistantTurn:
        reasoning, content = split_gemma4_completion(completion)
        turn: Gemma4AssistantTurn = {"role": "assistant", "content": content}
        if reasoning is not None:
            turn["reasoning"] = reasoning
        return turn

    def to_completion(self, assistant_turn: Gemma4AssistantTurn) -> str:
        return join_gemma4_completion(assistant_turn.get("reasoning"), assistant_turn["content"])
