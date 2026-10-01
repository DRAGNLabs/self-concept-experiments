import warnings
from typing import Any, Iterator, NamedTuple

from selfconcept.assistant_axis.internals.model_specifics.types import CoverallLayerGetter, HFStyleAssistantTurn, ModelSpecifics
from selfconcept.common.hf_strong_types import Conversation, HFTokenizer, configure_apply_chat_template


class HarmonyMessage(NamedTuple):
    role: str
    channel: str | None
    content_indices: list[int]


def _iter_harmony_messages(tokenizer: HFTokenizer, full_ids: list[int]) -> Iterator[HarmonyMessage]:
    """Messages are "<|start|>ROLE[<|channel|>CHANNEL ...]<|message|>CONTENT" ended by <|end|>, <|return|> or <|call|>."""
    start_id = tokenizer.convert_tokens_to_ids("<|start|>")
    message_id = tokenizer.convert_tokens_to_ids("<|message|>")
    message_end_ids = {tokenizer.convert_tokens_to_ids(token) for token in ("<|end|>", "<|return|>", "<|call|>")}

    position = 0
    while start_id in full_ids[position:]:
        header_start = full_ids.index(start_id, position) + 1
        content_start = full_ids.index(message_id, header_start) + 1
        content_end = next(
            (index for index in range(content_start, len(full_ids)) if full_ids[index] in message_end_ids),
            len(full_ids),
        )
        header = tokenizer.decode(full_ids[header_start:content_start - 1])
        role, _, channel_and_format = header.partition("<|channel|>")
        channel = channel_and_format.split()[0] if channel_and_format.strip() else None
        yield HarmonyMessage(role.strip(), channel, list(range(content_start, content_end)))
        position = content_end


def _pooled_messages(tokenizer: HFTokenizer, full_ids: list[int], **apply_chat_template_kwargs: Any) -> list[HarmonyMessage]:
    """User messages and assistant final-channel messages, which alternate like the conversation's turns."""
    if apply_chat_template_kwargs.get("enable_thinking"):
        warnings.warn("gpt-oss: pooling analysis-channel tokens is not supported yet; pooling the final channel only")
    return [
        message for message in _iter_harmony_messages(tokenizer, full_ids)
        if message.role == "user" or (message.role == "assistant" and message.channel == "final")
    ]


class GptModelSpecifics(CoverallLayerGetter, HFStyleAssistantTurn, ModelSpecifics):
    def get_response_indices(
        self,
        conversation: Conversation,
        tokenizer: HFTokenizer,
        **apply_chat_template_kwargs: Any
    ) -> list[list[int]]:
        full_ids = configure_apply_chat_template(tokenizer).tokenize(True)(
            conversation, add_generation_prompt=False, **apply_chat_template_kwargs
        )["input_ids"]
        return [
            message.content_indices
            for message in _pooled_messages(tokenizer, full_ids, **apply_chat_template_kwargs)
            if message.role == "assistant"
        ]

    def build_turn_spans(
        self,
        conversation: Conversation,
        tokenizer: HFTokenizer,
        full_ids: list[int],
        **apply_chat_template_kwargs,
    ) -> tuple[list[int], list[dict[str, Any]]]:
        turns = [turn for turn in conversation if turn["role"] != "system"]
        messages = _pooled_messages(tokenizer, full_ids, **apply_chat_template_kwargs)
        spans = [
            {
                "turn": turn_index,
                "role": turn["role"],
                "start": message.content_indices[0],
                "end": message.content_indices[-1] + 1,
                "n_tokens": len(message.content_indices),
                "text": turn["content"],
            }
            for turn_index, (turn, message) in enumerate(zip(turns, messages, strict=True))
            if message.content_indices
        ]
        return full_ids, spans

    def set_enable_thinking(self, old_chat_kwargs: dict[str, Any], enable_thinking: bool) -> dict[str, Any]:
        # The template ignores enable_thinking; it only tells the span methods whether CoT pooling was requested.
        return {**old_chat_kwargs, "enable_thinking": enable_thinking}

    def thinking_close_ids(self, tokenizer: HFTokenizer) -> list[int]:
        return []  # gpt-oss always reasons
