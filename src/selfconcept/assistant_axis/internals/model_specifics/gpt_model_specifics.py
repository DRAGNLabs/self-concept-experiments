import warnings
from typing import Any, Iterator, NamedTuple

from selfconcept.assistant_axis.internals.model_specifics.types import CoverallLayerGetter, ModelSpecifics
from selfconcept.common.generation.model_specifics import HarmonyAssistantTurn, HarmonyGenerationSpecifics
from selfconcept.common.hf_strong_types import AllRoles, Conversation, HFTokenizer, configure_apply_chat_template


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


def _pooled_messages(tokenizer: HFTokenizer, full_ids: list[int], include_thinking: bool) -> list[HarmonyMessage]:
    """User messages and assistant final-channel messages, which alternate like the conversation's turns."""
    if include_thinking:
        warnings.warn("gpt-oss: pooling analysis-channel tokens is not supported yet; pooling the final channel only")
    return [
        message for message in _iter_harmony_messages(tokenizer, full_ids)
        if message.role == "user" or (message.role == "assistant" and message.channel == "final")
    ]


class GptModelSpecifics(CoverallLayerGetter, HarmonyGenerationSpecifics, ModelSpecifics[AllRoles, HarmonyAssistantTurn]):
    def get_response_indices(
        self,
        conversation: Conversation,
        tokenizer: HFTokenizer,
        *,
        include_thinking: bool,
        **apply_chat_template_kwargs: Any
    ) -> list[list[int]]:
        full_ids = configure_apply_chat_template(tokenizer).tokenize(True)(
            conversation, add_generation_prompt=False, **apply_chat_template_kwargs
        )["input_ids"]
        return [
            message.content_indices
            for message in _pooled_messages(tokenizer, full_ids, include_thinking)
            if message.role == "assistant"
        ]

    def build_turn_spans(
        self,
        conversation: Conversation,
        tokenizer: HFTokenizer,
        full_ids: list[int],
        *,
        include_thinking: bool,
        **apply_chat_template_kwargs,
    ) -> tuple[list[int], list[dict[str, Any]]]:
        turns = [turn for turn in conversation if turn["role"] != "system"]
        messages = _pooled_messages(tokenizer, full_ids, include_thinking)
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
