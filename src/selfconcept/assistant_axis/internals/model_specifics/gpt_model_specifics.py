from typing import Any, Iterator, NamedTuple

from jaxtyping import Float
import torch

from selfconcept.assistant_axis.internals.model_specifics.message_spans import (
    MessageKind, MessageSpan, MessageSpanSpecifics, ToolCallCloseSpecifics, end_token_index_after,
    shortest_suffix_containing,
)
from selfconcept.assistant_axis.internals.model_specifics.types import CoverallLayerGetter, ModelSpecifics
from selfconcept.assistant_axis.internals.model_specifics.unembedding import CheckpointTensorLoader, UnembeddingSpecifics
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
    """One message per conversation turn, alternating like the turns: user messages and, per assistant turn, its
    final-channel message, or with include_thinking its analysis-channel (CoT) message when it has one.

    The gpt-oss chat template renders an assistant turn's `thinking` field as an analysis message only for the
    last turn of the conversation, so CoT pooling of earlier assistant turns falls back to their final message.
    """
    pooled: list[HarmonyMessage] = []
    for message in _iter_harmony_messages(tokenizer, full_ids):
        if message.role == "user":
            pooled.append(message)
        elif message.role == "assistant" and message.channel == "analysis":
            if include_thinking:
                pooled.append(message)
        elif message.role == "assistant" and message.channel == "final":
            if not (include_thinking and pooled and pooled[-1].role == "assistant"):
                pooled.append(message)
    return pooled


def _harmony_message_kind(message: HarmonyMessage) -> MessageKind:
    """Tool calls render as "assistant to=functions.NAME", tool results as "functions.NAME to=assistant"."""
    if message.role in ("system", "developer"):
        return "system"
    if message.role == "user":
        return "user"
    if not message.role.startswith("assistant"):
        return "tool_result"
    if " to=" in message.role:
        return "tool_call"
    return "reasoning" if message.channel == "analysis" else "response"


class GptModelSpecifics(CoverallLayerGetter, HarmonyGenerationSpecifics, ModelSpecifics[AllRoles, HarmonyAssistantTurn], MessageSpanSpecifics, ToolCallCloseSpecifics,
                        UnembeddingSpecifics):
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

    def message_spans(self, tokenizer: HFTokenizer, full_ids: list[int]) -> list[MessageSpan]:
        end_token_ids = {tokenizer.convert_tokens_to_ids(token) for token in ("<|end|>", "<|return|>", "<|call|>")}
        return [
            MessageSpan(
                _harmony_message_kind(message),
                range(message.content_indices[0], message.content_indices[-1] + 1) if message.content_indices else range(0),
                end_token_index_after(full_ids, message.content_indices, end_token_ids),
            )
            for message in _iter_harmony_messages(tokenizer, full_ids)
        ]

    def tool_call_close(self, tokenizer: HFTokenizer, full_ids: list[int], span: MessageSpan) -> range:
        """The closing brace of the JSON arguments."""
        return shortest_suffix_containing(tokenizer, full_ids, span.content, "}")

    def gain_scaled_unembedding(self, load_tensor: CheckpointTensorLoader) -> Float[torch.Tensor, "vocab hidden"]:
        """GptOssRMSNorm scales by weight."""
        return load_tensor("lm_head.weight") * load_tensor("model.norm.weight")
