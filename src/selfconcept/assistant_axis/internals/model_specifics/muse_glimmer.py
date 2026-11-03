"""Muse Glimmer model specifics: ``to=self`` channel aware span finding.

The Muse Glimmer chat template renders every message as ``<|start|>HEADER<|message|>CONTENT`` ended by ``<|eom|>``
or ``<|eot|>``, where HEADER is the role and, for assistant turns, the recipient (``assistant to=self`` for the
reasoning channel, ``assistant to=user`` for the answer); an assistant turn's ``reasoning_content`` field renders as
its own ``to=self`` message before the answer (for every assistant turn, unlike gpt-oss's template). Generation-side
handling (completion splitting, the forced user channel) lives in
``selfconcept.common.generation.model_specifics.muse_glimmer`` and is mixed in here; this module finds the token
spans. ``include_thinking`` selects, for activation pooling, the ``to=self`` channel (CoT region) over the answer.
"""
from typing import Any, Iterator, NamedTuple

import torch.nn as nn

from selfconcept.assistant_axis.internals.model_specifics.message_spans import (
    MessageKind, MessageSpan, end_token_index_after, shortest_suffix_containing,
)
from selfconcept.assistant_axis.internals.model_specifics.types import CoverallLayerGetter, ModelSpecifics
from selfconcept.common.generation.model_specifics.muse_glimmer import (  # noqa: F401  (split re-exported)
    EOM, EOT, MESSAGE, START, MuseGlimmerAssistantTurn, MuseGlimmerGenerationSpecifics, split_muse_glimmer_completion,
)
from selfconcept.common.hf_strong_types import AllRoles, Conversation, HFTokenizer, configure_apply_chat_template


class MuseGlimmerMessage(NamedTuple):
    role: str
    recipient: str | None  # "self" (reasoning), "user" (answer), a tool name, or None (system / user turns)
    content_indices: list[int]


def iter_muse_glimmer_messages(tokenizer: HFTokenizer, full_ids: list[int]) -> Iterator[MuseGlimmerMessage]:
    """Messages are "<|start|>ROLE[ to=RECIPIENT]<|message|>CONTENT" ended by <|eom|>, <|eot|>, the next <|start|>
    or the end of the ids. Header tokens are never included in the content indices."""
    start_id = tokenizer.convert_tokens_to_ids(START)
    message_id = tokenizer.convert_tokens_to_ids(MESSAGE)
    end_ids = {tokenizer.convert_tokens_to_ids(token) for token in (EOM, EOT)}

    position = 0
    while start_id in full_ids[position:]:
        header_start = full_ids.index(start_id, position) + 1
        if message_id not in full_ids[header_start:]:
            return
        content_start = full_ids.index(message_id, header_start) + 1
        content_end = next(
            (i for i in range(content_start, len(full_ids)) if full_ids[i] in end_ids or full_ids[i] == start_id),
            len(full_ids),
        )
        header = tokenizer.decode(full_ids[header_start:content_start - 1]).strip()
        role, _, recipient = header.partition("to=")
        yield MuseGlimmerMessage(role.strip(), recipient.strip() or None, list(range(content_start, content_end)))
        position = content_end


def pooled_muse_glimmer_messages(tokenizer: HFTokenizer, full_ids: list[int], include_thinking: bool = False) -> list[MuseGlimmerMessage]:
    """One message per non-system turn: user content, and per assistant turn its to=user answer, or with
    ``include_thinking`` its to=self channel when it has one (falling back to the answer)."""
    pooled: list[MuseGlimmerMessage] = []
    for message in iter_muse_glimmer_messages(tokenizer, full_ids):
        if message.role == "user":
            pooled.append(message)
        elif message.role == "assistant" and message.recipient == "self":
            if include_thinking:
                pooled.append(message)
        elif message.role == "assistant" and message.recipient in (None, "user"):
            if not (include_thinking and pooled and pooled[-1].role == "assistant"):
                pooled.append(message)
    return pooled


def _muse_glimmer_message_kind(message: MuseGlimmerMessage) -> MessageKind:
    """Tool calls render as "assistant to=NAME", tool results as "tool NAME"."""
    if message.role in ("system", "user"):
        return message.role
    if message.role != "assistant":
        return "tool_result"
    if message.recipient == "self":
        return "reasoning"
    return "response" if message.recipient in (None, "user") else "tool_call"


class MuseGlimmerModelSpecifics(CoverallLayerGetter, MuseGlimmerGenerationSpecifics, ModelSpecifics[AllRoles, MuseGlimmerAssistantTurn]):
    def get_response_indices(self, conversation: Conversation, tokenizer: HFTokenizer, *, include_thinking: bool,
                             **apply_chat_template_kwargs: Any) -> list[list[int]]:
        full_ids = configure_apply_chat_template(tokenizer).tokenize(True)(
            conversation, add_generation_prompt=False, **apply_chat_template_kwargs
        )["input_ids"]
        return [m.content_indices for m in pooled_muse_glimmer_messages(tokenizer, full_ids, include_thinking) if m.role == "assistant"]

    def build_turn_spans(self, conversation: Conversation, tokenizer: HFTokenizer, full_ids: list[int], *, include_thinking: bool,
                         **apply_chat_template_kwargs) -> tuple[list[int], list[dict[str, Any]]]:
        turns = [turn for turn in conversation if turn["role"] != "system"]
        messages = pooled_muse_glimmer_messages(tokenizer, full_ids, include_thinking)
        spans = [
            {"turn": i, "role": turn["role"], "start": m.content_indices[0], "end": m.content_indices[-1] + 1,
             "n_tokens": len(m.content_indices), "text": turn["content"]}
            for i, (turn, m) in enumerate(zip(turns, messages, strict=True))
            if m.content_indices
        ]
        return full_ids, spans

    def get_layers(self, model: Any) -> nn.ModuleList:
        try:  # MuseGlimmerForConditionalGeneration: text tower under model.language_model
            layers = model.model.language_model.layers
            if layers is not None and len(layers) > 0:
                return layers
        except AttributeError:
            pass
        return super().get_layers(model)

    def message_spans(self, tokenizer: HFTokenizer, full_ids: list[int]) -> list[MessageSpan]:
        end_token_ids = {tokenizer.convert_tokens_to_ids(token) for token in (EOM, EOT)}
        return [
            MessageSpan(
                _muse_glimmer_message_kind(message),
                range(message.content_indices[0], message.content_indices[-1] + 1) if message.content_indices else range(0),
                end_token_index_after(full_ids, message.content_indices, end_token_ids),
            )
            for message in iter_muse_glimmer_messages(tokenizer, full_ids)
        ]

    def tool_call_close(self, tokenizer: HFTokenizer, full_ids: list[int], span: MessageSpan) -> range:
        """From the last </atem:invoke> through </atem:function_calls>."""
        return shortest_suffix_containing(tokenizer, full_ids, span.content, "</atem:invoke>")
