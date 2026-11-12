"""Gemma 4 model specifics: thought-channel aware span finding.

The Gemma 4 chat template renders an assistant turn as
``<|turn>model\\n[<|channel>thought\\n{reasoning}\\n<channel|>]{content}<turn|>`` and injects ``<|think|>`` into the
system turn when ``enable_thinking`` is set; the model then opens the thought channel itself. Generation-side
handling (completion splitting, the thinking flag) lives in ``selfconcept.common.generation.model_specifics.gemma4``
and is mixed in here; this module finds the token spans. ``include_thinking`` selects, for activation pooling, the
thought channel (CoT region) over the final content.
"""
from typing import Any, Iterator, NamedTuple

from jaxtyping import Float
import torch
import torch.nn as nn

from selfconcept.assistant_axis.internals.model_specifics.message_spans import (
    MessageKind, MessageSpan, MessageSpanSpecifics, ToolCallCloseSpecifics, end_token_index_after, shortest_suffix_containing,
)
from selfconcept.assistant_axis.internals.model_specifics.types import CoverallLayerGetter, ModelSpecifics
from selfconcept.assistant_axis.internals.model_specifics.unembedding import CheckpointTensorLoader, UnembeddingSpecifics
from selfconcept.common.generation.model_specifics.gemma4 import (  # noqa: F401  (split_gemma4_completion is re-exported)
    THOUGHT_CLOSE, THOUGHT_OPEN, TURN_CLOSE, TURN_OPEN, Gemma4AssistantTurn, Gemma4GenerationSpecifics, split_gemma4_completion,
)
from selfconcept.common.hf_strong_types import AllRoles, Conversation, HFTokenizer, configure_apply_chat_template


class Gemma4Message(NamedTuple):
    role: str
    channel: str | None  # "thought" or None for the plain content
    content_indices: list[int]


def _newline_ids(tokenizer: HFTokenizer) -> set[int]:
    return {i for i in tokenizer.encode("\n", add_special_tokens=False)}


def iter_gemma4_messages(tokenizer: HFTokenizer, full_ids: list[int]) -> Iterator[Gemma4Message]:
    """One message per turn: ``(role, None, content ids)``, preceded for an assistant turn with a thought channel by
    ``(role, "thought", body ids)``. Header tokens (role name, newlines, channel markers) are never included."""
    turn_open = tokenizer.convert_tokens_to_ids(TURN_OPEN)
    turn_close = tokenizer.convert_tokens_to_ids(TURN_CLOSE)
    channel_open = tokenizer.convert_tokens_to_ids("<|channel>")
    channel_close = tokenizer.convert_tokens_to_ids(THOUGHT_CLOSE)
    newlines = _newline_ids(tokenizer)

    position = 0
    while turn_open in full_ids[position:]:
        header_start = full_ids.index(turn_open, position) + 1
        turn_end = full_ids.index(turn_close, header_start) if turn_close in full_ids[header_start:] else len(full_ids)
        # the header is "<|turn>ROLE\n": role tokens up to the first newline token
        content_start = next((i + 1 for i in range(header_start, turn_end) if full_ids[i] in newlines), turn_end)
        role = tokenizer.decode(full_ids[header_start:content_start]).strip()
        body = full_ids[content_start:turn_end]
        if role == "model" and body and body[0] == channel_open and channel_close in body:
            close = body.index(channel_close)
            thought_start = next((i + 1 for i in range(1, close) if body[i] in newlines), close)
            thought = list(range(content_start + thought_start, content_start + close))
            if thought and full_ids[thought[-1]] in newlines:
                thought.pop()
            yield Gemma4Message("assistant", "thought", thought)
            content_start = content_start + close + 1
            body = full_ids[content_start:turn_end]
        yield Gemma4Message("assistant" if role == "model" else role, None, list(range(content_start, turn_end)))
        position = turn_end + 1


def pooled_gemma4_messages(tokenizer: HFTokenizer, full_ids: list[int], include_thinking: bool = False) -> list[Gemma4Message]:
    """One message per non-system turn: user content, and per assistant turn its content, or with ``include_thinking``
    its thought channel when it has one (falling back to the content)."""
    pool_cot = include_thinking
    pooled: list[Gemma4Message] = []
    for message in iter_gemma4_messages(tokenizer, full_ids):
        if message.role == "system":
            continue
        if message.channel == "thought":
            if pool_cot:
                pooled.append(message)
        elif message.role == "assistant" and pool_cot and pooled and pooled[-1].role == "assistant" and pooled[-1].channel == "thought":
            continue
        else:
            pooled.append(message)
    return pooled


TOOL_CALL_OPEN, TOOL_CALL_CLOSE = "<|tool_call>", "<tool_call|>"
TOOL_RESPONSE_OPEN, TOOL_RESPONSE_CLOSE = "<|tool_response>", "<tool_response|>"


def _model_turn_spans(tokenizer: HFTokenizer, full_ids: list[int], content_indices: list[int]) -> list[MessageSpan]:
    """A model turn's content split at its tool blocks: each ``<|tool_call>…<tool_call|>`` is a tool_call and each
    ``<|tool_response>…<tool_response|>`` a tool_result (the template renders tool results inside the model turn);
    every non-blank stretch of text outside them is a response."""
    block_kind_by_open: dict[int, tuple[MessageKind, int]] = {
        tokenizer.convert_tokens_to_ids(TOOL_CALL_OPEN): ("tool_call", tokenizer.convert_tokens_to_ids(TOOL_CALL_CLOSE)),
        tokenizer.convert_tokens_to_ids(TOOL_RESPONSE_OPEN): ("tool_result", tokenizer.convert_tokens_to_ids(TOOL_RESPONSE_CLOSE)),
    }
    turn_close = {tokenizer.convert_tokens_to_ids(TURN_CLOSE)}
    spans: list[MessageSpan] = []
    text_run: list[int] = []

    def flush_text_run() -> None:
        if text_run and tokenizer.decode([full_ids[i] for i in text_run]).strip():
            spans.append(MessageSpan("response", range(text_run[0], text_run[-1] + 1), end_token_index_after(full_ids, text_run, turn_close)))
        text_run.clear()

    position, end = (content_indices[0], content_indices[-1] + 1) if content_indices else (0, 0)
    while position < end:
        block = block_kind_by_open.get(full_ids[position])
        if block is None:
            text_run.append(position)
            position += 1
            continue
        flush_text_run()
        kind, close_id = block
        close = full_ids.index(close_id, position + 1) if close_id in full_ids[position + 1:end] else end
        spans.append(MessageSpan(kind, range(position + 1, close), close if close < end else None))
        position = close + 1
    flush_text_run()
    return spans


class Gemma4ModelSpecifics(CoverallLayerGetter, Gemma4GenerationSpecifics, ModelSpecifics[AllRoles, Gemma4AssistantTurn],
                           MessageSpanSpecifics, ToolCallCloseSpecifics, UnembeddingSpecifics):
    def message_spans(self, tokenizer: HFTokenizer, full_ids: list[int]) -> list[MessageSpan]:
        thought_close = {tokenizer.convert_tokens_to_ids(THOUGHT_CLOSE)}
        spans: list[MessageSpan] = []
        for message in iter_gemma4_messages(tokenizer, full_ids):
            content = range(message.content_indices[0], message.content_indices[-1] + 1) if message.content_indices else range(0)
            if message.channel == "thought":
                spans.append(MessageSpan("reasoning", content, end_token_index_after(full_ids, message.content_indices, thought_close)))
            elif message.role == "assistant":
                spans += _model_turn_spans(tokenizer, full_ids, message.content_indices)
            else:
                spans.append(MessageSpan("system" if message.role == "system" else "user", content, None))
        return spans

    def tool_call_close(self, tokenizer: HFTokenizer, full_ids: list[int], span: MessageSpan) -> range:
        """The closing brace of the call's arguments."""
        return shortest_suffix_containing(tokenizer, full_ids, span.content, "}")

    def get_response_indices(self, conversation: Conversation, tokenizer: HFTokenizer, *, include_thinking: bool,
                             **apply_chat_template_kwargs: Any) -> list[list[int]]:
        full_ids = configure_apply_chat_template(tokenizer).tokenize(True)(
            conversation, add_generation_prompt=False, **apply_chat_template_kwargs
        )["input_ids"]
        return [m.content_indices for m in pooled_gemma4_messages(tokenizer, full_ids, include_thinking) if m.role == "assistant"]

    def build_turn_spans(self, conversation: Conversation, tokenizer: HFTokenizer, full_ids: list[int], *, include_thinking: bool,
                         **apply_chat_template_kwargs) -> tuple[list[int], list[dict[str, Any]]]:
        turns = [turn for turn in conversation if turn["role"] != "system"]
        messages = pooled_gemma4_messages(tokenizer, full_ids, include_thinking)
        spans = [
            {"turn": i, "role": turn["role"], "start": m.content_indices[0], "end": m.content_indices[-1] + 1,
             "n_tokens": len(m.content_indices), "text": turn["content"]}
            for i, (turn, m) in enumerate(zip(turns, messages, strict=True))
            if m.content_indices
        ]
        return full_ids, spans

    def get_layers(self, model: Any) -> nn.ModuleList:
        try:  # Gemma4ForConditionalGeneration: text tower under model.language_model
            layers = model.model.language_model.layers
            if layers is not None and len(layers) > 0:
                return layers
        except AttributeError:
            pass
        return super().get_layers(model)

    def gain_scaled_unembedding(self, load_tensor: CheckpointTensorLoader) -> Float[torch.Tensor, "vocab hidden"]:
        """Tied embeddings; Gemma4RMSNorm scales by weight (not Gemma 2/3's 1 + weight)."""
        return load_tensor("model.language_model.embed_tokens.weight") * load_tensor("model.language_model.norm.weight")
