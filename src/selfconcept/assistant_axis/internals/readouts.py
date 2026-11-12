"""Token positions to read activations at, located through the rendered messages of a conversation that may contain
tool calls and tool results. Each function asks only for the model-specifics protocol it uses."""
from typing import Any, Literal, NotRequired, TypedDict, cast

from selfconcept.assistant_axis.internals.model_specifics.message_spans import (
    MessageKind, MessageSpan, MessageSpanSpecifics, ReadoutSpecifics, shortest_suffix_containing,
)
from selfconcept.assistant_axis.internals.model_specifics.gemma4 import Gemma4ModelSpecifics
from selfconcept.assistant_axis.internals.model_specifics.gpt_model_specifics import GptModelSpecifics
from selfconcept.assistant_axis.internals.model_specifics.muse_glimmer import MuseGlimmerModelSpecifics
from selfconcept.assistant_axis.internals.model_specifics.qwen import QwenModelSpecifics
from selfconcept.common.hf_strong_types import HFTokenizer

READOUT_SPECIFICS_BY_MODEL_SUBSTRING: dict[str, ReadoutSpecifics] = {
    "gpt-oss": GptModelSpecifics(),
    "glimmer": MuseGlimmerModelSpecifics(),
    "gemma-4": Gemma4ModelSpecifics(),
    "qwen": QwenModelSpecifics(),
}


class ToolCallingMessage(TypedDict):
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    tool_calls: NotRequired[list[dict[str, Any]]]
    reasoning_content: NotRequired[str]


def tool_calling_conversation_ids(tokenizer: HFTokenizer, messages: list[ToolCallingMessage], **chat_kwargs: Any) -> list[int]:
    """The chat template renders tool calls and tool results, which the typed `Conversation` does not model."""
    return cast(list[int], tokenizer.apply_chat_template(cast(Any, messages), tokenize=True, return_dict=False, **chat_kwargs))


def _take(rendered: list[MessageSpan], cursor: int, kinds: list[MessageKind]) -> list[MessageSpan]:
    taken = rendered[cursor:cursor + len(kinds)]
    if [span.kind for span in taken] != kinds:
        raise ValueError(f"expected rendered messages {kinds} at {cursor}, found {[span.kind for span in taken]}")
    return taken


def spans_by_message(
    specifics: MessageSpanSpecifics, tokenizer: HFTokenizer, full_ids: list[int], messages: list[ToolCallingMessage]
) -> list[list[MessageSpan]]:
    """The rendered spans of each conversation message, in conversation order; system messages map to no spans.
    An assistant message is its reasoning (if rendered), one span per tool call, then its response (if any)."""
    rendered = [span for span in specifics.message_spans(tokenizer, full_ids) if span.kind != "system"]
    cursor = 0
    spans_per_message: list[list[MessageSpan]] = []
    for message in messages:
        if message["role"] == "system":
            spans_per_message.append([])
            continue
        if message["role"] == "assistant":
            reasoning = [span for span in rendered[cursor:cursor + 1] if span.kind == "reasoning"]
            tool_call_kinds: list[MessageKind] = ["tool_call"] * len(message.get("tool_calls", []))
            response_kinds: list[MessageKind] = ["response"] if message["content"] else []
            kinds = tool_call_kinds + response_kinds
        else:
            reasoning = []
            kinds: list[MessageKind] = ["user"] if message["role"] == "user" else ["tool_result"]
        taken = reasoning + _take(rendered, cursor + len(reasoning), kinds)
        spans_per_message.append(taken)
        cursor += len(taken)
    if cursor != len(rendered):
        raise ValueError(f"{len(rendered) - cursor} rendered messages left over after the conversation")
    return spans_per_message


def _final_action_span(spans: list[MessageSpan]) -> MessageSpan:
    return [span for span in spans if span.kind in ("tool_call", "response")][-1]


def last_token_positions(
    specifics: MessageSpanSpecifics, tokenizer: HFTokenizer, full_ids: list[int], messages: list[ToolCallingMessage],
    message_index: int,
) -> list[int]:
    """The end token of the message's last tool call, or of its response when it makes no tool call."""
    span = _final_action_span(spans_by_message(specifics, tokenizer, full_ids, messages)[message_index])
    if span.end_token_index is None:
        raise ValueError(f"message {message_index} has no end token")
    return [span.end_token_index]


def tool_call_close_positions(
    specifics: ReadoutSpecifics, tokenizer: HFTokenizer, full_ids: list[int], messages: list[ToolCallingMessage],
    message_index: int,
) -> list[int]:
    """The closing markup of the message's last tool call, through its end token."""
    span = _final_action_span(spans_by_message(specifics, tokenizer, full_ids, messages)[message_index])
    if span.kind != "tool_call" or span.end_token_index is None:
        raise ValueError(f"message {message_index} does not end in a complete tool call")
    return [*specifics.tool_call_close(tokenizer, full_ids, span), span.end_token_index]


def response_close_positions(
    specifics: MessageSpanSpecifics, tokenizer: HFTokenizer, full_ids: list[int], messages: list[ToolCallingMessage],
    message_index: int, marker: str = "```",
) -> list[int]:
    """A response's last `marker` (the closing code fence by default) through its end token."""
    span = _final_action_span(spans_by_message(specifics, tokenizer, full_ids, messages)[message_index])
    if span.kind != "response" or span.end_token_index is None:
        raise ValueError(f"message {message_index} does not end in a complete response")
    return [*shortest_suffix_containing(tokenizer, full_ids, span.content, marker), span.end_token_index]


def assistant_positions(
    specifics: MessageSpanSpecifics, tokenizer: HFTokenizer, full_ids: list[int], messages: list[ToolCallingMessage],
    kinds: frozenset[MessageKind],
) -> list[int]:
    """Every content token of the conversation's assistant messages of the given kinds."""
    return [
        position
        for spans in spans_by_message(specifics, tokenizer, full_ids, messages)
        for span in spans
        if span.kind in kinds
        for position in span.content
    ]


def all_response_positions(
    specifics: MessageSpanSpecifics, tokenizer: HFTokenizer, full_ids: list[int], messages: list[ToolCallingMessage]
) -> list[int]:
    """Every content token of every assistant tool call and response, reasoning excluded."""
    return assistant_positions(specifics, tokenizer, full_ids, messages, frozenset({"tool_call", "response"}))


def final_response_positions(
    specifics: MessageSpanSpecifics, tokenizer: HFTokenizer, full_ids: list[int], messages: list[ToolCallingMessage]
) -> list[int]:
    """The content tokens of the conversation's last assistant response (Goodfire's closing report)."""
    responses = [span for spans in spans_by_message(specifics, tokenizer, full_ids, messages) for span in spans if span.kind == "response"]
    if not responses:
        raise ValueError("the conversation has no assistant response")
    return list(responses[-1].content)


def readout_specifics_for(model_name: str) -> ReadoutSpecifics:
    """For scripts to resolve once at the top level; library functions take the specifics as a parameter."""
    matches = [specifics for substring, specifics in READOUT_SPECIFICS_BY_MODEL_SUBSTRING.items() if substring in model_name.lower()]
    if not matches:
        raise ValueError(f"no ReadoutSpecifics for {model_name}: implement message_spans and tool_call_close for it and "
                         "add it to READOUT_SPECIFICS_BY_MODEL_SUBSTRING")
    return matches[0]
