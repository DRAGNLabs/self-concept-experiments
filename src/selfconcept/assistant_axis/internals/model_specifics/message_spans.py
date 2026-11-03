"""Narrow model-specifics protocols for reading activations at chosen messages of a rendered conversation,
including tool calls and tool results, which the turn-span pooling of `ModelSpecifics` skips."""
from typing import Literal, NamedTuple, Protocol

from selfconcept.common.hf_strong_types import HFTokenizer

type MessageKind = Literal["system", "user", "reasoning", "response", "tool_call", "tool_result"]


class MessageSpan(NamedTuple):
    kind: MessageKind
    content: range  # header and terminating special token excluded
    end_token_index: int | None  # the terminating special token, None when the ids end inside the message


class MessageSpanSpecifics(Protocol):
    def message_spans(self, tokenizer: HFTokenizer, full_ids: list[int]) -> list[MessageSpan]:
        """Every rendered message in order, including messages the chat template inserts (kind "system")."""
        ...


class ToolCallCloseSpecifics(Protocol):
    def tool_call_close(self, tokenizer: HFTokenizer, full_ids: list[int], span: MessageSpan) -> range:
        """The closing markup at the end of a tool-call message's content, after its last argument value."""
        ...


class ReadoutSpecifics(MessageSpanSpecifics, ToolCallCloseSpecifics, Protocol): ...


def shortest_suffix_containing(tokenizer: HFTokenizer, full_ids: list[int], content: range, marker: str) -> range:
    for start in reversed(content):
        if marker in tokenizer.decode(full_ids[start:content.stop]):
            return range(start, content.stop)
    raise ValueError(f"{marker!r} not found in the message content")


def end_token_index_after(full_ids: list[int], content_indices: list[int], end_token_ids: set[int]) -> int | None:
    if not content_indices:
        return None
    following = content_indices[-1] + 1
    return following if following < len(full_ids) and full_ids[following] in end_token_ids else None
