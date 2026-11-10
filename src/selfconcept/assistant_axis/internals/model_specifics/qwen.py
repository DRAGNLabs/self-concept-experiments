from typing import Any

from selfconcept.assistant_axis.internals.conversation_utils import content_only_ids_and_offset_standard, find_subsequence
from selfconcept.assistant_axis.internals.model_specifics._shared import build_turn_spans_chatml, get_response_indices_chatml
from selfconcept.assistant_axis.internals.model_specifics.message_spans import (
    MessageKind, MessageSpan, MessageSpanSpecifics, ToolCallCloseSpecifics, shortest_suffix_containing,
)
from selfconcept.assistant_axis.internals.model_specifics.types import CoverallLayerGetter, ModelSpecifics
from selfconcept.common.generation.model_specifics import QwenThinkGenerationSpecifics, ReasoningContentAssistantTurn
from selfconcept.common.hf_strong_types import AllRoles, Conversation, HFTokenizer, configure_apply_chat_template


IM_START, IM_END = "<|im_start|>", "<|im_end|>"
THINK_OPEN, THINK_CLOSE = "<think>", "</think>"
TOOL_CALL_OPEN, TOOL_CALL_CLOSE = "<tool_call>", "</tool_call>"
TOOL_RESPONSE_OPEN, TOOL_RESPONSE_CLOSE = "<tool_response>", "</tool_response>"


def _trimmed(tokenizer: HFTokenizer, full_ids: list[int], start: int, end: int) -> range:
    """[start, end) without the whitespace-only tokens at either edge (the template's newlines around a block)."""
    while start < end and not tokenizer.decode([full_ids[start]]).strip():
        start += 1
    while end > start and not tokenizer.decode([full_ids[end - 1]]).strip():
        end -= 1
    return range(start, end)


def qwen_message_spans(tokenizer: HFTokenizer, full_ids: list[int]) -> list[MessageSpan]:
    """Rendered messages of a Qwen3.x ChatML conversation (``<|im_start|>ROLE\n … <|im_end|>``), tool calls included.

    An assistant turn is ``[<think>\n{reasoning}\n</think>\n\n]{content}[\n<tool_call>\n…\n</tool_call>]…``: a
    non-empty think block is a reasoning span (the empty block the template writes with thinking off is none), each
    ``<tool_call>…</tool_call>`` a tool_call and the remaining text a response. Tool results are rendered inside user
    turns as ``<tool_response>\n…\n</tool_response>`` blocks, one tool_result each. A tool call's ``end_token_index``
    is the ``<|im_end|>`` that follows its ``</tool_call>`` when it closes the turn (Goodfire's readout suffix is
    ``</function>\n</tool_call><|im_end|>``), otherwise the ``</tool_call>`` itself."""
    token_id = tokenizer.convert_tokens_to_ids
    im_start, im_end = token_id(IM_START), token_id(IM_END)
    think_open, think_close = token_id(THINK_OPEN), token_id(THINK_CLOSE)
    blocks: dict[int, tuple[MessageKind, int]] = {
        token_id(TOOL_CALL_OPEN): ("tool_call", token_id(TOOL_CALL_CLOSE)),
        token_id(TOOL_RESPONSE_OPEN): ("tool_result", token_id(TOOL_RESPONSE_CLOSE)),
    }
    spans: list[MessageSpan] = []
    position = 0
    while im_start in full_ids[position:]:
        header_start = full_ids.index(im_start, position) + 1
        turn_end = full_ids.index(im_end, header_start) if im_end in full_ids[header_start:] else len(full_ids)
        closed = turn_end < len(full_ids)
        # header "ROLE\n": the role's tokens up to the first token that contains a newline
        content_start = next((i + 1 for i in range(header_start, turn_end) if "\n" in tokenizer.decode([full_ids[i]])), turn_end)
        role = tokenizer.decode(full_ids[header_start:content_start]).strip()
        position = turn_end + 1
        if role == "system":
            spans.append(MessageSpan("system", _trimmed(tokenizer, full_ids, content_start, turn_end), turn_end if closed else None))
            continue
        cursor = content_start
        if role == "assistant" and cursor < turn_end and full_ids[cursor] == think_open:
            close = full_ids.index(think_close, cursor) if think_close in full_ids[cursor:turn_end] else turn_end
            thought = _trimmed(tokenizer, full_ids, cursor + 1, close)
            if len(thought):
                spans.append(MessageSpan("reasoning", thought, close if close < turn_end else None))
            cursor = min(close + 1, turn_end)
        text_kind: MessageKind = "response" if role == "assistant" else "user"
        run_start = cursor

        def flush_text(until: int) -> None:
            text = _trimmed(tokenizer, full_ids, run_start, until)
            if len(text):
                spans.append(MessageSpan(text_kind, text, turn_end if closed and until == turn_end else None))

        while cursor < turn_end:
            block = blocks.get(full_ids[cursor])
            if block is None:
                cursor += 1
                continue
            flush_text(cursor)
            kind, close_id = block
            close = full_ids.index(close_id, cursor + 1) if close_id in full_ids[cursor + 1:turn_end] else turn_end
            if close == turn_end:
                end_token = None  # the ids end inside the block
            elif kind == "tool_call" and closed and close + 1 == turn_end:
                end_token = turn_end
            else:
                end_token = close
            spans.append(MessageSpan(kind, _trimmed(tokenizer, full_ids, cursor + 1, close), end_token))
            cursor = min(close + 1, turn_end)
            run_start = cursor
        flush_text(turn_end)
    return spans


class QwenModelSpecifics(CoverallLayerGetter, QwenThinkGenerationSpecifics, ModelSpecifics[AllRoles, ReasoningContentAssistantTurn],
                         MessageSpanSpecifics, ToolCallCloseSpecifics):
    def message_spans(self, tokenizer: HFTokenizer, full_ids: list[int]) -> list[MessageSpan]:
        return qwen_message_spans(tokenizer, full_ids)

    def tool_call_close(self, tokenizer: HFTokenizer, full_ids: list[int], span: MessageSpan) -> range:
        """``</function>`` through ``</tool_call>``: with the span's end token (``<|im_end|>``) this is Goodfire's own
        readout suffix ``</function>\n</tool_call><|im_end|>``."""
        closing = shortest_suffix_containing(tokenizer, full_ids, span.content, "</function>")
        tool_call_close = tokenizer.convert_tokens_to_ids(TOOL_CALL_CLOSE)
        stop = next((i + 1 for i in range(span.content.stop, min(span.content.stop + 4, len(full_ids))) if full_ids[i] == tool_call_close), span.content.stop)
        return range(closing.start, stop)

    def get_response_indices(self, conversation: Conversation, tokenizer: HFTokenizer, *, include_thinking: bool, **apply_chat_template_kwargs: Any) -> list[list[int]]:
        return get_response_indices_chatml(conversation, tokenizer, include_thinking=include_thinking, **apply_chat_template_kwargs)

    def build_turn_spans(
        self,
        conversation: Conversation,
        tokenizer: HFTokenizer,
        full_ids: list[int],
        *,
        include_thinking: bool,
        **apply_chat_template_kwargs,
    ) -> tuple[list[int], list[dict[str, Any]]]:
        return build_turn_spans_chatml(conversation, tokenizer, full_ids, include_thinking=include_thinking, **apply_chat_template_kwargs)
