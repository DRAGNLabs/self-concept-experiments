from collections.abc import Sequence
from typing import Any

from selfconcept.common.hf_strong_types import Conversation, HFTokenizer, configure_apply_chat_template, configure_call
from selfconcept.common.span_targeting.model_specifics._shared import decode_token_pieces, trim_span
from selfconcept.common.span_targeting.model_specifics.think_tag import (
    CharSpan,
    token_piece_char_offsets,
    tokens_within_chars,
)
from selfconcept.common.span_targeting.types import TokenSpan


def tokenize_generation_prompt(
    conversation: Conversation,
    tokenizer: HFTokenizer,
    **chat_template_kwargs: Any,
) -> list[int]:
    prompt_text = configure_apply_chat_template(tokenizer).tokenize(False)(
        conversation, add_generation_prompt=True, **chat_template_kwargs
    )
    return configure_call(tokenizer)(prompt_text, add_special_tokens=False)["input_ids"]


def locate_text_spans(token_ids: list[int], tokenizer: HFTokenizer, texts: Sequence[str]) -> list[TokenSpan]:
    """The tokens of each text, searched for in order in the decoded token_ids; edge whitespace is trimmed."""
    token_pieces = decode_token_pieces(tokenizer, token_ids)
    piece_starts, piece_ends = token_piece_char_offsets(token_pieces)
    decoded_text = "".join(piece or "" for piece in token_pieces)
    special_token_ids = set(tokenizer.all_special_ids)

    token_spans: list[TokenSpan] = []
    search_start = 0
    for text in texts:
        char_start = decoded_text.find(text, search_start)
        if char_start == -1:
            raise ValueError(f"text not found in the decoded tokens after char {search_start}: {text[:80]!r}")
        char_span: CharSpan = (char_start, char_start + len(text))
        token_span = trim_span(
            tokens_within_chars(char_span, piece_starts, piece_ends), token_ids, token_pieces, special_token_ids
        )
        if token_span.start == token_span.end:
            raise ValueError(f"no whole tokens inside the located text: {text[:80]!r}")
        token_spans.append(token_span)
        search_start = char_span[1]
    return token_spans
