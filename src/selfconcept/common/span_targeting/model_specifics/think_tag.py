from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from itertools import accumulate

from selfconcept.common.hf_strong_types import HFTokenizer
from selfconcept.common.span_targeting.model_specifics._shared import (
    TokenPiece,
    decode_token_pieces,
    place_after_prompt,
    trim_span,
)
from selfconcept.common.span_targeting.types import CompletionSpans, PromptCompletion, TokenSpan

type CharSpan = tuple[int, int]


def _thinking_and_response_char_spans(
    completion_text: str,
    open_tag: str,
    close_tag: str,
    prompt_left_thinking_open: bool,
) -> tuple[CharSpan | None, CharSpan | None]:
    open_char = completion_text.find(open_tag)
    close_char = completion_text.find(close_tag, max(open_char, 0))
    if open_char != -1:
        thinking_char_start = open_char + len(open_tag)
    elif close_char != -1 or prompt_left_thinking_open:
        thinking_char_start = 0
    else:
        return None, (0, len(completion_text))

    if close_char == -1:
        return (thinking_char_start, len(completion_text)), None
    return (thinking_char_start, close_char), (close_char + len(close_tag), len(completion_text))


def token_piece_char_offsets(token_pieces: list[TokenPiece]) -> tuple[list[int], list[int]]:
    """(piece_starts, piece_ends): each token's character offsets in the joined decoded text."""
    piece_ends = list(accumulate(len(piece or "") for piece in token_pieces))
    piece_starts = [0, *piece_ends[:-1]]
    return piece_starts, piece_ends


def tokens_within_chars(char_span: CharSpan, piece_starts: list[int], piece_ends: list[int]) -> TokenSpan:
    """Tokens lying entirely inside char_span, so a token straddling a tag belongs to neither side."""
    char_start, char_end = char_span
    token_start = bisect_left(piece_starts, char_start)
    token_end = bisect_right(piece_ends, char_end)
    return TokenSpan(token_start, max(token_start, token_end))


@dataclass(frozen=True)
class ThinkTagModelSpecifics:
    """Tags are located in decoded text, since some tokenizers (OLMo 3) fuse them with neighbouring characters."""

    tokenizer: HFTokenizer
    open_tag: str = "<think>"
    close_tag: str = "</think>"

    def split_completion(self, example: PromptCompletion) -> CompletionSpans:
        completion_token_ids = example.completion_token_ids
        token_pieces = decode_token_pieces(self.tokenizer, completion_token_ids)
        piece_starts, piece_ends = token_piece_char_offsets(token_pieces)
        special_token_ids = set(self.tokenizer.all_special_ids)

        thinking_char_span, response_char_span = _thinking_and_response_char_spans(
            "".join(piece or "" for piece in token_pieces),
            self.open_tag,
            self.close_tag,
            self._prompt_left_thinking_open(example),
        )

        def to_token_spans(char_span: CharSpan | None) -> list[TokenSpan]:
            if char_span is None:
                return []
            token_span = tokens_within_chars(char_span, piece_starts, piece_ends)
            return place_after_prompt(
                [trim_span(token_span, completion_token_ids, token_pieces, special_token_ids)], example
            )

        return CompletionSpans(thinking=to_token_spans(thinking_char_span), response=to_token_spans(response_char_span))

    def _prompt_left_thinking_open(self, example: PromptCompletion) -> bool:
        prompt_text = self.tokenizer.decode(example.prompt_token_ids)
        return prompt_text.rfind(self.open_tag) > prompt_text.rfind(self.close_tag)
