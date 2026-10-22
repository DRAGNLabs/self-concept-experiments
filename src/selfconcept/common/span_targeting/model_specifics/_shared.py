from tokenizers.decoders import DecodeStream

from selfconcept.common.hf_strong_types import HFTokenizer
from selfconcept.common.span_targeting.types import PromptCompletion, TokenSpan

type TokenPiece = str | None
"""A token's decoded text, or None when its bytes only decode together with a later token's."""


def decode_token_pieces(tokenizer: HFTokenizer, token_ids: list[int]) -> list[TokenPiece]:
    decode_stream = DecodeStream(skip_special_tokens=False)
    return [decode_stream.step(tokenizer.backend_tokenizer, token_id) for token_id in token_ids]


def _is_edge_noise(token_id: int, token_piece: TokenPiece, special_token_ids: set[int]) -> bool:
    is_whitespace = token_piece is not None and token_piece.strip() == ""
    return is_whitespace or token_id in special_token_ids


def trim_span(
    span: TokenSpan,
    token_ids: list[int],
    token_pieces: list[TokenPiece],
    special_token_ids: set[int],
) -> TokenSpan:
    """Drops whitespace-only and special tokens from both edges; may return an empty span."""
    start, end = span
    while start < end and _is_edge_noise(token_ids[start], token_pieces[start], special_token_ids):
        start += 1
    while end > start and _is_edge_noise(token_ids[end - 1], token_pieces[end - 1], special_token_ids):
        end -= 1
    return TokenSpan(start, end)


def place_after_prompt(completion_spans: list[TokenSpan], example: PromptCompletion) -> list[TokenSpan]:
    """Shifts completion-indexed spans to full-sequence indices, dropping empty ones."""
    prompt_length = len(example.prompt_token_ids)
    return [
        TokenSpan(span.start + prompt_length, span.end + prompt_length)
        for span in completion_spans
        if span.start < span.end
    ]
