from typing import NamedTuple


class PromptCompletion(NamedTuple):
    prompt_token_ids: list[int]
    completion_token_ids: list[int]


class TokenSpan(NamedTuple):
    """Indexes the concatenated prompt and completion; end is exclusive."""
    start: int
    end: int


class CompletionSpans(NamedTuple):
    thinking: list[TokenSpan]
    response: list[TokenSpan]
