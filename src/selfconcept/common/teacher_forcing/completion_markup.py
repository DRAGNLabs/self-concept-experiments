from typing import NamedTuple, Protocol


class CompletionMarkup(NamedTuple):
    """The raw text a model writes around its reasoning and answer after the generation prompt."""
    thought_open: str
    thought_close: str  # also opens the answer
    end_of_turn: str
    direct_answer_open: str  # opens the answer of a turn without reasoning


class CompletionMarkupSpecifics(Protocol):
    def completion_markup(self) -> CompletionMarkup: ...
