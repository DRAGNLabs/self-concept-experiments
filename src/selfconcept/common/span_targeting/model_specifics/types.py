from typing import Protocol

from selfconcept.common.span_targeting.types import CompletionSpans, PromptCompletion


class ModelSpecifics(Protocol):
    def split_completion(self, example: PromptCompletion) -> CompletionSpans:
        """Spans hold content tokens only: no tags, headers or terminators."""
        ...
