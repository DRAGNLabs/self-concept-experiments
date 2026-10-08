from collections.abc import Sequence
from typing import Protocol

import torch
from jaxtyping import Bool

from selfconcept.common.span_targeting.model_specifics import ModelSpecifics
from selfconcept.common.span_targeting.types import CompletionSpans, PromptCompletion, TokenSpan


class SpanTargeter(Protocol):
    def target_spans(self, completion_spans: CompletionSpans) -> list[TokenSpan]: ...


class AssistantTargeter:
    def target_spans(self, completion_spans: CompletionSpans) -> list[TokenSpan]:
        return completion_spans.thinking + completion_spans.response


class ResponseTargeter:
    def target_spans(self, completion_spans: CompletionSpans) -> list[TokenSpan]:
        return completion_spans.response


class ThinkingTargeter:
    def target_spans(self, completion_spans: CompletionSpans) -> list[TokenSpan]:
        return completion_spans.thinking


def build_target_mask(
    examples: Sequence[PromptCompletion],
    targeter: SpanTargeter,
    model_specifics: ModelSpecifics,
    seq_len: int,
    device: torch.device,
) -> Bool[torch.Tensor, "batch seq"]:
    """Aligned with right-padded prompt+completion rows; spans past seq_len are clipped."""
    return build_span_mask(
        [targeter.target_spans(model_specifics.split_completion(example)) for example in examples], seq_len, device
    )


def build_span_mask(
    token_spans_by_example: Sequence[list[TokenSpan]],
    seq_len: int,
    device: torch.device,
) -> Bool[torch.Tensor, "batch seq"]:
    """Aligned with right-padded rows; spans past seq_len are clipped."""
    example_indices: list[int] = []
    span_starts: list[int] = []
    span_ends: list[int] = []
    for example_index, token_spans in enumerate(token_spans_by_example):
        for span in token_spans:
            example_indices.append(example_index)
            span_starts.append(span.start)
            span_ends.append(span.end)

    positions = torch.arange(seq_len, device=device)
    span_starts_tensor = torch.tensor(span_starts, dtype=torch.long, device=device)
    span_ends_tensor = torch.tensor(span_ends, dtype=torch.long, device=device)
    is_in_span_by_span = (positions >= span_starts_tensor[:, None]) & (positions < span_ends_tensor[:, None])
    targeted_span_count = torch.zeros(len(token_spans_by_example), seq_len, dtype=torch.long, device=device).index_add_(
        0, torch.tensor(example_indices, dtype=torch.long, device=device), is_in_span_by_span.long()
    )
    return targeted_span_count > 0
