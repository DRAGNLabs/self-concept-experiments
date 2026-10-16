"""Difference-in-means self-recognition vectors: own-text minus other-text spans of the stage-2 judge prompts."""
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any, TypedDict

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor, nn
from transformers import PreTrainedModel

from selfconcept.common.activation_extraction.extraction import extract_token_span_activations
from selfconcept.common.hf_strong_types import HFTokenizer
from selfconcept.common.span_targeting.types import TokenSpan
from selfconcept.self_recognition.judge_prompt_spans import locate_text_spans, tokenize_generation_prompt
from selfconcept.self_recognition.judging import judge_candidate_text
from selfconcept.self_recognition.schemas import (
    JudgementRecord,
    PooledSpanMeans,
    SelfRecognitionVectors,
    TranscriptRecord,
)
from selfconcept.self_recognition.success_filter import LENGTH_MATCHED_MAX_LOG_RATIO, spearman_or_nan

type TranscriptKey = tuple[str, int, int]
"""(model, prompt_id, sample_index)"""
type JudgePromptSpans = tuple[list[int], TokenSpan, TokenSpan]
"""(prompt token ids, own-text span, other-text span)"""
type JudgementsInBothOrders = tuple[JudgementRecord, JudgementRecord]
"""(target_position 1, target_position 2)"""


def transcript_key(transcript: TranscriptRecord) -> TranscriptKey:
    return transcript["model"], transcript["prompt_id"], transcript["sample_index"]


def judge_prompt_token_spans(
    judgement: JudgementRecord,
    transcripts_by_key: Mapping[TranscriptKey, TranscriptRecord],
    tokenizer: HFTokenizer,
    chat_template_kwargs: Mapping[str, Any],
) -> JudgePromptSpans:
    prompt_id = judgement["prompt_id"]
    own_text = judge_candidate_text(
        transcripts_by_key[judgement["target_model"], prompt_id, judgement["target_sample_index"]]
    ).strip()
    other_text = judge_candidate_text(
        transcripts_by_key[judgement["other_model"], prompt_id, judgement["other_sample_index"]]
    ).strip()
    token_ids = tokenize_generation_prompt(judgement["judge_conversation"], tokenizer, **chat_template_kwargs)
    if judgement["target_position"] == 1:
        own_span, other_span = locate_text_spans(token_ids, tokenizer, [own_text, other_text])
    else:
        other_span, own_span = locate_text_spans(token_ids, tokenizer, [other_text, own_text])
    return token_ids, own_span, other_span


def candidate_char_counts(
    judgement: JudgementRecord,
    transcripts_by_key: Mapping[TranscriptKey, TranscriptRecord],
) -> tuple[int, int]:
    """(own, other) characters of the candidate texts, as success_filter measures them."""
    prompt_id = judgement["prompt_id"]
    return (
        len(judge_candidate_text(transcripts_by_key[judgement["target_model"], prompt_id, judgement["target_sample_index"]])),
        len(judge_candidate_text(transcripts_by_key[judgement["other_model"], prompt_id, judgement["other_sample_index"]])),
    )


def span_length(span: TokenSpan) -> int:
    return span.end - span.start


def pooled_own_and_other(
    model: PreTrainedModel,
    examples: Sequence[JudgePromptSpans],
    decoder_layers: nn.ModuleList,
    pad_token_id: int,
    max_length: int,
) -> tuple[Float[Tensor, "example layer hidden"], Float[Tensor, "example layer hidden"]]:
    """Mean over each span's tokens at every decoder layer, as float32 on CPU."""
    spans_in_position_order = [sorted([own_span, other_span]) for _, own_span, other_span in examples]
    span_activations_by_example = extract_token_span_activations(
        model,
        [token_ids for token_ids, _, _ in examples],
        spans_in_position_order,
        decoder_layers,
        pad_token_id,
        max_length,
    )
    own_means: list[Float[Tensor, "layer hidden"]] = []
    other_means: list[Float[Tensor, "layer hidden"]] = []
    for (_, own_span, _), spans, span_activations in zip(
        examples, spans_in_position_order, span_activations_by_example, strict=True
    ):
        span_lengths = [span_length(span) for span in spans]
        if span_activations.shape[1] != sum(span_lengths):
            raise ValueError(f"spans were truncated: {span_activations.shape[1]} of {sum(span_lengths)} tokens")
        first_span_mean, second_span_mean = (
            activations.float().mean(dim=1).cpu() for activations in span_activations.split(span_lengths, dim=1)
        )
        own_is_first = spans[0] == own_span
        own_means.append(first_span_mean if own_is_first else second_span_mean)
        other_means.append(second_span_mean if own_is_first else first_span_mean)
    return torch.stack(own_means), torch.stack(other_means)


def difference_in_means(
    own: Float[Tensor, "example layer hidden"],
    other: Float[Tensor, "example layer hidden"],
) -> Float[Tensor, "layer hidden"]:
    return own.mean(dim=0) - other.mean(dim=0)


def build_self_recognition_vectors(
    model_name: str,
    own: Float[Tensor, "example layer hidden"],
    other: Float[Tensor, "example layer hidden"],
    train_prompt_ids: Sequence[int],
) -> SelfRecognitionVectors:
    return {
        "model": model_name,
        "axis": difference_in_means(own, other),
        "train_prompt_ids": sorted(set(train_prompt_ids)),
        "positive_count": own.shape[0],
        "negative_count": other.shape[0],
    }


def non_recognised_judgement_pairs(judgements: Sequence[JudgementRecord]) -> list[JudgementsInBothOrders]:
    """Items judged in both orders but not correctly in both."""
    judgements_by_position_by_item: defaultdict[tuple[int, str, int, int], dict[int, JudgementRecord]] = (
        defaultdict(dict)
    )
    for judgement in judgements:
        item_key = (
            judgement["prompt_id"],
            judgement["other_model"],
            judgement["target_sample_index"],
            judgement["other_sample_index"],
        )
        judgements_by_position_by_item[item_key][judgement["target_position"]] = judgement
    return [
        (judgements_by_position[1], judgements_by_position[2])
        for judgements_by_position in judgements_by_position_by_item.values()
        if judgements_by_position.keys() == {1, 2}
        and not (judgements_by_position[1]["correct"] and judgements_by_position[2]["correct"])
    ]


def sample_without_replacement[T](items: Sequence[T], count: int, seed: int) -> list[T]:
    return random.Random(seed).sample(list(items), min(count, len(items)))


def per_layer_cosine(
    first: Float[Tensor, "layer hidden"],
    second: Float[Tensor, "layer hidden"],
) -> Float[Tensor, " layer"]:
    return torch.nn.functional.cosine_similarity(first, second, dim=-1)


def split_half_cosine(
    own: Float[Tensor, "example layer hidden"],
    other: Float[Tensor, "example layer hidden"],
    pair_indices: Int[Tensor, " example"],
    seed: int,
) -> Float[Tensor, " layer"]:
    """Cosine between the difference in means of two random halves of the pairs, keeping each pair's
    examples in the same half."""
    unique_pair_indices = pair_indices.unique()
    shuffled = unique_pair_indices[torch.randperm(len(unique_pair_indices), generator=torch.Generator().manual_seed(seed))]
    is_in_first_half = torch.isin(pair_indices, shuffled[: len(shuffled) // 2])
    return per_layer_cosine(
        difference_in_means(own[is_in_first_half], other[is_in_first_half]),
        difference_in_means(own[~is_in_first_half], other[~is_in_first_half]),
    )


class PairDifferences(TypedDict):
    own_minus_other: Float[Tensor, "pair layer hidden"]
    log_length_ratio: Float[Tensor, " pair"]
    """log(own characters / other characters)"""


def log_length_ratios(pooled: PooledSpanMeans) -> Float[Tensor, " example"]:
    own_char_counts = torch.tensor(pooled["own_char_counts"], dtype=torch.float64).clamp_min(1)
    other_char_counts = torch.tensor(pooled["other_char_counts"], dtype=torch.float64).clamp_min(1)
    return (own_char_counts / other_char_counts).log()


def pair_differences(pooled: PooledSpanMeans) -> PairDifferences:
    """Own minus other averaged over each pair's two presentation orders, so every pair counts once with both
    slots weighted equally."""
    _, pair_rows = pooled["pair_indices"].unique(return_inverse=True)
    pair_count = int(pair_rows.max()) + 1
    example_counts = torch.bincount(pair_rows, minlength=pair_count)
    own_minus_other_sums = torch.zeros(pair_count, *pooled["own"].shape[1:]).index_add_(
        0, pair_rows, pooled["own"] - pooled["other"]
    )
    log_length_ratio_sums = torch.zeros(pair_count, dtype=torch.float64).index_add_(
        0, pair_rows, log_length_ratios(pooled)
    )
    return {
        "own_minus_other": own_minus_other_sums / example_counts[:, None, None],
        "log_length_ratio": log_length_ratio_sums / example_counts,
    }


def longer_minus_shorter_axis(pairs: PairDifferences) -> Float[Tensor, "layer hidden"]:
    """Mean over pairs of the longer text's span mean minus the shorter's; exact length ties are skipped."""
    own_longer_sign = pairs["log_length_ratio"].sign()
    untied = own_longer_sign != 0
    return (own_longer_sign[untied, None, None].float() * pairs["own_minus_other"][untied]).mean(dim=0)


def authorship_balanced_length_axis(pairs: PairDifferences) -> Float[Tensor, "layer hidden"] | None:
    """Half the own-longer pairs' mean minus the own-shorter pairs' mean, so the own-minus-other part cancels."""
    own_longer = pairs["log_length_ratio"] > 0
    own_shorter = pairs["log_length_ratio"] < 0
    if not (own_longer.any() and own_shorter.any()):
        return None
    return (pairs["own_minus_other"][own_longer].mean(dim=0) - pairs["own_minus_other"][own_shorter].mean(dim=0)) / 2


def length_projection_spearman(
    pairs: PairDifferences,
    axis: Float[Tensor, "layer hidden"],
    layers: Sequence[int],
) -> list[float]:
    """Per layer, Spearman across pairs between log length ratio and the pair difference's projection onto the
    unit axis."""
    unit_axis = torch.nn.functional.normalize(axis[list(layers)], dim=-1)
    projections = torch.einsum("plh,lh->lp", pairs["own_minus_other"][:, list(layers)], unit_axis)
    return [
        spearman_or_nan(pairs["log_length_ratio"].numpy(), layer_projections.double().numpy())
        for layer_projections in projections
    ]


def is_length_matched(pooled: PooledSpanMeans) -> Bool[Tensor, " example"]:
    return log_length_ratios(pooled).abs() <= LENGTH_MATCHED_MAX_LOG_RATIO
