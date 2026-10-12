"""Which paired judgements count as recognised, and per-(target, other) statistics for the go/no-go check."""
import math
import re
import statistics
from collections import defaultdict
from collections.abc import Sequence
from typing import Literal, TypedDict

import numpy as np
from numpy.typing import NDArray
from scipy.stats import bootstrap, rankdata

from selfconcept.self_recognition.judging import (
    TranscriptKey,
    is_judgeable,
    judge_candidate_text,
    judged_candidates_text,
    transcripts_by_key,
)
from selfconcept.self_recognition.schemas import CandidatePosition, JudgementRecord, PairKeyTuple, TranscriptRecord

NAME_LEAK_PATTERN = re.compile(r"Qwen|Alibaba|Gemma|Google|DeepMind|GPT|OpenAI|GLM|Zhipu|Z\.ai")
NAME_LEAK_ACCURACY_MARGIN = 0.05
POOLED_OTHER_MODEL = "all"
MIN_ACCURACY_CI_LOW = 0.5
MIN_PURITY = 0.5
LENGTH_MATCHED_MAX_LOG_RATIO = math.log(1.2)
LENGTH_BOOTSTRAP_RESAMPLES = 1000


class PairKey(TypedDict):
    prompt_id: int
    target_model: str
    other_model: str
    target_sample_index: int
    other_sample_index: int


class RecognisedPair(PairKey):
    target_first: JudgementRecord
    target_second: JudgementRecord


class AccuracyEstimate(TypedDict):
    """Per-presentation accuracy; the 95% CI resamples prompts, since samples of one prompt are correlated."""

    accuracy: float
    ci_low: float
    ci_high: float
    presentations: int


class LengthSelection(TypedDict):
    """Candidate length is log(own characters / other characters). An item's selection is the share of its two
    presentations in which the target chose its own text, so position bias cancels."""

    selection_spearman: float
    selection_spearman_ci_low: float
    selection_spearman_ci_high: float
    own_probability_spearman: float
    own_longer_rate: float
    own_longer_rate_among_recognised: float
    length_matched_item_rate: float
    length_matched_accuracy: AccuracyEstimate | None


class PairSummary(TypedDict):
    target_model: str
    other_model: str
    items: int
    skipped_truncated_or_empty: int
    accuracy: AccuracyEstimate | None
    debiased_accuracy: float | None
    position_bias_rate: float
    p_choose_1: float | None
    unparseable_rate: float
    robust_count: int
    robust_rate: float
    null_robust_rate: float | None
    purity: float | None
    longer_text_baseline: float
    length_selection: LengthSelection
    name_leak_rate: float
    accuracy_with_name_leak: AccuracyEstimate | None
    accuracy_without_name_leak: AccuracyEstimate | None


type GoCriterion = Literal["projected_count", "accuracy_ci_low", "purity"]


class GoDecision(TypedDict):
    target_model: str
    observed_robust_count: int
    projected_robust_count: float
    accuracy_ci_low: float | None
    purity: float | None
    verdict: Literal["GO", "NO-GO"]
    failed_criteria: list[GoCriterion]
    samples_needed_for_min_robust: int | None


def pair_key_tuple(judgement: JudgementRecord) -> PairKeyTuple:
    return (
        judgement["prompt_id"],
        judgement["target_model"],
        judgement["other_model"],
        judgement["target_sample_index"],
        judgement["other_sample_index"],
    )


def pair_key(key: PairKeyTuple) -> PairKey:
    prompt_id, target_model, other_model, target_sample_index, other_sample_index = key
    return {
        "prompt_id": prompt_id,
        "target_model": target_model,
        "other_model": other_model,
        "target_sample_index": target_sample_index,
        "other_sample_index": other_sample_index,
    }


def complete_items(judgements: Sequence[JudgementRecord]) -> dict[PairKeyTuple, dict[CandidatePosition, JudgementRecord]]:
    """Items judged in both orders, keyed by pair."""
    judgement_by_position_by_pair: defaultdict[PairKeyTuple, dict[CandidatePosition, JudgementRecord]] = defaultdict(dict)
    for judgement in judgements:
        judgement_by_position_by_pair[pair_key_tuple(judgement)][judgement["target_position"]] = judgement
    return {key: by_position for key, by_position in judgement_by_position_by_pair.items() if len(by_position) == 2}


def names_a_model(judgement: JudgementRecord) -> bool:
    return NAME_LEAK_PATTERN.search(judged_candidates_text(judgement["judge_conversation"])) is not None


def recognised_pairs(judgements: Sequence[JudgementRecord], exclude_name_leaks: bool = False) -> list[RecognisedPair]:
    """Items the target judged correctly in both presentation orders."""
    return [
        {**pair_key(key), "target_first": judgement_by_position[1], "target_second": judgement_by_position[2]}
        for key, judgement_by_position in complete_items(judgements).items()
        if judgement_by_position[1]["correct"] and judgement_by_position[2]["correct"]
        and not (exclude_name_leaks and names_a_model(judgement_by_position[1]))
    ]


def accuracy_estimate(judgements: Sequence[JudgementRecord]) -> AccuracyEstimate | None:
    if not judgements:
        return None
    correctness_by_prompt: defaultdict[int, list[bool]] = defaultdict(list)
    for judgement in judgements:
        correctness_by_prompt[judgement["prompt_id"]].append(judgement["correct"])
    correct_counts = np.array([sum(correctness) for correctness in correctness_by_prompt.values()], dtype=float)
    presentation_counts = np.array([len(correctness) for correctness in correctness_by_prompt.values()], dtype=float)
    accuracy = float(correct_counts.sum() / presentation_counts.sum())
    if len(correct_counts) < 2:
        return {"accuracy": accuracy, "ci_low": math.nan, "ci_high": math.nan, "presentations": len(judgements)}

    def pooled_accuracy(correct: NDArray[np.float64], presentations: NDArray[np.float64], axis: int) -> NDArray[np.float64]:
        return correct.sum(axis=axis) / presentations.sum(axis=axis)

    interval = bootstrap(
        (correct_counts, presentation_counts),
        pooled_accuracy,
        paired=True,
        vectorized=True,
        method="percentile",
        rng=np.random.default_rng(0),
    ).confidence_interval
    return {
        "accuracy": accuracy,
        "ci_low": float(interval.low),
        "ci_high": float(interval.high),
        "presentations": len(judgements),
    }


def self_probability(judgement: JudgementRecord) -> float | None:
    """P(target's label) renormalised over the two labels."""
    probabilities = judgement["answer_probabilities"]
    if probabilities is None:
        return None
    mass = probabilities["position_1"] + probabilities["position_2"]
    if mass == 0:
        return None
    target_probability = probabilities["position_1"] if judgement["target_position"] == 1 else probabilities["position_2"]
    return target_probability / mass


def debiased_accuracy(items: Sequence[dict[CandidatePosition, JudgementRecord]]) -> float | None:
    item_self_probabilities = [
        (first, second)
        for judgement_by_position in items
        if (first := self_probability(judgement_by_position[1])) is not None
        and (second := self_probability(judgement_by_position[2])) is not None
    ]
    if not item_self_probabilities:
        return None
    return statistics.mean((first + second) / 2 for first, second in item_self_probabilities)


def longer_text_correctness(
    key: PairKeyTuple,
    transcript_by_key_by_model: dict[str, dict[TranscriptKey, TranscriptRecord]],
) -> float:
    """1 if always picking the longer text picks the target's, 0 if it picks the other's, 0.5 on a tie."""
    prompt_id, target_model, other_model, target_sample_index, other_sample_index = key
    target_length = len(judge_candidate_text(transcript_by_key_by_model[target_model][(prompt_id, target_sample_index)]))
    other_length = len(judge_candidate_text(transcript_by_key_by_model[other_model][(prompt_id, other_sample_index)]))
    return 0.5 if target_length == other_length else float(target_length > other_length)


def spearman_or_nan(first: NDArray[np.float64], second: NDArray[np.float64]) -> float:
    if len(first) < 3 or np.ptp(first) == 0 or np.ptp(second) == 0:
        return math.nan
    return float(np.corrcoef(rankdata(first), rankdata(second))[0, 1])


def own_to_other_log_length(
    key: PairKeyTuple,
    transcript_by_key_by_model: dict[str, dict[TranscriptKey, TranscriptRecord]],
) -> float:
    prompt_id, target_model, other_model, target_sample_index, other_sample_index = key
    own_length = len(judge_candidate_text(transcript_by_key_by_model[target_model][(prompt_id, target_sample_index)]))
    other_length = len(judge_candidate_text(transcript_by_key_by_model[other_model][(prompt_id, other_sample_index)]))
    return math.log(max(own_length, 1) / max(other_length, 1))


def length_selection(
    judgements: Sequence[JudgementRecord],
    transcript_by_key_by_model: dict[str, dict[TranscriptKey, TranscriptRecord]],
) -> LengthSelection:
    judgement_by_position_by_pair = complete_items(judgements)
    keys = list(judgement_by_position_by_pair)
    log_ratios = np.array([own_to_other_log_length(key, transcript_by_key_by_model) for key in keys])
    selections = np.array(
        [
            (by_position[1]["correct"] + by_position[2]["correct"]) / 2
            for by_position in judgement_by_position_by_pair.values()
        ]
    )
    prompt_ids = np.array([key[0] for key in keys])
    rows_by_prompt_id = {prompt_id: np.flatnonzero(prompt_ids == prompt_id) for prompt_id in np.unique(prompt_ids)}
    rng = np.random.default_rng(0)
    resampled_spearmans = [
        spearman_or_nan(log_ratios[rows], selections[rows])
        for rows in (
            np.concatenate([rows_by_prompt_id[prompt_id] for prompt_id in rng.choice(list(rows_by_prompt_id), len(rows_by_prompt_id))])
            for _ in range(LENGTH_BOOTSTRAP_RESAMPLES if len(rows_by_prompt_id) > 1 else 0)
        )
    ]
    interval = (
        np.nanpercentile(resampled_spearmans, [2.5, 97.5])
        if resampled_spearmans and not np.all(np.isnan(resampled_spearmans))
        else (math.nan, math.nan)
    )
    probabilities = [
        (first + second) / 2 if (first := self_probability(by_position[1])) is not None and (second := self_probability(by_position[2])) is not None else math.nan
        for by_position in judgement_by_position_by_pair.values()
    ]
    has_probability = ~np.isnan(probabilities)
    robust = selections == 1
    matched_keys = {key for key, ratio in zip(keys, log_ratios) if abs(ratio) <= LENGTH_MATCHED_MAX_LOG_RATIO}
    return {
        "selection_spearman": spearman_or_nan(log_ratios, selections),
        "selection_spearman_ci_low": float(interval[0]),
        "selection_spearman_ci_high": float(interval[1]),
        "own_probability_spearman": spearman_or_nan(log_ratios[has_probability], np.array(probabilities)[has_probability]),
        "own_longer_rate": float(np.mean(log_ratios > 0)) if len(keys) else math.nan,
        "own_longer_rate_among_recognised": float(np.mean(log_ratios[robust] > 0)) if robust.any() else math.nan,
        "length_matched_item_rate": len(matched_keys) / len(keys) if keys else math.nan,
        "length_matched_accuracy": accuracy_estimate(
            [judgement for judgement in judgements if pair_key_tuple(judgement) in matched_keys]
        ),
    }


def skipped_item_count(
    judgements: Sequence[JudgementRecord],
    target_model: str,
    other_models: Sequence[str],
    transcript_by_key_by_model: dict[str, dict[TranscriptKey, TranscriptRecord]],
) -> int:
    """Items with both transcripts present but either one truncated or empty, over the judged prompts and samples."""
    judged_prompt_ids = {judgement["prompt_id"] for judgement in judgements}
    judged_sample_indices = {judgement["target_sample_index"] for judgement in judgements}
    target_by_key = transcript_by_key_by_model[target_model]
    return sum(
        1
        for other_model in other_models
        for key, target in target_by_key.items()
        if key[0] in judged_prompt_ids and key[1] in judged_sample_indices
        and (other := transcript_by_key_by_model[other_model].get(key)) is not None
        and not (is_judgeable(target) and is_judgeable(other))
    )


def pair_summary(
    target_model: str,
    other_model: str,
    other_models: Sequence[str],
    judgements: Sequence[JudgementRecord],
    transcript_by_key_by_model: dict[str, dict[TranscriptKey, TranscriptRecord]],
) -> PairSummary:
    judgement_by_position_by_pair = complete_items(judgements)
    items = list(judgement_by_position_by_pair.values())
    item_count = len(items)
    parsed_choices = [judgement["chosen_position"] for judgement in judgements if judgement["chosen_position"] is not None]
    p_choose_1 = parsed_choices.count(1) / len(parsed_choices) if parsed_choices else None
    robust_count = sum(by_position[1]["correct"] and by_position[2]["correct"] for by_position in items)
    robust_rate = robust_count / item_count if item_count else 0.0
    null_robust_rate = None if p_choose_1 is None else p_choose_1 * (1 - p_choose_1)
    name_leak_keys = {key for key, by_position in judgement_by_position_by_pair.items() if names_a_model(by_position[1])}
    return {
        "target_model": target_model,
        "other_model": other_model,
        "items": item_count,
        "skipped_truncated_or_empty": skipped_item_count(judgements, target_model, other_models, transcript_by_key_by_model),
        "accuracy": accuracy_estimate(judgements),
        "debiased_accuracy": debiased_accuracy(items),
        "position_bias_rate": (
            sum(
                by_position[1]["chosen_position"] is not None
                and by_position[1]["chosen_position"] == by_position[2]["chosen_position"]
                for by_position in items
            ) / item_count if item_count else 0.0
        ),
        "p_choose_1": p_choose_1,
        "unparseable_rate": sum(judgement["chosen_position"] is None for judgement in judgements) / len(judgements),
        "robust_count": robust_count,
        "robust_rate": robust_rate,
        "null_robust_rate": null_robust_rate,
        "purity": None if null_robust_rate is None or robust_rate == 0 else 1 - null_robust_rate / robust_rate,
        "longer_text_baseline": (
            statistics.mean(longer_text_correctness(key, transcript_by_key_by_model) for key in judgement_by_position_by_pair)
            if item_count else math.nan
        ),
        "length_selection": length_selection(judgements, transcript_by_key_by_model),
        "name_leak_rate": len(name_leak_keys) / item_count if item_count else 0.0,
        "accuracy_with_name_leak": accuracy_estimate(
            [judgement for judgement in judgements if pair_key_tuple(judgement) in name_leak_keys]
        ),
        "accuracy_without_name_leak": accuracy_estimate(
            [judgement for judgement in judgements if pair_key_tuple(judgement) not in name_leak_keys]
        ),
    }


def summarize_pairs(
    judgements: Sequence[JudgementRecord],
    transcripts_by_model: dict[str, list[TranscriptRecord]],
) -> list[PairSummary]:
    """One row per (target, other), then a pooled row per target with other_model == POOLED_OTHER_MODEL."""
    transcript_by_key_by_model = {model: transcripts_by_key(transcripts) for model, transcripts in transcripts_by_model.items()}
    judgements_by_other_by_target: defaultdict[str, defaultdict[str, list[JudgementRecord]]] = defaultdict(lambda: defaultdict(list))
    for judgement in judgements:
        judgements_by_other_by_target[judgement["target_model"]][judgement["other_model"]].append(judgement)
    return [
        summary
        for target_model, judgements_by_other in sorted(judgements_by_other_by_target.items())
        for summary in [
            *(
                pair_summary(target_model, other_model, [other_model], other_judgements, transcript_by_key_by_model)
                for other_model, other_judgements in sorted(judgements_by_other.items())
            ),
            pair_summary(
                target_model,
                POOLED_OTHER_MODEL,
                sorted(judgements_by_other),
                [judgement for other_judgements in judgements_by_other.values() for judgement in other_judgements],
                transcript_by_key_by_model,
            ),
        ]
    ]


def name_leaks_inflate_accuracy(pooled_summary: PairSummary) -> bool:
    """Whether leak items' accuracy beats non-leak accuracy by the margin with non-overlapping CIs."""
    with_leak = pooled_summary["accuracy_with_name_leak"]
    without_leak = pooled_summary["accuracy_without_name_leak"]
    if with_leak is None or without_leak is None:
        return False
    return (
        with_leak["accuracy"] - without_leak["accuracy"] > NAME_LEAK_ACCURACY_MARGIN
        and with_leak["ci_low"] > without_leak["ci_high"]
    )


def go_decision(
    summaries: Sequence[PairSummary],
    n_train_prompts: int,
    target_samples: int,
    min_robust: int,
) -> GoDecision:
    """GO/NO-GO for one target, from its rows of summarize_pairs."""
    pooled = next(summary for summary in summaries if summary["other_model"] == POOLED_OTHER_MODEL)
    assert all(summary["target_model"] == pooled["target_model"] for summary in summaries)
    other_count = sum(summary["other_model"] != POOLED_OTHER_MODEL for summary in summaries)
    robust_per_sample = pooled["robust_rate"] * other_count * n_train_prompts
    projected_robust_count = robust_per_sample * target_samples
    accuracy_ci_low = None if pooled["accuracy"] is None else pooled["accuracy"]["ci_low"]
    passed_by_criterion: dict[GoCriterion, bool] = {
        "projected_count": projected_robust_count >= min_robust,
        "accuracy_ci_low": accuracy_ci_low is not None and accuracy_ci_low > MIN_ACCURACY_CI_LOW,
        "purity": pooled["purity"] is not None and pooled["purity"] >= MIN_PURITY,
    }
    failed_criteria: list[GoCriterion] = [criterion for criterion, passed in passed_by_criterion.items() if not passed]
    return {
        "target_model": pooled["target_model"],
        "observed_robust_count": pooled["robust_count"],
        "projected_robust_count": projected_robust_count,
        "accuracy_ci_low": accuracy_ci_low,
        "purity": pooled["purity"],
        "verdict": "NO-GO" if failed_criteria else "GO",
        "failed_criteria": failed_criteria,
        "samples_needed_for_min_robust": math.ceil(min_robust / robust_per_sample) if robust_per_sample > 0 else None,
    }
