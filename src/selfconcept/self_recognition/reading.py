"""Reading the self-recognition vector: projections of the judge-prompt candidate spans (Reading A) and of plain
transcript responses (Reading B), with metrics whose 95% CIs resample whole prompts."""
import math
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import torch
from jaxtyping import Float
from numpy.typing import NDArray
from scipy.stats import bootstrap, pearsonr, spearmanr
from sklearn.metrics import roc_auc_score
from torch import Tensor

from selfconcept.common.chat import Conversation
from selfconcept.common.hf_strong_types import HFTokenizer, configure_apply_chat_template, configure_call
from selfconcept.common.span_targeting.types import PromptCompletion
from selfconcept.measurement.interface import TokenScorer
from selfconcept.measurement.unit_axes import load_unit_axes, unit_axis_scorer
from selfconcept.self_recognition.judging import is_judgeable, judge_candidate_text, transcripts_by_key
from selfconcept.self_recognition.schemas import (
    EstimateWithInterval,
    JudgementRecord,
    ReadingMetrics,
    ReadingSummary,
    SpanProjectionRecord,
    Split,
    TranscriptProjectionRecord,
    TranscriptReadingSummary,
    TranscriptRecord,
)
from selfconcept.self_recognition.success_filter import complete_items

RANDOM_DIRECTION_COUNT = 100
BOOTSTRAP_RESAMPLE_COUNT = 1000
BEST_LAYER_DEPTH_RANGE = (0.2, 0.8)
RANDOM_AUROC_PERCENTILE = 95

type RowIndices = NDArray[np.intp]
type RowStatistic = Callable[[RowIndices], float]
type PairedDifferenceKey = tuple[int, str, int]
"""(prompt_id, other model, sample_index)"""


class TranscriptReadingItem(NamedTuple):
    prompt_id: int
    split: Split
    sample_index: int
    response_model: str
    is_own: bool
    conversation: Conversation


class PairedDifferences(NamedTuple):
    keys: list[PairedDifferenceKey]
    projection_differences: NDArray[np.float64]
    token_count_differences: NDArray[np.float64]


def axis_scorer(vectors_file: Path, model: str, num_layers: int, hidden_size: int) -> TokenScorer:
    return unit_axis_scorer(load_unit_axes(vectors_file, list(range(num_layers)), num_layers, hidden_size, model))


def unit_direction(vectors_file: Path, model: str, layer: int, num_layers: int, hidden_size: int) -> Float[Tensor, "hidden"]:
    return torch.from_numpy(load_unit_axes(vectors_file, [layer], num_layers, hidden_size, model)[layer]["direction"])


def project_by_layer(pooled: Float[Tensor, "example layer hidden"], scorer: TokenScorer) -> Float[Tensor, "example layer"]:
    return torch.stack([scorer(layer=layer, residual=pooled[:, layer]) for layer in range(pooled.shape[1])], dim=1)


def random_unit_directions(count: int, hidden_size: int) -> Float[Tensor, "seed hidden"]:
    """Direction i is Gaussian from a generator seeded with i, so a direction does not depend on count."""
    directions = torch.stack(
        [torch.randn(hidden_size, generator=torch.Generator().manual_seed(seed)) for seed in range(count)]
    )
    return directions / directions.norm(dim=-1, keepdim=True)


def token_norms(span_activations: Sequence[Float[Tensor, "layer token hidden"]]) -> list[Float[Tensor, "layer token"]]:
    return [activations.float().norm(dim=-1).cpu() for activations in span_activations]


def mean_token_norm_by_layer(token_norms_by_example: Sequence[Float[Tensor, "layer token"]]) -> list[float]:
    return torch.cat(list(token_norms_by_example), dim=1).mean(dim=1).tolist()


def clustered_bootstrap(statistic: RowStatistic, cluster_ids: Sequence[int], seed: int = 0) -> EstimateWithInterval:
    """The statistic on all rows, with a percentile 95% CI from resampling whole clusters with replacement."""
    row_indices_by_cluster: defaultdict[int, list[int]] = defaultdict(list)
    for row_index, cluster_id in enumerate(cluster_ids):
        row_indices_by_cluster[cluster_id].append(row_index)
    cluster_rows = [np.array(rows, dtype=np.intp) for rows in row_indices_by_cluster.values()]
    estimate = statistic(np.arange(len(cluster_ids), dtype=np.intp))
    if len(cluster_rows) < 2:
        return estimate, math.nan, math.nan

    def resampled_statistic(cluster_sample: NDArray[np.intp]) -> float:
        return statistic(np.concatenate([cluster_rows[cluster_index] for cluster_index in cluster_sample]))

    interval = bootstrap(
        (np.arange(len(cluster_rows)),),
        resampled_statistic,
        vectorized=False,
        n_resamples=BOOTSTRAP_RESAMPLE_COUNT,
        method="percentile",
        rng=np.random.default_rng(seed),
    ).confidence_interval
    return estimate, float(interval.low), float(interval.high)


def auroc(positive_scores: NDArray[np.float64], negative_scores: NDArray[np.float64]) -> float:
    if len(positive_scores) == 0 or len(negative_scores) == 0:
        return math.nan
    labels = np.concatenate([np.ones(len(positive_scores)), np.zeros(len(negative_scores))])
    return float(roc_auc_score(labels, np.concatenate([positive_scores, negative_scores])))


def labelled_auroc(scores: NDArray[np.float64], is_positive: NDArray[np.bool_]) -> float:
    return auroc(scores[is_positive], scores[~is_positive])


def correlation_or_nan(
    correlation: Callable[[NDArray[np.float64], NDArray[np.float64]], float],
    first: NDArray[np.float64],
    second: NDArray[np.float64],
) -> float:
    if len(first) < 3 or np.ptp(first) == 0 or np.ptp(second) == 0:
        return math.nan
    return correlation(first, second)


def pearson(first: NDArray[np.float64], second: NDArray[np.float64]) -> float:
    return float(pearsonr(first, second).statistic)  # pyright: ignore[reportAttributeAccessIssue]


def spearman(first: NDArray[np.float64], second: NDArray[np.float64]) -> float:
    return float(spearmanr(first, second).statistic)  # pyright: ignore[reportAttributeAccessIssue]


def select_best_layer(auroc_by_layer: Sequence[float]) -> int:
    """Argmax within the middle of the stack (20-80% of depth), where steering directions usually live."""
    layer_count = len(auroc_by_layer)
    low_fraction, high_fraction = BEST_LAYER_DEPTH_RANGE
    candidate_layers = [layer for layer in range(layer_count) if low_fraction <= layer / layer_count <= high_fraction]
    return max(candidate_layers, key=lambda layer: -math.inf if math.isnan(auroc_by_layer[layer]) else auroc_by_layer[layer])


def random_auroc_percentile(
    positive_by_seed: NDArray[np.float64],
    negative_by_seed: NDArray[np.float64],
) -> float:
    """The 95th percentile over random directions of the positive-vs-negative AUROC; inputs are example x seed."""
    if positive_by_seed.size == 0 or negative_by_seed.size == 0:
        return math.nan
    return float(
        np.percentile(
            [auroc(positive_by_seed[:, seed], negative_by_seed[:, seed]) for seed in range(positive_by_seed.shape[1])],
            RANDOM_AUROC_PERCENTILE,
        )
    )


def span_projection_record(
    judgement: JudgementRecord,
    split: Split,
    own_token_count: int,
    other_token_count: int,
    own_projection_by_layer: Sequence[float],
    other_projection_by_layer: Sequence[float],
    random_own_projection_by_seed: Sequence[float] | None,
    random_other_projection_by_seed: Sequence[float] | None,
) -> SpanProjectionRecord:
    return {
        "prompt_id": judgement["prompt_id"],
        "split": split,
        "other_model": judgement["other_model"],
        "target_sample_index": judgement["target_sample_index"],
        "other_sample_index": judgement["other_sample_index"],
        "target_position": judgement["target_position"],
        "chosen_position": judgement["chosen_position"],
        "correct": judgement["correct"],
        "answer_probabilities": judgement["answer_probabilities"],
        "own_token_count": own_token_count,
        "other_token_count": other_token_count,
        "own_projection_by_layer": list(own_projection_by_layer),
        "other_projection_by_layer": list(other_projection_by_layer),
        "random_own_projection_by_seed": None if random_own_projection_by_seed is None else list(random_own_projection_by_seed),
        "random_other_projection_by_seed": (
            None if random_other_projection_by_seed is None else list(random_other_projection_by_seed)
        ),
    }


def span_projections_at_layer(
    records: Sequence[SpanProjectionRecord],
    layer: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """(own, other) projections."""
    return (
        np.array([record["own_projection_by_layer"][layer] for record in records]),
        np.array([record["other_projection_by_layer"][layer] for record in records]),
    )


def separation_auroc_by_layer(records: Sequence[SpanProjectionRecord]) -> list[float]:
    layer_count = len(records[0]["own_projection_by_layer"])
    return [auroc(*span_projections_at_layer(records, layer)) for layer in range(layer_count)]


def text_1_minus_text_2_logit(record: SpanProjectionRecord) -> float:
    """Stage 2's log(p1 / p2); nan when either label has no probability mass."""
    probabilities = record["answer_probabilities"]
    if probabilities is None or probabilities["position_1"] <= 0 or probabilities["position_2"] <= 0:
        return math.nan
    return math.log(probabilities["position_1"]) - math.log(probabilities["position_2"])


def reading_metrics(records: Sequence[SpanProjectionRecord], layer: int) -> ReadingMetrics:
    own, other = span_projections_at_layer(records, layer)
    target_is_text_1 = np.array([record["target_position"] == 1 for record in records])
    text_1 = np.where(target_is_text_1, own, other)
    text_2 = np.where(target_is_text_1, other, own)
    text_1_minus_text_2 = text_1 - text_2
    chosen_position = np.array([record["chosen_position"] or 0 for record in records])
    correct = np.array([record["correct"] for record in records])
    answer_logit = np.array([text_1_minus_text_2_logit(record) for record in records])
    own_token_count = np.array([record["own_token_count"] for record in records], dtype=float)
    other_token_count = np.array([record["other_token_count"] for record in records], dtype=float)
    prompt_ids = [record["prompt_id"] for record in records]

    def judgement_auroc(rows: RowIndices) -> float:
        parsed_rows = rows[chosen_position[rows] != 0]
        return labelled_auroc(text_1_minus_text_2[parsed_rows], chosen_position[parsed_rows] == 1)

    def judgement_logit_pearson(rows: RowIndices) -> float:
        scored_rows = rows[~np.isnan(answer_logit[rows])]
        return correlation_or_nan(pearson, text_1_minus_text_2[scored_rows], answer_logit[scored_rows])

    return {
        "holdout_auroc": clustered_bootstrap(lambda rows: auroc(own[rows], other[rows]), prompt_ids),
        "probe_paired_accuracy": clustered_bootstrap(lambda rows: float(np.mean(own[rows] > other[rows])), prompt_ids),
        "model_paired_accuracy": float(np.mean(correct)),
        "judgement_auroc": clustered_bootstrap(judgement_auroc, prompt_ids),
        "judgement_logit_pearson": clustered_bootstrap(judgement_logit_pearson, prompt_ids),
        "correctness_auroc": clustered_bootstrap(lambda rows: labelled_auroc(own[rows] - other[rows], correct[rows]), prompt_ids),
        "position_auroc": auroc(text_1, text_2),
        "length_spearman": correlation_or_nan(
            spearman, np.concatenate([own, other]), np.concatenate([own_token_count, other_token_count])
        ),
    }


def reading_summary(
    train_records: Sequence[SpanProjectionRecord],
    holdout_records: Sequence[SpanProjectionRecord],
    layer_norm_by_layer: Sequence[float],
) -> ReadingSummary:
    train_auroc_by_layer = separation_auroc_by_layer(train_records)
    best_layer = select_best_layer(train_auroc_by_layer)
    random_records = [record for record in holdout_records if record["random_own_projection_by_seed"] is not None]
    other_models = sorted({record["other_model"] for record in holdout_records})
    return {
        **reading_metrics(holdout_records, best_layer),
        "best_layer": best_layer,
        "train_auroc_by_layer": train_auroc_by_layer,
        "holdout_auroc_by_layer": separation_auroc_by_layer(holdout_records),
        "random_auroc_95th_percentile": random_auroc_percentile(
            np.array([record["random_own_projection_by_seed"] for record in random_records]),
            np.array([record["random_other_projection_by_seed"] for record in random_records]),
        ),
        "layer_norm_by_layer": list(layer_norm_by_layer),
        "metrics_by_other_model": {
            other_model: reading_metrics(
                [record for record in holdout_records if record["other_model"] == other_model], best_layer
            )
            for other_model in other_models
        },
    }


def teacher_forced_conversation(question_transcript: TranscriptRecord, response_transcript: TranscriptRecord) -> Conversation:
    """The question, answered by the response transcript's final answer (without any gpt-oss thinking)."""
    return [
        question_transcript["conversation"][0],
        {"role": "assistant", "content": judge_candidate_text(response_transcript)},
    ]


def generation_prompt_completion(conversation: Conversation, tokenizer: HFTokenizer, **chat_template_kwargs: Any) -> PromptCompletion:
    """Like tokenize_as_prompt_completion, but the prompt is the exact generation prompt, including anything
    the template adds there that a full-conversation render drops (gemma-4's empty thought block)."""
    apply_chat_template = configure_apply_chat_template(tokenizer).tokenize(False)
    generation_prompt = apply_chat_template(conversation[:-1], add_generation_prompt=True, **chat_template_kwargs)
    full_text = apply_chat_template(conversation, add_generation_prompt=False, **chat_template_kwargs)
    shared_length = next(
        length for length in range(len(generation_prompt), -1, -1) if full_text.startswith(generation_prompt[:length])
    )
    tokenize = configure_call(tokenizer)
    return PromptCompletion(
        tokenize(generation_prompt, add_special_tokens=False)["input_ids"],
        tokenize(full_text[shared_length:], add_special_tokens=False)["input_ids"],
    )


def transcript_reading_items(
    target_model: str,
    transcripts_by_model: Mapping[str, Sequence[TranscriptRecord]],
    split_by_prompt_id: Mapping[int, Split],
) -> list[TranscriptReadingItem]:
    """For each judgeable target transcript: its own response, then every other model's judgeable response to
    the same prompt with the same sample index, all in the target's assistant turn."""
    transcript_by_key_by_model = {model: transcripts_by_key(transcripts) for model, transcripts in transcripts_by_model.items()}
    response_models = [target_model, *sorted(model for model in transcripts_by_model if model != target_model)]
    return [
        TranscriptReadingItem(
            prompt_id=prompt_id,
            split=split_by_prompt_id[prompt_id],
            sample_index=sample_index,
            response_model=response_model,
            is_own=response_model == target_model,
            conversation=teacher_forced_conversation(target, response),
        )
        for (prompt_id, sample_index), target in sorted(transcript_by_key_by_model[target_model].items())
        if prompt_id in split_by_prompt_id and is_judgeable(target)
        for response_model in response_models
        if (response := transcript_by_key_by_model[response_model].get((prompt_id, sample_index))) is not None
        and is_judgeable(response)
    ]


def transcript_projection_record(
    item: TranscriptReadingItem,
    response_token_count: int,
    projection_by_layer: Sequence[float],
    random_projection_by_seed: Sequence[float] | None,
) -> TranscriptProjectionRecord:
    return {
        "prompt_id": item.prompt_id,
        "split": item.split,
        "sample_index": item.sample_index,
        "response_model": item.response_model,
        "is_own": item.is_own,
        "response_token_count": response_token_count,
        "projection_by_layer": list(projection_by_layer),
        "random_projection_by_seed": None if random_projection_by_seed is None else list(random_projection_by_seed),
    }


def transcript_projections_at_layer(records: Sequence[TranscriptProjectionRecord], layer: int) -> NDArray[np.float64]:
    return np.array([record["projection_by_layer"][layer] for record in records])


def own_vs_other_auroc(records: Sequence[TranscriptProjectionRecord], layer: int) -> EstimateWithInterval:
    projections = transcript_projections_at_layer(records, layer)
    is_own = np.array([record["is_own"] for record in records])
    return clustered_bootstrap(
        lambda rows: labelled_auroc(projections[rows], is_own[rows]), [record["prompt_id"] for record in records]
    )


def own_vs_other_auroc_by_layer(records: Sequence[TranscriptProjectionRecord]) -> list[float]:
    is_own = np.array([record["is_own"] for record in records])
    layer_count = len(records[0]["projection_by_layer"])
    return [labelled_auroc(transcript_projections_at_layer(records, layer), is_own) for layer in range(layer_count)]


def own_and_one_other(records: Sequence[TranscriptProjectionRecord], other_model: str) -> list[TranscriptProjectionRecord]:
    return [record for record in records if record["is_own"] or record["response_model"] == other_model]


def paired_differences(records: Sequence[TranscriptProjectionRecord], layer: int) -> PairedDifferences:
    """Own minus other, on the same prompt and sample index."""
    own_by_key = {(record["prompt_id"], record["sample_index"]): record for record in records if record["is_own"]}
    pairs = [
        (own, other)
        for other in records
        if not other["is_own"] and (own := own_by_key.get((other["prompt_id"], other["sample_index"]))) is not None
    ]
    return PairedDifferences(
        keys=[(other["prompt_id"], other["response_model"], other["sample_index"]) for _, other in pairs],
        projection_differences=np.array(
            [own["projection_by_layer"][layer] - other["projection_by_layer"][layer] for own, other in pairs]
        ),
        token_count_differences=np.array(
            [own["response_token_count"] - other["response_token_count"] for own, other in pairs], dtype=float
        ),
    )


def clustered_mean(values: NDArray[np.float64], keys: Sequence[PairedDifferenceKey]) -> EstimateWithInterval:
    return clustered_bootstrap(lambda rows: float(np.mean(values[rows])), [prompt_id for prompt_id, _, _ in keys])


def length_adjusted_own_coefficient(records: Sequence[TranscriptProjectionRecord], layer: int) -> EstimateWithInterval:
    """The is_own coefficient of projection ~ is_own + log(response tokens) + other-model effects. The effects are
    sum-to-zero coded, so the coefficient compares own text with the unweighted mean over other models."""
    other_models = sorted({record["response_model"] for record in records if not record["is_own"]})

    def other_model_effects(record: TranscriptProjectionRecord) -> list[float]:
        if record["is_own"]:
            return [0.0] * (len(other_models) - 1)
        if record["response_model"] == other_models[-1]:
            return [-1.0] * (len(other_models) - 1)
        return [float(record["response_model"] == model) for model in other_models[:-1]]

    design = np.array(
        [
            [1.0, float(record["is_own"]), math.log(record["response_token_count"]), *other_model_effects(record)]
            for record in records
        ]
    )
    projections = transcript_projections_at_layer(records, layer)
    return clustered_bootstrap(
        lambda rows: float(np.linalg.lstsq(design[rows], projections[rows])[0][1]),
        [record["prompt_id"] for record in records],
    )


def recognised_minus_unrecognised_difference(
    differences: PairedDifferences,
    judgements: Sequence[JudgementRecord],
) -> EstimateWithInterval | None:
    """Mean own-minus-other difference on pairs the target recognised in both stage-2 orders, minus that on pairs
    judged in both orders but not recognised. None when either group is empty."""
    is_recognised_by_key = {
        (prompt_id, other_model, target_sample_index): judgement_by_position[1]["correct"]
        and judgement_by_position[2]["correct"]
        for (prompt_id, _, other_model, target_sample_index, other_sample_index), judgement_by_position in complete_items(
            judgements
        ).items()
        if target_sample_index == other_sample_index
    }
    judged_rows = [row for row, key in enumerate(differences.keys) if key in is_recognised_by_key]
    judged_keys = [differences.keys[row] for row in judged_rows]
    judged_differences = differences.projection_differences[judged_rows]
    is_recognised = np.array([is_recognised_by_key[key] for key in judged_keys], dtype=bool)
    if not is_recognised.any() or is_recognised.all():
        return None

    def difference_of_means(rows: RowIndices) -> float:
        recognised = judged_differences[rows][is_recognised[rows]]
        unrecognised = judged_differences[rows][~is_recognised[rows]]
        if len(recognised) == 0 or len(unrecognised) == 0:
            return math.nan
        return float(np.mean(recognised) - np.mean(unrecognised))

    return clustered_bootstrap(difference_of_means, [prompt_id for prompt_id, _, _ in judged_keys])


def paired_difference_with(differences: PairedDifferences, other_model: str) -> EstimateWithInterval:
    rows = [row for row, (_, model, _) in enumerate(differences.keys) if model == other_model]
    return clustered_mean(differences.projection_differences[rows], [differences.keys[row] for row in rows])


def transcript_reading_summary(
    train_records: Sequence[TranscriptProjectionRecord],
    holdout_records: Sequence[TranscriptProjectionRecord],
    judge_best_layer: int,
    judgements: Sequence[JudgementRecord],
) -> TranscriptReadingSummary:
    train_auroc_by_layer = own_vs_other_auroc_by_layer(train_records)
    best_layer = select_best_layer(train_auroc_by_layer)
    other_models = sorted({record["response_model"] for record in holdout_records if not record["is_own"]})
    differences = paired_differences(holdout_records, best_layer)
    random_records = [record for record in holdout_records if record["random_projection_by_seed"] is not None]
    return {
        "best_layer": best_layer,
        "train_auroc_by_layer": train_auroc_by_layer,
        "holdout_auroc_by_layer": own_vs_other_auroc_by_layer(holdout_records),
        "holdout_auroc": own_vs_other_auroc(holdout_records, best_layer),
        "holdout_auroc_at_judge_best_layer": own_vs_other_auroc(holdout_records, judge_best_layer),
        "holdout_auroc_by_other_model": {
            other_model: own_vs_other_auroc(own_and_one_other(holdout_records, other_model), best_layer)
            for other_model in other_models
        },
        "paired_difference": clustered_mean(differences.projection_differences, differences.keys),
        "paired_difference_by_other_model": {
            other_model: paired_difference_with(differences, other_model) for other_model in other_models
        },
        "length_adjusted_own_coefficient": length_adjusted_own_coefficient(holdout_records, best_layer),
        "own_minus_other_token_count": clustered_mean(differences.token_count_differences, differences.keys),
        "recognised_minus_unrecognised_difference": recognised_minus_unrecognised_difference(differences, judgements),
        "random_auroc_95th_percentile": random_auroc_percentile(
            np.array([record["random_projection_by_seed"] for record in random_records if record["is_own"]]),
            np.array([record["random_projection_by_seed"] for record in random_records if not record["is_own"]]),
        ),
    }
