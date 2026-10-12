"""Records passed between the self-recognition stages: prompts -> transcripts -> judgements -> vectors."""
from pathlib import Path
from typing import Literal, TypedDict, cast

from jaxtyping import Float, Int
from torch import Tensor

from selfconcept.common.chat import Conversation
from selfconcept.common.jsonl import read_jsonl
from selfconcept.common.paths import scratch_dir
from selfconcept.measurement.unit_axes import ModelAndAxis

DEFAULT_RUN_DIR = scratch_dir("self-recognition")

type Split = Literal["train", "holdout"]
type CandidatePosition = Literal[1, 2]
type PairKeyTuple = tuple[int, str, str, int, int]
"""(prompt_id, target_model, other_model, target_sample_index, other_sample_index)"""


class PromptRecord(TypedDict):
    id: int
    question: str
    split: Split


class TranscriptRecord(TypedDict):
    prompt_id: int
    model: str
    sample_index: int
    conversation: Conversation
    truncated: bool


class AnswerProbabilities(TypedDict):
    """Summed probability of the tokens decoding to each label, at the answer position."""

    position_1: float
    position_2: float


class JudgementRecord(TypedDict):
    prompt_id: int
    target_model: str
    other_model: str
    target_sample_index: int
    other_sample_index: int
    target_position: CandidatePosition
    judge_conversation: Conversation
    raw_response: str
    chosen_position: CandidatePosition | None
    correct: bool
    answer_probabilities: AnswerProbabilities | None  # None when the prompt exceeded max_model_len


class SelfRecognitionVectors(ModelAndAxis):
    """Saved with torch.save"""

    model: str
    axis: Float[Tensor, "layer hidden"]
    train_prompt_ids: list[int]
    positive_count: int
    negative_count: int


class PooledSpanMeans(TypedDict):
    """One example per judge prompt (both orders of each pair), longest prompt first. pair_indices index the
    cache's pair keys; character counts are of the full candidate text, token counts of the located span."""

    own: Float[Tensor, "example layer hidden"]
    other: Float[Tensor, "example layer hidden"]
    pair_indices: Int[Tensor, " example"]
    prompt_ids: list[int]
    other_models: list[str]
    target_sample_indices: list[int]
    other_sample_indices: list[int]
    target_positions: list[CandidatePosition]
    own_token_counts: list[int]
    other_token_counts: list[int]
    own_char_counts: list[int]
    other_char_counts: list[int]


class ActivationMeansInputs(TypedDict):
    model: str
    limit_pairs: int | None
    seed: int
    max_length: int
    recognised_pair_keys: list[PairKeyTuple]
    non_recognised_pair_keys: list[PairKeyTuple]


class ActivationMeans(ActivationMeansInputs):
    """Stage 3's pooled span means, cached so the vector and its diagnostics can be rebuilt without the model.
    non_recognised is None when no non-recognised pair fits in max_length."""

    recognised: PooledSpanMeans
    non_recognised: PooledSpanMeans | None


type SteeringCondition = Literal["own_span", "other_span", "all_positions"]
type SteeringDirection = Literal["self_recognition", "random"]
type EstimateWithInterval = tuple[float, float, float]
"""(estimate, ci_low, ci_high)"""


class SpanProjectionRecord(TypedDict):
    """Reading A: one per judge record. The random projections, onto the seeded random unit directions at the
    train-chosen best layer, are recorded on holdout records only."""

    prompt_id: int
    split: Split
    other_model: str
    target_sample_index: int
    other_sample_index: int
    target_position: CandidatePosition
    chosen_position: CandidatePosition | None
    correct: bool
    answer_probabilities: AnswerProbabilities | None
    own_token_count: int
    other_token_count: int
    own_projection_by_layer: list[float]
    other_projection_by_layer: list[float]
    random_own_projection_by_seed: list[float] | None
    random_other_projection_by_seed: list[float] | None


class SteeringRecord(TypedDict):
    """One per (judge record, condition, direction, alpha)."""

    prompt_id: int
    other_model: str
    target_sample_index: int
    other_sample_index: int
    target_position: CandidatePosition
    condition: SteeringCondition
    direction: SteeringDirection
    random_seed: int | None
    layer: int
    alpha: float
    coefficient: float
    position_1_probability: float
    position_2_probability: float
    answer_mass: float


class ContinuationRecord(TypedDict):
    """Greedy continuation of a steered judge prompt, for the coherence check."""

    prompt_id: int
    other_model: str
    target_sample_index: int
    other_sample_index: int
    target_position: CandidatePosition
    condition: SteeringCondition
    layer: int
    alpha: float
    continuation: str
    generated_token_count: int
    stopped: bool
    distinct_bigram_ratio: float
    max_token_share: float
    degenerate: bool


class IndividualRecord(TypedDict):
    """Qualitative "Did you write this?" run on one holdout text."""

    prompt_id: int
    sample_index: int
    response_model: str
    is_own: bool
    layer: int
    alpha: float
    coefficient: float
    yes_probability: float
    no_probability: float
    sampled_generations: list[str]


class ReadingMetrics(TypedDict):
    holdout_auroc: EstimateWithInterval
    probe_paired_accuracy: EstimateWithInterval
    model_paired_accuracy: float
    judgement_auroc: EstimateWithInterval
    judgement_logit_pearson: EstimateWithInterval
    correctness_auroc: EstimateWithInterval
    position_auroc: float
    length_spearman: float


class ReadingSummary(ReadingMetrics):
    best_layer: int
    train_auroc_by_layer: list[float]
    holdout_auroc_by_layer: list[float]
    random_auroc_95th_percentile: float
    layer_norm_by_layer: list[float]
    metrics_by_other_model: dict[str, ReadingMetrics]


class TranscriptProjectionRecord(TypedDict):
    """Reading B: one per (transcript, response source). The random projections are as in SpanProjectionRecord,
    at Reading B's train-chosen best layer."""

    prompt_id: int
    split: Split
    sample_index: int
    response_model: str
    is_own: bool
    response_token_count: int
    projection_by_layer: list[float]
    random_projection_by_seed: list[float] | None


class TranscriptReadingSummary(TypedDict):
    best_layer: int
    train_auroc_by_layer: list[float]
    holdout_auroc_by_layer: list[float]
    holdout_auroc: EstimateWithInterval
    holdout_auroc_at_judge_best_layer: EstimateWithInterval
    holdout_auroc_by_other_model: dict[str, EstimateWithInterval]
    paired_difference: EstimateWithInterval
    paired_difference_by_other_model: dict[str, EstimateWithInterval]
    length_adjusted_own_coefficient: EstimateWithInterval
    own_minus_other_token_count: EstimateWithInterval
    recognised_minus_unrecognised_difference: EstimateWithInterval | None
    random_auroc_95th_percentile: float


class SteeringCell(TypedDict):
    condition: SteeringCondition
    direction: SteeringDirection
    random_seed: int | None
    alpha: float
    own_choice_probability_shift: EstimateWithInterval
    position_1_probability_shift: EstimateWithInterval
    self_attribution_rate: EstimateWithInterval
    answer_mass_ratio: float
    degenerate_fraction: float | None
    coherent: bool
    wilcoxon_p_value: float


def model_slug(model: str) -> str:
    return model.replace("/", "__")


def prompts_path(data_dir: Path) -> Path:
    return data_dir / "prompts.jsonl"


def transcripts_path(run_dir: Path, model: str) -> Path:
    return run_dir / "transcripts" / f"{model_slug(model)}.jsonl"


def judgements_path(run_dir: Path, target_model: str) -> Path:
    return run_dir / "judgements" / f"{model_slug(target_model)}.jsonl"


def vectors_path(run_dir: Path, model: str) -> Path:
    return run_dir / "vectors" / f"{model_slug(model)}.pt"


def length_matched_vectors_path(run_dir: Path, model: str) -> Path:
    return run_dir / "vectors" / "length_matched" / f"{model_slug(model)}.pt"


def activation_means_path(run_dir: Path, model: str) -> Path:
    return run_dir / "vectors" / "activation_means" / f"{model_slug(model)}.pt"


def reading_path(run_dir: Path, model: str) -> Path:
    return run_dir / "validation" / f"{model_slug(model)}_reading.jsonl"


def transcript_reading_path(run_dir: Path, model: str) -> Path:
    return run_dir / "validation" / f"{model_slug(model)}_transcript_reading.jsonl"


def steering_path(run_dir: Path, model: str, shard_tag: str) -> Path:
    return run_dir / "validation" / f"{model_slug(model)}_steering_{shard_tag}.jsonl"


def steering_paths(run_dir: Path, model: str) -> list[Path]:
    """Every shard's steering file, and the untagged file of an unsharded run."""
    return sorted((run_dir / "validation").glob(f"{model_slug(model)}_steering*.jsonl"))


def continuation_path(run_dir: Path, model: str, shard_tag: str) -> Path:
    return run_dir / "validation" / f"{model_slug(model)}_continuations_{shard_tag}.jsonl"


def continuation_paths(run_dir: Path, model: str) -> list[Path]:
    """Every shard's continuation file, and the untagged file of an unsharded run."""
    return sorted((run_dir / "validation").glob(f"{model_slug(model)}_continuations*.jsonl"))


def individual_path(run_dir: Path, model: str) -> Path:
    return run_dir / "validation" / f"{model_slug(model)}_individual.jsonl"


def layer_norms_path(run_dir: Path, model: str) -> Path:
    return run_dir / "validation" / f"{model_slug(model)}_layer_norms.json"


def read_prompts(path: Path) -> list[PromptRecord]:
    return cast(list[PromptRecord], read_jsonl(path))


def read_transcripts(path: Path) -> list[TranscriptRecord]:
    return cast(list[TranscriptRecord], read_jsonl(path))


def read_judgements(path: Path) -> list[JudgementRecord]:
    return cast(list[JudgementRecord], read_jsonl(path))


def read_span_projections(path: Path) -> list[SpanProjectionRecord]:
    return cast(list[SpanProjectionRecord], read_jsonl(path))


def read_transcript_projections(path: Path) -> list[TranscriptProjectionRecord]:
    return cast(list[TranscriptProjectionRecord], read_jsonl(path))


def read_steering_records(path: Path) -> list[SteeringRecord]:
    return cast(list[SteeringRecord], read_jsonl(path))


def read_continuations(path: Path) -> list[ContinuationRecord]:
    return cast(list[ContinuationRecord], read_jsonl(path))
