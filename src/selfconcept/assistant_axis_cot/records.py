from typing import Literal, TypedDict

import numpy as np
from jaxtyping import Float32, Float64, Int8, Int32, Int64

type AxisName = Literal["all_tokens", "response_only"]
type Condition = Literal["unprompted", "persona"]
type CompletionRegion = Literal["cot", "final", "delimiter"]
type SequenceRegion = Literal["prompt", "cot", "final", "unlabelled"]

COMPLETION_REGIONS: tuple[CompletionRegion, ...] = ("cot", "final", "delimiter")
SEQUENCE_REGIONS: tuple[SequenceRegion, ...] = ("prompt", "cot", "final", "unlabelled")
LABELLED_SEQUENCE_REGIONS: tuple[SequenceRegion, ...] = tuple(
    region for region in SEQUENCE_REGIONS if region != "unlabelled"
)


class SpanProjections[DirectionName: str](TypedDict):
    """Per token of one extraction span, up to the extraction max_length."""

    projections_by_axis: dict[DirectionName, Float32[np.ndarray, " n_tokens"]]
    residual_norms: Float32[np.ndarray, " n_tokens"]


class CompletionProjections(SpanProjections[AxisName]):
    """Per completion token. region_codes index into COMPLETION_REGIONS."""

    completion_token_ids: Int32[np.ndarray, " n_tokens"]
    region_codes: Int8[np.ndarray, " n_tokens"]
    truncated: bool


class SequenceProjections[DirectionName: str](SpanProjections[DirectionName]):
    """Per token of the concatenated prompt and completion. region_codes index into SEQUENCE_REGIONS."""

    token_ids: Int32[np.ndarray, " n_tokens"]
    region_codes: Int8[np.ndarray, " n_tokens"]


class AblationResult(TypedDict):
    """One conversation's region mean ablations. Axes: direction, in the run's direction order; ablated region,
    indexing LABELLED_SEQUENCE_REGIONS; measured region, indexing SEQUENCE_REGIONS; layer, the target layer onward."""

    kl_sum: Float64[np.ndarray, "direction ablated measured"]
    kl_token_count: Int64[np.ndarray, " measured"]
    projection_shift_sum: Float64[np.ndarray, "direction ablated measured layer"]
    squared_projection_shift_sum: Float64[np.ndarray, "direction ablated measured layer"]
    projection_token_count: Int64[np.ndarray, " measured"]
    truncated: bool


class ConversationMetadata(TypedDict):
    role: str
    prompt_index: int
    question_index: int
    condition: Condition


class TokenProjectionRecord(ConversationMetadata, CompletionProjections):
    pass


class SequenceProjectionRecord(ConversationMetadata, SequenceProjections[str]):
    pass


class AblationRecord(ConversationMetadata, AblationResult):
    pass


def condition_of(role: str, prompt_index: int) -> Condition | None:
    """None for the axis-training default role's system-prompted variants, which are neither unprompted nor personas."""
    if role == "unprompted" or (role == "default" and prompt_index == 0):
        return "unprompted"
    if role == "default":
        return None
    return "persona"
