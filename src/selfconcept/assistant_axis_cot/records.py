from typing import Literal, TypedDict

import numpy as np
from jaxtyping import Float32, Int8, Int32

type AxisName = Literal["all_tokens", "response_only"]
type Condition = Literal["unprompted", "persona"]
type Region = Literal["cot", "final", "delimiter"]

REGIONS: tuple[Region, ...] = ("cot", "final", "delimiter")


class CompletionProjections(TypedDict):
    """Per completion token, up to the extraction max_length. region_codes index into REGIONS."""

    completion_token_ids: Int32[np.ndarray, " n_tokens"]
    region_codes: Int8[np.ndarray, " n_tokens"]
    projections_by_axis: dict[AxisName, Float32[np.ndarray, " n_tokens"]]
    residual_norms: Float32[np.ndarray, " n_tokens"]
    truncated: bool


class TokenProjectionRecord(CompletionProjections):
    role: str
    prompt_index: int
    question_index: int
    condition: Condition


def condition_of(role: str, prompt_index: int) -> Condition | None:
    """None for the axis-training default role's system-prompted variants, which are neither unprompted nor personas."""
    if role == "unprompted" or (role == "default" and prompt_index == 0):
        return "unprompted"
    if role == "default":
        return None
    return "persona"
