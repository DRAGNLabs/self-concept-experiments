from typing import Literal, NewType, Protocol, TypedDict

import numpy as np
from jaxtyping import Float

ArmId = NewType("ArmId", str)
GroupId = NewType("GroupId", str)
FactorName = NewType("FactorName", str)
CategoryName = NewType("CategoryName", str)
SettingName = NewType("SettingName", str)
TierName = NewType("TierName", str)


class Arm(TypedDict):
    arm_id: ArmId
    group_id: GroupId
    gate_value_by_setting: dict[SettingName, str]


class SuccessDefinition(TypedDict):
    factor_name: FactorName
    success_category_names: list[CategoryName]


class Tier(TypedDict):
    """An arm is eligible if each listed gate setting takes one of the listed values; settings not listed are
    unconstrained."""

    tier_name: TierName
    gate_values_by_setting: dict[SettingName, list[str]]
    success: SuccessDefinition


class BatchUnit(TypedDict):
    arm: Arm
    seeds: list[int]


class EpisodeResult(TypedDict):
    arm: Arm
    seed: int
    category_by_factor: dict[FactorName, CategoryName]
    cost: float


class CandidatePosterior(TypedDict):
    """Posterior draws for the arms an application proposes at one tier; `hit_probability` is per seed."""

    arms: list[Arm]
    hit_probability: Float[np.ndarray, "draw arm"]
    expected_cost_per_seed: Float[np.ndarray, "draw arm"]


class EffectEstimate(TypedDict):
    effect_name: str
    level_name: str
    mean: float
    lower: float
    upper: float


class FitSucceeded(TypedDict):
    """`warnings` flag a usable but suspect fit (e.g. high R-hat); the search keeps planning from it."""

    kind: Literal["fit_succeeded"]
    candidate_posterior_by_tier: dict[TierName, CandidatePosterior]
    diagnostic_value_by_name: dict[str, float]
    warnings: list[str]
    effect_estimates: list[EffectEstimate]


class FitFailed(TypedDict):
    kind: Literal["fit_failed"]
    reason: str
    diagnostic_value_by_name: dict[str, float]


FitOutcome = FitSucceeded | FitFailed


class SearchApplication(Protocol):
    def fit(self, episodes: list[EpisodeResult], ladder: list[Tier]) -> FitOutcome: ...

    def run_batch(self, units: list[BatchUnit], round_index: int) -> list[EpisodeResult]: ...
