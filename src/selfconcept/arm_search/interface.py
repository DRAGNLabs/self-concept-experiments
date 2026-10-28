from typing import Literal, NewType, Protocol, TypedDict

import numpy as np
from jaxtyping import Float

ArmId = NewType("ArmId", str) # see Arm
GroupId = NewType("GroupId", str) # a group is a collection of arms that is dropped all at once when one of them hits the target
FactorName = NewType("FactorName", str) # outcomes are made of a set of factors, each of which has a number of categorical possibilities
CategoryName = NewType("CategoryName", str) # categories are the possible values a factor can take on in an outcome
SettingName = NewType("SettingName", str) # knobs that we control. Together with group, defined an arm. Come in "gate" and "free" varieties, where gate settings help define the target. Only gate settings are realized in the arm_search code
TierName = NewType("TierName", str) # See Tier


class Arm(TypedDict):
    """A full configuration that, when run, produces an outcome that may or may not be our target."""

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
