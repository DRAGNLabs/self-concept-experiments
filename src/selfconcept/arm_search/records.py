from pathlib import Path
from typing import Literal, TypedDict

from selfconcept.arm_search.interface import ArmId, BatchUnit, EffectEstimate, GroupId, Tier, TierName

StopReason = Literal["reached_reward_target", "below_value_threshold", "fit_failed"]


class StopConditions(TypedDict):
    """`reward_target` counts groups covered at the target (last) tier."""

    reward_target: float | None
    min_value_per_cost: float | None


class SearchConfig(TypedDict):
    """`ladder` runs from the most lenient tier to the target."""

    state_dir: Path
    ladder: list[Tier]
    seeds_per_unit: int
    units_per_batch: int
    min_expected_hits: float
    reported_arm_count: int
    random_seed: int
    stop_conditions: StopConditions


class ArmEstimate(TypedDict):
    arm_id: ArmId
    group_id: GroupId
    success_probability_mean: float
    success_probability_lower: float
    success_probability_upper: float


class PlannedUnit(TypedDict):
    arm_id: ArmId
    group_id: GroupId
    draw_index: int
    hit_probability_mean: float
    expected_new_group: float
    expected_cost: float


class RoundPlan(TypedDict):
    """`expected_hits_by_tier` holds, for the batch each tier would have played, the expected number of groups not yet
    covered at the target that get a first hit at that tier."""

    round_index: int
    planned_at: str
    tier_name: TierName
    units: list[BatchUnit]
    expected_hits_by_tier: dict[TierName, float]
    expected_cost: float
    fit_diagnostic_value_by_name: dict[str, float]
    fit_warnings: list[str]
    effect_estimates: list[EffectEstimate]
    planned_units: list[PlannedUnit]
    top_estimated_arms: list[ArmEstimate]


class RoundResult(TypedDict):
    """`realized_reward_by_tier` counts groups newly covered at each tier; `realized_hits_by_tier` counts groups the
    target had not covered before the round that hit each tier in it (the counterpart of the plan's expected hits)."""

    completed_at: str
    realized_reward_by_tier: dict[TierName, float]
    realized_hits_by_tier: dict[TierName, float]
    realized_cost: float


class RoundRecord(TypedDict):
    plan: RoundPlan
    result: RoundResult | None


class RoundPlanned(TypedDict):
    kind: Literal["round_planned"]
    plan: RoundPlan


class SearchStopped(TypedDict):
    kind: Literal["search_stopped"]
    stop_reason: StopReason
    detail: str
    fit_diagnostic_value_by_name: dict[str, float]


PlanningOutcome = RoundPlanned | SearchStopped


class SearchSummary(TypedDict):
    rounds: list[RoundRecord]
    total_reward_by_tier: dict[TierName, float]
    total_cost: float
    stop_reason: StopReason
    detail: str
