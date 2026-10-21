import logging
from collections.abc import Callable
from datetime import UTC, datetime
from typing import TypedDict

import numpy as np

from selfconcept.arm_search.episode_tally import assign_unit_seeds, covered_group_ids
from selfconcept.arm_search.interface import (
    CandidatePosterior,
    EpisodeResult,
    FitSucceeded,
    GroupId,
    SearchApplication,
    Tier,
)
from selfconcept.arm_search.records import (
    ArmEstimate,
    PlannedUnit,
    PlanningOutcome,
    RoundPlan,
    RoundPlanned,
    SearchConfig,
    SearchStopped,
)
from selfconcept.arm_search.state import initialize_state, load_episodes, load_rounds, write_planning_outcome
from selfconcept.arm_search.thompson import ThompsonPick, thompson_batch

logger = logging.getLogger(__name__)


class TierBatch(TypedDict):
    tier: Tier
    candidates: CandidatePosterior
    picks: list[ThompsonPick]
    expected_hits: float
    expected_cost: float


def _tier_batch(
    tier: Tier,
    candidates: CandidatePosterior,
    target_covered_group_ids: set[GroupId],
    config: SearchConfig,
    rng: np.random.Generator,
) -> TierBatch:
    """Every tier aims at groups the target has not covered, so a lenient tier keeps working a group after its first
    lenient hit."""
    picks = (
        thompson_batch(candidates, target_covered_group_ids, config["seeds_per_unit"], config["units_per_batch"], rng)
        if candidates["arms"]
        else []
    )
    return TierBatch(
        tier=tier,
        candidates=candidates,
        picks=picks,
        expected_hits=sum(pick["expected_new_group"] for pick in picks),
        expected_cost=sum(pick["expected_cost"] for pick in picks),
    )


def _top_estimated_arms(candidates: CandidatePosterior, count: int) -> list[ArmEstimate]:
    hit_probability = candidates["hit_probability"]
    mean_by_arm = hit_probability.mean(axis=0)
    lower_by_arm, upper_by_arm = np.quantile(hit_probability, [0.05, 0.95], axis=0)
    return [
        ArmEstimate(
            arm_id=candidates["arms"][arm_index]["arm_id"],
            group_id=candidates["arms"][arm_index]["group_id"],
            success_probability_mean=float(mean_by_arm[arm_index]),
            success_probability_lower=float(lower_by_arm[arm_index]),
            success_probability_upper=float(upper_by_arm[arm_index]),
        )
        for arm_index in np.argsort(-mean_by_arm)[:count]
    ]


def _round_plan(
    round_index: int,
    played: TierBatch,
    tier_batches: list[TierBatch],
    episodes: list[EpisodeResult],
    fit_outcome: FitSucceeded,
    config: SearchConfig,
) -> RoundPlan:
    arms = played["candidates"]["arms"]
    return RoundPlan(
        round_index=round_index,
        planned_at=datetime.now(UTC).isoformat(),
        tier_name=played["tier"]["tier_name"],
        units=assign_unit_seeds(
            [arms[pick["arm_index"]] for pick in played["picks"]], episodes, config["seeds_per_unit"]
        ),
        expected_hits_by_tier={batch["tier"]["tier_name"]: batch["expected_hits"] for batch in tier_batches},
        expected_cost=played["expected_cost"],
        fit_diagnostic_value_by_name=fit_outcome["diagnostic_value_by_name"],
        fit_warnings=fit_outcome["warnings"],
        effect_estimates=fit_outcome["effect_estimates"],
        planned_units=[
            PlannedUnit(
                arm_id=arms[pick["arm_index"]]["arm_id"],
                group_id=arms[pick["arm_index"]]["group_id"],
                draw_index=pick["draw_index"],
                hit_probability_mean=pick["hit_probability_mean"],
                expected_new_group=pick["expected_new_group"],
                expected_cost=pick["expected_cost"],
            )
            for pick in played["picks"]
        ],
        top_estimated_arms=_top_estimated_arms(played["candidates"], config["reported_arm_count"]),
    )


def _plan_round(application: SearchApplication, config: SearchConfig) -> PlanningOutcome:
    """Play the strictest tier whose Thompson batch expects at least `min_expected_hits` new groups, else the tier
    expecting the most, while its expected hits per cost clear `min_value_per_cost`."""
    episodes = load_episodes(config["state_dir"])
    ladder = config["ladder"]
    fit_outcome = application.fit(episodes, ladder)
    diagnostic_value_by_name = fit_outcome["diagnostic_value_by_name"]
    if fit_outcome["kind"] == "fit_failed":
        return SearchStopped(
            kind="search_stopped",
            stop_reason="fit_failed",
            detail=fit_outcome["reason"],
            fit_diagnostic_value_by_name=diagnostic_value_by_name,
        )
    for warning in fit_outcome["warnings"]:
        logger.warning("fit warning: %s", warning)
    round_index = len(load_rounds(config["state_dir"]))
    target_covered_group_ids = covered_group_ids(episodes, ladder[-1])
    tier_batches = [
        _tier_batch(
            tier,
            fit_outcome["candidate_posterior_by_tier"][tier["tier_name"]],
            target_covered_group_ids,
            config,
            np.random.default_rng([config["random_seed"], round_index, tier_position]),
        )
        for tier_position, tier in enumerate(ladder)
    ]
    for batch in tier_batches:
        logger.info("tier %s: expected hits %.3g", batch["tier"]["tier_name"], batch["expected_hits"])
    reaching_batches = [batch for batch in tier_batches if batch["expected_hits"] >= config["min_expected_hits"]]
    played = reaching_batches[-1] if reaching_batches else max(tier_batches, key=lambda batch: batch["expected_hits"])
    min_value_per_cost = config["stop_conditions"]["min_value_per_cost"]
    if played["expected_hits"] == 0 or (
        not reaching_batches
        and min_value_per_cost is not None
        and played["expected_hits"] / played["expected_cost"] < min_value_per_cost
    ):
        return SearchStopped(
            kind="search_stopped",
            stop_reason="below_value_threshold",
            detail=f"tier {played['tier']['tier_name']} expects {played['expected_hits']:.4g} hits"
            f" for cost {played['expected_cost']:.4g}",
            fit_diagnostic_value_by_name=diagnostic_value_by_name,
        )
    return RoundPlanned(
        kind="round_planned", plan=_round_plan(round_index, played, tier_batches, episodes, fit_outcome, config)
    )


def plan_round(application: SearchApplication, config: SearchConfig) -> PlanningOutcome:
    """Fit, choose the next batch, and persist the outcome so a planner in another process can hand it back."""
    initialize_state(config)
    outcome = _plan_round(application, config)
    write_planning_outcome(config["state_dir"], outcome)
    logger.info("planning outcome: %s", outcome["kind"])
    return outcome


def in_process_planner(application: SearchApplication) -> Callable[[SearchConfig], PlanningOutcome]:
    return lambda config: plan_round(application, config)
