from collections.abc import Callable
from datetime import UTC, datetime

import numpy as np

from selfconcept.arm_search.episode_tally import assign_unit_seeds, covered_group_ids, is_eligible, realized_cost
from selfconcept.arm_search.records import (
    PlanningOutcome,
    RoundPlan,
    RoundPlanned,
    SearchConfig,
    SearchStopped,
)
from selfconcept.arm_search.state import load_episodes, load_rounds
from selfconcept.arm_search.toy.truth import ToyWorld, toy_arm, toy_conditions

type Planner = Callable[[SearchConfig], PlanningOutcome]


def _stopped(detail: str) -> SearchStopped:
    return SearchStopped(
        kind="search_stopped", stop_reason="below_value_threshold", detail=detail, fit_diagnostic_value_by_name={}
    )


def budget_limited(planner: Planner, cost_budget: float) -> Planner:
    """Every policy stops at the same spent cost, so coverage curves compare over one range."""

    def plan_within_budget(config: SearchConfig) -> PlanningOutcome:
        spent = realized_cost(load_episodes(config["state_dir"]))
        return _stopped(f"spent {spent:.4g} of budget {cost_budget:.4g}") if spent >= cost_budget else planner(config)

    return plan_within_budget


def random_planner(world: ToyWorld) -> Planner:
    """Uniform over the target tier's eligible arms on groups it has not covered."""

    def plan_random_round(config: SearchConfig) -> PlanningOutcome:
        target = config["ladder"][-1]
        episodes = load_episodes(config["state_dir"])
        round_index = len(load_rounds(config["state_dir"]))
        covered = covered_group_ids(episodes, target)
        eligible_arms = [
            arm
            for problem_index in range(world["problem_count"])
            for arm in (toy_arm(problem_index, condition) for condition in toy_conditions(world))
            if is_eligible(arm, target) and arm["group_id"] not in covered
        ]
        if not eligible_arms:
            return _stopped("every group is covered")
        rng = np.random.default_rng([config["random_seed"], round_index])
        chosen_indices = rng.integers(len(eligible_arms), size=config["units_per_batch"])
        chosen_arms = [eligible_arms[index] for index in chosen_indices]
        return RoundPlanned(
            kind="round_planned",
            plan=RoundPlan(
                round_index=round_index,
                planned_at=datetime.now(UTC).isoformat(),
                tier_name=target["tier_name"],
                units=assign_unit_seeds(chosen_arms, episodes, config["seeds_per_unit"]),
                expected_hits_by_tier={target["tier_name"]: 0.0},
                expected_cost=0.0,
                fit_diagnostic_value_by_name={},
                fit_warnings=[],
                effect_estimates=[],
                planned_units=[],
                top_estimated_arms=[],
            ),
        )

    return plan_random_round
