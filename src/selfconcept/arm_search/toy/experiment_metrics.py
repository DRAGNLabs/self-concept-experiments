from typing import TypedDict

import numpy as np

from selfconcept.arm_search.episode_tally import covered_group_ids, is_hit
from selfconcept.arm_search.interface import EpisodeResult, GroupId, Tier, TierName
from selfconcept.arm_search.records import RoundRecord
from selfconcept.arm_search.toy.truth import (
    WORDING_VALUES,
    ToyWorld,
    draw_truth,
    toy_conditions,
    toy_group_id,
    true_term_probability,
)


class CoveragePoint(TypedDict):
    cumulative_cost: float
    covered_count_by_tier: dict[TierName, int]


def coverage_curve(episodes: list[EpisodeResult], ladder: list[Tier]) -> list[CoveragePoint]:
    """Groups covered at each tier after each episode, in the order episodes were run."""
    covered_by_tier: dict[TierName, set[GroupId]] = {tier["tier_name"]: set() for tier in ladder}
    cumulative_cost = 0.0
    curve: list[CoveragePoint] = []
    for episode in episodes:
        cumulative_cost += episode["cost"]
        for tier in ladder:
            if is_hit(episode, tier):
                covered_by_tier[tier["tier_name"]].add(episode["arm"]["group_id"])
        curve.append(
            CoveragePoint(
                cumulative_cost=cumulative_cost,
                covered_count_by_tier={tier_name: len(covered) for tier_name, covered in covered_by_tier.items()},
            )
        )
    return curve


def covered_at_cost(curve: list[CoveragePoint], tier_name: TierName, cost: float) -> int:
    within_cost = [point for point in curve if point["cumulative_cost"] <= cost]
    return within_cost[-1]["covered_count_by_tier"][tier_name] if within_cost else 0


def first_hit_cost(curve: list[CoveragePoint], tier_name: TierName) -> float | None:
    return next((point["cumulative_cost"] for point in curve if point["covered_count_by_tier"][tier_name] > 0), None)


def best_target_probability_by_group(world: ToyWorld) -> dict[GroupId, float]:
    """Per group, the true per-seed target probability of its best neutral-wording condition."""
    truth = draw_truth(world)
    neutral_conditions = [condition for condition in toy_conditions(world) if condition["wording"] == WORDING_VALUES[0]]
    return {
        toy_group_id(problem_index): max(
            float(np.prod(true_term_probability(world, truth, problem_index, condition)))
            for condition in neutral_conditions
        )
        for problem_index in range(world["problem_count"])
    }


def played_tier_new_hits(rounds: list[RoundRecord], episodes: list[EpisodeResult], ladder: list[Tier]) -> list[int]:
    """Per completed round, groups the target had not covered before the round that hit the played tier in it; the
    realized counterpart of the plan's expected hits."""
    tier_by_name = {tier["tier_name"]: tier for tier in ladder}
    target_covered: set[GroupId] = set()
    new_hit_counts: list[int] = []
    round_start = 0
    for record in rounds:
        round_end = round_start + sum(len(unit["seeds"]) for unit in record["plan"]["units"])
        round_episodes = episodes[round_start:round_end]
        if record["result"] is not None:
            played_tier = tier_by_name[record["plan"]["tier_name"]]
            hit_group_ids = {episode["arm"]["group_id"] for episode in round_episodes if is_hit(episode, played_tier)}
            new_hit_counts.append(len(hit_group_ids - target_covered))
        target_covered |= covered_group_ids(round_episodes, ladder[-1])
        round_start = round_end
    return new_hit_counts
