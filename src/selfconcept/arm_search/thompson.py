from typing import TypedDict

import numpy as np

from selfconcept.arm_search.interface import CandidatePosterior, GroupId
from selfconcept.arm_search.reward import add_pending_unit, initial_group_coverage


class ThompsonPick(TypedDict):
    arm_index: int
    draw_index: int
    hit_probability_mean: float
    expected_new_group: float
    expected_cost: float


def thompson_batch(
    candidates: CandidatePosterior,
    covered_group_ids: set[GroupId],
    seeds_per_unit: int,
    units_per_batch: int,
    rng: np.random.Generator,
) -> list[ThompsonPick]:
    """Each unit samples one posterior draw and takes the arm with the most chance per cost of a first hit on its
    group, given the units already picked; the expected values average over all draws."""
    hit_probability = candidates["hit_probability"]
    draw_count = hit_probability.shape[0]
    any_hit_probability = 1 - (1 - hit_probability) ** seeds_per_unit
    unit_costs = seeds_per_unit * candidates["expected_cost_per_seed"]
    coverage = initial_group_coverage(candidates["arms"], covered_group_ids, draw_count)
    picks: list[ThompsonPick] = []
    for _ in range(units_per_batch):
        draw_index = int(rng.integers(draw_count))
        uncovered_probability = coverage["uncovered_probability"][:, coverage["arm_group_indices"]]
        new_group_probability = uncovered_probability * any_hit_probability
        arm_index = int(np.argmax(new_group_probability[draw_index] / unit_costs[draw_index]))
        picks.append(
            ThompsonPick(
                arm_index=arm_index,
                draw_index=draw_index,
                hit_probability_mean=float(hit_probability[:, arm_index].mean()),
                expected_new_group=float(new_group_probability[:, arm_index].mean()),
                expected_cost=float(unit_costs[:, arm_index].mean()),
            )
        )
        coverage = add_pending_unit(coverage, hit_probability, arm_index, seeds_per_unit)
    return picks
