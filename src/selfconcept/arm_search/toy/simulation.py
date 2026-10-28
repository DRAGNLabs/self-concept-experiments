import zlib

import numpy as np

from selfconcept.arm_search.interface import BatchUnit, EpisodeResult
from selfconcept.arm_search.ladder_spec import observable_criterion_count
from selfconcept.arm_search.toy.truth import (
    OUTCOME_FACTOR,
    OUTCOME_LEVELS,
    TOY_LADDER_SPEC,
    ToyTruth,
    ToyWorld,
    condition_cost_per_seed,
    toy_problem_and_condition,
    true_term_probability,
)


def simulate_toy_episodes(world: ToyWorld, truth: ToyTruth, units: list[BatchUnit]) -> list[EpisodeResult]:
    """Each criterion is reached only through the previous ones; STRONG is judged (so observed) only under neutral
    wording. Outcomes depend only on the world, arm and seed, so every policy sees the same episode for the same
    arm and seed."""
    episodes: list[EpisodeResult] = []
    for unit in units:
        problem_index, condition = toy_problem_and_condition(world, unit["arm"])
        term_probability = true_term_probability(world, truth, problem_index, condition)
        observed_term_count = observable_criterion_count(TOY_LADDER_SPEC, unit["arm"])
        for seed in unit["seeds"]:
            uniforms = np.random.default_rng(
                [world["truth_seed"], zlib.crc32(unit["arm"]["arm_id"].encode()), seed]
            ).random(len(term_probability))
            met_criteria = np.cumprod(uniforms < term_probability)[:observed_term_count]
            episodes.append(
                EpisodeResult(
                    arm=unit["arm"],
                    seed=seed,
                    category_by_factor={OUTCOME_FACTOR: OUTCOME_LEVELS[int(met_criteria.sum())]},
                    cost=condition_cost_per_seed(world, condition),
                )
            )
    return episodes
