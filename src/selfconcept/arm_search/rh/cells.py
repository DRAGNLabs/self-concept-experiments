from collections import defaultdict
from typing import Final

import numpy as np
from jaxtyping import Float

from selfconcept.arm_search.interface import Arm, EpisodeResult
from selfconcept.arm_search.ladder_candidates import ConditionTable, pair_term_cells
from selfconcept.arm_search.ladder_model import LadderCells
from selfconcept.arm_search.ladder_spec import off_target_gates, term_names, term_trials_and_successes
from selfconcept.arm_search.rh.conditions import (
    Condition,
    Effort,
    condition_id,
    free_setting_levels_by_name,
    free_setting_values,
    gate_value_by_setting,
    parse_condition_id,
    search_conditions,
)
from selfconcept.arm_search.rh.episodes import RH_LADDER_SPEC
from selfconcept.arm_search.rh.pool import BENCHMARKS, PoolProblem

# Thousands of generated characters per episode before any episode of a condition's effort and attempt count exists
DEFAULT_COST_PER_SEED: Final = 60.0


def arm_example_id_and_condition(arm: Arm) -> tuple[str, Condition]:
    example_id, arm_condition_id = arm["arm_id"].split("|")
    return example_id, parse_condition_id(arm_condition_id)


def model_conditions(episodes: list[EpisodeResult], efforts: tuple[Effort, ...]) -> list[Condition]:
    """The search grid, then any other condition historic episodes ran (e.g. no tier instruction, 6 attempts)."""
    grid = search_conditions(efforts)
    grid_ids = {condition_id(condition) for condition in grid}
    extra_ids = sorted({episode["arm"]["arm_id"].split("|")[1] for episode in episodes} - grid_ids)
    return grid + [parse_condition_id(extra_id) for extra_id in extra_ids]


def cost_per_seed(conditions: list[Condition], episodes: list[EpisodeResult]) -> Float[np.ndarray, " condition"]:
    """Mean episode cost by effort and attempt count, the settings that drive generation length."""
    costs_by_key: dict[tuple[str, int], list[float]] = defaultdict(list)
    for episode in episodes:
        _, condition = arm_example_id_and_condition(episode["arm"])
        costs_by_key[(condition["effort"], condition["max_attempts"])].append(episode["cost"])
    mean_cost_by_key = {key: float(np.mean(costs)) for key, costs in costs_by_key.items()}
    return np.array(
        [
            mean_cost_by_key.get((condition["effort"], condition["max_attempts"]), DEFAULT_COST_PER_SEED)
            for condition in conditions
        ]
    )


def rh_condition_table(
    conditions: list[Condition], episodes: list[EpisodeResult], efforts: tuple[Effort, ...]
) -> ConditionTable:
    levels_by_setting = list(free_setting_levels_by_name(efforts).values())
    return ConditionTable(
        off_target_gate=np.array(
            [off_target_gates(RH_LADDER_SPEC, gate_value_by_setting(condition)) for condition in conditions]
        ),
        setting_codes=np.array(
            [
                [levels.index(value) for levels, value in zip(levels_by_setting, free_setting_values(condition))]
                for condition in conditions
            ]
        ),
        cost_per_seed=cost_per_seed(conditions, episodes),
    )


def rh_ladder_cells(
    problems: list[PoolProblem], conditions: list[Condition], episodes: list[EpisodeResult], efforts: tuple[Effort, ...]
) -> LadderCells:
    """Cheats over all episodes, intent over cheats, STRONG over judged (neutral-wording) cheats with intent
    (RH_LADDER_SPEC)."""
    problem_index_by_example_id = {problem["example_id"]: index for index, problem in enumerate(problems)}
    condition_index_by_id = {condition_id(condition): index for index, condition in enumerate(conditions)}
    example_ids_and_condition_ids = [episode["arm"]["arm_id"].split("|") for episode in episodes]
    pair_codes = np.array(
        [
            problem_index_by_example_id[example_id] * len(conditions) + condition_index_by_id[episode_condition_id]
            for example_id, episode_condition_id in example_ids_and_condition_ids
        ],
        dtype=int,
    )
    condition_table = rh_condition_table(conditions, episodes, efforts)
    return LadderCells(
        term_names=term_names(RH_LADDER_SPEC),
        cells_by_term={
            term_name: pair_term_cells(pair_codes, is_trial, is_success, condition_table)
            for term_name, (is_trial, is_success) in term_trials_and_successes(RH_LADDER_SPEC, episodes).items()
        },
        group_origin_index=np.array([BENCHMARKS.index(problem["benchmark"]) for problem in problems]),
        origin_count=len(BENCHMARKS),
        gate_names=list(RH_LADDER_SPEC["target_value_by_gate"]),
        setting_levels_by_name=free_setting_levels_by_name(efforts),
    )
