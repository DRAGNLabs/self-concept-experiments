import numpy as np
from jaxtyping import Int

from selfconcept.arm_search.interface import EpisodeResult, SettingName
from selfconcept.arm_search.ladder_candidates import ConditionTable, pair_term_cells
from selfconcept.arm_search.ladder_model import LadderCells
from selfconcept.arm_search.ladder_spec import off_target_gates, term_names, term_trials_and_successes
from selfconcept.arm_search.toy.truth import (
    TOY_LADDER_SPEC,
    WORDING,
    ToyWorld,
    condition_cost_per_seed,
    toy_condition_key,
    toy_conditions,
    toy_problem_and_condition,
)


def toy_condition_table(world: ToyWorld) -> ConditionTable:
    conditions = toy_conditions(world)
    return ConditionTable(
        off_target_gate=np.array(
            [off_target_gates(TOY_LADDER_SPEC, {WORDING: condition["wording"]}) for condition in conditions]
        ),
        setting_codes=np.array(
            [
                [
                    world["setting_levels_by_name"][setting_name].index(level)
                    for setting_name, level in condition["level_by_setting"].items()
                ]
                for condition in conditions
            ]
        ),
        cost_per_seed=np.array([condition_cost_per_seed(world, condition) for condition in conditions]),
    )


def toy_ladder_cells(
    world: ToyWorld, group_origin_index: Int[np.ndarray, " group"], episodes: list[EpisodeResult]
) -> LadderCells:
    """Cheats over all episodes, intent over cheats, STRONG over judged (neutral-wording) cheats with intent
    (TOY_LADDER_SPEC)."""
    condition_table = toy_condition_table(world)
    condition_index_by_key = {
        toy_condition_key(condition): condition_index for condition_index, condition in enumerate(toy_conditions(world))
    }
    problem_and_condition = [toy_problem_and_condition(world, episode["arm"]) for episode in episodes]
    pair_codes = np.array(
        [
            problem_index * len(condition_index_by_key) + condition_index_by_key[toy_condition_key(condition)]
            for problem_index, condition in problem_and_condition
        ],
        dtype=int,
    )
    return LadderCells(
        term_names=term_names(TOY_LADDER_SPEC),
        cells_by_term={
            term_name: pair_term_cells(pair_codes, is_trial, is_success, condition_table)
            for term_name, (is_trial, is_success) in term_trials_and_successes(TOY_LADDER_SPEC, episodes).items()
        },
        group_origin_index=group_origin_index,
        origin_count=world["benchmark_count"],
        gate_names=list(TOY_LADDER_SPEC["target_value_by_gate"]),
        setting_levels_by_name={
            SettingName(setting_name): levels for setting_name, levels in world["setting_levels_by_name"].items()
        },
    )
