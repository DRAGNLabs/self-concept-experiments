from collections.abc import Callable
from typing import TypedDict

import numpy as np
from jaxtyping import Bool, Float, Int
from scipy.special import log_expit

from selfconcept.arm_search.interface import Arm, CandidatePosterior
from selfconcept.arm_search.ladder_draws import ArmFeatures, LadderDraws, TermDraws, hit_probability
from selfconcept.arm_search.ladder_model import TermCells


class ConditionTable(TypedDict):
    """Every condition (gate and free setting values) a tier allows, with its expected cost per seed."""

    off_target_gate: Float[np.ndarray, "condition gate"]
    setting_codes: Int[np.ndarray, "condition setting"]
    cost_per_seed: Float[np.ndarray, " condition"]


class CandidateLimits(TypedDict):
    groups_per_draw: int
    conditions_per_draw: int
    max_candidates: int


class Candidates(TypedDict):
    group_index: Int[np.ndarray, " candidate"]
    condition_index: Int[np.ndarray, " candidate"]
    hit_probability: Float[np.ndarray, "draw candidate"]


def pair_term_cells(
    pair_codes: Int[np.ndarray, " episode"],
    is_trial: Bool[np.ndarray, " episode"],
    is_success: Bool[np.ndarray, " episode"],
    condition_table: ConditionTable,
) -> TermCells:
    """Binomial cells of one term; pair code = group index × condition count + condition index."""
    condition_count = len(condition_table["cost_per_seed"])
    cell_codes, cell_by_trial = np.unique(pair_codes[is_trial], return_inverse=True)
    condition_index = cell_codes % condition_count
    return TermCells(
        group_index=cell_codes // condition_count,
        off_target_gate=condition_table["off_target_gate"][condition_index],
        setting_codes=condition_table["setting_codes"][condition_index],
        trial_count=np.bincount(cell_by_trial, minlength=len(cell_codes)),
        success_count=np.bincount(cell_by_trial, weights=is_success[is_trial], minlength=len(cell_codes)).astype(int),
    )


def _group_logit(
    term_draws: TermDraws,
    group_origin_index: Int[np.ndarray, " group"],
    group_indices: Int[np.ndarray, " eligible_group"],
    gate_pattern: Float[np.ndarray, " gate"],
) -> Float[np.ndarray, "draw eligible_group"]:
    return (
        term_draws["origin_intercept"][:, group_origin_index[group_indices]]
        + term_draws["group_effect"][:, group_indices]
        + np.einsum("dgp,g->dp", term_draws["gate_group_deviation"][:, :, group_indices], gate_pattern)
    )


def _condition_logit(term_draws: TermDraws, conditions: ConditionTable) -> Float[np.ndarray, "draw condition"]:
    return term_draws["gate_effect"] @ conditions["off_target_gate"].T + sum(
        (
            effect[:, conditions["setting_codes"][:, position]]
            for position, effect in enumerate(term_draws["setting_effects"])
        ),
        np.zeros((term_draws["gate_effect"].shape[0], len(conditions["cost_per_seed"]))),
    )


def _draw_top_pairs(
    group_logits: list[Float[np.ndarray, "draw group"]],
    condition_logits: list[Float[np.ndarray, "draw condition"]],
    log_cost: Float[np.ndarray, " condition"],
    limits: CandidateLimits,
) -> tuple[Int[np.ndarray, "draw top_group"], Int[np.ndarray, "draw top_condition"]]:
    """One round of coordinate ascent per draw: conditions ranked for the draw's best group under an average
    condition, then groups ranked for the draw's best condition."""
    average_condition_logits = [logit.mean(axis=1, keepdims=True) for logit in condition_logits]
    average_condition_group_score = sum(
        log_expit(group_logit + condition_logit)
        for group_logit, condition_logit in zip(group_logits, average_condition_logits, strict=True)
    )
    best_group = np.argmax(average_condition_group_score, axis=1)
    draw_range = np.arange(len(best_group))
    condition_score = (
        sum(
            log_expit(group_logit[draw_range, best_group][:, None] + condition_logit)
            for group_logit, condition_logit in zip(group_logits, condition_logits, strict=True)
        )
        - log_cost
    )
    top_conditions = np.argsort(-condition_score, axis=1)[:, : limits["conditions_per_draw"]]
    group_score = sum(
        log_expit(group_logit + condition_logit[draw_range, top_conditions[:, 0]][:, None])
        for group_logit, condition_logit in zip(group_logits, condition_logits, strict=True)
    )
    top_groups = np.argsort(-group_score, axis=1)[:, : limits["groups_per_draw"]]
    return top_groups, top_conditions


def tier_candidates(
    draws: LadderDraws,
    term_count: int,
    conditions: ConditionTable,
    eligible_group_indices: Int[np.ndarray, " group"],
    limits: CandidateLimits,
) -> Candidates:
    """Union over draws and gate patterns of each draw's best groups × best conditions, so the full group ×
    condition space is never scored. Trimming keeps each draw's best pair first (Thompson picks need it), then the
    pairs the most draws proposed."""
    term_draws = list(draws["term_draws_by_name"].values())[:term_count]
    gate_patterns, pattern_by_condition = np.unique(conditions["off_target_gate"], axis=0, return_inverse=True)
    proposed_pairs: list[Int[np.ndarray, "pair 2"]] = []
    draw_best_pairs: list[Int[np.ndarray, "draw 2"]] = []
    for pattern_position, gate_pattern in enumerate(gate_patterns):
        pattern_conditions = np.flatnonzero(pattern_by_condition == pattern_position)
        pattern_table = ConditionTable(
            off_target_gate=conditions["off_target_gate"][pattern_conditions],
            setting_codes=conditions["setting_codes"][pattern_conditions],
            cost_per_seed=conditions["cost_per_seed"][pattern_conditions],
        )
        top_groups, top_conditions = _draw_top_pairs(
            [
                _group_logit(term, draws["group_origin_index"], eligible_group_indices, gate_pattern)
                for term in term_draws
            ],
            [_condition_logit(term, pattern_table) for term in term_draws],
            np.log(pattern_table["cost_per_seed"]),
            limits,
        )
        proposed_pairs.append(
            np.stack(
                [
                    np.repeat(eligible_group_indices[top_groups], top_conditions.shape[1], axis=1).ravel(),
                    np.tile(pattern_conditions[top_conditions], (1, top_groups.shape[1])).ravel(),
                ],
                axis=1,
            )
        )
        draw_best_pairs.append(
            np.stack(
                [eligible_group_indices[top_groups[:, 0]], pattern_conditions[top_conditions[:, 0]]], axis=1
            )
        )
    all_proposed_pairs = np.concatenate(proposed_pairs)
    unique_pairs, vote_count = np.unique(all_proposed_pairs, axis=0, return_counts=True)
    condition_count = len(conditions["cost_per_seed"])
    draw_best_codes = np.concatenate(draw_best_pairs) @ np.array([condition_count, 1])
    is_draw_best = np.isin(unique_pairs @ np.array([condition_count, 1]), draw_best_codes)
    priority = vote_count + is_draw_best * len(all_proposed_pairs)
    kept_pairs = unique_pairs[np.argsort(-priority, kind="stable")[: limits["max_candidates"]]]
    features = ArmFeatures(
        group_index=kept_pairs[:, 0],
        off_target_gate=conditions["off_target_gate"][kept_pairs[:, 1]],
        setting_codes=conditions["setting_codes"][kept_pairs[:, 1]],
    )
    return Candidates(
        group_index=kept_pairs[:, 0],
        condition_index=kept_pairs[:, 1],
        hit_probability=hit_probability(draws, term_count, features),
    )


def tier_candidate_posterior(
    draws: LadderDraws,
    term_count: int,
    condition_table: ConditionTable,
    tier_condition_indices: Int[np.ndarray, " tier_condition"],
    eligible_group_indices: Int[np.ndarray, " group"],
    limits: CandidateLimits,
    arm_for: Callable[[int, int], Arm],
) -> CandidatePosterior:
    """Candidates among the tier's conditions and the eligible groups; `arm_for(group index, condition index)`."""
    draw_count = next(iter(draws["term_draws_by_name"].values()))["group_effect"].shape[0]
    if len(eligible_group_indices) == 0:
        return CandidatePosterior(
            arms=[], hit_probability=np.zeros((draw_count, 0)), expected_cost_per_seed=np.zeros((draw_count, 0))
        )
    candidates = tier_candidates(
        draws,
        term_count,
        ConditionTable(
            off_target_gate=condition_table["off_target_gate"][tier_condition_indices],
            setting_codes=condition_table["setting_codes"][tier_condition_indices],
            cost_per_seed=condition_table["cost_per_seed"][tier_condition_indices],
        ),
        eligible_group_indices,
        limits,
    )
    candidate_condition_indices = tier_condition_indices[candidates["condition_index"]]
    return CandidatePosterior(
        arms=[
            arm_for(int(group_index), int(condition_index))
            for group_index, condition_index in zip(
                candidates["group_index"], candidate_condition_indices, strict=True
            )
        ],
        hit_probability=candidates["hit_probability"],
        expected_cost_per_seed=np.broadcast_to(
            condition_table["cost_per_seed"][candidate_condition_indices], candidates["hit_probability"].shape
        ),
    )
