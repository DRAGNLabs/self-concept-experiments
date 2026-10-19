"""One definition of a ladder application's chain of criteria, from which its outcome levels, the ladder model's terms
and their cells, and the search's tiers all follow, so they cannot drift apart."""

from collections.abc import Mapping
from typing import TypedDict

import numpy as np
from jaxtyping import Bool

from selfconcept.arm_search.interface import (
    Arm,
    CategoryName,
    EpisodeResult,
    FactorName,
    SettingName,
    SuccessDefinition,
    Tier,
    TierName,
)
from selfconcept.arm_search.ladder_model import TermName


class LadderCriterion(TypedDict):
    """`category_name` is the outcome level of an episode that met this criterion and every earlier one;
    `on_target_only` criteria are only observed (so their term is only fit) where every gate is at its target."""

    term_name: TermName
    category_name: CategoryName
    on_target_only: bool


class LadderTierSpec(TypedDict):
    """A hit meets the first `criterion_count` criteria, with every gate at its target if `on_target_only`."""

    tier_name: TierName
    criterion_count: int
    on_target_only: bool


class LadderSpec(TypedDict):
    """`criteria` in chain order; `base_category_name` is the level of an episode that met none; `tiers` from the most
    lenient to the target."""

    factor_name: FactorName
    base_category_name: CategoryName
    criteria: list[LadderCriterion]
    target_value_by_gate: dict[SettingName, str]
    tiers: list[LadderTierSpec]


def outcome_levels(spec: LadderSpec) -> list[CategoryName]:
    """Ordered: index k is "met the first k criteria"."""
    return [spec["base_category_name"], *(criterion["category_name"] for criterion in spec["criteria"])]


def term_names(spec: LadderSpec) -> list[TermName]:
    return [criterion["term_name"] for criterion in spec["criteria"]]


def ladder_tiers(spec: LadderSpec) -> list[Tier]:
    levels = outcome_levels(spec)
    target_gate_values = {gate: [value] for gate, value in spec["target_value_by_gate"].items()}
    return [
        Tier(
            tier_name=tier["tier_name"],
            gate_values_by_setting=target_gate_values if tier["on_target_only"] else {},
            success=SuccessDefinition(
                factor_name=spec["factor_name"], success_category_names=levels[tier["criterion_count"] :]
            ),
        )
        for tier in spec["tiers"]
    ]


def tier_term_count(spec: LadderSpec, tier: Tier) -> int:
    """Criteria a tier needs, read off its most lenient success level."""
    levels = outcome_levels(spec)
    return min(levels.index(category) for category in tier["success"]["success_category_names"])


def off_target_gates(spec: LadderSpec, gate_value_by_setting: Mapping[SettingName, str]) -> list[float]:
    """1 for each gate (in spec order) away from its target value: the ladder model's `off_target_gate` row."""
    return [float(gate_value_by_setting[gate] != value) for gate, value in spec["target_value_by_gate"].items()]


def is_on_target(spec: LadderSpec, arm: Arm) -> bool:
    return not any(off_target_gates(spec, arm["gate_value_by_setting"]))


def observable_criterion_count(spec: LadderSpec, arm: Arm) -> int:
    """How far along the chain an episode of the arm can be observed to get."""
    if is_on_target(spec, arm):
        return len(spec["criteria"])
    return next(
        (position for position, criterion in enumerate(spec["criteria"]) if criterion["on_target_only"]),
        len(spec["criteria"]),
    )


def term_trials_and_successes(
    spec: LadderSpec, episodes: list[EpisodeResult]
) -> dict[TermName, tuple[Bool[np.ndarray, " episode"], Bool[np.ndarray, " episode"]]]:
    """A term's trials are the episodes that met every earlier criterion (and are on target where the criterion needs
    it); its successes are those that also met it."""
    levels = outcome_levels(spec)
    level = np.array([levels.index(episode["category_by_factor"][spec["factor_name"]]) for episode in episodes])
    on_target = np.array([is_on_target(spec, episode["arm"]) for episode in episodes], dtype=bool)
    return {
        criterion["term_name"]: (
            (level >= position) & (on_target | (not criterion["on_target_only"])),
            level >= position + 1,
        )
        for position, criterion in enumerate(spec["criteria"])
    }
