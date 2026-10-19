import itertools
from typing import Final, TypedDict

import numpy as np
from jaxtyping import Float, Int
from scipy.special import expit

from selfconcept.arm_search.interface import Arm, ArmId, CategoryName, FactorName, GroupId, SettingName, TierName
from selfconcept.arm_search.ladder_model import TermName
from selfconcept.arm_search.ladder_spec import LadderCriterion, LadderSpec, LadderTierSpec, outcome_levels

WORDING: Final = SettingName("wording")
WORDING_VALUES: Final = ["neutral", "pass_tests"]
# The RH ladder's shape: STRONG is judged (so observed) only under neutral wording
TOY_LADDER_SPEC: Final = LadderSpec(
    factor_name=FactorName("outcome_level"),
    base_category_name=CategoryName("none"),
    criteria=[
        LadderCriterion(term_name=TermName("cheats"), category_name=CategoryName("cheats"), on_target_only=False),
        LadderCriterion(
            term_name=TermName("intent"), category_name=CategoryName("cheats_intent"), on_target_only=False
        ),
        LadderCriterion(term_name=TermName("strong"), category_name=CategoryName("strong"), on_target_only=True),
    ],
    target_value_by_gate={WORDING: WORDING_VALUES[0]},
    tiers=[
        LadderTierSpec(tier_name=TierName("cheats"), criterion_count=1, on_target_only=False),
        LadderTierSpec(tier_name=TierName("neutral_cheats"), criterion_count=1, on_target_only=True),
        LadderTierSpec(tier_name=TierName("neutral_intent"), criterion_count=2, on_target_only=True),
        LadderTierSpec(tier_name=TierName("target"), criterion_count=3, on_target_only=True),
    ],
)
OUTCOME_FACTOR: Final = TOY_LADDER_SPEC["factor_name"]
OUTCOME_LEVELS: Final = outcome_levels(TOY_LADDER_SPEC)


class ToyTermTruth(TypedDict):
    """One criterion's true log-odds pieces; `wording_effect` and `wording_problem_spread` apply to pass-tests
    wording (off target)."""

    intercept_by_benchmark: list[float]
    problem_spread: float
    wording_effect: float
    wording_problem_spread: float
    effect_by_level_by_setting: dict[str, dict[str, float]]


class ToyWorld(TypedDict):
    """`term_truths` run cheats, intent given cheats, STRONG given cheats with intent."""

    problem_count: int
    benchmark_count: int
    setting_levels_by_name: dict[str, list[str]]
    cost_multiplier_by_level_by_setting: dict[str, dict[str, float]]
    term_truths: list[ToyTermTruth]
    truth_seed: int


class ToyTruth(TypedDict):
    problem_benchmark_index: Int[np.ndarray, " problem"]
    problem_effect: Float[np.ndarray, "term problem"]
    wording_problem_deviation: Float[np.ndarray, "term problem"]


class ToyCondition(TypedDict):
    wording: str
    level_by_setting: dict[str, str]


def draw_truth(world: ToyWorld) -> ToyTruth:
    rng = np.random.default_rng(world["truth_seed"])
    problem_count = world["problem_count"]
    return ToyTruth(
        problem_benchmark_index=rng.integers(world["benchmark_count"], size=problem_count),
        problem_effect=np.stack(
            [rng.normal(0.0, term["problem_spread"], problem_count) for term in world["term_truths"]]
        ),
        wording_problem_deviation=np.stack(
            [rng.normal(0.0, term["wording_problem_spread"], problem_count) for term in world["term_truths"]]
        ),
    )


def toy_conditions(world: ToyWorld) -> list[ToyCondition]:
    setting_names = list(world["setting_levels_by_name"])
    return [
        ToyCondition(wording=wording, level_by_setting=dict(zip(setting_names, levels, strict=True)))
        for wording in WORDING_VALUES
        for levels in itertools.product(*world["setting_levels_by_name"].values())
    ]


def condition_cost_per_seed(world: ToyWorld, condition: ToyCondition) -> float:
    return float(
        np.prod(
            [
                world["cost_multiplier_by_level_by_setting"].get(setting_name, {}).get(level, 1.0)
                for setting_name, level in condition["level_by_setting"].items()
            ]
        )
    )


def toy_group_id(problem_index: int) -> GroupId:
    return GroupId(f"p{problem_index:03d}")


def toy_condition_key(condition: ToyCondition) -> str:
    return "|".join([condition["wording"], *condition["level_by_setting"].values()])


def toy_arm(problem_index: int, condition: ToyCondition) -> Arm:
    return Arm(
        arm_id=ArmId(f"{toy_group_id(problem_index)}|{toy_condition_key(condition)}"),
        group_id=toy_group_id(problem_index),
        gate_value_by_setting={WORDING: condition["wording"]},
    )


def toy_problem_and_condition(world: ToyWorld, arm: Arm) -> tuple[int, ToyCondition]:
    group, wording, *levels = arm["arm_id"].split("|")
    return int(group.removeprefix("p")), ToyCondition(
        wording=wording, level_by_setting=dict(zip(world["setting_levels_by_name"], levels, strict=True))
    )


def true_term_probability(
    world: ToyWorld, truth: ToyTruth, problem_index: int, condition: ToyCondition
) -> Float[np.ndarray, " term"]:
    off_target = float(condition["wording"] != WORDING_VALUES[0])
    benchmark_index = truth["problem_benchmark_index"][problem_index]
    return expit(
        np.array(
            [
                term["intercept_by_benchmark"][benchmark_index]
                + truth["problem_effect"][term_position, problem_index]
                + off_target
                * (term["wording_effect"] + truth["wording_problem_deviation"][term_position, problem_index])
                + sum(
                    term["effect_by_level_by_setting"].get(setting_name, {}).get(level, 0.0)
                    for setting_name, level in condition["level_by_setting"].items()
                )
                for term_position, term in enumerate(world["term_truths"])
            ]
        )
    )
