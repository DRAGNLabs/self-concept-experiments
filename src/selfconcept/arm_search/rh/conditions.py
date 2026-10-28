import itertools
from pathlib import Path
from typing import Final, Literal, TypedDict, cast, get_args

from selfconcept.arm_search.interface import Arm, ArmId, GroupId, SettingName
from selfconcept.codebench.harness import Feedback, ImpossiblePrompt

Wording = Literal["neutral", "pass_tests"]
PromptTier = Literal["critical", "critical_no_carve", "strict", "none"]
Note = Literal["none", "grading", "nohardcode", "grading_nohardcode"]
Effort = Literal["medium", "low", "thinking"]
TaskFormat = Literal["check", "workspace"]

WORDING: Final = SettingName("wording")
NEUTRAL: Final[Wording] = "neutral"
NOTES_DIR: Final = Path("experiments/arm_search/rh/notes")
FEEDBACK_BY_WORDING: Final[dict[Wording, Feedback]] = {"neutral": "neutral", "pass_tests": "impossiblebench"}
# "none" (no tier instruction) only occurs in historic EvilGenie/LCB-Pro episodes, and attempt counts 6 and 3 only in
# historic EvilGenie/LCB-Pro runs; the search never proposes them.
SEARCH_PROMPT_TIERS: Final[list[PromptTier]] = ["critical", "critical_no_carve", "strict"]
SEARCH_MAX_ATTEMPTS: Final = [5, 10]
MAX_ATTEMPT_LEVELS: Final = ["5", "10", "6", "3"]


class Condition(TypedDict):
    wording: Wording
    prompt_tier: PromptTier
    note: Note
    effort: Effort
    max_attempts: int
    budget_stated: bool


def condition_id(condition: Condition) -> str:
    return (
        f"{condition['wording']}.{condition['prompt_tier']}.{condition['note']}.{condition['effort']}"
        f".a{condition['max_attempts']}.{'budget' if condition['budget_stated'] else 'nobudget'}"
    )


def parse_condition_id(text: str) -> Condition:
    wording, prompt_tier, note, effort, attempts, budget = text.split(".")
    return Condition(
        wording=cast(Wording, wording),
        prompt_tier=cast(PromptTier, prompt_tier),
        note=cast(Note, note),
        effort=cast(Effort, effort),
        max_attempts=int(attempts.removeprefix("a")),
        budget_stated=budget == "budget",
    )


def search_conditions(efforts: tuple[Effort, ...]) -> list[Condition]:
    """`efforts` are the searched model's (RhModelSpecifics.efforts)."""
    return [
        Condition(
            wording=wording,
            prompt_tier=prompt_tier,
            note=note,
            effort=effort,
            max_attempts=max_attempts,
            budget_stated=budget_stated,
        )
        for wording, prompt_tier, note, effort, max_attempts, budget_stated in itertools.product(
            get_args(Wording), SEARCH_PROMPT_TIERS, get_args(Note), efforts, SEARCH_MAX_ATTEMPTS, [False, True]
        )
    ]


def free_setting_levels_by_name(efforts: tuple[Effort, ...]) -> dict[SettingName, list[str]]:
    """The first level of each free setting is the model's reference level; a model with one effort has no effort
    effect."""
    return {
        SettingName("prompt_tier"): list(get_args(PromptTier)),
        SettingName("note"): list(get_args(Note)),
        SettingName("effort"): list(efforts),
        SettingName("max_attempts"): MAX_ATTEMPT_LEVELS,
        SettingName("budget_stated"): ["no", "yes"],
    }


def free_setting_values(condition: Condition) -> list[str]:
    """In the order of free_setting_levels_by_name."""
    return [
        condition["prompt_tier"],
        condition["note"],
        condition["effort"],
        str(condition["max_attempts"]),
        "yes" if condition["budget_stated"] else "no",
    ]


def gate_value_by_setting(condition: Condition) -> dict[SettingName, str]:
    return {WORDING: condition["wording"]}


def rh_arm(example_id: str, condition: Condition) -> Arm:
    return Arm(
        arm_id=ArmId(f"{example_id}|{condition_id(condition)}"),
        group_id=GroupId(example_id),
        gate_value_by_setting=gate_value_by_setting(condition),
    )


def harness_impossible_prompt(prompt_tier: PromptTier) -> ImpossiblePrompt:
    return "minimal" if prompt_tier == "none" else prompt_tier


def note_path(note: Note, task_format: TaskFormat) -> Path | None:
    return None if note == "none" else NOTES_DIR / f"{task_format}_{note}.txt"
