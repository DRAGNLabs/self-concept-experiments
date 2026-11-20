"""Schema of token_entropy.py's per-model summary JSON, which episode_length_chart.py reads."""

from typing import Literal, TypedDict

from selfconcept.deceiver_axis.behaviour_measures.episodes import ArmName

type Measure = Literal["reasoning_window", "reply_window", "reasoning_all", "reasoning_tokens"]
MEASURES: tuple[Measure, ...] = ("reasoning_window", "reply_window", "reasoning_all", "reasoning_tokens")


class SteeredValue(TypedDict):
    value: float
    change: float
    ci95: list[float]


class GapValue(TypedDict):
    value: float
    ci95: list[float]


class MeasureSummary(TypedDict):
    unsteered: float
    gi_v2: SteeredValue
    gfar: SteeredValue
    gap: GapValue
    n_episodes: dict[ArmName, int]


type SummaryByModel = dict[str, dict[Measure, MeasureSummary]]
