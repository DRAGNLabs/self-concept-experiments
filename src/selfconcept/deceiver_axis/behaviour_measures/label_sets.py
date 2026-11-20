"""Schemas of the blind label sets: what each sampler writes to key.json per item, and what the labelers write back."""

from typing import Literal, TypedDict

from selfconcept.deceiver_axis.behaviour_measures.blind_labels import LabelRow

# --- abstention / admission set (abstention_admission_sample.py, abstention_admission_tally.py) ---

type Pool = Literal["abstention", "hack"]
type Role = Literal["pool", "control"]
type AbstentionAdmissionLabelField = Literal["abstained", "stuck_to_previous", "admitted_hack", "tests_incorrect"]
type AbstentionAdmissionQuoteField = Literal["abstained_quote", "stuck_quote", "admitted_quote", "tests_incorrect_quote"]


class ShownTurn(TypedDict):
    turn: int
    text: str


class AbstentionAdmissionItem(TypedDict):
    """One entry of items.json: the turns shown to the labeler."""
    item_id: str
    n_turns: int
    shown_turns: list[ShownTurn]


class AbstentionAdmissionKey(TypedDict):
    episode_key: str
    arm: str
    pool: Pool
    role: Role
    verdict: str | None
    hack_turn: int | None


class AbstentionAdmissionLabel(LabelRow):
    abstained: str
    abstained_quote: str
    stuck_to_previous: str
    stuck_quote: str
    admitted_hack: str
    admitted_quote: str
    tests_incorrect: str
    tests_incorrect_quote: str
    themes: list[str]
    note: str
