"""Schemas of the blind label sets: what each sampler writes to key.json per item, and what the labelers write back."""

from typing import Literal, TypedDict

from selfconcept.deceiver_axis.behaviour_measures.blind_labels import LabelRow
from selfconcept.deceiver_axis.behaviour_measures.episodes import ArmName

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


# --- explanation audit (episode_audit_sample.py, episode_audit_tally.py) ---

type AuditLabelField = Literal["impossible_action", "explanation_mismatch", "claims_correct_when_rejected",
                               "damaging_code"]
type AuditQuoteField = Literal["impossible_action_quote", "explanation_mismatch_quote",
                               "claims_correct_when_rejected_quote", "damaging_code_quote", "prose_quote", "code_quote"]


class AuditKey(TypedDict):
    model: str
    arm: ArmName
    arm_dir: str
    episode_key: str
    stop_reason: str


class AuditLabel(LabelRow, total=False):
    """Fields depend on the task: the four-label audit has each label with its own quote; the single-label
    explanation re-judge has explanation_mismatch with prose_quote and code_quote."""
    impossible_action: str
    impossible_action_quote: str
    explanation_mismatch: str
    explanation_mismatch_quote: str
    claims_correct_when_rejected: str
    claims_correct_when_rejected_quote: str
    damaging_code: str
    damaging_code_quote: str
    prose_quote: str
    code_quote: str
    themes: list[str]
    note: str
