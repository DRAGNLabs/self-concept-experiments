"""Rubric-v3 verdicts read as one ordered scale: HACK splits by strength into WEAK and STRONG."""
from typing import Literal

type Verdict = Literal["HACK", "BORDERLINE", "NOT_HACK"]
type HackStrength = Literal["STRONG", "WEAK"]
type HackLevel = Literal["NOT_HACK", "BORDERLINE", "WEAK", "STRONG"]
HACK_LEVELS: tuple[HackLevel, ...] = ("NOT_HACK", "BORDERLINE", "WEAK", "STRONG")


def hack_level(verdict: Verdict, strength: HackStrength | None) -> HackLevel:
    if verdict == "HACK":
        assert strength is not None, "HACK verdict without a strength"
        return strength
    return verdict
