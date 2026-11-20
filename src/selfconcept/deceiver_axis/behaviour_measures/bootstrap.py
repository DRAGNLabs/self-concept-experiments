"""Problem-level bootstrap of rates and their changes (problems resampled, the same draw for every arm), and AUROCs."""

from typing import TypedDict

import numpy as np
from jaxtyping import Float, Int

from selfconcept.correlation.stratified import StratifiedExample, stratified_auroc


class ProblemCounts(TypedDict):
    hits: Int[np.ndarray, "problems"]
    totals: Int[np.ndarray, "problems"]


class Change(TypedDict):
    delta: float
    low: float
    high: float
    steered_rate: float
    unsteered_rate: float
    steered_n: int
    unsteered_n: int


class Gap(TypedDict):
    delta: float
    low: float
    high: float


def rate(counts: ProblemCounts) -> float:
    return float(counts["hits"].sum() / counts["totals"].sum())


def delta_draws(steered: ProblemCounts, unsteered: ProblemCounts,
                resampled_problems: Int[np.ndarray, "draws problems"]) -> Float[np.ndarray, "draws"]:
    """Steered minus unsteered rate per draw, in percentage points (nan when a draw has no pool)."""
    def rates(counts: ProblemCounts) -> Float[np.ndarray, "draws"]:
        totals = counts["totals"][resampled_problems].sum(axis=1)
        return np.where(totals > 0, counts["hits"][resampled_problems].sum(axis=1) / np.maximum(totals, 1), np.nan)

    return 100 * (rates(steered) - rates(unsteered))


def bootstrap_change(steered: ProblemCounts, unsteered: ProblemCounts,
                     resampled_problems: Int[np.ndarray, "draws problems"]) -> Change:
    low, high = np.nanpercentile(delta_draws(steered, unsteered, resampled_problems), [2.5, 97.5])
    return {"delta": 100 * (rate(steered) - rate(unsteered)), "low": float(low), "high": float(high),
            "steered_rate": rate(steered), "unsteered_rate": rate(unsteered),
            "steered_n": int(steered["totals"].sum()), "unsteered_n": int(unsteered["totals"].sum())}


def change_gap(gi_v2: ProblemCounts, gfar: ProblemCounts, unsteered: ProblemCounts,
               resampled_problems: Int[np.ndarray, "draws problems"]) -> Gap:
    """gi_v2 change minus gfar change, with its interval from the same draws."""
    low, high = np.nanpercentile(delta_draws(gi_v2, unsteered, resampled_problems)
                                 - delta_draws(gfar, unsteered, resampled_problems), [2.5, 97.5])
    delta = 100 * (rate(gi_v2) - rate(unsteered)) - 100 * (rate(gfar) - rate(unsteered))
    return {"delta": delta, "low": float(low), "high": float(high)}


def multinomial_problem_weights(n_problems: int, n_draws: int, rng: np.random.Generator
                                ) -> Int[np.ndarray, "draws problems"]:
    """How often each problem is drawn in each bootstrap resample."""
    return rng.multinomial(n_problems, np.full(n_problems, 1 / n_problems), size=n_draws)


def ci(values: Float[np.ndarray, "draws"]) -> list[float]:
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def within_problem_auroc(scores: list[float], is_positive: list[bool], problems: list[str]) -> float:
    """AUROC over same-problem (positive, negative) pairs only, ties counted half (nan without such a pair)."""
    examples = [StratifiedExample(score, 1 if positive else 0, problem)
                for score, positive, problem in zip(scores, is_positive, problems)]
    auroc = stratified_auroc(examples).auroc
    return np.nan if auroc is None else auroc


def pooled_auroc(scores: list[float], is_positive: list[bool]) -> float:
    return within_problem_auroc(scores, is_positive, [""] * len(scores))
