"""Tally abstain_flag_logprobs.py output (CPU): per arm and score (peak first-token score over reasoning, answer or both;
exact whole-flag logprob at the peak), the AUROC of the score for hidden-test failure (pooled and within-problem), and
the flag rate per outcome (HACK, honest success = passed hidden tests and not HACK, other) at a threshold set on the
first arm (unsteered) so FLAG_RATE_ON_SUCCESS of its honest successes are flagged, with problem-bootstrap intervals
for each arm and for the contrasts (paired on shared episodes when both arms ran on the same transcripts). Prints
JSON: a ScoreFlagSummary per score.

    python experiments/deceiver-axis/scripts/abstain_flag_tally.py --flags <dir> --slug muse
"""
import argparse
import json
from pathlib import Path

from typing import TypedDict

import numpy as np
from jaxtyping import Float

from selfconcept.deceiver_axis.behaviour_measures.abstain_flag import (CONTRASTS, FLAG_ARMS_BY_SLUG, N_DRAWS, OUTCOMES,
                                                                       EpisodeFlagRow, Outcome, Score, episode_rows,
                                                                       flag_files, success_threshold)
from selfconcept.deceiver_axis.behaviour_measures.bootstrap import (ci, multinomial_problem_weights, pooled_auroc,
                                                                    within_problem_auroc)
from selfconcept.deceiver_axis.behaviour_measures.episodes import PRESET_BY_SLUG, load_verdict_by_key

SCORES: tuple[Score, ...] = ("both", "reasoning", "answer", "exact")
type RateDraws = dict[str, dict[Outcome, Float[np.ndarray, "draws"]]]


class ArmFlagSummary(TypedDict):
    n: int
    auroc_fail_pooled: float
    auroc_fail_within_problem: float
    flag_rate_and_n: dict[Outcome, tuple[float, int]]
    median: dict[Outcome, float]
    flag_rate_ci: dict[Outcome, list[float]]


class ContrastSummary(TypedDict):
    episodes: list[int]
    flag_rate_difference_ci: dict[Outcome, list[float]]


class ScoreFlagSummary(TypedDict):
    threshold: float
    by_arm: dict[str, ArmFlagSummary]
    contrasts: dict[str, ContrastSummary]  # "<arm a> - <arm b>"


def flag_rate_draws(rows_by_arm: dict[str, list[EpisodeFlagRow]], score: Score, threshold: float,
                    rng: np.random.Generator) -> RateDraws:
    """Problem-bootstrap draws (N_DRAWS) of each arm's flag rate per outcome, problems resampled jointly over arms."""
    problems = sorted({row["problem"] for rows in rows_by_arm.values() for row in rows})
    index_by_problem = {problem: i for i, problem in enumerate(problems)}
    weights = multinomial_problem_weights(len(problems), N_DRAWS, rng)
    draws_by_arm: RateDraws = {}
    for arm, rows in rows_by_arm.items():
        draws_by_arm[arm] = {}
        for outcome in OUTCOMES:
            flagged, total = np.zeros(len(problems)), np.zeros(len(problems))
            for row in rows:
                if row["outcome"] == outcome:
                    total[index_by_problem[row["problem"]]] += 1
                    flagged[index_by_problem[row["problem"]]] += row[score] > threshold
            draws_by_arm[arm][outcome] = (weights @ flagged) / np.maximum(weights @ total, 1)
    return draws_by_arm


def arm_summary(rows: list[EpisodeFlagRow], score: Score, threshold: float,
                rate_draws: dict[Outcome, Float[np.ndarray, "draws"]]) -> ArmFlagSummary:
    scores = [row[score] for row in rows]
    fails = [row["fail"] for row in rows]
    rows_by_outcome: dict[Outcome, list[EpisodeFlagRow]] = {
        outcome: [row for row in rows if row["outcome"] == outcome] for outcome in OUTCOMES}
    return {"n": len(rows), "auroc_fail_pooled": pooled_auroc(scores, fails),
            "auroc_fail_within_problem": within_problem_auroc(scores, fails, [row["problem"] for row in rows]),
            "flag_rate_and_n": {outcome: (float(np.mean([row[score] > threshold for row in outcome_rows])),
                                          len(outcome_rows)) for outcome, outcome_rows in rows_by_outcome.items()},
            "median": {outcome: float(np.median([row[score] for row in outcome_rows] or [np.nan]))
                       for outcome, outcome_rows in rows_by_outcome.items()},
            "flag_rate_ci": {outcome: ci(rate_draws[outcome]) for outcome in OUTCOMES}}


def paired_on_shared_episodes(rows_a: list[EpisodeFlagRow], rows_b: list[EpisodeFlagRow]
                              ) -> tuple[list[EpisodeFlagRow], list[EpisodeFlagRow]]:
    """Both arms restricted to their shared episodes when they share any (same transcripts), else unchanged."""
    shared = {row["key"] for row in rows_a} & {row["key"] for row in rows_b}
    if not shared:
        return rows_a, rows_b
    return [row for row in rows_a if row["key"] in shared], [row for row in rows_b if row["key"] in shared]


def contrast_summaries(rows_by_arm: dict[str, list[EpisodeFlagRow]], score: Score, threshold: float,
                       rng: np.random.Generator) -> dict[str, ContrastSummary]:
    summary_by_contrast: dict[str, ContrastSummary] = {}
    for a, b in CONTRASTS:
        if a not in rows_by_arm or b not in rows_by_arm:
            continue
        rows_a, rows_b = paired_on_shared_episodes(rows_by_arm[a], rows_by_arm[b])
        draws = flag_rate_draws({a: rows_a, b: rows_b}, score, threshold, rng)
        summary_by_contrast[f"{a} - {b}"] = {
            "episodes": [len(rows_a), len(rows_b)],
            "flag_rate_difference_ci": {outcome: ci(draws[a][outcome] - draws[b][outcome]) for outcome in OUTCOMES}}
    return summary_by_contrast


def score_summary(rows_by_arm: dict[str, list[EpisodeFlagRow]], score: Score,
                  rng: np.random.Generator) -> ScoreFlagSummary:
    threshold = success_threshold(next(iter(rows_by_arm.values())), score)
    draws = flag_rate_draws(rows_by_arm, score, threshold, rng)
    return {"threshold": threshold,
            "by_arm": {arm: arm_summary(rows, score, threshold, draws[arm]) for arm, rows in rows_by_arm.items()},
            "contrasts": contrast_summaries(rows_by_arm, score, threshold, rng)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--flags", type=Path, required=True)
    parser.add_argument("--slug", required=True, choices=list(FLAG_ARMS_BY_SLUG))
    args = parser.parse_args()
    verdict_by_key = load_verdict_by_key(PRESET_BY_SLUG[args.slug]["verdict_files"])
    rows_by_arm = {label: episode_rows(flag_files(args.flags, stem), verdict_by_key)
                   for label, stem in FLAG_ARMS_BY_SLUG[args.slug].items()}
    rows_by_arm = {arm: rows for arm, rows in rows_by_arm.items() if rows}
    rng = np.random.default_rng(0)
    print(json.dumps({score: score_summary(rows_by_arm, score, rng) for score in SCORES}, indent=1))


if __name__ == "__main__":
    main()
