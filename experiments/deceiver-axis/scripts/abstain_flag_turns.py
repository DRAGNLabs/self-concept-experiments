"""Turn-level abstain-flag rates (CPU): per arm of FLAG_ARMS_BY_SLUG, the share of turns whose peak first-token flag
score (reasoning and answer) is over the threshold that flags FLAG_RATE_ON_SUCCESS of the unsteered honest-success
EPISODES (the tally's "both" score), per turn class (abstain_flag.classify_turn; not_honest_pass pools honest_fail and
hack: turns that did not pass honestly). Per class also the length-normalized rate: tokens over the threshold per 10k
tokens of those turns (pooled ratio). Problem-bootstrap 95% intervals per arm and for CONTRASTS (problems resampled
jointly). Prints JSON (abstain_flag.TurnFlagSummary).

    python experiments/deceiver-axis/scripts/abstain_flag_turns.py --flags <dir> --slug gptoss
"""
import argparse
import json
from pathlib import Path
from typing import Literal, TypedDict

import numpy as np
from jaxtyping import Float, Int

from selfconcept.deceiver_axis.behaviour_measures.abstain_flag import (CONTRASTS, FLAG_ARMS_BY_SLUG,
                                                                       MEMBERS_BY_POOLED_CLASS, N_DRAWS, TURN_CLASSES,
                                                                       ClassSummary, TurnClass, TurnFlagRow,
                                                                       TurnFlagSummary, episode_rows, flag_files,
                                                                       success_threshold, turn_rows)
from selfconcept.deceiver_axis.behaviour_measures.bootstrap import ci, multinomial_problem_weights
from selfconcept.deceiver_axis.behaviour_measures.episodes import (PRESET_BY_SLUG, load_judge_details,
                                                                   load_verdict_by_key)


class ClassTotals(TypedDict):
    turns: Float[np.ndarray, "problems"]
    flagged: Float[np.ndarray, "problems"]
    tokens: Float[np.ndarray, "problems"]
    over: Float[np.ndarray, "problems"]


class ClassDraws(TypedDict):
    flag_rate: Float[np.ndarray, "draws"]
    over_per_10k_tokens: Float[np.ndarray, "draws"]


def class_totals(rows: list[TurnFlagRow], turn_class: TurnClass, index_by_problem: dict[str, int]) -> ClassTotals:
    members = MEMBERS_BY_POOLED_CLASS.get(turn_class, {turn_class})
    totals: ClassTotals = {field: np.zeros(len(index_by_problem))  # type: ignore[misc]
                           for field in ("turns", "flagged", "tokens", "over")}
    for row in rows:
        if row["turn_class"] in members:
            problem = index_by_problem[row["problem"]]
            totals["turns"][problem] += 1
            totals["flagged"][problem] += row["flagged"]
            totals["tokens"][problem] += row["tokens"]
            totals["over"][problem] += row["over"]
    return totals


def class_draws(totals: ClassTotals, weights: Int[np.ndarray, "draws problems"]) -> ClassDraws:
    return {"flag_rate": (weights @ totals["flagged"]) / np.maximum(weights @ totals["turns"], 1),
            "over_per_10k_tokens": 1e4 * (weights @ totals["over"]) / np.maximum(weights @ totals["tokens"], 1)}


def class_summary(totals: ClassTotals, draws: ClassDraws) -> ClassSummary:
    return {"turns": int(totals["turns"].sum()),
            "flag_rate": float(totals["flagged"].sum() / max(totals["turns"].sum(), 1)),
            "ci": ci(draws["flag_rate"]), "tokens": int(totals["tokens"].sum()),
            "over_per_10k_tokens": float(1e4 * totals["over"].sum() / max(totals["tokens"].sum(), 1)),
            "over_per_10k_tokens_ci": ci(draws["over_per_10k_tokens"])}


def contrast_intervals(draws_by_arm: dict[str, dict[TurnClass, ClassDraws]],
                       measure: Literal["flag_rate", "over_per_10k_tokens"]) -> dict[str, dict[TurnClass, list[float]]]:
    return {f"{a} - {b}": {turn_class: ci(draws_by_arm[a][turn_class][measure] - draws_by_arm[b][turn_class][measure])
                           for turn_class in TURN_CLASSES}
            for a, b in CONTRASTS if a in draws_by_arm and b in draws_by_arm}


def load_rows_by_arm(flag_dir: Path, slug: str, excluded_problem_ids: frozenset[str]
                     ) -> tuple[float, dict[str, list[TurnFlagRow]]]:
    """Excluded problems are left out of the threshold's honest-success episodes as well as the turns."""
    preset = PRESET_BY_SLUG[slug]
    verdict_by_key = load_verdict_by_key(preset["verdict_files"])
    detail_by_key = load_judge_details(preset, verdict_by_key)
    unsteered_episodes = [row for row in episode_rows(flag_files(flag_dir, "unsteered"), verdict_by_key)
                          if row["problem"] not in excluded_problem_ids]
    threshold = success_threshold(unsteered_episodes, "both")
    rows_by_arm = {label: [row for row in turn_rows(flag_files(flag_dir, stem), verdict_by_key, detail_by_key, threshold)
                           if row["problem"] not in excluded_problem_ids]
                   for label, stem in FLAG_ARMS_BY_SLUG[slug].items()}
    return threshold, {arm: rows for arm, rows in rows_by_arm.items() if rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--flags", type=Path, required=True)
    parser.add_argument("--slug", required=True, choices=list(FLAG_ARMS_BY_SLUG))
    parser.add_argument("--exclude-problems", nargs="+", default=[], help="example ids to leave out")
    args = parser.parse_args()
    threshold, rows_by_arm = load_rows_by_arm(args.flags, args.slug, frozenset(args.exclude_problems))
    problems = sorted({row["problem"] for rows in rows_by_arm.values() for row in rows})
    index_by_problem = {problem: i for i, problem in enumerate(problems)}
    weights = multinomial_problem_weights(len(problems), N_DRAWS, np.random.default_rng(0))
    draws_by_arm: dict[str, dict[TurnClass, ClassDraws]] = {}
    by_arm: dict[str, dict[TurnClass, ClassSummary]] = {}
    for arm, rows in rows_by_arm.items():
        totals_by_class: dict[TurnClass, ClassTotals] = {
            turn_class: class_totals(rows, turn_class, index_by_problem) for turn_class in TURN_CLASSES}
        draws_by_arm[arm] = {turn_class: class_draws(totals, weights) for turn_class, totals in totals_by_class.items()}
        by_arm[arm] = {turn_class: class_summary(totals, draws_by_arm[arm][turn_class])
                       for turn_class, totals in totals_by_class.items()}
    summary: TurnFlagSummary = {"threshold_first_token": threshold, "by_arm": by_arm,
                                "contrasts": contrast_intervals(draws_by_arm, "flag_rate"),
                                "contrasts_over_per_10k_tokens": contrast_intervals(draws_by_arm, "over_per_10k_tokens")}
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
