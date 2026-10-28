import logging
from collections import Counter

import numpy as np

from selfconcept.arm_search.episode_tally import covered_group_ids
from selfconcept.arm_search.interface import EpisodeResult, GroupId
from selfconcept.arm_search.records import RoundRecord
from selfconcept.arm_search.state import load_episodes, load_rounds
from selfconcept.arm_search.toy.experiment import (
    POLICIES,
    TOY_LADDER,
    Policy,
    ToyExperimentConfig,
    policy_state_dir,
    replicate_world,
)
from selfconcept.arm_search.toy.experiment_metrics import (
    CoveragePoint,
    best_target_probability_by_group,
    covered_at_cost,
    coverage_curve,
    first_hit_cost,
    played_tier_new_hits,
)

logger = logging.getLogger(__name__)
TARGET = TOY_LADDER[-1]["tier_name"]
HACKABLE_THRESHOLDS = [0.05, 0.01]


def _mean_sd(values: list[float]) -> str:
    return f"{np.mean(values):.1f} ± {np.std(values):.1f}" if values else "–"


def _coverage_lines(
    experiment: ToyExperimentConfig, curves_by_policy: dict[Policy, list[list[CoveragePoint]]]
) -> list[str]:
    checkpoints = experiment.reported_cost_checkpoints
    lines = [
        f"## Target groups covered by cost (mean ± sd over {experiment.replicate_count} replicates)",
        "",
        "| policy | "
        + " | ".join(f"cost ≤ {checkpoint:g}" for checkpoint in checkpoints)
        + " | first target hit cost |",
        "|---" * (len(checkpoints) + 2) + "|",
    ]
    for policy, curves in curves_by_policy.items():
        first_costs = [first_hit_cost(curve, TARGET) for curve in curves]
        found_costs = [cost for cost in first_costs if cost is not None]
        first_cell = (
            f"median {np.median(found_costs):.0f} ({len(found_costs)}/{len(curves)} found)" if found_costs else "never"
        )
        cells = [
            _mean_sd([float(covered_at_cost(curve, TARGET, checkpoint)) for curve in curves])
            for checkpoint in checkpoints
        ]
        lines.append(f"| {policy} | " + " | ".join(cells) + f" | {first_cell} |")
    return lines


def _tier_coverage_lines(curves_by_policy: dict[Policy, list[list[CoveragePoint]]]) -> list[str]:
    lines = [
        "",
        "## Groups covered at each tier, end of budget",
        "",
        "| policy | " + " | ".join(tier["tier_name"] for tier in TOY_LADDER) + " |",
        "|---" * (len(TOY_LADDER) + 1) + "|",
    ]
    for policy, curves in curves_by_policy.items():
        cells = [
            _mean_sd(
                [float(curve[-1]["covered_count_by_tier"][tier["tier_name"]]) if curve else 0.0 for curve in curves]
            )
            for tier in TOY_LADDER
        ]
        lines.append(f"| {policy} | " + " | ".join(cells) + " |")
    return lines


def _hackable_lines(
    experiment: ToyExperimentConfig, target_covered_by_policy_by_replicate: list[dict[Policy, set[GroupId]]]
) -> list[str]:
    lines = [
        "",
        "## Hackable groups found (true best per-seed target probability above a threshold)",
        "",
        "| threshold | hackable per world | " + " | ".join(POLICIES) + " |",
        "|---" * (len(POLICIES) + 2) + "|",
    ]
    best_by_replicate = [
        best_target_probability_by_group(replicate_world(experiment, replicate_index))
        for replicate_index in range(experiment.replicate_count)
    ]
    for threshold in HACKABLE_THRESHOLDS:
        hackable_by_replicate = [
            {group_id for group_id, probability in best.items() if probability > threshold}
            for best in best_by_replicate
        ]
        hackable_count = sum(len(hackable) for hackable in hackable_by_replicate)
        hackable_and_covered = list(zip(hackable_by_replicate, target_covered_by_policy_by_replicate, strict=True))
        found_cells = [
            f"{sum(len(hackable & covered_by_policy[policy]) for hackable, covered_by_policy in hackable_and_covered)}"
            f"/{hackable_count}"
            for policy in POLICIES
        ]
        lines.append(
            f"| > {threshold:g} | {_mean_sd([float(len(hackable)) for hackable in hackable_by_replicate])} | "
            + " | ".join(found_cells)
            + " |"
        )
    return lines


def _round_lines(
    rounds_by_policy: dict[Policy, list[list[RoundRecord]]],
    episodes_by_replicate_by_policy: dict[Policy, list[list[EpisodeResult]]],
) -> list[str]:
    lines = [
        "",
        "## Rounds: tiers played, calibration at the played tier (groups not yet covered at the target that hit the"
        " played tier), fit health",
        "",
        "| policy | rounds per tier played | first round at target | Σ expected hits | Σ realized hits"
        " | rounds with fit warnings |",
        "|---|---|---|---|---|---|",
    ]
    for policy, rounds_by_replicate in rounds_by_policy.items():
        records = [record for rounds in rounds_by_replicate for record in rounds if record["result"] is not None]
        tier_counts = Counter(record["plan"]["tier_name"] for record in records)
        first_target_rounds = [
            next((record["plan"]["round_index"] for record in rounds if record["plan"]["tier_name"] == TARGET), None)
            for rounds in rounds_by_replicate
        ]
        expected = sum(
            record["plan"]["expected_hits_by_tier"].get(record["plan"]["tier_name"], 0.0) for record in records
        )
        realized = sum(
            sum(played_tier_new_hits(rounds, episodes, TOY_LADDER))
            for rounds, episodes in zip(rounds_by_replicate, episodes_by_replicate_by_policy[policy], strict=True)
        )
        warned = sum(bool(record["plan"]["fit_warnings"]) for record in records)
        tier_cell = ", ".join(
            f"{tier['tier_name']} {tier_counts[tier['tier_name']]}"
            for tier in TOY_LADDER
            if tier_counts[tier["tier_name"]]
        )
        first_cell = ", ".join("–" if round_index is None else str(round_index) for round_index in first_target_rounds)
        lines.append(
            f"| {policy} | {tier_cell} | {first_cell} | {expected:.1f} | {realized:.0f} | {warned}/{len(records)} |"
        )
    return lines


def _spread_recovery_lines(
    experiment: ToyExperimentConfig, rounds_by_policy: dict[Policy, list[list[RoundRecord]]]
) -> list[str]:
    lines = [
        "",
        "## Problem-spread recovery at the last fit (tiered policy; mean of posterior means over replicates)",
        "",
        "| term | truth | estimate |",
        "|---|---|---|",
    ]
    last_estimates = [rounds[-1]["plan"]["effect_estimates"] for rounds in rounds_by_policy["tiered"] if rounds]
    for term_position, term_name in enumerate(["cheats", "intent", "strong"]):
        means = [
            estimate["mean"]
            for estimates in last_estimates
            for estimate in estimates
            if estimate["effect_name"] == "group spread" and estimate["level_name"] == term_name
        ]
        truth = experiment.world["term_truths"][term_position]["problem_spread"]
        estimate_cell = f"{np.mean(means):.2f}" if means else "–"
        lines.append(f"| {term_name} | {truth:g} | {estimate_cell} |")
    return lines


def summarize_experiment(experiment: ToyExperimentConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    replicate_indices = range(experiment.replicate_count)
    episodes_by_replicate_by_policy: dict[Policy, list[list[EpisodeResult]]] = {
        policy: [load_episodes(policy_state_dir(experiment, index, policy)) for index in replicate_indices]
        for policy in POLICIES
    }
    curves_by_policy: dict[Policy, list[list[CoveragePoint]]] = {
        policy: [coverage_curve(episodes, TOY_LADDER) for episodes in episodes_by_replicate]
        for policy, episodes_by_replicate in episodes_by_replicate_by_policy.items()
    }
    rounds_by_policy: dict[Policy, list[list[RoundRecord]]] = {
        policy: [load_rounds(policy_state_dir(experiment, index, policy)) for index in replicate_indices]
        for policy in POLICIES
    }
    target_covered_by_policy_by_replicate: list[dict[Policy, set[GroupId]]] = [
        {
            policy: covered_group_ids(episodes_by_replicate_by_policy[policy][index], TOY_LADDER[-1])
            for policy in POLICIES
        }
        for index in replicate_indices
    ]
    report = "\n".join(
        [
            "# Toy arm search v2: tiered Thompson sampling",
            "",
            *_coverage_lines(experiment, curves_by_policy),
            *_tier_coverage_lines(curves_by_policy),
            *_hackable_lines(experiment, target_covered_by_policy_by_replicate),
            *_round_lines(rounds_by_policy, episodes_by_replicate_by_policy),
            *_spread_recovery_lines(experiment, rounds_by_policy),
            "",
        ]
    )
    report_path = experiment.output_dir.expanduser() / "report.md"
    report_path.write_text(report)
    logger.info("wrote %s", report_path)
