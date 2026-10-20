"""Fit health and calibration of the RH ladder model on a search's episodes so far (e.g. a model's calibration round),
before the search plans from them.

    python -m selfconcept.arm_search.rh.model_check --search <search yaml> <report md>

Fits the model to every episode in the search's state dir and reports the fit diagnostics, the effect estimates and the
in-sample calibration per tier and benchmark (observed against predicted hits).
"""

import logging
from pathlib import Path

import numpy as np
from jaxtyping import Float

from selfconcept.arm_search.episode_tally import is_eligible, is_hit
from selfconcept.arm_search.interface import EpisodeResult, Tier
from selfconcept.arm_search.ladder_draws import (
    ArmFeatures,
    LadderDraws,
    fit_diagnostics,
    hit_probability,
    ladder_draws,
    ladder_effect_estimates,
)
from selfconcept.arm_search.ladder_model import sample_ladder_model
from selfconcept.arm_search.ladder_spec import tier_term_count
from selfconcept.arm_search.rh.cells import model_conditions, rh_condition_table, rh_ladder_cells
from selfconcept.arm_search.rh.cli import RH_LADDER, RhSearchConfig, pool_problems
from selfconcept.arm_search.rh.conditions import Effort, condition_id
from selfconcept.arm_search.rh.episodes import RH_LADDER_SPEC
from selfconcept.arm_search.rh.model_specifics import rh_model_specifics
from selfconcept.arm_search.rh.pool import BENCHMARKS, PoolProblem
from selfconcept.arm_search.state import load_episodes

logger = logging.getLogger(__name__)


def _episode_hit_probability(
    problems: list[PoolProblem],
    episodes: list[EpisodeResult],
    scored: list[EpisodeResult],
    tier: Tier,
    draws: LadderDraws,
    efforts: tuple[Effort, ...],
) -> Float[np.ndarray, " episode"]:
    """Posterior-mean probability of a hit at the tier for each scored episode's arm."""
    conditions = model_conditions(episodes, efforts)
    condition_index_by_id = {condition_id(condition): index for index, condition in enumerate(conditions)}
    table = rh_condition_table(conditions, episodes, efforts)
    problem_index_by_example_id = {problem["example_id"]: index for index, problem in enumerate(problems)}
    example_and_condition = [episode["arm"]["arm_id"].split("|") for episode in scored]
    condition_indices = np.array([condition_index_by_id[cid] for _, cid in example_and_condition], dtype=int)
    features = ArmFeatures(
        group_index=np.array(
            [problem_index_by_example_id[example_id] for example_id, _ in example_and_condition], dtype=int
        ),
        off_target_gate=table["off_target_gate"][condition_indices],
        setting_codes=table["setting_codes"][condition_indices],
    )
    return hit_probability(draws, tier_term_count(RH_LADDER_SPEC, tier), features).mean(axis=0)


def _calibration_lines(
    problems: list[PoolProblem], episodes: list[EpisodeResult], draws: LadderDraws, efforts: tuple[Effort, ...]
) -> list[str]:
    benchmark_by_example_id = {problem["example_id"]: problem["benchmark"] for problem in problems}
    lines = ["| tier | benchmark | episodes | observed hits | predicted hits |", "|---|---|---|---|---|"]
    for tier in RH_LADDER:
        eligible = [episode for episode in episodes if is_eligible(episode["arm"], tier)]
        predicted = _episode_hit_probability(problems, episodes, eligible, tier, draws, efforts)
        for benchmark in BENCHMARKS:
            in_benchmark = np.array(
                [benchmark_by_example_id[e["arm"]["group_id"]] == benchmark for e in eligible], dtype=bool
            )
            observed = sum(is_hit(e, tier) for e, keep in zip(eligible, in_benchmark, strict=True) if keep)
            lines.append(
                f"| {tier['tier_name']} | {benchmark} | {int(in_benchmark.sum())} | {observed} "
                f"| {predicted[in_benchmark].sum():.1f} |"
            )
    return lines


def model_check(search: RhSearchConfig, output_path: Path) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    problems = pool_problems(search)
    pool_example_ids = {problem["example_id"] for problem in problems}
    episodes = [
        episode
        for episode in load_episodes(search.state_dir.expanduser())
        if episode["arm"]["group_id"] in pool_example_ids
    ]
    efforts = rh_model_specifics(search.model).efforts
    cells = rh_ladder_cells(problems, model_conditions(episodes, efforts), episodes, efforts)
    trace = sample_ladder_model(cells, search.priors, search.sampler_settings)
    diagnostics = fit_diagnostics(trace, search.max_r_hat, search.max_divergences)
    lines = [
        f"# RH ladder model check: {search.model}, {search.state_dir}",
        "",
        f"{len(episodes)} episodes; diagnostics {diagnostics['diagnostic_value_by_name']}; "
        f"warnings {diagnostics['warnings'] or 'none'}",
        "",
        "## Effect estimates (mean, 90% interval)",
        "",
        "| effect | level | mean | 5% | 95% |",
        "|---|---|---|---|---|",
        *(
            f"| {e['effect_name']} | {e['level_name']} | {e['mean']:.2f} | {e['lower']:.2f} | {e['upper']:.2f} |"
            for e in ladder_effect_estimates(trace, cells)
        ),
        "",
        "## Calibration (in-sample)",
        "",
        *_calibration_lines(problems, episodes, ladder_draws(trace, cells), efforts),
    ]
    output_path.expanduser().parent.mkdir(parents=True, exist_ok=True)
    output_path.expanduser().write_text("\n".join(lines) + "\n")
    logger.info("wrote %s", output_path.expanduser())


if __name__ == "__main__":
    from jsonargparse import auto_cli

    auto_cli(model_check)
