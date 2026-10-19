"""Reward-hack arm search on one model (`model` in the config: gpt-oss, Qwen3 or Gemma 4).

    python -m selfconcept.arm_search.rh.cli manage --search experiments/arm_search/rh/search.yaml   # login node
    python -m selfconcept.arm_search.rh.cli plan --search <same>   # one round's fit and plan, run by manage in sbatch

`manage` alternates planning (a CPU job per round) and batches (generation and judge jobs) until a stop condition,
and resumes from the state dir after any interruption.
"""

import logging
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from selfconcept.arm_search.ladder_candidates import CandidateLimits
from selfconcept.arm_search.ladder_model import LadderPriors, LadderSamplerSettings
from selfconcept.arm_search.ladder_spec import ladder_tiers
from selfconcept.arm_search.manager import run_search
from selfconcept.arm_search.planning import plan_round
from selfconcept.arm_search.records import PlanningOutcome, SearchConfig, SearchStopped, StopConditions
from selfconcept.arm_search.rh.application import RhApplication
from selfconcept.arm_search.rh.episodes import RH_LADDER_SPEC
from selfconcept.arm_search.rh.model_specifics import rh_model_specifics
from selfconcept.arm_search.rh.pool import load_pool
from selfconcept.arm_search.rh.runner import RunnerSettings
from selfconcept.arm_search.state import load_episodes, load_planning_outcome, load_rounds

logger = logging.getLogger(__name__)

PLAN_SBATCH: Final = Path("experiments/arm_search/slurm/rh_plan.sbatch")


RH_LADDER: Final = ladder_tiers(RH_LADDER_SPEC)


@dataclass(frozen=True)
class RhSearchConfig:
    """`config_path` is this file, handed to each round's plan job; `model` is the searched model's HF id;
    `max_episodes` and `max_rounds` cap the whole search; `example_ids` limits the pool (dry runs)."""

    config_path: Path
    model: str
    state_dir: Path
    priors: LadderPriors
    sampler_settings: LadderSamplerSettings
    candidate_limits: CandidateLimits
    max_r_hat: float
    max_divergences: int
    seeds_per_unit: int
    units_per_batch: int
    min_expected_hits: float
    reported_arm_count: int
    random_seed: int
    stop_conditions: StopConditions
    max_episodes: int | None
    max_rounds: int | None
    runner: RunnerSettings
    example_ids: list[str] | None = None


def _application(config: RhSearchConfig) -> RhApplication:
    return RhApplication(
        state_dir=config.state_dir.expanduser(),
        model=rh_model_specifics(config.model),
        problems=[
            problem
            for problem in load_pool()
            if config.example_ids is None or problem["example_id"] in config.example_ids
        ],
        priors=config.priors,
        sampler_settings=config.sampler_settings,
        candidate_limits=config.candidate_limits,
        max_r_hat=config.max_r_hat,
        max_divergences=config.max_divergences,
        runner_settings=config.runner,
    )


def _search_config(config: RhSearchConfig) -> SearchConfig:
    return SearchConfig(
        state_dir=config.state_dir.expanduser(),
        ladder=RH_LADDER,
        seeds_per_unit=config.seeds_per_unit,
        units_per_batch=config.units_per_batch,
        min_expected_hits=config.min_expected_hits,
        reported_arm_count=config.reported_arm_count,
        random_seed=config.random_seed,
        stop_conditions=config.stop_conditions,
    )


def _setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy_logger_name in ["pymc", "pytensor"]:
        logging.getLogger(noisy_logger_name).setLevel(logging.WARNING)


def sbatch_planner(config: RhSearchConfig) -> Callable[[SearchConfig], PlanningOutcome]:
    """Plans each round in a CPU job; stops once the search has max_episodes episodes or max_rounds rounds."""

    def plan_in_job(search_config: SearchConfig) -> PlanningOutcome:
        episode_count = len(load_episodes(search_config["state_dir"]))
        round_count = sum(record["result"] is not None for record in load_rounds(search_config["state_dir"]))
        count_cap_and_units = [
            (episode_count, config.max_episodes, "episodes"),
            (round_count, config.max_rounds, "rounds"),
        ]
        for count, cap, unit in count_cap_and_units:
            if cap is not None and count >= cap:
                return SearchStopped(
                    kind="search_stopped",
                    stop_reason="below_value_threshold",
                    detail=f"{count} {unit} reached the cap of {cap}",
                    fit_diagnostic_value_by_name={},
                )
        subprocess.run(
            ["sbatch", "--wait", str(PLAN_SBATCH), str(config.config_path)], cwd=config.runner["repo_root"], check=True
        )
        return load_planning_outcome(search_config["state_dir"])

    return plan_in_job


def manage(search: RhSearchConfig) -> None:
    _setup_logging()
    summary = run_search(_application(search), _search_config(search), sbatch_planner(search))
    logger.info(
        "stopped (%s: %s) after %d rounds, reward by tier %s",
        summary["stop_reason"],
        summary["detail"],
        len(summary["rounds"]),
        summary["total_reward_by_tier"],
    )


def plan(search: RhSearchConfig) -> None:
    _setup_logging()
    plan_round(_application(search), _search_config(search))


if __name__ == "__main__":
    from jsonargparse import auto_cli

    auto_cli([manage, plan])
