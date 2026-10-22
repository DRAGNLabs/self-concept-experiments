import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, TypedDict

from selfconcept.arm_search.ladder_candidates import CandidateLimits
from selfconcept.arm_search.ladder_model import LadderPriors, LadderSamplerSettings
from selfconcept.arm_search.ladder_spec import ladder_tiers
from selfconcept.arm_search.manager import run_search
from selfconcept.arm_search.planning import in_process_planner
from selfconcept.arm_search.records import SearchConfig, StopConditions
from selfconcept.arm_search.toy.application import ToyApplication
from selfconcept.arm_search.toy.baselines import Planner, budget_limited, random_planner
from selfconcept.arm_search.toy.truth import TOY_LADDER_SPEC, ToyWorld

logger = logging.getLogger(__name__)

Policy = Literal["tiered", "target_only", "random"]
POLICIES: list[Policy] = ["tiered", "target_only", "random"]


TOY_LADDER: Final = ladder_tiers(TOY_LADDER_SPEC)


class ToySearchSettings(TypedDict):
    seeds_per_unit: int
    units_per_batch: int
    min_expected_hits: float
    reported_arm_count: int
    stop_conditions: StopConditions


@dataclass(frozen=True)
class ToyExperimentConfig:
    output_dir: Path
    replicate_count: int
    world: ToyWorld
    priors: LadderPriors
    sampler_settings: LadderSamplerSettings
    candidate_limits: CandidateLimits
    max_r_hat: float
    max_divergences: int
    search: ToySearchSettings
    cost_budget: float
    reported_cost_checkpoints: list[float]


def replicate_world(experiment: ToyExperimentConfig, replicate_index: int) -> ToyWorld:
    return {**experiment.world, "truth_seed": experiment.world["truth_seed"] + replicate_index}


def policy_state_dir(experiment: ToyExperimentConfig, replicate_index: int, policy: Policy) -> Path:
    return experiment.output_dir.expanduser() / f"replicate_{replicate_index:02d}" / policy


def policy_search_config(experiment: ToyExperimentConfig, replicate_index: int, policy: Policy) -> SearchConfig:
    """Only the tiered policy sees the lenient tiers; the others aim at the target directly."""
    search = experiment.search
    return SearchConfig(
        state_dir=policy_state_dir(experiment, replicate_index, policy),
        ladder=TOY_LADDER if policy == "tiered" else TOY_LADDER[-1:],
        seeds_per_unit=search["seeds_per_unit"],
        units_per_batch=search["units_per_batch"],
        min_expected_hits=search["min_expected_hits"],
        reported_arm_count=search["reported_arm_count"],
        random_seed=replicate_index,
        stop_conditions=search["stop_conditions"],
    )


def policy_planner(application: ToyApplication, experiment: ToyExperimentConfig, policy: Policy) -> Planner:
    planner_by_policy: dict[Policy, Planner] = {
        "tiered": in_process_planner(application),
        "target_only": in_process_planner(application),
        "random": random_planner(application.world),
    }
    return budget_limited(planner_by_policy[policy], experiment.cost_budget)


def run_replicate(experiment: ToyExperimentConfig, replicate_index: int, policy: Policy | None = None) -> None:
    """Run one policy (default: every policy) on one replicate world; reruns resume from the policy's state dir."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy_logger_name in ["pymc", "pytensor"]:
        logging.getLogger(noisy_logger_name).setLevel(logging.WARNING)
    application = ToyApplication(
        world=replicate_world(experiment, replicate_index),
        priors=experiment.priors,
        sampler_settings={**experiment.sampler_settings, "random_seed": replicate_index},
        candidate_limits=experiment.candidate_limits,
        max_r_hat=experiment.max_r_hat,
        max_divergences=experiment.max_divergences,
    )
    run_policies: list[Policy] = POLICIES if policy is None else [policy]
    for run_policy in run_policies:
        summary = run_search(
            application,
            policy_search_config(experiment, replicate_index, run_policy),
            policy_planner(application, experiment, run_policy),
        )
        logger.info(
            "replicate %d %s: reward by tier %s for cost %.0f in %d rounds (%s)",
            replicate_index,
            run_policy,
            summary["total_reward_by_tier"],
            summary["total_cost"],
            len(summary["rounds"]),
            summary["detail"],
        )
