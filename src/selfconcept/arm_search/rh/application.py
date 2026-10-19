from dataclasses import dataclass
from pathlib import Path

import numpy as np
from jaxtyping import Int

from selfconcept.arm_search.episode_tally import covered_group_ids
from selfconcept.arm_search.interface import (
    BatchUnit,
    CandidatePosterior,
    EpisodeResult,
    FitFailed,
    FitOutcome,
    FitSucceeded,
    Tier,
)
from selfconcept.arm_search.ladder_candidates import CandidateLimits, tier_candidate_posterior
from selfconcept.arm_search.ladder_draws import fit_diagnostics, ladder_draws, ladder_effect_estimates
from selfconcept.arm_search.ladder_model import LadderPriors, LadderSamplerSettings, sample_ladder_model
from selfconcept.arm_search.ladder_spec import tier_term_count
from selfconcept.arm_search.rh.cells import model_conditions, rh_condition_table, rh_ladder_cells
from selfconcept.arm_search.rh.conditions import WORDING, Condition, Effort, condition_id, rh_arm, search_conditions
from selfconcept.arm_search.rh.model_specifics import RhModelSpecifics
from selfconcept.arm_search.rh.episodes import RH_LADDER_SPEC
from selfconcept.arm_search.rh.pool import PoolProblem
from selfconcept.arm_search.rh.runner import RunnerSettings, run_round


def _tier_condition_indices(
    conditions: list[Condition], tier: Tier, proposable_condition_ids: set[str]
) -> Int[np.ndarray, " condition"]:
    allowed_wordings = tier["gate_values_by_setting"].get(WORDING)
    return np.array(
        [
            index
            for index, condition in enumerate(conditions)
            if condition_id(condition) in proposable_condition_ids
            and (allowed_wordings is None or condition["wording"] in allowed_wordings)
        ],
        dtype=int,
    )


def _only_available(candidates: CandidatePosterior, available_arm_ids: set[str]) -> CandidatePosterior:
    keep = np.array([arm["arm_id"] in available_arm_ids for arm in candidates["arms"]], dtype=bool)
    return CandidatePosterior(
        arms=[arm for arm, is_kept in zip(candidates["arms"], keep, strict=True) if is_kept],
        hit_probability=candidates["hit_probability"][:, keep],
        expected_cost_per_seed=candidates["expected_cost_per_seed"][:, keep],
    )


def fit_rh_model(
    problems: list[PoolProblem],
    priors: LadderPriors,
    sampler_settings: LadderSamplerSettings,
    candidate_limits: CandidateLimits,
    max_r_hat: float,
    max_divergences: int,
    episodes: list[EpisodeResult],
    ladder: list[Tier],
    efforts: tuple[Effort, ...],
    proposable_condition_ids: set[str] | None = None,
    available_arm_ids: set[str] | None = None,
) -> FitOutcome:
    """Proposes only the search grid unless `proposable_condition_ids` says otherwise; `available_arm_ids` drops every
    other candidate (the replay, which can only run arms that have unused historic episodes)."""
    conditions = model_conditions(episodes, efforts)
    cells = rh_ladder_cells(problems, conditions, episodes, efforts)
    trace = sample_ladder_model(cells, priors, sampler_settings)
    diagnostics = fit_diagnostics(trace, max_r_hat, max_divergences)
    draws = ladder_draws(trace, cells)
    condition_table = rh_condition_table(conditions, episodes, efforts)
    target_covered = covered_group_ids(episodes, ladder[-1])
    eligible_group_indices = np.array(
        [index for index, problem in enumerate(problems) if problem["example_id"] not in target_covered], dtype=int
    )
    proposable = proposable_condition_ids or {condition_id(condition) for condition in search_conditions(efforts)}
    candidate_posterior_by_tier = {
        tier["tier_name"]: tier_candidate_posterior(
            draws,
            tier_term_count(RH_LADDER_SPEC, tier),
            condition_table,
            _tier_condition_indices(conditions, tier, proposable),
            eligible_group_indices,
            candidate_limits,
            lambda problem_index, condition_index: rh_arm(
                problems[problem_index]["example_id"], conditions[condition_index]
            ),
        )
        for tier in ladder
    }
    if available_arm_ids is not None:
        candidate_posterior_by_tier = {
            tier_name: _only_available(candidates, available_arm_ids)
            for tier_name, candidates in candidate_posterior_by_tier.items()
        }
    hit_probabilities = [candidates["hit_probability"] for candidates in candidate_posterior_by_tier.values()]
    if not all(np.isfinite(hit_probability).all() for hit_probability in hit_probabilities):
        return FitFailed(
            kind="fit_failed",
            reason="non-finite hit probabilities",
            diagnostic_value_by_name=diagnostics["diagnostic_value_by_name"],
        )
    return FitSucceeded(
        kind="fit_succeeded",
        candidate_posterior_by_tier=candidate_posterior_by_tier,
        diagnostic_value_by_name=diagnostics["diagnostic_value_by_name"],
        warnings=diagnostics["warnings"],
        effect_estimates=ladder_effect_estimates(trace, cells),
    )


@dataclass(frozen=True)
class RhApplication:
    """A SearchApplication over one model's reward hacking on the rh_hillclimb pool, fit by the ladder model.

    `max_r_hat` and `max_divergences` only raise fit warnings; the search continues from a suspect fit.
    """

    state_dir: Path
    model: RhModelSpecifics
    problems: list[PoolProblem]
    priors: LadderPriors
    sampler_settings: LadderSamplerSettings
    candidate_limits: CandidateLimits
    max_r_hat: float
    max_divergences: int
    runner_settings: RunnerSettings

    def fit(self, episodes: list[EpisodeResult], ladder: list[Tier]) -> FitOutcome:
        return fit_rh_model(
            self.problems,
            self.priors,
            self.sampler_settings,
            self.candidate_limits,
            self.max_r_hat,
            self.max_divergences,
            episodes,
            ladder,
            self.model.efforts,
        )

    def run_batch(self, units: list[BatchUnit], round_index: int) -> list[EpisodeResult]:
        problem_by_example_id = {problem["example_id"]: problem for problem in self.problems}
        return run_round(units, round_index, self.state_dir, problem_by_example_id, self.runner_settings, self.model)
