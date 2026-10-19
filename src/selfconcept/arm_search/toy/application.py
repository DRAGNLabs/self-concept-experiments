from dataclasses import dataclass

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
    GroupId,
    Tier,
)
from selfconcept.arm_search.ladder_candidates import CandidateLimits, tier_candidate_posterior
from selfconcept.arm_search.ladder_draws import LadderDraws, fit_diagnostics, ladder_draws, ladder_effect_estimates
from selfconcept.arm_search.ladder_model import LadderPriors, LadderSamplerSettings, sample_ladder_model
from selfconcept.arm_search.ladder_spec import tier_term_count
from selfconcept.arm_search.toy.cells import toy_condition_table, toy_ladder_cells
from selfconcept.arm_search.toy.simulation import simulate_toy_episodes
from selfconcept.arm_search.toy.truth import (
    TOY_LADDER_SPEC,
    WORDING,
    ToyWorld,
    draw_truth,
    toy_arm,
    toy_conditions,
    toy_group_id,
)


def _tier_condition_indices(world: ToyWorld, tier: Tier) -> Int[np.ndarray, " condition"]:
    allowed_wordings = tier["gate_values_by_setting"].get(WORDING)
    return np.array(
        [
            condition_index
            for condition_index, condition in enumerate(toy_conditions(world))
            if allowed_wordings is None or condition["wording"] in allowed_wordings
        ]
    )


def toy_tier_candidates(
    world: ToyWorld, draws: LadderDraws, tier: Tier, target_covered_group_ids: set[GroupId], limits: CandidateLimits
) -> CandidatePosterior:
    eligible_group_indices = np.array(
        [
            problem_index
            for problem_index in range(world["problem_count"])
            if toy_group_id(problem_index) not in target_covered_group_ids
        ],
        dtype=int,
    )
    conditions = toy_conditions(world)
    return tier_candidate_posterior(
        draws,
        tier_term_count(TOY_LADDER_SPEC, tier),
        toy_condition_table(world),
        _tier_condition_indices(world, tier),
        eligible_group_indices,
        limits,
        lambda problem_index, condition_index: toy_arm(problem_index, conditions[condition_index]),
    )


@dataclass(frozen=True)
class ToyApplication:
    """A SearchApplication over a simulated world, fit by the ladder model (correctly specified up to priors).

    `max_r_hat` and `max_divergences` only raise fit warnings; the search continues from a suspect fit.
    """

    world: ToyWorld
    priors: LadderPriors
    sampler_settings: LadderSamplerSettings
    candidate_limits: CandidateLimits
    max_r_hat: float
    max_divergences: int

    def fit(self, episodes: list[EpisodeResult], ladder: list[Tier]) -> FitOutcome:
        cells = toy_ladder_cells(self.world, draw_truth(self.world)["problem_benchmark_index"], episodes)
        trace = sample_ladder_model(cells, self.priors, self.sampler_settings)
        diagnostics = fit_diagnostics(trace, self.max_r_hat, self.max_divergences)
        draws = ladder_draws(trace, cells)
        target_covered_group_ids = covered_group_ids(episodes, ladder[-1])
        candidate_posterior_by_tier = {
            tier["tier_name"]: toy_tier_candidates(
                self.world, draws, tier, target_covered_group_ids, self.candidate_limits
            )
            for tier in ladder
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

    def run_batch(self, units: list[BatchUnit], round_index: int) -> list[EpisodeResult]:
        return simulate_toy_episodes(self.world, draw_truth(self.world), units)
