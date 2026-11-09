"""gi_v2's percentile among antonym concepts across models (PLAN "Cross-model percentile of gi_v2 among antonym
concepts"): q_cross = the share of concepts whose true equal-weight mean logit lift over the models is at least gi_v2's.

Concept j is the same antonym pair on every model (arm directory names match across models). Lift on model m:
d_{j,m} = elogit(hits_{j,m}) − elogit(unsteered_m); D_j = mean over models, with binomial noise variance
V = (1/M²) Σ_m 1 / (n_m p̄_m (1 − p̄_m)). Shape-free (NPMLE) and normal profile likelihood-ratio intervals for q, each
widened for gi_v2's own noise. Reported for all models and with each model left out, plus the pairwise Spearman of
lifts over concepts (descriptive).

Hierarchical version (`hierarchical_q`): d_{j,m} = b_m + μ_j + ε_{j,m} + noise with model-specific deviations
ε ~ N(0, τ²); q = the share of concepts whose shared effect μ_j is at least gi_v2's, so an effect on one model only
counts as deviation, not as shared effect.

    python scripts/controls/cross_model_percentile.py --models data/controls/cross_model_models.json --outcome proxy --out <json>

`--outcome judged` counts HACK verdicts; each model entry then needs "verdicts" (collect_verdicts.py output keyed
'<arm dir>/<seed dir>/<example_id>'; unjudged episodes count as not hacks).
"""
import argparse
import itertools
import json
import warnings
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal, NotRequired, TypedDict

import numpy as np
from jaxtyping import Float, Int
from scipy import stats

from control_percentile import LIKELIHOOD_RATIO_CUTOFF, shape_free_bound, shape_free_interval, unconstrained_npmle
from selfconcept.codebench.harness import CheckRecord
from selfconcept.codebench.hardcoded_cases import is_proxy_hack, load_check_records

# Support of a concept's true mean logit lift (shape-free estimator).
LIFT_GRID = np.linspace(-4, 4, 301)
# Normal family for the true lifts: means and sds (sd 0 = every concept identical).
NORMAL_MEANS = np.linspace(-2, 2, 401)
NORMAL_SDS = np.concatenate([[0.0], np.geomspace(1e-3, 3, 160)])
# Hierarchical model: model-specific deviation sds searched (0 = concepts act the same on every model), and a coarser
# normal family for the shared effects (the grid is repeated per deviation sd).
TAU_GRID = np.concatenate([[0.0], np.geomspace(0.02, 2.0, 20)])
HIERARCHICAL_NORMAL_MEANS = np.linspace(-2, 2, 201)
HIERARCHICAL_NORMAL_SDS = np.concatenate([[0.0], np.geomspace(1e-3, 3, 80)])
Z_95 = 1.959964
PERMUTATIONS = 10_000

Outcome = Literal["proxy", "judged"]
NormalGridEdge = Literal["lowest normal mean", "highest normal mean", "highest normal sd"]
TauGridEdge = Literal["highest model-specific sd"]


class ModelRuns(TypedDict):
    name: str
    controls_root: str
    controls_glob: str
    records_glob: str
    unsteered: str
    target: str
    verdicts: NotRequired[str]


class ArmCount(TypedDict):
    hits: int
    episodes: int


class ModelCounts(TypedDict):
    hits_by_concept: dict[str, int]
    episodes_per_concept: int
    unsteered: ArmCount
    target: ArmCount


class ModelLifts(TypedDict):
    lift_by_concept: Float[np.ndarray, "concepts"]
    target_lift: float
    control_variance: float
    target_variance: float


class ShapeFreeQInterval(TypedDict):
    low: float
    high: float


class QInterval(ShapeFreeQInterval):
    estimate: float


class NormalQInterval(QInterval):
    grid_edges: list[NormalGridEdge]


class CrossModelQ(TypedDict):
    models: list[str]
    target_mean_lift: float
    target_sd: float
    control_mean_lift: float
    control_noise_sd: float
    concepts_at_or_above_target: int
    expected_at_or_above_without_spread: float
    shape_free: ShapeFreeQInterval
    normal: NormalQInterval


class HierarchicalQ(TypedDict):
    models: list[str]
    model_specific_sd_estimate: float
    model_specific_sd_low: float
    model_specific_sd_high: float
    target_mean_lift: float
    target_heterogeneity_q: float
    target_heterogeneity_p: float
    target_model_specific_sd: float
    control_mean_lift: float
    shape_free: ShapeFreeQInterval
    normal: NormalQInterval
    tau_grid_edges: list[TauGridEdge]


class PairCorrelation(TypedDict):
    models: tuple[str, str]
    spearman: float


class LiftCorrelations(TypedDict):
    pairs: list[PairCorrelation]
    mean_spearman: float
    shuffle_p: float


class ModelSummary(TypedDict):
    episodes_per_concept: int
    unsteered_rate: float
    target_rate: float
    pooled_control_rate: float
    target_lift: float
    mean_control_lift: float


def empirical_logit(hits: Int[np.ndarray, "..."] | int, episodes: int) -> Float[np.ndarray, "..."]:
    return np.log((np.asarray(hits) + 0.5) / (episodes - np.asarray(hits) + 0.5))


def logit_variance(rate: float, episodes: int) -> float:
    return 1 / (episodes * rate * (1 - rate))


def model_lifts(counts: ModelCounts, concepts: Sequence[str]) -> ModelLifts:
    hits = np.array([counts["hits_by_concept"][concept] for concept in concepts])
    episodes = counts["episodes_per_concept"]
    unsteered_logit = empirical_logit(counts["unsteered"]["hits"], counts["unsteered"]["episodes"])
    pooled_control_rate = hits.sum() / (episodes * len(hits))
    target_rate = counts["target"]["hits"] / counts["target"]["episodes"]
    return {"lift_by_concept": empirical_logit(hits, episodes) - unsteered_logit,
            "target_lift": float(empirical_logit(counts["target"]["hits"], counts["target"]["episodes"]) - unsteered_logit),
            "control_variance": logit_variance(pooled_control_rate, episodes),
            "target_variance": logit_variance(target_rate, counts["target"]["episodes"])}


def shape_free_q(mean_lifts: Float[np.ndarray, "concepts"], noise_variance: float, threshold: float) -> tuple[float, float]:
    likelihood = stats.norm.pdf(mean_lifts[:, None], LIFT_GRID[None, :], np.sqrt(noise_variance))
    return shape_free_interval(likelihood, LIFT_GRID >= threshold)


def normal_tail(means: Float[np.ndarray, "means sds"], sds: Float[np.ndarray, "means sds"], threshold: float) -> Float[np.ndarray, "means sds"]:
    with np.errstate(divide="ignore", invalid="ignore"):
        spread_tail = stats.norm.sf((threshold - means) / sds)
    return np.where(sds > 0, spread_tail, (means >= threshold).astype(float))


def normal_q(mean_lifts: Float[np.ndarray, "concepts"], noise_variance: float, threshold: float) -> tuple[QInterval, Float[np.ndarray, "means sds"]]:
    """Point estimate and interval at one threshold, plus the 95% region mask (for the grid-edge check)."""
    means, sds = np.meshgrid(NORMAL_MEANS, NORMAL_SDS, indexing="ij")
    total_sd = np.sqrt(sds ** 2 + noise_variance)
    loglik = stats.norm.logpdf(mean_lifts[:, None, None], means[None], total_sd[None]).sum(axis=0)
    inside = loglik >= loglik.max() - LIKELIHOOD_RATIO_CUTOFF
    tail = normal_tail(means, sds, threshold)
    best = np.unravel_index(loglik.argmax(), loglik.shape)
    return {"estimate": float(tail[best]), "low": float(tail[inside].min()), "high": float(tail[inside].max())}, inside


def normal_grid_edges(inside: Float[np.ndarray, "means sds"]) -> list[NormalGridEdge]:
    edges: list[NormalGridEdge] = []
    if inside[0].any():
        edges.append("lowest normal mean")
    if inside[-1].any():
        edges.append("highest normal mean")
    if inside[:, -1].any():
        edges.append("highest normal sd")
    return edges


def cross_model_q(lifts_by_model: dict[str, ModelLifts]) -> CrossModelQ:
    """q falls as the threshold rises, so the interval for an uncertain threshold runs from the lower bound at
    D_gi + 1.96 sd to the upper bound at D_gi − 1.96 sd."""
    model_count = len(lifts_by_model)
    mean_lifts = np.mean([lifts["lift_by_concept"] for lifts in lifts_by_model.values()], axis=0)
    noise_variance = sum(lifts["control_variance"] for lifts in lifts_by_model.values()) / model_count ** 2
    target_mean_lift = float(np.mean([lifts["target_lift"] for lifts in lifts_by_model.values()]))
    target_sd = float(np.sqrt(sum(lifts["target_variance"] for lifts in lifts_by_model.values()) / model_count ** 2))
    threshold_low, threshold_high = target_mean_lift - Z_95 * target_sd, target_mean_lift + Z_95 * target_sd
    shape_free_low, _ = shape_free_q(mean_lifts, noise_variance, threshold_high)
    _, shape_free_high = shape_free_q(mean_lifts, noise_variance, threshold_low)
    normal_at_target, inside = normal_q(mean_lifts, noise_variance, target_mean_lift)
    normal_low, _ = normal_q(mean_lifts, noise_variance, threshold_high)
    normal_high, _ = normal_q(mean_lifts, noise_variance, threshold_low)
    control_mean_lift = float(mean_lifts.mean())
    return {"models": list(lifts_by_model), "target_mean_lift": target_mean_lift, "target_sd": target_sd,
            "control_mean_lift": control_mean_lift, "control_noise_sd": float(np.sqrt(noise_variance)),
            "concepts_at_or_above_target": int((mean_lifts >= target_mean_lift).sum()),
            "expected_at_or_above_without_spread": float(len(mean_lifts) * stats.norm.sf(target_mean_lift, control_mean_lift, np.sqrt(noise_variance))),
            "shape_free": {"low": shape_free_low, "high": shape_free_high},
            "normal": {"estimate": normal_at_target["estimate"], "low": normal_low["low"], "high": normal_high["high"],
                       "grid_edges": normal_grid_edges(inside)}}


def lift_matrix(lifts_by_model: dict[str, ModelLifts]) -> Float[np.ndarray, "concepts models"]:
    return np.stack([lifts["lift_by_concept"] for lifts in lifts_by_model.values()], axis=1)


def model_offsets(lifts: Float[np.ndarray, "concepts models"]) -> Float[np.ndarray, "models"]:
    """b_m, constrained to average 0 over models so that concept means are unchanged."""
    return lifts.mean(axis=0) - lifts.mean()


def contrast_log_likelihood(centred_lifts: Float[np.ndarray, "concepts models"], noise_variances: Float[np.ndarray, "models"], model_specific_sd: float) -> float:
    """Restricted likelihood of each concept's deviations from its own mean, with model-specific deviations ~ N(0, τ²)
    plus sampling noise (the concept's shared effect integrated out)."""
    variances = model_specific_sd ** 2 + noise_variances
    weights = 1 / variances
    weighted_sums = centred_lifts @ weights
    quadratic = (centred_lifts ** 2) @ weights - weighted_sums ** 2 / weights.sum()
    return float(-0.5 * (len(centred_lifts) * (np.log(variances).sum() + np.log(weights.sum())) + quadratic.sum()))


def heterogeneity(lifts: Float[np.ndarray, "models"], variances: Float[np.ndarray, "models"]) -> tuple[float, float, float]:
    """Cochran's Q, its χ² p, and the DerSimonian–Laird model-specific sd."""
    weights = 1 / variances
    q = float((weights * (lifts - (weights @ lifts) / weights.sum()) ** 2).sum())
    tau_squared = max(0.0, (q - (len(lifts) - 1)) / (weights.sum() - (weights ** 2).sum() / weights.sum()))
    return q, float(stats.chi2.sf(q, len(lifts) - 1)), float(np.sqrt(tau_squared))


def hierarchical_q(lifts_by_model: dict[str, ModelLifts]) -> HierarchicalQ:
    """d_{j,m} = b_m + μ_j + ε_{j,m} + noise, ε ~ N(0, τ²). The concept means carry noise V + τ²/M and are deconvolved
    for the distribution of μ; τ comes from the within-concept deviations; both intervals are profiled over τ. gi_v2's
    shared effect is its four-model mean, uncertain by its own noise plus τ²/M."""
    model_count = len(lifts_by_model)
    concept_model_lifts = lift_matrix(lifts_by_model)
    offsets = model_offsets(concept_model_lifts)
    noise_variances = np.array([model["control_variance"] for model in lifts_by_model.values()])
    mean_lifts = concept_model_lifts.mean(axis=1)
    mean_noise_variance = noise_variances.sum() / model_count ** 2
    target_lifts = np.array([model["target_lift"] for model in lifts_by_model.values()])
    target_variances = np.array([model["target_variance"] for model in lifts_by_model.values()])
    target_mean_lift = float(target_lifts.mean())
    contrast_logliks = np.array([contrast_log_likelihood(concept_model_lifts - offsets, noise_variances, tau) for tau in TAU_GRID])
    concept_noise_variances = mean_noise_variance + TAU_GRID ** 2 / model_count
    threshold_sds = np.sqrt(target_variances.sum() / model_count ** 2 + TAU_GRID ** 2 / model_count)

    likelihoods = [stats.norm.pdf(mean_lifts[:, None], LIFT_GRID[None, :], np.sqrt(variance)) for variance in concept_noise_variances]
    peak_fits = [unconstrained_npmle(likelihood) for likelihood in likelihoods]
    shape_free_totals = np.array([peak for peak, _ in peak_fits]) + contrast_logliks
    shape_free_cutoffs = shape_free_totals - shape_free_totals.max() + LIKELIHOOD_RATIO_CUTOFF
    shape_free_bounds = [(shape_free_bound(likelihood, LIFT_GRID >= target_mean_lift + Z_95 * sd, "low", cutoff, peak_fit),
                          shape_free_bound(likelihood, LIFT_GRID >= target_mean_lift - Z_95 * sd, "high", cutoff, peak_fit))
                         for likelihood, peak_fit, cutoff, sd in zip(likelihoods, peak_fits, shape_free_cutoffs, threshold_sds) if cutoff >= 0]
    tau_inside = shape_free_cutoffs >= 0

    means, sds = np.meshgrid(HIERARCHICAL_NORMAL_MEANS, HIERARCHICAL_NORMAL_SDS, indexing="ij")
    normal_logliks = np.stack([stats.norm.logpdf(mean_lifts[:, None, None], means[None], np.sqrt(sds ** 2 + variance)[None]).sum(axis=0) + contrast
                               for variance, contrast in zip(concept_noise_variances, contrast_logliks)])
    normal_inside = normal_logliks >= normal_logliks.max() - LIKELIHOOD_RATIO_CUTOFF
    best_tau, best_mean, best_sd = np.unravel_index(normal_logliks.argmax(), normal_logliks.shape)
    normal_lows = [normal_tail(means, sds, target_mean_lift + Z_95 * sd)[inside] for inside, sd in zip(normal_inside, threshold_sds)]
    normal_highs = [normal_tail(means, sds, target_mean_lift - Z_95 * sd)[inside] for inside, sd in zip(normal_inside, threshold_sds)]
    target_q, target_p, target_tau = heterogeneity(target_lifts - offsets, target_variances)
    return {"models": list(lifts_by_model),
            "model_specific_sd_estimate": float(TAU_GRID[shape_free_totals.argmax()]),
            "model_specific_sd_low": float(TAU_GRID[tau_inside].min()), "model_specific_sd_high": float(TAU_GRID[tau_inside].max()),
            "target_mean_lift": target_mean_lift, "target_heterogeneity_q": target_q, "target_heterogeneity_p": target_p,
            "target_model_specific_sd": target_tau, "control_mean_lift": float(mean_lifts.mean()),
            "shape_free": {"low": min(low for low, _ in shape_free_bounds), "high": max(high for _, high in shape_free_bounds)},
            "normal": {"estimate": float(normal_tail(means, sds, target_mean_lift)[best_mean, best_sd]),
                       "low": float(min(tails.min() for tails in normal_lows if tails.size)),
                       "high": float(max(tails.max() for tails in normal_highs if tails.size)),
                       "grid_edges": normal_grid_edges(normal_inside.any(axis=0))},
            "tau_grid_edges": ["highest model-specific sd"] if tau_inside[-1] or normal_inside[-1].any() else []}


def lift_correlations(lifts_by_model: dict[str, ModelLifts], rng: np.random.Generator) -> LiftCorrelations:
    names = list(lifts_by_model)
    rank_rows = stats.rankdata([lifts_by_model[name]["lift_by_concept"] for name in names], axis=1)
    model_pairs = list(itertools.combinations(range(len(names)), 2))

    def pairwise_spearman(rows: Float[np.ndarray, "models concepts"]) -> list[float]:
        correlations = np.corrcoef(rows)
        return [float(correlations[a, b]) for a, b in model_pairs]

    observed = pairwise_spearman(rank_rows)
    shuffled_means = [np.mean(pairwise_spearman(np.array([rng.permutation(row) for row in rank_rows]))) for _ in range(PERMUTATIONS)]
    return {"pairs": [{"models": (names[a], names[b]), "spearman": rho} for (a, b), rho in zip(model_pairs, observed)],
            "mean_spearman": float(np.mean(observed)),
            "shuffle_p": float((1 + np.sum(np.array(shuffled_means) >= np.mean(observed))) / (1 + PERMUTATIONS))}


def model_subsets(names: Sequence[str]) -> dict[str, list[str]]:
    return {"all": list(names)} | {f"without {name}": [other for other in names if other != name] for name in names}


def hack_predicate(outcome: Outcome, verdict_by_key: dict[str, str] | None) -> Callable[[CheckRecord, str], bool]:
    """Takes the record and its '<arm dir>/<seed dir>' prefix."""
    if outcome == "proxy":
        return lambda record, _: is_proxy_hack(record)
    assert verdict_by_key is not None, "--outcome judged needs a verdicts file for every model"
    return lambda record, prefix: verdict_by_key.get(f"{prefix}/{record['example_id']}", "").upper() == "HACK"


def arm_count(arm_dir: Path, records_glob: str, is_hack: Callable[[CheckRecord, str], bool]) -> ArmCount:
    hits_and_episodes = [(sum(is_hack(record, f"{arm_dir.name}/{path.parent.name}") for record in records), len(records))
                         for path in sorted(arm_dir.glob(records_glob)) for records in [load_check_records([path])]]
    return {"hits": sum(hits for hits, _ in hits_and_episodes), "episodes": sum(episodes for _, episodes in hits_and_episodes)}


def load_model_counts(runs: ModelRuns, outcome: Outcome) -> ModelCounts:
    verdict_by_key = json.loads(Path(runs["verdicts"]).read_text()) if "verdicts" in runs else None
    is_hack = hack_predicate(outcome, verdict_by_key)
    count_by_concept = {arm_dir.name: arm_count(arm_dir, runs["records_glob"], is_hack) for arm_dir in sorted(Path(runs["controls_root"]).glob(runs["controls_glob"]))}
    episode_counts = {count["episodes"] for count in count_by_concept.values()}
    assert len(episode_counts) == 1, f"{runs['name']}: control arms differ in episode count: {sorted(episode_counts)}"
    return {"hits_by_concept": {concept: count["hits"] for concept, count in count_by_concept.items()}, "episodes_per_concept": episode_counts.pop(),
            "unsteered": arm_count(Path(runs["unsteered"]), runs["records_glob"], is_hack),
            "target": arm_count(Path(runs["target"]), runs["records_glob"], is_hack)}


def model_summary(counts: ModelCounts, lifts: ModelLifts) -> ModelSummary:
    hits = sum(counts["hits_by_concept"].values())
    return {"episodes_per_concept": counts["episodes_per_concept"],
            "unsteered_rate": counts["unsteered"]["hits"] / counts["unsteered"]["episodes"],
            "target_rate": counts["target"]["hits"] / counts["target"]["episodes"],
            "pooled_control_rate": hits / (counts["episodes_per_concept"] * len(counts["hits_by_concept"])),
            "target_lift": lifts["target_lift"], "mean_control_lift": float(lifts["lift_by_concept"].mean())}


def shared_concepts(counts_by_model: dict[str, ModelCounts]) -> list[str]:
    concept_sets = [set(counts["hits_by_concept"]) for counts in counts_by_model.values()]
    assert all(concepts == concept_sets[0] for concepts in concept_sets), "models differ in their control concepts"
    return sorted(concept_sets[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", type=Path, required=True, help="JSON list of model run entries (see ModelRuns)")
    parser.add_argument("--outcome", choices=["proxy", "judged"], required=True)
    parser.add_argument("--seed", type=int, default=0, help="label-shuffle RNG seed")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    runs_list: list[ModelRuns] = json.loads(args.models.read_text())
    counts_by_model = {runs["name"]: load_model_counts(runs, args.outcome) for runs in runs_list}
    concepts = shared_concepts(counts_by_model)
    lifts_by_model = {name: model_lifts(counts, concepts) for name, counts in counts_by_model.items()}
    subsets = model_subsets(list(lifts_by_model))
    q_by_subset = {label: cross_model_q({name: lifts_by_model[name] for name in subset}) for label, subset in subsets.items()}
    hierarchical_q_by_subset = {label: hierarchical_q({name: lifts_by_model[name] for name in subset}) for label, subset in subsets.items()}
    for label, q in q_by_subset.items():
        if q["normal"]["grid_edges"]:
            warnings.warn(f"{label}: the normal fit's 95% region reaches {q['normal']['grid_edges']}; widen NORMAL_MEANS / NORMAL_SDS", stacklevel=1)
    for label, hierarchical in hierarchical_q_by_subset.items():
        if edges := hierarchical["normal"]["grid_edges"] + hierarchical["tau_grid_edges"]:
            warnings.warn(f"{label}: the hierarchical fit's 95% region reaches {edges}; widen the HIERARCHICAL_NORMAL_* / TAU_GRID grids", stacklevel=1)
    result = {"outcome": args.outcome, "concepts": len(concepts),
              "models": {name: model_summary(counts_by_model[name], lifts_by_model[name]) for name in lifts_by_model},
              "q_cross": q_by_subset, "hierarchical_q": hierarchical_q_by_subset, "lift_correlations": lift_correlations(lifts_by_model, np.random.default_rng(args.seed))}
    args.out.write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
