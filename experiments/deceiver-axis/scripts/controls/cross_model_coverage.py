"""Coverage check for cross_model_percentile.py before it is run on the data (PLAN "Cross-model percentile of gi_v2
among antonym concepts"). Simulates the antonym design at the real episode counts, pooled control rates and gi_v2
rates (proxy outcome, read from the runs in --models) and records how often each 95% interval covers the true q,
and how often its upper bound alone does (the bound a "gi_v2 is rare" claim rests on; at q = 0 the only meaningful one).

Mean-lift estimator (`cross_model_q`; truth = share of concepts whose mean true lift is at least gi_v2's; gi_v2 keeps
its observed per-model rates); spread of the controls' true per-model logit lifts δ_{j,m}:
- none: δ = 0 (q = 0);
- shared normal: δ_{j,m} = μ_j ~ N(0, s), s so that q = 5% (and 20%);
- model-specific normal: δ_{j,m} iid N(0, τ), τ so that q = 5%;
- one-model outliers: 10% of concepts lift one random model by M·μ_gi + 0.5, the rest 0 (q = 10%).

Hierarchical estimator (`hierarchical_q`; truth = share of concepts whose shared effect μ_j is at least gi_v2's μ_gi,
the mean of its observed lifts): δ_{j,m} = μ_j + ε_{j,m}, and gi_v2's lifts are μ_gi + ε_{gi,m} with the same ε sd:
- none (q = 0); shared normal q 5% without model-specific deviations; shared normal q 5% and 20% with ε sd 0.3
  (about gi_v2's own heterogeneity); ε sd 0.3 only (q = 0); one-model outliers as above (no shared effect, q = 0).

    python scripts/controls/cross_model_coverage.py --models data/controls/cross_model_models.json --estimator hierarchical --replicates 200 --out <json>
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Literal, TypedDict

import numpy as np
from jaxtyping import Float
from scipy import special, stats

from cross_model_percentile import ModelCounts, ModelLifts, ModelRuns, cross_model_q, hierarchical_q, load_model_counts, model_lifts, model_subsets

CONCEPTS = 100
MODEL_SPECIFIC_SD = 0.3
Estimator = Literal["mean_lift", "hierarchical"]
Scenario = Literal["none", "shared normal q 5%", "shared normal q 20%", "model-specific normal q 5%", "one-model outliers q 10%",
                   "shared normal q 5%, model-specific sd 0.3", "shared normal q 20%, model-specific sd 0.3", "model-specific sd 0.3 only",
                   "one-model outliers"]
SCENARIOS_BY_ESTIMATOR: dict[Estimator, tuple[Scenario, ...]] = {
    "mean_lift": ("none", "shared normal q 5%", "shared normal q 20%", "model-specific normal q 5%", "one-model outliers q 10%"),
    "hierarchical": ("none", "shared normal q 5%", "shared normal q 5%, model-specific sd 0.3", "shared normal q 20%, model-specific sd 0.3",
                     "model-specific sd 0.3 only", "one-model outliers"),
}
TRUE_Q_BY_SCENARIO: dict[Scenario, float] = {
    "none": 0.0, "shared normal q 5%": 0.05, "shared normal q 20%": 0.20, "model-specific normal q 5%": 0.05, "one-model outliers q 10%": 0.10,
    "shared normal q 5%, model-specific sd 0.3": 0.05, "shared normal q 20%, model-specific sd 0.3": 0.20, "model-specific sd 0.3 only": 0.0,
    "one-model outliers": 0.0,
}
SUBSETS = ("all", "without gpt-oss-120b", "without Gemma-4-31B")


class ModelRates(TypedDict):
    episodes_per_concept: int
    control_rate: float
    target_rate: float
    target_episodes: int
    unsteered_rate: float
    unsteered_episodes: int


class Replicate(TypedDict):
    shape_free_low: float
    shape_free_high: float
    normal_low: float
    normal_high: float


class ScenarioCoverage(TypedDict):
    true_q: float
    shape_free_coverage: float
    normal_coverage: float
    shape_free_upper_bound_coverage: float
    normal_upper_bound_coverage: float
    shape_free_median_high: float
    normal_median_high: float
    replicates: int


def model_rates(counts: ModelCounts) -> ModelRates:
    hits = sum(counts["hits_by_concept"].values())
    return {"episodes_per_concept": counts["episodes_per_concept"],
            "control_rate": hits / (counts["episodes_per_concept"] * len(counts["hits_by_concept"])),
            "target_rate": counts["target"]["hits"] / counts["target"]["episodes"], "target_episodes": counts["target"]["episodes"],
            "unsteered_rate": counts["unsteered"]["hits"] / counts["unsteered"]["episodes"], "unsteered_episodes": counts["unsteered"]["episodes"]}


def observed_target_lifts(rates_by_model: dict[str, ModelRates]) -> Float[np.ndarray, "models"]:
    return np.array([special.logit(rates["target_rate"]) - special.logit(rates["control_rate"]) for rates in rates_by_model.values()])


def one_model_outlier_lifts(share: float, model_count: int, target_mean_lift: float, rng: np.random.Generator) -> Float[np.ndarray, "concepts models"]:
    outlier = rng.random(CONCEPTS) < share
    lifted_model = rng.integers(model_count, size=CONCEPTS)
    return np.where(outlier[:, None] & (np.arange(model_count)[None, :] == lifted_model[:, None]), model_count * target_mean_lift + 0.5, 0.0)


def shared_normal_lifts(q: float, model_count: int, target_mean_lift: float, rng: np.random.Generator) -> Float[np.ndarray, "concepts models"]:
    return np.repeat(rng.normal(0, target_mean_lift / stats.norm.isf(q), (CONCEPTS, 1)), model_count, axis=1)


def mean_lift_scenario(scenario: Scenario, target_lifts: Float[np.ndarray, "models"], rng: np.random.Generator) -> tuple[Float[np.ndarray, "concepts models"], Float[np.ndarray, "models"]]:
    model_count, target_mean_lift = len(target_lifts), float(target_lifts.mean())
    if scenario in ("shared normal q 5%", "shared normal q 20%"):
        return shared_normal_lifts(TRUE_Q_BY_SCENARIO[scenario], model_count, target_mean_lift, rng), target_lifts
    if scenario == "model-specific normal q 5%":
        return rng.normal(0, np.sqrt(model_count) * target_mean_lift / stats.norm.isf(TRUE_Q_BY_SCENARIO[scenario]), (CONCEPTS, model_count)), target_lifts
    if scenario == "one-model outliers q 10%":
        return one_model_outlier_lifts(TRUE_Q_BY_SCENARIO[scenario], model_count, target_mean_lift, rng), target_lifts
    return np.zeros((CONCEPTS, model_count)), target_lifts


def hierarchical_scenario(scenario: Scenario, target_lifts: Float[np.ndarray, "models"], rng: np.random.Generator) -> tuple[Float[np.ndarray, "concepts models"], Float[np.ndarray, "models"]]:
    model_count, target_mean_lift = len(target_lifts), float(target_lifts.mean())
    model_specific_sd = MODEL_SPECIFIC_SD if "model-specific sd" in scenario else 0.0
    if scenario == "one-model outliers":
        shared = one_model_outlier_lifts(0.10, model_count, target_mean_lift, rng)
    elif scenario.startswith("shared normal"):
        shared = shared_normal_lifts(TRUE_Q_BY_SCENARIO[scenario], model_count, target_mean_lift, rng)
    else:
        shared = np.zeros((CONCEPTS, model_count))
    return (shared + rng.normal(0, model_specific_sd, (CONCEPTS, model_count)),
            target_mean_lift + rng.normal(0, model_specific_sd, model_count))


def simulated_counts(rates: ModelRates, control_lifts: Float[np.ndarray, "concepts"], target_lift: float, rng: np.random.Generator) -> ModelCounts:
    baseline = special.logit(rates["control_rate"])
    hits = rng.binomial(rates["episodes_per_concept"], special.expit(baseline + control_lifts))
    return {"hits_by_concept": {f"c{index:03d}": int(count) for index, count in enumerate(hits)}, "episodes_per_concept": rates["episodes_per_concept"],
            "unsteered": {"hits": int(rng.binomial(rates["unsteered_episodes"], rates["unsteered_rate"])), "episodes": rates["unsteered_episodes"]},
            "target": {"hits": int(rng.binomial(rates["target_episodes"], special.expit(baseline + target_lift))), "episodes": rates["target_episodes"]}}


def replicate(rates_by_model: dict[str, ModelRates], estimator: Estimator, scenario: Scenario, seed: int) -> Replicate:
    rng = np.random.default_rng(seed)
    scenario_lifts = mean_lift_scenario if estimator == "mean_lift" else hierarchical_scenario
    control_lifts, target_lifts = scenario_lifts(scenario, observed_target_lifts(rates_by_model), rng)
    concepts = [f"c{index:03d}" for index in range(CONCEPTS)]
    lifts_by_model: dict[str, ModelLifts] = {name: model_lifts(simulated_counts(rates, control_lifts[:, column], float(target_lifts[column]), rng), concepts)
                                             for column, (name, rates) in enumerate(rates_by_model.items())}
    q = cross_model_q(lifts_by_model) if estimator == "mean_lift" else hierarchical_q(lifts_by_model)
    return {"shape_free_low": q["shape_free"]["low"], "shape_free_high": q["shape_free"]["high"], "normal_low": q["normal"]["low"], "normal_high": q["normal"]["high"]}


def coverage(replicates: list[Replicate], q: float) -> ScenarioCoverage:
    def covered(low_key: Literal["shape_free_low", "normal_low"], high_key: Literal["shape_free_high", "normal_high"]) -> float:
        return float(np.mean([run[low_key] <= q <= run[high_key] for run in replicates]))

    def upper_bound_covered(high_key: Literal["shape_free_high", "normal_high"]) -> float:
        return float(np.mean([q <= run[high_key] for run in replicates]))

    return {"true_q": q, "shape_free_coverage": covered("shape_free_low", "shape_free_high"), "normal_coverage": covered("normal_low", "normal_high"),
            "shape_free_upper_bound_coverage": upper_bound_covered("shape_free_high"), "normal_upper_bound_coverage": upper_bound_covered("normal_high"),
            "shape_free_median_high": float(np.median([run["shape_free_high"] for run in replicates])),
            "normal_median_high": float(np.median([run["normal_high"] for run in replicates])), "replicates": len(replicates)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--estimator", choices=["mean_lift", "hierarchical"], default="mean_lift")
    parser.add_argument("--replicates", type=int, default=200)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    estimator: Estimator = args.estimator
    scenarios = SCENARIOS_BY_ESTIMATOR[estimator]
    runs_list: list[ModelRuns] = json.loads(args.models.read_text())
    rates_by_model = {runs["name"]: model_rates(load_model_counts(runs, "proxy")) for runs in runs_list}
    subset_models = model_subsets(list(rates_by_model))
    tasks: list[tuple[str, Scenario, int]] = [(subset, scenario, rep) for subset in SUBSETS for scenario in scenarios for rep in range(args.replicates)]
    with ProcessPoolExecutor(args.workers) as pool:
        futures = [pool.submit(replicate, {name: rates_by_model[name] for name in subset_models[subset]}, estimator, scenario, task_index)
                   for task_index, (subset, scenario, _) in enumerate(tasks)]
        replicates = [future.result() for future in futures]
    runs_by_case = {(subset, scenario): [run for (task_subset, task_scenario, _), run in zip(tasks, replicates) if (task_subset, task_scenario) == (subset, scenario)]
                    for subset in SUBSETS for scenario in scenarios}
    result = {"estimator": estimator, "models": rates_by_model,
              "true_target_mean_lift_by_subset": {subset: float(observed_target_lifts({name: rates_by_model[name] for name in subset_models[subset]}).mean()) for subset in SUBSETS},
              "coverage": {subset: {scenario: coverage(runs_by_case[(subset, scenario)], TRUE_Q_BY_SCENARIO[scenario]) for scenario in scenarios} for subset in SUBSETS}}
    args.out.write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
