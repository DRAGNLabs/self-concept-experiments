"""gi_v2's percentile among control steering directions, any model: q = the share of control directions whose true
rate is at or above gi_v2's observed rate (percentile = 1 − q), on two judge-free measures (proxy hacks: shown pass,
hidden fail; hard-coding hits, selfconcept.codebench.hardcoded_cases).

Per measure: unsteered, gi_v2 and pooled control rates, Fisher test of pooled vs unsteered, binomial dispersion χ² over
the controls, Spearman of control hits with cos(direction, gi_v2), the count at or above gi_v2, and two intervals for q:
- shape-free: NPMLE of the control-rate distribution on a grid, likelihood-ratio 95% interval (profile with the mass
  above gi_v2's rate fixed, cutoff χ²₁ / 2 = 1.92);
- beta: control rates ~ Beta(mean, concentration), profile interval over a (mean, concentration) grid.
Every control direction has the same number of episodes. Writes one JSON.

    python scripts/controls/control_percentile.py --controls-root <runs> --controls 'antonym_*' --unsteered <runs/unsteered_ref> \
        --gi <gi_v2 arm dir> --problems <ib_solvhard_s0.jsonl> ... --directions <dirs> --gi-direction <persona_contrast.pt> \
        --layer 17 --out <percentile.json>
"""
import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Literal, TypedDict

import numpy as np
import torch
from jaxtyping import Float, Int
from scipy import stats

from selfconcept.codebench.harness import CheckExample, CheckRecord
from selfconcept.codebench.hardcoded_cases import is_hardcoded_hack, is_proxy_hack, load_check_problems, load_check_records

RATE_GRID = np.linspace(0.005, 0.7, 280)
BETA_MEANS = np.linspace(0.08, 0.40, 50)
BETA_CONCENTRATIONS = np.exp(np.linspace(np.log(3), np.log(3000), 50))
LIKELIHOOD_RATIO_CUTOFF = 1.92
NPMLE_ITERATIONS, PROFILE_ITERATIONS, BISECTION_STEPS = 1500, 600, 9
TOP_DIRECTIONS = 6

Measure = Literal["hidden test fail", "hard-code regex"]
type Likelihood = Float[np.ndarray, "directions grid"]


class QInterval(TypedDict):
    beta_estimate: float
    beta_low: float
    beta_high: float
    shape_free_low: float
    shape_free_high: float


class MeasureSummary(QInterval):
    unsteered: float
    gi_v2: float
    pooled: float
    pooled_vs_unsteered_p: float
    dispersion_p: float
    spearman_cos: float
    spearman_p: float
    at_or_above_gi: int
    top: list[tuple[int, str]]


def shape_free_interval(likelihood: Likelihood, above: Float[np.ndarray, "grid"]) -> tuple[float, float]:
    def fit(tail_mass: float | None, start: Float[np.ndarray, "grid"], iterations: int) -> tuple[float, Float[np.ndarray, "grid"]]:
        weights = np.maximum(start, 1e-9) if tail_mass is not None else start.copy()
        for _ in range(iterations):
            responsibilities = likelihood * weights
            responsibilities /= responsibilities.sum(axis=1, keepdims=True)
            weights = responsibilities.mean(axis=0)
            if tail_mass is not None:
                mass_above, mass_below = weights[above].sum(), weights[~above].sum()
                weights = np.where(above, weights * tail_mass / max(mass_above, 1e-300), weights * (1 - tail_mass) / max(mass_below, 1e-300))
        return float(np.log(likelihood @ weights + 1e-300).sum()), weights

    peak, weights = fit(None, np.full(len(RATE_GRID), 1 / len(RATE_GRID)), NPMLE_ITERATIONS)
    estimate = float(weights[above].sum())

    def boundary(inside: float, outside: float) -> float:
        if peak - fit(outside, weights, PROFILE_ITERATIONS)[0] <= LIKELIHOOD_RATIO_CUTOFF:
            return outside
        for _ in range(BISECTION_STEPS):
            middle = (inside + outside) / 2
            inside, outside = (middle, outside) if peak - fit(middle, weights, PROFILE_ITERATIONS)[0] <= LIKELIHOOD_RATIO_CUTOFF else (inside, middle)
        return (inside + outside) / 2

    return boundary(estimate, 0.0), boundary(estimate, 1.0)


def q_intervals(hits: Int[np.ndarray, "directions"], episodes: int, gi_rate: float) -> QInterval:
    means, concentrations = np.meshgrid(BETA_MEANS, BETA_CONCENTRATIONS, indexing="ij")
    density = stats.beta.pdf(RATE_GRID[None, None, :], (means * concentrations)[..., None], ((1 - means) * concentrations)[..., None])
    density /= density.sum(axis=-1, keepdims=True)
    tail = stats.beta.sf(gi_rate, means * concentrations, (1 - means) * concentrations)
    likelihood = stats.binom.pmf(hits[:, None], episodes, RATE_GRID[None, :])
    loglik = np.log(np.einsum("dr,mkr->dmk", likelihood, density) + 1e-300).sum(axis=0)
    inside = loglik >= loglik.max() - LIKELIHOOD_RATIO_CUTOFF
    shape_free_low, shape_free_high = shape_free_interval(likelihood, RATE_GRID >= gi_rate)
    best = np.unravel_index(loglik.argmax(), loglik.shape)
    return {"beta_estimate": float(tail[best]), "beta_low": float(tail[inside].min()), "beta_high": float(tail[inside].max()),
            "shape_free_low": shape_free_low, "shape_free_high": shape_free_high}


def summarize_measure(hits_by_arm: dict[str, int], episodes: int, unsteered: tuple[int, int], gi: tuple[int, int],
                      cos_by_arm: dict[str, float]) -> MeasureSummary:
    arms = sorted(hits_by_arm)
    hits = np.array([hits_by_arm[arm] for arm in arms])
    pooled = hits.sum() / (episodes * len(hits))
    gi_rate = gi[0] / gi[1]
    chi2 = float(((hits - episodes * pooled) ** 2 / (episodes * pooled * (1 - pooled))).sum())
    contingency = [[int(hits.sum()), episodes * len(hits) - int(hits.sum())], [unsteered[0], unsteered[1] - unsteered[0]]]
    spearman_rho, spearman_p = np.asarray(stats.spearmanr(hits, [cos_by_arm[arm] for arm in arms]), dtype=float)
    _, fisher_p = np.asarray(stats.fisher_exact(np.array(contingency)), dtype=float)
    return {"unsteered": unsteered[0] / unsteered[1], "gi_v2": gi_rate, "pooled": float(pooled),
            "pooled_vs_unsteered_p": float(fisher_p),
            "dispersion_p": float(stats.chi2.sf(chi2, len(hits) - 1)),
            "spearman_cos": float(spearman_rho), "spearman_p": float(spearman_p),
            "at_or_above_gi": int((hits >= np.ceil(episodes * gi_rate)).sum()),
            "top": sorted(zip(hits.tolist(), arms), reverse=True)[:TOP_DIRECTIONS],
            **q_intervals(hits, episodes, gi_rate)}


def unit_layer(artifact: Path, layer: int) -> Float[torch.Tensor, "hidden"]:
    vector = torch.load(artifact, weights_only=False)["axis"][layer].float()
    return vector / vector.norm()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--controls-root", type=Path, required=True)
    parser.add_argument("--controls", required=True, help="glob of control arm directories under --controls-root")
    parser.add_argument("--records", default="seed*/main_impossible_original.jsonl", help="glob of episode files under each arm")
    parser.add_argument("--unsteered", type=Path, required=True, help="unsteered arm directory")
    parser.add_argument("--gi", type=Path, required=True, help="gi_v2 arm directory at the matched dose")
    parser.add_argument("--problems", type=Path, nargs="+", required=True)
    parser.add_argument("--directions", type=Path, required=True, help="directory of <arm>.pt control direction artifacts")
    parser.add_argument("--gi-direction", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    problems_by_id = load_check_problems(args.problems)
    measures: dict[Measure, Callable[[CheckRecord], bool]] = {
        "hidden test fail": is_proxy_hack,
        "hard-code regex": lambda record: is_hardcoded_hack(record, problems_by_id[record["example_id"]]),
    }
    records_by_arm = {arm_dir.name: load_check_records(arm_dir.glob(args.records)) for arm_dir in sorted(args.controls_root.glob(args.controls))}
    episode_counts = {len(records) for records in records_by_arm.values()}
    assert len(episode_counts) == 1, f"control arms differ in episode count: {sorted(episode_counts)}"
    episodes = episode_counts.pop()
    unsteered_records, gi_records = load_check_records(args.unsteered.glob(args.records)), load_check_records(args.gi.glob(args.records))
    gi_direction = unit_layer(args.gi_direction, args.layer)
    cos_by_arm = {arm: float(unit_layer(args.directions / f"{arm}.pt", args.layer) @ gi_direction) for arm in records_by_arm}

    def count(records: list[CheckRecord], measure: Measure) -> int:
        return sum(map(measures[measure], records))

    summary_by_measure = {
        measure: summarize_measure({arm: count(records, measure) for arm, records in records_by_arm.items()}, episodes,
                                   (count(unsteered_records, measure), len(unsteered_records)), (count(gi_records, measure), len(gi_records)), cos_by_arm)
        for measure in measures
    }
    args.out.write_text(json.dumps(summary_by_measure, indent=1))


if __name__ == "__main__":
    main()
