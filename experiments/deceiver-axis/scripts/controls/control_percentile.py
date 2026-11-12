"""A target steering direction's percentile among a population of control directions, any model and any target:
q = the share of control directions whose true hack rate is at or above the target's observed rate (percentile = 1 − q),
on two judge-free measures (proxy hacks: shown pass, hidden fail; hard-coding hits, selfconcept.codebench.hardcoded_cases).

Per measure: unsteered, target and pooled control rates, Fisher test of pooled vs unsteered, binomial dispersion χ²
over the controls, Spearman of control hits with cos(control, target), the count of controls at or above the target,
and two intervals for q:
- shape-free: NPMLE of the control-rate distribution on a grid, likelihood-ratio 95% interval (profile with the mass
  above the target's rate fixed);
- beta: control rates ~ Beta(mean, concentration), profile likelihood-ratio interval over a (mean, concentration) grid.
Every control direction has the same number of episodes. Writes one JSON.

    python scripts/controls/control_percentile.py --controls-root <runs> --controls 'antonym_*' --unsteered <runs/unsteered_ref> \
        --target <target arm dir> --problems <ib_solvhard_s0.jsonl> ... --directions <dirs> --target-direction <axis.pt> \
        --layer 17 --out <percentile.json>
"""
import argparse
import json
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Literal, TypedDict

import numpy as np
import torch
from jaxtyping import Bool, Float, Int
from scipy import stats

from selfconcept.codebench.harness import CheckExample, CheckRecord
from selfconcept.codebench.hardcoded_cases import is_hardcoded_hack, is_proxy_hack, load_check_problems, load_check_records

# Support of a control direction's true hack rate (both estimators discretise it here).
RATE_GRID = np.linspace(0.005, 0.7, 280)
# Beta family searched by the beta estimator: means and concentrations (a + b; sd shrinks as it grows, log-spaced from
# very spread to nearly identical directions). Wide enough that the 95% region should not reach an edge; q_intervals
# reports any edge it does reach.
BETA_MEANS = np.linspace(0.005, 0.6, 120)
BETA_CONCENTRATIONS = np.exp(np.linspace(np.log(1), np.log(1e5), 81))
# χ²₁(0.95) / 2: a point is inside the 95% likelihood-ratio interval when its log-likelihood is within this of the peak.
LIKELIHOOD_RATIO_CUTOFF = 1.92
# EM steps for the unconstrained NPMLE, EM steps per constrained refit (warm-started), bisection steps per interval end.
NPMLE_ITERATIONS, PROFILE_ITERATIONS, BISECTION_STEPS = 1500, 600, 9
TOP_DIRECTIONS = 6

Measure = Literal["hidden test fail", "hard-code regex"]
GridEdge = Literal["lowest beta mean", "highest beta mean", "lowest beta concentration", "highest beta concentration", "target rate outside the rate grid"]
type Likelihood = Float[np.ndarray, "directions grid"]


class QInterval(TypedDict):
    beta_estimate: float
    beta_low: float
    beta_high: float
    shape_free_low: float
    shape_free_high: float
    grid_edges: list[GridEdge]
    grid_edges_affecting_upper_bound: list[GridEdge]


class MeasureSummary(QInterval):
    unsteered: float
    target: float
    pooled: float
    pooled_vs_unsteered_p: float
    dispersion_p: float
    spearman_cos: float
    spearman_p: float
    at_or_above_target: int
    top: list[tuple[int, str]]


def npmle_fit(likelihood: Likelihood, start: Float[np.ndarray, "grid"], iterations: int,
              above: Bool[np.ndarray, "grid"] | None = None, tail_mass: float | None = None) -> tuple[float, Float[np.ndarray, "grid"]]:
    """EM for the mixing weights on the grid (with the mass on `above` held at `tail_mass` when given); returns the
    log-likelihood and the weights."""
    weights = np.maximum(start, 1e-9) if tail_mass is not None else start.copy()
    for _ in range(iterations):
        responsibilities = likelihood * weights
        responsibilities /= responsibilities.sum(axis=1, keepdims=True)
        weights = responsibilities.mean(axis=0)
        if above is not None and tail_mass is not None:
            mass_above, mass_below = weights[above].sum(), weights[~above].sum()
            weights = np.where(above, weights * tail_mass / max(mass_above, 1e-300), weights * (1 - tail_mass) / max(mass_below, 1e-300))
    return float(np.log(likelihood @ weights + 1e-300).sum()), weights


def unconstrained_npmle(likelihood: Likelihood) -> tuple[float, Float[np.ndarray, "grid"]]:
    grid_size = likelihood.shape[1]
    return npmle_fit(likelihood, np.full(grid_size, 1 / grid_size), NPMLE_ITERATIONS)


def shape_free_bound(likelihood: Likelihood, above: Bool[np.ndarray, "grid"], side: Literal["low", "high"], cutoff: float,
                     peak_fit: tuple[float, Float[np.ndarray, "grid"]]) -> float:
    """The lowest or highest tail mass whose profile log-likelihood is within `cutoff` of the unconstrained peak."""
    peak, weights = peak_fit

    def within(tail_mass: float) -> bool:
        return peak - npmle_fit(likelihood, weights, PROFILE_ITERATIONS, above, tail_mass)[0] <= cutoff

    inside, outside = float(weights[above].sum()), 0.0 if side == "low" else 1.0
    if within(outside):
        return outside
    for _ in range(BISECTION_STEPS):
        middle = (inside + outside) / 2
        inside, outside = (middle, outside) if within(middle) else (inside, middle)
    return (inside + outside) / 2


def shape_free_interval(likelihood: Likelihood, above: Bool[np.ndarray, "grid"]) -> tuple[float, float]:
    peak_fit = unconstrained_npmle(likelihood)
    return (shape_free_bound(likelihood, above, "low", LIKELIHOOD_RATIO_CUTOFF, peak_fit),
            shape_free_bound(likelihood, above, "high", LIKELIHOOD_RATIO_CUTOFF, peak_fit))


def grid_edges(inside: Bool[np.ndarray, "means concentrations"], target_rate: float) -> list[GridEdge]:
    """Grid edges the beta 95% region reaches (the interval may be cut off there), and a target rate the rate grid does
    not cover (both estimators then misplace the tail)."""
    edges: list[GridEdge] = []
    if inside[0].any():
        edges.append("lowest beta mean")
    if inside[-1].any():
        edges.append("highest beta mean")
    if inside[:, 0].any():
        edges.append("lowest beta concentration")
    if inside[:, -1].any():
        edges.append("highest beta concentration")
    if not RATE_GRID[0] <= target_rate <= RATE_GRID[-1]:
        edges.append("target rate outside the rate grid")
    return edges


def edges_that_can_cut_the_upper_bound(edges: list[GridEdge], region_means: Float[np.ndarray, "inside"], target_rate: float) -> list[GridEdge]:
    """A tighter spread only lowers q while the target sits above every mean in the region, so the highest-concentration
    edge (reached whenever the controls show no overdispersion) can only cut off the lower bound then."""
    harmless = "highest beta concentration" if target_rate > region_means.max() else None
    return [edge for edge in edges if edge != harmless]


def q_intervals(hits: Int[np.ndarray, "directions"], episodes: int, target_rate: float) -> QInterval:
    means, concentrations = np.meshgrid(BETA_MEANS, BETA_CONCENTRATIONS, indexing="ij")
    density = stats.beta.pdf(RATE_GRID[None, None, :], (means * concentrations)[..., None], ((1 - means) * concentrations)[..., None])
    density /= density.sum(axis=-1, keepdims=True)
    tail = stats.beta.sf(target_rate, means * concentrations, (1 - means) * concentrations)
    likelihood = stats.binom.pmf(hits[:, None], episodes, RATE_GRID[None, :])
    loglik = np.log(np.einsum("dr,mkr->dmk", likelihood, density) + 1e-300).sum(axis=0)
    inside = loglik >= loglik.max() - LIKELIHOOD_RATIO_CUTOFF
    shape_free_low, shape_free_high = shape_free_interval(likelihood, RATE_GRID >= target_rate)
    best = np.unravel_index(loglik.argmax(), loglik.shape)
    return {"beta_estimate": float(tail[best]), "beta_low": float(tail[inside].min()), "beta_high": float(tail[inside].max()),
            "shape_free_low": shape_free_low, "shape_free_high": shape_free_high,
            "grid_edges": (edges := grid_edges(inside, target_rate)),
            "grid_edges_affecting_upper_bound": edges_that_can_cut_the_upper_bound(edges, means[inside], target_rate)}


def summarize_measure(hits_by_arm: dict[str, int], episodes: int, unsteered: tuple[int, int], target: tuple[int, int],
                      cos_by_arm: dict[str, float]) -> MeasureSummary:
    arms = sorted(hits_by_arm)
    hits = np.array([hits_by_arm[arm] for arm in arms])
    pooled = hits.sum() / (episodes * len(hits))
    target_rate = target[0] / target[1]
    chi2 = float(((hits - episodes * pooled) ** 2 / (episodes * pooled * (1 - pooled))).sum())
    contingency = [[int(hits.sum()), episodes * len(hits) - int(hits.sum())], [unsteered[0], unsteered[1] - unsteered[0]]]
    spearman_rho, spearman_p = np.asarray(stats.spearmanr(hits, [cos_by_arm[arm] for arm in arms]), dtype=float)
    _, fisher_p = np.asarray(stats.fisher_exact(np.array(contingency)), dtype=float)
    return {"unsteered": unsteered[0] / unsteered[1], "target": target_rate, "pooled": float(pooled),
            "pooled_vs_unsteered_p": float(fisher_p),
            "dispersion_p": float(stats.chi2.sf(chi2, len(hits) - 1)),
            "spearman_cos": float(spearman_rho), "spearman_p": float(spearman_p),
            "at_or_above_target": int((hits >= np.ceil(episodes * target_rate)).sum()),
            "top": sorted(zip(hits.tolist(), arms), reverse=True)[:TOP_DIRECTIONS],
            **q_intervals(hits, episodes, target_rate)}


def unit_layer(artifact: Path, layer: int) -> Float[torch.Tensor, "hidden"]:
    vector = torch.load(artifact, weights_only=False)["axis"][layer].float()
    return vector / vector.norm()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--controls-root", type=Path, required=True)
    parser.add_argument("--controls", required=True, help="glob of control arm directories under --controls-root")
    parser.add_argument("--records", default="seed*/main_impossible_original.jsonl", help="glob of episode files under each arm")
    parser.add_argument("--unsteered", type=Path, required=True, help="unsteered arm directory")
    parser.add_argument("--target", type=Path, required=True, help="the target direction's arm directory (at the dose the controls are matched to)")
    parser.add_argument("--problems", type=Path, nargs="+", required=True)
    parser.add_argument("--directions", type=Path, required=True, help="directory of <arm>.pt control direction artifacts")
    parser.add_argument("--target-direction", type=Path, required=True, help="(layers, hidden) axis artifact of the target")
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
    unsteered_records, target_records = load_check_records(args.unsteered.glob(args.records)), load_check_records(args.target.glob(args.records))
    target_direction = unit_layer(args.target_direction, args.layer)
    cos_by_arm = {arm: float(unit_layer(args.directions / f"{arm}.pt", args.layer) @ target_direction) for arm in records_by_arm}

    def count(records: list[CheckRecord], measure: Measure) -> int:
        return sum(map(measures[measure], records))

    summary_by_measure = {
        measure: summarize_measure({arm: count(records, measure) for arm, records in records_by_arm.items()}, episodes,
                                   (count(unsteered_records, measure), len(unsteered_records)), (count(target_records, measure), len(target_records)), cos_by_arm)
        for measure in measures
    }
    for measure, summary in summary_by_measure.items():
        if summary["grid_edges_affecting_upper_bound"]:
            warnings.warn(f"{measure}: the beta fit's 95% region reaches the grid edge(s) {summary['grid_edges_affecting_upper_bound']}; its interval may be cut "
                          f"off there (widen BETA_MEANS / BETA_CONCENTRATIONS or RATE_GRID)", stacklevel=1)
    args.out.write_text(json.dumps(summary_by_measure, indent=1))


if __name__ == "__main__":
    main()
