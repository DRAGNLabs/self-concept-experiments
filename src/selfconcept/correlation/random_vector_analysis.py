"""Compare Assistant-Axis associations against an isotropic random-direction ensemble."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from selfconcept.assistant_axis.projection import load_unit_axes
from selfconcept.common.jsonl import read_jsonl
from .analyze import binary_outcome
from .random_vectors import random_directions, REGIONS, sha256


def group_indices(values):
    groups = defaultdict(list)
    for i, value in enumerate(values):
        groups[value].append(i)
    return {key: np.asarray(indices) for key, indices in groups.items()}


def pair_totals(scores, labels, strata, problems):
    """Vectorized Mann–Whitney counts, with totals retained by problem for bootstrap."""
    by_problem = {}
    n_strata = 0
    for indices in group_indices(strata).values():
        positive = labels[indices] == 1
        n_pos, n_neg = int(positive.sum()), int((~positive).sum())
        if not n_pos or not n_neg:
            continue
        problem = problems[indices[0]]
        if any(problems[i] != problem for i in indices):
            raise ValueError("Stratum crosses base problems")
        ranks = rankdata(scores[indices], axis=0)
        u = ranks[positive].sum(axis=0) - n_pos * (n_pos + 1) / 2
        old_u, old_pairs = by_problem.get(problem, (np.zeros(scores.shape[1]), 0))
        by_problem[problem] = (old_u + u, old_pairs + n_pos * n_neg)
        n_strata += 1
    if not by_problem:
        return np.empty((0, scores.shape[1])), np.empty(0), 0
    return np.stack([v[0] for v in by_problem.values()]), np.asarray([v[1] for v in by_problem.values()]), n_strata


def project(residuals, directions):
    """Identical residuals (same prompt, e.g. every turn-0 stratum) must tie exactly, whatever BLAS rounds per row."""
    unique, inverse = np.unique(residuals, axis=0, return_inverse=True)
    return (unique.astype(np.float64) @ directions.T.astype(np.float64))[inverse.reshape(-1)]


def baseline_aurocs(region_tokens, residuals, labels, strata, problems):
    """Direction-free references: the region's token count and the norm of its mean residual."""
    scores = np.column_stack([region_tokens, np.linalg.norm(residuals.astype(np.float64), axis=1)])
    u, pairs, _ = pair_totals(scores, labels, strata, problems)
    if not len(pairs):
        return None
    tokens_auroc, norm_auroc = (u.sum(axis=0) / pairs.sum()).tolist()
    return {"region_tokens": tokens_auroc, "residual_norm": norm_auroc}


def adjusted_correlations(scores, labels, lengths, strata):
    """Partial rank correlations controlling strata and prompt/CoT/final token lengths.

    Global ranks, then within-stratum centering, then least-squares residualization
    of both outcome and projection ranks on centered length ranks.
    """
    values = np.column_stack([rankdata(scores, axis=0), rankdata(labels), rankdata(lengths, axis=0)])
    for indices in group_indices(strata).values():
        values[indices] -= values[indices].mean(axis=0)
    targets, controls = values[:, :-3], values[:, -3:]
    residuals = targets - controls @ np.linalg.lstsq(controls, targets, rcond=None)[0]
    x, y = residuals[:, :-1], residuals[:, -1]
    denominator = np.linalg.norm(x, axis=0) * np.linalg.norm(y)
    return np.divide(x.T @ y, denominator, out=np.full(x.shape[1], np.nan), where=denominator > 1e-10)


def compare_to_random(statistics, center):
    axis, random = statistics[0], statistics[1:]
    valid = random[np.isfinite(random)]
    if not np.isfinite(axis) or not len(valid):
        return None
    exceed = int(np.sum(np.abs(valid - center) >= abs(axis - center)))
    return {"assistant_axis": float(axis), "random_quantiles": np.quantile(valid, [.025, .5, .975]).tolist(),
            "n_random": len(valid), "n_random_at_least_as_extreme": exceed,
            "random_direction_tail_fraction": (1 + exceed) / (1 + len(valid))}


def analyze_cell(scores, labels, lengths, strata, problems, bootstrap=1000, seed=1729):
    n_available = len(labels)
    informative = [indices for indices in group_indices(strata).values() if len(np.unique(labels[indices])) == 2]
    if not informative:
        return {"status": "no_matched_pairs", "n": 0, "n_available": n_available}
    keep = np.sort(np.concatenate(informative))
    scores, labels, lengths = scores[keep], labels[keep], lengths[keep]
    strata, problems = [strata[i] for i in keep], [problems[i] for i in keep]
    u, pairs, n_strata = pair_totals(scores, labels, strata, problems)
    if not len(pairs):
        return {"status": "no_matched_pairs", "n": len(labels)}
    aucs = u.sum(axis=0) / pairs.sum()
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(bootstrap):
        indices = rng.integers(len(pairs), size=len(pairs))
        draws.append(u[indices, 0].sum() / pairs[indices].sum())
    adjusted = adjusted_correlations(scores, labels, lengths, strata)
    return {"status": "exploratory", "n": len(labels), "n_available": n_available, "n_positive": int(labels.sum()),
            "n_strata": n_strata, "n_problems": len(pairs), "n_pairs": int(pairs.sum()),
            "auroc": compare_to_random(aucs, .5),
            "assistant_axis_problem_bootstrap_ci": np.quantile(draws, [.025, .975]).tolist() if draws else None,
            "length_adjusted_rank_correlation": compare_to_random(adjusted, 0),
            "all_aurocs": aucs.tolist(),
            "all_length_adjusted_correlations": [float(x) if np.isfinite(x) else None for x in adjusted]}


def load_cache(directories):
    records, seen, manifests = [], set(), []
    for directory in directories:
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest["stage"] != "complete":
            raise ValueError(f"Scoring is incomplete: {directory}")
        manifests.append(manifest)
        # The most recent attempt wins when an OOM was retried.
        latest = {(r["scenario"], r["example_id"]): r for r in read_jsonl(directory / "records.jsonl")}
        if len(latest) != manifest["identity"]["n_expected"]:
            raise ValueError(f"Missing records in {directory}")
        if seen & latest.keys():
            raise ValueError("Duplicate examples across shards")
        seen.update(latest)
        records.extend((directory, r) for r in latest.values())
    identities = [{k: v for k, v in m["identity"].items() if k not in ("shard", "n_expected")} for m in manifests]
    if not identities or any(i != identities[0] for i in identities):
        raise ValueError("Residual cache identities disagree")
    if sorted(m["identity"]["shard"] for m in manifests) != list(range(identities[0]["shards"])):
        raise ValueError("Must include every shard exactly once")
    if identities[0]["limit"] is not None:
        raise ValueError("Smoke-test caches cannot be used for the full analysis")
    if len({(m["num_layers"], m["hidden_size"]) for m in manifests}) != 1:
        raise ValueError("Model dimensions disagree")
    return records, manifests[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", type=Path)
    parser.add_argument("--axis", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--random-count", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    if args.random_count < 1 or args.seed < 0 or args.bootstrap < 0:
        parser.error("Invalid random count, seed or bootstrap count")
    records, manifest = load_cache(args.directories)
    identity = manifest["identity"]
    layers = identity["layers"]
    axis = load_unit_axes(args.axis, layers, manifest["num_layers"], manifest["hidden_size"], identity["model"])
    completed = [(d, r) for d, r in records if r["status"] == "complete" and binary_outcome(r["outcome"])[0] is not None]
    if not completed:
        raise ValueError("No complete scorable records")
    args.out.mkdir(parents=True, exist_ok=True)
    matrices = {layer: np.stack([np.load(d / r["residuals"])[f"layer_{layer}"] for d, r in completed]) for layer in layers}
    rows = [r for _, r in completed]
    directions = {layer: np.vstack([axis[layer]["direction"], random_directions(manifest["hidden_size"], args.random_count, args.seed, layer)]) for layer in layers}
    np.savez_compressed(args.out / "directions.npz", **{f"layer_{l}": d for l, d in directions.items()})
    labels = np.asarray([binary_outcome(r["outcome"])[0] for r in rows])
    lengths = np.asarray([[r["prompt_tokens"], *r["region_counts"][1:]] for r in rows])
    region_tokens = np.asarray([r["region_counts"] for r in rows])
    problems = [json.dumps([r["scenario"], r["outcome"]["problem"]]) for r in rows]
    strata = [r["outcome"]["stratum"] for r in rows]
    cells = []
    for layer in layers:
        for region_index, region in enumerate(REGIONS):
            residuals = matrices[layer][:, region_index, :]
            valid = np.isfinite(residuals).all(axis=1)
            for scenario in ["all", *sorted({r["scenario"] for r in rows})]:
                subset = np.flatnonzero(valid & np.asarray([scenario == "all" or r["scenario"] == scenario for r in rows]))
                scores = project(residuals[subset], directions[layer])
                for grouping in ("problem", "problem_family_turn"):
                    grouping_values = problems if grouping == "problem" else strata
                    subset_strata, subset_problems = [grouping_values[i] for i in subset], [problems[i] for i in subset]
                    cell = analyze_cell(scores, labels[subset], lengths[subset], subset_strata, subset_problems,
                                        args.bootstrap, args.seed)
                    cell["baseline_aurocs"] = baseline_aurocs(region_tokens[subset, region_index], residuals[subset],
                                                              labels[subset], subset_strata, subset_problems)
                    cells.append({"layer": layer, "region": region, "scenario": scenario, "grouping": grouping,
                                  "primary": layer == layers[0] and region == "final" and scenario == "all" and grouping == "problem_family_turn",
                                  **cell})
    result = {"identity": identity, "axis_sha256": sha256(args.axis), "random_count": args.random_count,
              "seed": args.seed, "coverage": {"expected": len(records), "complete": len(completed),
              "error_oom": sum(r["status"] == "error_oom" for _, r in records)}, "cells": cells,
              "limitations": ["Negative labels are unflagged, not judge-confirmed non-hacks.",
                  "Random-direction tail fractions describe axis specificity, not a causal or label-permutation test.",
                  "Exact turn matching does not equal exact token-length matching; partial-rank adjustment is a sensitivity analysis.",
                  "Problem-only comparison uses this same strictly selected cohort, not the original broader cohort.",
                  "Bootstrap resamples whole base problems, keeping repeated episodes and turns together.",
                  "Bare legacy axis artifact has matching dimensions but cannot authenticate model provenance.",
                  "Incomplete/OOM or empty-region cases may bias complete-case results."]}
    (args.out / "analysis.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    lines = ["# Random-vector reward-hacking control", "", f"Coverage: {result['coverage']}", "",
             "Random directions are independent isotropic unit vectors. Two-sided extremeness uses |AUROC − 0.5|.", "",
             "Baselines need no direction: the region's token count and the norm of its mean residual.", "",
             "| Region | Grouping | Pairs | AA AUROC | Random AUROC 2.5/50/97.5% | Tail fraction | Token-count AUROC | Residual-norm AUROC |",
             "|---|---|---:|---:|---|---:|---:|---:|"]
    for cell in cells:
        if cell["layer"] == layers[0] and cell["scenario"] == "all" and cell["status"] == "exploratory":
            comparison = cell["auroc"]
            quantiles = " / ".join(f"{quantile:.3f}" for quantile in comparison["random_quantiles"])
            baselines = cell["baseline_aurocs"]
            lines.append(f"| {cell['region']} | {cell['grouping']} | {cell['n_pairs']} | {comparison['assistant_axis']:.4f} | {quantiles} "
                         f"| {comparison['random_direction_tail_fraction']:.4f} | {baselines['region_tokens']:.3f} | {baselines['residual_norm']:.3f} |")
    lines.extend(["", *result["limitations"]])
    (args.out / "analysis.md").write_text("\n".join(lines) + "\n")
    primary = next(c for c in cells if c["primary"])
    if primary["status"] == "exploratory":
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, field, center, label in zip(axes, ("all_aurocs", "all_length_adjusted_correlations"),
                                           (.5, 0), ("|AUROC − 0.5|", "|Length-adjusted rank correlation|")):
            values = np.asarray([np.nan if v is None else v for v in primary[field]])
            ax.hist(np.abs(values[1:][np.isfinite(values[1:])] - center), bins=25, color="#587fa3", label="Random directions")
            if np.isfinite(values[0]):
                ax.axvline(abs(values[0] - center), color="#b74635", label="Assistant Axis", linewidth=2)
            ax.set_xlabel(label)
            ax.set_ylabel("Direction count")
            ax.legend()
        fig.suptitle(f"Layer {layers[0]}, final region; problem/family/turn matched")
        fig.tight_layout()
        fig.savefig(args.out / "random_controls.png", dpi=180)
        plt.close(fig)
    print(json.dumps({k: v for k, v in primary.items() if not k.startswith("all_")}, indent=2))


if __name__ == "__main__":
    main()
