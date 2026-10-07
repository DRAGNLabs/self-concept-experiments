"""Score named candidate directions against the cached reward-hacking cohort.

Generalizes random_vector_analysis (one axis vs random directions) to several named directions, each also
compared after projecting out a reference axis (the Assistant Axis) and with that axis' projection as a control.
Reuses its cache loader, pair counting, projection and baseline code; adds STRONG-only AUROCs and the
cosine of each direction with the in-cohort hack-minus-non-hack mean difference (a ceiling, fit with the
scored problem held out).

usage: python -m selfconcept.correlation.direction_analysis SHARD_DIR [...] --direction NAME=PATH [...]
           --reference NAME=PATH --out DIR
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from selfconcept.assistant_axis.projection import load_unit_axes
from .analyze import binary_outcome
from .random_vector_analysis import baseline_aurocs, group_indices, load_cache, pair_totals, project
from .random_vectors import random_directions, REGIONS, sha256


def partial_rank_correlations(scores, labels, controls, strata):
    """Rank, center within stratum, residualize scores and labels on the controls, correlate."""
    values = np.column_stack([rankdata(scores, axis=0), rankdata(labels), rankdata(controls, axis=0)])
    for indices in group_indices(strata).values():
        values[indices] -= values[indices].mean(axis=0)
    n_controls = controls.shape[1]
    targets, control_ranks = values[:, :-n_controls], values[:, -n_controls:]
    residuals = targets - control_ranks @ np.linalg.lstsq(control_ranks, targets, rcond=None)[0]
    x, y = residuals[:, :-1], residuals[:, -1]
    denominator = np.linalg.norm(x, axis=0) * np.linalg.norm(y)
    return np.divide(x.T @ y, denominator, out=np.full(x.shape[1], np.nan), where=denominator > 1e-10)


def tail(statistic, random, center):
    valid = random[np.isfinite(random)]
    if not np.isfinite(statistic) or not len(valid):
        return None
    exceed = int(np.sum(np.abs(valid - center) >= abs(statistic - center)))
    return {"value": float(statistic), "random_quantiles": np.quantile(valid, [.025, .5, .975]).tolist(),
            "n_random_at_least_as_extreme": exceed, "random_direction_tail_fraction": (1 + exceed) / (1 + len(valid))}


def orthogonalize(direction, reference):
    residual = direction - reference * (direction @ reference)
    norm = np.linalg.norm(residual)
    return residual / norm if norm > 0 else residual


def held_out_diff_means(residuals, labels, problems):
    """Leave-one-problem-out hack-minus-non-hack mean direction; each row is scored by a fit that excluded its problem."""
    scores = np.full(len(labels), np.nan)
    for problem, indices in group_indices(problems).items():
        mask = np.ones(len(labels), bool)
        mask[indices] = False
        positives, negatives = residuals[mask & (labels == 1)], residuals[mask & (labels == 0)]
        if not len(positives) or not len(negatives):
            continue
        direction = positives.mean(axis=0) - negatives.mean(axis=0)
        scores[indices] = residuals[indices] @ (direction / np.linalg.norm(direction))
    return scores


def auroc_by_column(scores, labels, strata, problems, bootstrap, rng):
    u, pairs, n_strata = pair_totals(scores, labels, strata, problems)
    if not len(pairs):
        return None, None, 0
    aucs = u.sum(axis=0) / pairs.sum()
    draws = []
    for _ in range(bootstrap):
        indices = rng.integers(len(pairs), size=len(pairs))
        draws.append(u[indices].sum(axis=0) / pairs[indices].sum())
    cis = np.quantile(np.stack(draws), [.025, .975], axis=0).T if draws else None
    return aucs, cis, n_strata


def analyze_cell(residuals, columns, names, n_named, labels, strong, lengths, strata, problems, bootstrap, seed):
    """columns: named directions, then the reference axis, then random directions (all unit vectors)."""
    informative = [i for i in group_indices(strata).values() if len(np.unique(labels[i])) == 2]
    if not informative:
        return {"status": "no_matched_pairs", "n": len(labels)}
    keep = np.sort(np.concatenate(informative))
    residuals, labels, strong, lengths = residuals[keep], labels[keep], strong[keep], lengths[keep]
    strata, problems = [strata[i] for i in keep], [problems[i] for i in keep]
    scores = project(residuals, columns)
    reference_index = n_named
    random_slice = slice(n_named + 1, None)
    rng = np.random.default_rng(seed)
    aucs, cis, n_strata = auroc_by_column(scores, labels, strata, problems, bootstrap, rng)
    if aucs is None:
        return {"status": "no_matched_pairs", "n": len(labels)}
    length_adjusted = partial_rank_correlations(scores, labels, lengths, strata)
    reference_adjusted = partial_rank_correlations(scores, labels, np.column_stack([lengths, scores[:, reference_index]]), strata)
    strong_mask = (labels == 0) | strong
    strong_aucs = auroc_by_column(scores[strong_mask], labels[strong_mask], [strata[i] for i in np.flatnonzero(strong_mask)],
                                  [problems[i] for i in np.flatnonzero(strong_mask)], 0, rng)[0]
    ceiling = held_out_diff_means(residuals, labels, problems)
    ceiling_valid = np.isfinite(ceiling)
    ceiling_auroc = None
    if ceiling_valid.sum():
        u, pairs, _ = pair_totals(ceiling[ceiling_valid, None], labels[ceiling_valid],
                                  [strata[i] for i in np.flatnonzero(ceiling_valid)], [problems[i] for i in np.flatnonzero(ceiling_valid)])
        ceiling_auroc = float(u.sum() / pairs.sum()) if len(pairs) else None
    directions = {}
    for index, name in enumerate(names):
        directions[name] = {
            "auroc": tail(aucs[index], aucs[random_slice], .5),
            "problem_bootstrap_ci": cis[index].tolist() if cis is not None else None,
            "strong_only_auroc": float(strong_aucs[index]) if strong_aucs is not None and np.isfinite(strong_aucs[index]) else None,
            "length_adjusted_rank_correlation": tail(length_adjusted[index], length_adjusted[random_slice], 0),
            "length_and_reference_adjusted_rank_correlation": tail(reference_adjusted[index], reference_adjusted[random_slice], 0),
        }
    return {"status": "exploratory", "n": int(len(labels)), "n_positive": int(labels.sum()), "n_strong": int(strong[labels == 1].sum()),
            "n_strata": n_strata, "n_problems": len(set(problems)), "directions": directions,
            "held_out_diff_means_auroc": ceiling_auroc,
            "baseline_aurocs": None}


def parse_named(items):
    named = {}
    for item in items or []:
        name, _, path = item.partition("=")
        if not name or not path or name in named:
            raise ValueError(f"Expected unique NAME=PATH, got {item}")
        named[name] = Path(path)
    return named


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directories", nargs="+", type=Path)
    parser.add_argument("--direction", action="append", required=True, help="NAME=PATH of a (layers, hidden) axis artifact")
    parser.add_argument("--reference", required=True, help="NAME=PATH of the reference axis to orthogonalize against and control for")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--random-count", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    named, reference = parse_named(args.direction), parse_named([args.reference])
    [(reference_name, reference_path)] = reference.items()
    if reference_name in named:
        parser.error("The reference axis cannot also be a candidate direction")
    records, manifest = load_cache(args.directories)
    identity, layers = manifest["identity"], manifest["identity"]["layers"]
    hidden, num_layers = manifest["hidden_size"], manifest["num_layers"]
    axes = {name: load_unit_axes(path, layers, num_layers, hidden, identity["model"]) for name, path in (named | reference).items()}
    completed = [(d, r) for d, r in records if r["status"] == "complete" and binary_outcome(r["outcome"])[0] is not None]
    if not completed:
        raise ValueError("No complete scorable records")
    rows = [r for _, r in completed]
    labels = np.asarray([binary_outcome(r["outcome"])[0] for r in rows])
    strong = np.asarray([r["outcome"].get("strength") == "STRONG" for r in rows])
    lengths = np.asarray([[r["prompt_tokens"], *r["region_counts"][1:]] for r in rows])
    region_tokens = np.asarray([r["region_counts"] for r in rows])
    problems = [json.dumps([r["scenario"], r["outcome"]["problem"]]) for r in rows]
    strata = [r["outcome"]["stratum"] for r in rows]
    args.out.mkdir(parents=True, exist_ok=True)
    cells, geometry = [], {}
    for layer in layers:
        reference_direction = axes[reference_name][layer]["direction"]
        base = [axes[name][layer]["direction"] for name in named]
        orthogonal = [orthogonalize(direction, reference_direction) for direction in base]
        names = [*named, *(f"{name}|orth_{reference_name}" for name in named)]
        randoms = random_directions(hidden, args.random_count, args.seed, layer)
        columns = np.vstack([*base, *orthogonal, reference_direction, randoms])
        geometry[str(layer)] = {name: {f"cosine_with_{reference_name}": float(direction @ reference_direction)} for name, direction in zip(named, base)}
        matrices = np.stack([np.load(d / r["residuals"])[f"layer_{layer}"] for d, r in completed])
        for region_index, region in enumerate(REGIONS):
            residuals = matrices[:, region_index, :]
            valid = np.isfinite(residuals).all(axis=1)
            for scenario in ["all", *sorted({r["scenario"] for r in rows})]:
                subset = np.flatnonzero(valid & np.asarray([scenario == "all" or r["scenario"] == scenario for r in rows]))
                for grouping in ("problem", "problem_family_turn"):
                    grouping_values = problems if grouping == "problem" else strata
                    subset_strata, subset_problems = [grouping_values[i] for i in subset], [problems[i] for i in subset]
                    cell = analyze_cell(residuals[subset], columns, [*names, reference_name], len(names), labels[subset], strong[subset],
                                        lengths[subset], subset_strata, subset_problems, args.bootstrap, args.seed)
                    if cell["status"] == "exploratory":
                        cell["baseline_aurocs"] = baseline_aurocs(region_tokens[subset, region_index], residuals[subset],
                                                                  labels[subset], subset_strata, subset_problems)
                        # Cosine of each candidate with the whole-cohort hack-minus-non-hack direction, against random directions.
                        positives, negatives = residuals[subset][labels[subset] == 1], residuals[subset][labels[subset] == 0]
                        if len(positives) and len(negatives):
                            difference = positives.mean(axis=0) - negatives.mean(axis=0)
                            difference = difference / np.linalg.norm(difference)
                            cosines = columns @ difference
                            cell["cosine_with_diff_means"] = {name: tail(cosines[i], cosines[len(names) + 1:], 0)
                                                              for i, name in enumerate([*names, reference_name])}
                    cells.append({"layer": layer, "region": region, "scenario": scenario, "grouping": grouping,
                                  "primary": layer == layers[0] and region == "cot" and scenario == "all" and grouping == "problem_family_turn",
                                  **cell})
    result = {"identity": identity, "directions": {name: str(path) for name, path in named.items()},
              "direction_sha256": {name: sha256(path) for name, path in (named | reference).items()},
              "reference": reference_name, "random_count": args.random_count, "seed": args.seed,
              "coverage": {"expected": len(records), "complete": len(completed)}, "geometry": geometry, "cells": cells,
              "limitations": ["Negative labels are unflagged, not judge-confirmed non-hacks.",
                              "Random-direction tail fractions describe specificity, not a label-permutation test.",
                              "The held-out diff-of-means AUROC is a ceiling for layer-wise linear directions on this cohort, not a schemer vector.",
                              "Orthogonalized directions change the hypothesis being tested; both versions are reported."]}
    (args.out / "analysis.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    (args.out / "analysis.md").write_text(render_markdown(result))
    primary = next(c for c in cells if c["primary"])
    print(json.dumps({k: v for k, v in primary.items() if k != "cosine_with_diff_means"}, indent=2))


def render_markdown(result):
    layer = result["identity"]["layers"][0]
    lines = [f"# Candidate directions vs reward hacking (layer {layer}, all scenarios)", "",
             f"Coverage: {result['coverage']}. Reference axis: {result['reference']}. Random directions: {result['random_count']}.", ""]
    for grouping in ("problem_family_turn", "problem"):
        lines += [f"## Grouping: {grouping}", "",
                  "| Region | Direction | AUROC | 95% CI (problems) | Random 2.5/50/97.5% | Tail | STRONG-only | Length-adj. corr (tail) | +ref-adj. corr (tail) | cos(diff-means) (tail) |",
                  "|---|---|---:|---|---|---:|---:|---|---|---|"]
        for cell in result["cells"]:
            if cell["layer"] != layer or cell["scenario"] != "all" or cell["grouping"] != grouping or cell["status"] != "exploratory":
                continue
            for name, stats in cell["directions"].items():
                auroc = stats["auroc"]
                if auroc is None:
                    continue
                ci = stats["problem_bootstrap_ci"]
                la, ra = stats["length_adjusted_rank_correlation"], stats["length_and_reference_adjusted_rank_correlation"]
                cos = cell.get("cosine_with_diff_means", {}).get(name)
                fmt = lambda t: f"{t['value']:.3f} ({t['random_direction_tail_fraction']:.3f})" if t else "—"
                lines.append(f"| {cell['region']} | {name} | {auroc['value']:.3f} | {ci[0]:.3f}–{ci[1]:.3f} | "
                             f"{' / '.join(f'{q:.3f}' for q in auroc['random_quantiles'])} | {auroc['random_direction_tail_fraction']:.3f} | "
                             f"{stats['strong_only_auroc'] if stats['strong_only_auroc'] is None else f'{stats['strong_only_auroc']:.3f}'} | {fmt(la)} | {fmt(ra)} | {fmt(cos)} |")
            baselines = cell["baseline_aurocs"]
            lines.append(f"| {cell['region']} | *baselines: token count / residual norm / held-out diff-means* | "
                         f"{baselines['region_tokens']:.3f} / {baselines['residual_norm']:.3f} / "
                         f"{cell['held_out_diff_means_auroc'] if cell['held_out_diff_means_auroc'] is None else f'{cell['held_out_diff_means_auroc']:.3f}'} | | | | | | | |")
        lines.append("")
    lines += result["limitations"]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
