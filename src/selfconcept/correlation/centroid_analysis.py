"""Does the activation centroid explain the residual-norm and random-direction reward-hacking AUROCs?

Koby's point: if the centroid of the region-mean residuals is far from the origin, the norm of a residual is
nearly its projection onto the centroid direction, so "shorter norm predicts hacking" is a statement about one
direction, and random directions inherit it in proportion to their (small) cosine with that direction.

For each region and grouping this reports, on the strict matched cohort:
  geometry   norm of the centroid, norms of the residuals and of their deviations from the centroid, the cosine of
             each residual with the centroid direction, and how the within-group variance splits between the
             centroid direction and the top principal components of the within-group deviations
  aurocs     stratified AUROCs for the norm, the centroid projection, the norm of the part orthogonal to the
             centroid, the distance to the centroid, the Assistant Axis, the AA with the centroid direction (and
             the top PCs) projected out, the top PCs themselves, and random directions before and after the same
             projections
  random     how well a random direction's cosine with the centroid direction (and with PC1) predicts its AUROC

usage: python -m selfconcept.correlation.centroid_analysis SHARD_DIR [...] --axis PATH --out DIR
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from selfconcept.assistant_axis.projection import load_unit_axes
from .analyze import binary_outcome
from .random_vector_analysis import group_indices, load_cache, pair_totals, project
from .random_vectors import random_directions, REGIONS


def unit(v):
    return v / np.linalg.norm(v)


def project_out(directions, basis):
    """Remove the span of the orthonormal rows of `basis` from each row of `directions`; renormalize."""
    residual = directions - (directions @ basis.T) @ basis
    return residual / np.linalg.norm(residual, axis=1, keepdims=True)


def aurocs(scores, labels, strata, problems, bootstrap=0, seed=1729):
    u, pairs, _ = pair_totals(scores, labels, strata, problems)
    aucs = u.sum(axis=0) / pairs.sum()
    if not bootstrap:
        return aucs, None
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(bootstrap):
        indices = rng.integers(len(pairs), size=len(pairs))
        draws.append(u[indices].sum(axis=0) / pairs[indices].sum())
    return aucs, np.quantile(np.stack(draws), [.025, .975], axis=0).T


def quantiles(values, qs=(.05, .25, .5, .75, .95)):
    return {str(q): float(np.quantile(values, q)) for q in qs}


def summarize_random(values, center=.5):
    return {"quantiles": quantiles(values, (.025, .25, .5, .75, .975)),
            "fraction_beyond_0.6_either_side": float(np.mean(np.abs(values - center) >= .1))}


def analyze(residuals, labels, strata, problems, region_tokens, aa, randoms, n_pcs, bootstrap, seed):
    informative = [i for i in group_indices(strata).values() if len(np.unique(labels[i])) == 2]
    keep = np.sort(np.concatenate(informative))
    h = residuals[keep].astype(np.float64)
    labels, region_tokens = labels[keep], region_tokens[keep]
    strata, problems = [strata[i] for i in keep], [problems[i] for i in keep]

    centroid = h.mean(axis=0)
    c_hat = unit(centroid)
    norms = np.linalg.norm(h, axis=1)
    along = h @ c_hat
    perp = h - np.outer(along, c_hat)
    perp_norm = np.linalg.norm(perp, axis=1)
    deviation = h - centroid
    deviation_norm = np.linalg.norm(deviation, axis=1)
    cosines = along / norms

    # Within-group deviations (the only variation the stratified AUROC can see) and their principal components.
    within = deviation.copy()
    for indices in group_indices(strata).values():
        within[indices] -= within[indices].mean(axis=0)
    total_var = float((within ** 2).sum())
    var_along_centroid = float(((within @ c_hat) ** 2).sum())
    _, s, vt = np.linalg.svd(within, full_matrices=False)
    pcs = vt[:n_pcs]
    pc_var = (s[:n_pcs] ** 2) / total_var

    geometry = {
        "n": int(len(h)), "n_positive": int(labels.sum()),
        "centroid_norm": float(np.linalg.norm(centroid)),
        "residual_norm": quantiles(norms),
        "distance_to_centroid": quantiles(deviation_norm),
        "norm_orthogonal_to_centroid": quantiles(perp_norm),
        "cosine_with_centroid": quantiles(cosines),
        "within_group_variance_fraction_along_centroid": var_along_centroid / total_var,
        "within_group_variance_fraction_top_pcs": pc_var.tolist(),
        "cosine_pcs_with_centroid": (pcs @ c_hat).tolist(),
        "cosine_aa_with_centroid": float(aa @ c_hat),
        "cosine_aa_with_pcs": (pcs @ aa).tolist(),
        "spearman_norm_vs_region_tokens": float(spearmanr(norms, region_tokens).correlation),
        "spearman_centroid_projection_vs_region_tokens": float(spearmanr(along, region_tokens).correlation),
        "spearman_norm_vs_centroid_projection": float(spearmanr(norms, along).correlation),
    }

    basis_c = c_hat[None, :]
    basis_pcs = np.vstack([c_hat, pcs])
    basis_pcs = np.linalg.qr(basis_pcs.T)[0].T  # orthonormalize centroid + PCs jointly
    aa_perp_c = project_out(aa[None, :], basis_c)[0]
    aa_perp_pcs = project_out(aa[None, :], basis_pcs)[0]
    randoms_perp_c = project_out(randoms, basis_c)
    randoms_perp_pcs = project_out(randoms, basis_pcs)

    named = {"residual_norm": norms, "centroid_projection": along, "norm_orthogonal_to_centroid": perp_norm,
             "distance_to_centroid": deviation_norm, "region_tokens": region_tokens.astype(float)}
    direction_columns = np.vstack([aa, aa_perp_c, aa_perp_pcs, pcs, randoms, randoms_perp_c, randoms_perp_pcs])
    scores = np.column_stack([np.column_stack(list(named.values())), project(h, direction_columns)])
    aucs, cis = aurocs(scores, labels, strata, problems, bootstrap, seed)
    k = len(named)
    aa_auc, aa_c, aa_pcs = aucs[k], aucs[k + 1], aucs[k + 2]
    pc_aucs = aucs[k + 3:k + 3 + n_pcs]
    r0 = k + 3 + n_pcs
    rand, rand_c, rand_pcs = aucs[r0:r0 + len(randoms)], aucs[r0 + len(randoms):r0 + 2 * len(randoms)], aucs[r0 + 2 * len(randoms):]
    cos_rand_c = randoms @ c_hat
    cos_rand_pc1 = randoms @ pcs[0]

    def with_ci(index):
        return {"auroc": float(aucs[index]), "ci": cis[index].tolist() if cis is not None else None}

    def tail(value, random):
        exceed = int(np.sum(np.abs(random - .5) >= abs(value - .5)))
        return (1 + exceed) / (1 + len(random))

    result = {
        "geometry": geometry,
        "direction_free_aurocs": {name: with_ci(i) for i, name in enumerate(named)},
        "assistant_axis": {
            "raw": {**with_ci(k), "random_tail": tail(aa_auc, rand)},
            "centroid_projected_out": {**with_ci(k + 1), "random_tail": tail(aa_c, rand_c)},
            "centroid_and_top_pcs_projected_out": {**with_ci(k + 2), "random_tail": tail(aa_pcs, rand_pcs)},
        },
        "top_pc_aurocs": pc_aucs.tolist(),
        "random_directions": {
            "raw": summarize_random(rand),
            "centroid_projected_out": summarize_random(rand_c),
            "centroid_and_top_pcs_projected_out": summarize_random(rand_pcs),
            "cosine_with_centroid": quantiles(np.abs(cos_rand_c), (.5, .95)),
            "pearson_auroc_vs_cosine_with_centroid": float(np.corrcoef(cos_rand_c, rand - .5)[0, 1]),
            "pearson_auroc_vs_cosine_with_pc1": float(np.corrcoef(cos_rand_pc1, rand - .5)[0, 1]),
            "r2_auroc_on_cosines_with_centroid_and_pcs": None,
        },
    }
    # How much of the spread in random-direction AUROCs is explained by loadings on the centroid + top PCs?
    loadings = np.column_stack([cos_rand_c, randoms @ pcs.T, np.ones(len(randoms))])
    coef, *_ = np.linalg.lstsq(loadings, rand - .5, rcond=None)
    fitted = loadings @ coef
    result["random_directions"]["r2_auroc_on_cosines_with_centroid_and_pcs"] = float(
        1 - ((rand - .5 - fitted) ** 2).sum() / ((rand - .5 - (rand - .5).mean()) ** 2).sum())
    return result


def render(results, layer):
    lines = [f"# Centroid check for the random-direction reward-hacking control (layer {layer}, all scenarios)", ""]
    for grouping, regions in results.items():
        lines += [f"## Grouping: {grouping}", "",
                  "| Region | n (hacks) | centroid norm | residual norm (median) | distance to centroid (median) | cos(h, centroid) 5/50/95% | within-group var along centroid | top-3 PC var | cos(AA, centroid) | AA AUROC | AA ⊥ centroid | AA ⊥ centroid+PCs |",
                  "|---|---|---:|---:|---:|---|---:|---|---:|---|---|---|"]
        for region, r in regions.items():
            g, a = r["geometry"], r["assistant_axis"]
            cos = g["cosine_with_centroid"]
            fmt = lambda d: f"{d['auroc']:.3f} [{d['ci'][0]:.2f}–{d['ci'][1]:.2f}] tail {d['random_tail']:.3f}"
            lines.append(f"| {region} | {g['n']} ({g['n_positive']}) | {g['centroid_norm']:.0f} | {g['residual_norm']['0.5']:.0f} | "
                         f"{g['distance_to_centroid']['0.5']:.0f} | {cos['0.05']:.3f} / {cos['0.5']:.3f} / {cos['0.95']:.3f} | "
                         f"{g['within_group_variance_fraction_along_centroid']:.3f} | "
                         f"{' / '.join(f'{v:.3f}' for v in g['within_group_variance_fraction_top_pcs'][:3])} | "
                         f"{g['cosine_aa_with_centroid']:+.3f} | {fmt(a['raw'])} | {fmt(a['centroid_projected_out'])} | {fmt(a['centroid_and_top_pcs_projected_out'])} |")
        lines += ["", "| Region | norm AUROC | centroid-projection AUROC | ⊥-norm AUROC | distance-to-centroid AUROC | token-count AUROC | PC1..PC5 AUROC | random 2.5/50/97.5% | random ⊥ centroid | random ⊥ centroid+PCs | corr(random AUROC, cos centroid) | corr(.., cos PC1) | R² on centroid+PC loadings | ρ(norm, tokens) |",
                  "|---|---:|---:|---:|---:|---:|---|---|---|---|---:|---:|---:|---:|"]
        for region, r in regions.items():
            d, rd, g = r["direction_free_aurocs"], r["random_directions"], r["geometry"]
            q = lambda s: " / ".join(f"{s['quantiles'][k]:.3f}" for k in ("0.025", "0.5", "0.975"))
            lines.append(f"| {region} | {d['residual_norm']['auroc']:.3f} | {d['centroid_projection']['auroc']:.3f} | "
                         f"{d['norm_orthogonal_to_centroid']['auroc']:.3f} | {d['distance_to_centroid']['auroc']:.3f} | {d['region_tokens']['auroc']:.3f} | "
                         f"{' / '.join(f'{v:.2f}' for v in r['top_pc_aurocs'][:5])} | {q(rd['raw'])} | {q(rd['centroid_projected_out'])} | "
                         f"{q(rd['centroid_and_top_pcs_projected_out'])} | {rd['pearson_auroc_vs_cosine_with_centroid']:+.3f} | "
                         f"{rd['pearson_auroc_vs_cosine_with_pc1']:+.3f} | {rd['r2_auroc_on_cosines_with_centroid_and_pcs']:.3f} | "
                         f"{g['spearman_norm_vs_region_tokens']:+.3f} |")
        lines.append("")
    lines += ["Notes: AUROCs are stratified (hacks vs non-hacks within the grouping, pooled); 'random' is 256 isotropic unit",
              "directions. Projecting a fixed direction cannot be changed by subtracting any per-stratum constant, so centering",
              "alone leaves every direction's AUROC untouched; what the centroid changes is which directions carry the",
              "within-group variance. 'AA ⊥ centroid+PCs' removes the centroid direction and the top PCs of the within-group",
              "deviations (fit on the whole cohort, so this is descriptive, not held out). The norm of the whole residual splits",
              "as norm² = (centroid projection)² + (⊥-norm)²."]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directories", nargs="+", type=Path)
    parser.add_argument("--axis", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--random-count", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--pcs", type=int, default=10)
    args = parser.parse_args()
    records, manifest = load_cache(args.directories)
    identity = manifest["identity"]
    [layer] = identity["layers"]
    hidden = manifest["hidden_size"]
    aa = load_unit_axes(args.axis, [layer], manifest["num_layers"], hidden, identity["model"])[layer]["direction"].astype(np.float64)
    randoms = random_directions(hidden, args.random_count, args.seed, layer).astype(np.float64)
    completed = [(d, r) for d, r in records if r["status"] == "complete" and binary_outcome(r["outcome"])[0] is not None]
    rows = [r for _, r in completed]
    matrices = np.stack([np.load(d / r["residuals"])[f"layer_{layer}"] for d, r in completed])
    labels = np.asarray([binary_outcome(r["outcome"])[0] for r in rows])
    region_tokens = np.asarray([r["region_counts"] for r in rows])
    problems = [json.dumps([r["scenario"], r["outcome"]["problem"]]) for r in rows]
    strata = [r["outcome"]["stratum"] for r in rows]
    results = {}
    for grouping in ("problem_family_turn", "problem"):
        grouping_values = strata if grouping == "problem_family_turn" else problems
        results[grouping] = {}
        for region_index, region in enumerate(REGIONS):
            residuals = matrices[:, region_index, :]
            valid = np.flatnonzero(np.isfinite(residuals).all(axis=1))
            results[grouping][region] = analyze(residuals[valid], labels[valid], [grouping_values[i] for i in valid],
                                                [problems[i] for i in valid], region_tokens[valid, region_index],
                                                aa, randoms, args.pcs, args.bootstrap, args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "analysis.json").write_text(json.dumps({"identity": identity, "axis": str(args.axis), "results": results}, indent=1))
    markdown = render(results, layer)
    (args.out / "analysis.md").write_text(markdown)
    print(markdown)


if __name__ == "__main__":
    main()
