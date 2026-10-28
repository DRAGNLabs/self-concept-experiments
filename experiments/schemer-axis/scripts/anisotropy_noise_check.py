# Run: PYTHONPATH=src .venv-313/bin/python -I experiments/schemer-axis/scripts/anisotropy_noise_check.py  (CPU, ~1 min)
"""Does the random-direction AUROC spread come from the separation or from where the within-class variance sits?
Adds isotropic noise that barely changes the separation along fixed directions but spreads the within-class
variance over all 2880 dims, then re-draws random directions."""
import json, sys
from pathlib import Path
import numpy as np
from scipy.stats import norm, pearsonr
from selfconcept.assistant_axis.projection import load_unit_axes
from selfconcept.correlation.analyze import binary_outcome
from selfconcept.correlation.random_vector_analysis import group_indices, load_cache, pair_totals, project
from selfconcept.correlation.random_vectors import random_directions, REGIONS

root = Path.home() / "nobackup/autodelete/rh-random-control-v3"
records, manifest = load_cache(sorted(root.glob("full/shard_*")))
identity = manifest["identity"]; [layer] = identity["layers"]; hidden = manifest["hidden_size"]
aa = load_unit_axes(root / "assistant_axis.pt", [layer], manifest["num_layers"], hidden, identity["model"])[layer]["direction"].astype(np.float64)
completed = [(d, r) for d, r in records if r["status"] == "complete" and binary_outcome(r["outcome"])[0] is not None]
rows = [r for _, r in completed]
matrices = np.stack([np.load(d / r["residuals"])[f"layer_{layer}"] for d, r in completed])
labels_all = np.asarray([binary_outcome(r["outcome"])[0] for r in rows])
problems_all = [json.dumps([r["scenario"], r["outcome"]["problem"]]) for r in rows]
strata_all = [r["outcome"]["stratum"] for r in rows]
randoms = random_directions(hidden, 256, 1729, layer).astype(np.float64)

def aurocs(scores, labels, strata, problems):
    u, pairs, _ = pair_totals(scores, labels, strata, problems)
    return u.sum(axis=0) / pairs.sum()

def q(v): return " / ".join(f"{x:.3f}" for x in np.quantile(v, [.025, .5, .975]))

for region in ("cot", "final"):
    ri = REGIONS.index(region)
    res = matrices[:, ri, :]
    valid = np.flatnonzero(np.isfinite(res).all(axis=1))
    labels, strata, problems = labels_all[valid], [strata_all[i] for i in valid], [problems_all[i] for i in valid]
    inf = [i for i in group_indices(strata).values() if len(np.unique(labels[i])) == 2]
    keep = np.sort(np.concatenate(inf))
    h = res[valid][keep].astype(np.float64); labels = labels[keep]
    strata = [strata[i] for i in keep]; problems = [problems[i] for i in keep]
    groups = list(group_indices(strata).values())
    # stratified mean difference (pair-weighted) and within-class residuals
    delta = np.zeros(hidden); w = 0.0; within = h.copy()
    for g in groups:
        hk, nk = g[labels[g] == 1], g[labels[g] == 0]
        npairs = len(hk) * len(nk)
        delta += npairs * (h[hk].mean(0) - h[nk].mean(0)); w += npairs
        within[hk] -= h[hk].mean(0); within[nk] -= h[nk].mean(0)
    delta /= w
    n = len(h)
    _, s, vt = np.linalg.svd(within, full_matrices=False)
    lam = s ** 2 / n                      # eigenvalues of within-class covariance
    tr = lam.sum()
    d_hat = delta / np.linalg.norm(delta)
    var_delta = ((within @ d_hat) ** 2).mean()
    dprime = np.linalg.norm(delta) / np.sqrt(var_delta)
    deff = tr ** 2 / (lam ** 2).sum()
    cos_pcs = vt[:10] @ d_hat
    print(f"\n=== {region} region, n={n}, hacks={labels.sum()} ===")
    print(f"|delta| = {np.linalg.norm(delta):.1f}; within-class var along delta = {var_delta:.1f}; d' along delta = {dprime:.2f} "
          f"(Gaussian AUROC along delta = {norm.cdf(dprime/np.sqrt(2)):.3f})")
    print(f"tr(Sigma) = {tr:.1f}; var fraction along delta = {var_delta/tr:.3f} (isotropic would be {1/hidden:.5f}); "
          f"effective dim of within-class var = {deff:.1f}; top-10 PC var fractions = {np.round(lam[:10]/tr, 3).tolist()}")
    print(f"cos(delta, PC1..10) = {np.round(cos_pcs, 2).tolist()}; |delta|^2 fraction in top-10 PCs = {(cos_pcs**2).sum():.2f}")
    sd_z = np.linalg.norm(delta) / np.sqrt(2 * tr)
    print(f"Gaussian prediction for random directions: z sd = {sd_z:.3f} -> 2.5/97.5% AUROC = "
          f"{norm.cdf(-1.96*sd_z):.3f} / {norm.cdf(1.96*sd_z):.3f}")
    sd_iso = dprime / np.sqrt(2 * hidden)
    print(f"Same d' but isotropic within-class variance: z sd = {sd_iso:.4f} -> {norm.cdf(-1.96*sd_iso):.3f} / {norm.cdf(1.96*sd_iso):.3f}")
    # Observed vs per-direction Gaussian prediction
    pc2 = vt[1]
    dirs = np.vstack([aa, pc2, d_hat, randoms])
    auc = aurocs(project(h, dirs), labels, strata, problems)
    print(f"Observed: AA {auc[0]:.3f}, PC2 {auc[1]:.3f}, delta (in-sample) {auc[2]:.3f}, random 2.5/50/97.5 = {q(auc[3:])}")
    pred = norm.cdf((randoms @ delta) / np.sqrt(2 * ((within @ randoms.T) ** 2).mean(0)))
    r, _ = pearsonr(pred, auc[3:])
    print(f"Per-direction Gaussian prediction vs observed random AUROC: r = {r:.3f}; predicted 2.5/50/97.5 = {q(pred)}")
    # Isotropic-noise test: add N(0, var_delta * f) per coordinate. Separation along delta barely moves;
    # the total within-class variance explodes.
    rng = np.random.default_rng(7)
    for f in (0.1, 1.0):
        noise = rng.standard_normal(h.shape) * np.sqrt(var_delta * f)
        hn = h + noise
        aucn = aurocs(project(hn, dirs), labels, strata, problems)
        tr_n = tr + hidden * var_delta * f
        print(f"+ isotropic noise, var per coord = {f} x var along delta (tr(Sigma) x{tr_n/tr:.0f}, d' along delta -> {dprime/np.sqrt(1+f):.2f}): "
              f"AA {aucn[0]:.3f}, PC2 {aucn[1]:.3f}, delta {aucn[2]:.3f}, random 2.5/50/97.5 = {q(aucn[3:])}")
    # Converse: keep only the top-10 PC subspace (same separation, almost same variance, dimension 10)
    sub = vt[:10]
    coords = h @ sub.T
    r10 = rng.standard_normal((256, 10)); r10 /= np.linalg.norm(r10, axis=1, keepdims=True)
    auc10 = aurocs(coords @ r10.T, labels, strata, problems)
    print(f"Random directions restricted to the top-10 PC subspace: 2.5/50/97.5 = {q(auc10)}")
