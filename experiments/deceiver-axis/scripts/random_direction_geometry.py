#!/usr/bin/env python
"""Where does a seeded random steering direction sit in the model's own activation geometry? (CPU)

Loads per-response pooled residual means at one layer (the `act_*` cache of the extraction stage: `<role>.pt` with
`pos_p<i>_q<j>` -> (layers, hidden)), estimates the mean activation, the per-dimension spread, the "massive" outlier
dimensions and the top principal components over responses, and then scores each `--seeds` direction (regenerated
exactly as `steered_rh_vllm.py --random-direction-seed` draws it) and each `--axis name=path` against an isotropic
null of `--null-count` fresh random unit directions from the same generator (other seeds). Statistics per direction:

    cos_mean          cosine with the mean activation direction (how much the shift changes the residual norm)
    e_massive_k       energy (sum of squares) on the k dimensions with the largest |mean|  (k = 1, 5, 20)
    e_spread_20       energy on the 20 dimensions with the largest response-to-response sd
    e_pc_k            energy in the span of the top-k principal components of the response means (k = 5, 50)
    whitened_norm     ||v / sd||  (diagonal whitening: a shift along low-variance dimensions is a bigger shock)
    weighted_norm     ||v * sd|| / ||sd||  (how much of the shift lies where activations already vary)

Each is reported with its percentile in the null, so "is 1729 unusual?" has a number. Caveat: response-pooled means,
not per-token activations, so the covariance is of response means (massive dimensions and the mean are robust to that,
the PC and whitening numbers less so).

    python experiments/deceiver-axis/scripts/random_direction_geometry.py --act-dir <act_response_only> --layer 30 \
        --seeds 1729 1730 1731 --axis gi_v2=<persona_contrast.pt> --out <dir>
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from selfconcept.correlation.random_vectors import random_directions  # noqa: E402


def load_responses(act_dir: Path, roles: list[str] | None, layer: int) -> tuple[np.ndarray, list[str]]:
    rows, names = [], []
    for path in sorted(act_dir.glob("*.pt")):
        if roles and path.stem not in roles:
            continue
        data = torch.load(path, map_location="cpu", weights_only=False)
        for key, value in data.items():
            if hasattr(value, "shape") and value.ndim == 2:
                rows.append(value[layer].float().numpy())
                names.append(f"{path.stem}/{key}")
    if not rows:
        raise SystemExit(f"no response means under {act_dir}")
    return np.stack(rows), names


def load_axis(path: Path, layer: int) -> np.ndarray:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    tensor = obj["axis"] if isinstance(obj, dict) and "axis" in obj else obj["vector"] if isinstance(obj, dict) else obj
    vector = tensor[layer].float().numpy()
    return vector / np.linalg.norm(vector)


def stats(v: np.ndarray, geo: dict) -> dict[str, float]:
    out = {"cos_mean": float(v @ geo["mean_unit"])}
    for k in (1, 5, 20):
        out[f"e_massive_{k}"] = float(np.sum(v[geo["massive_order"][:k]] ** 2))
    out["e_spread_20"] = float(np.sum(v[geo["spread_order"][:20]] ** 2))
    for k in (5, 50):
        out[f"e_pc_{k}"] = float(np.sum((geo["pcs"][:k] @ v) ** 2))
    out["whitened_norm"] = float(np.linalg.norm(v / geo["sd"]))
    out["weighted_norm"] = float(np.linalg.norm(v * geo["sd"]) / np.linalg.norm(geo["sd"]))
    return out


def percentile(value: float, null: np.ndarray) -> float:
    return float(100.0 * np.mean(null <= value))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--act-dir", type=Path, required=True)
    ap.add_argument("--roles", nargs="*", help="role stems to use (default: every <role>.pt)")
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--axis", nargs="*", default=[], help="name=path of a direction artifact (layers x hidden)")
    ap.add_argument("--null-count", type=int, default=2000)
    ap.add_argument("--null-first-seed", type=int, default=100000, help="null seeds are first .. first + count - 1")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    X, names = load_responses(args.act_dir, args.roles, args.layer)
    n, hidden = X.shape
    mean = X.mean(0)
    sd = X.std(0) + 1e-6
    centred = X - mean
    # PCs of the response means (n << hidden, so via the n x n Gram matrix)
    u, s, vt = np.linalg.svd(centred, full_matrices=False)
    pcs = vt[:50]
    explained = (s ** 2) / np.sum(s ** 2)
    massive_order = np.argsort(-np.abs(mean))
    spread_order = np.argsort(-sd)
    geo = {"mean_unit": mean / np.linalg.norm(mean), "sd": sd, "pcs": pcs, "massive_order": massive_order, "spread_order": spread_order}

    med_abs = float(np.median(np.abs(mean)))
    massive = [{"dim": int(d), "mean": float(mean[d]), "sd": float(sd[d]), "ratio_to_median_abs_mean": float(abs(mean[d]) / med_abs)} for d in massive_order[:20]]

    null = np.stack([np.asarray(random_directions(hidden, 1, args.null_first_seed + i, args.layer)[0], dtype=np.float64) for i in range(args.null_count)])
    null_stats = {k: np.array([stats(v, geo)[k] for v in null]) for k in stats(null[0], geo)}

    directions = {f"random {s}": np.asarray(random_directions(hidden, 1, s, args.layer)[0], dtype=np.float64) for s in args.seeds}
    for spec in args.axis:
        name, _, path = spec.partition("=")
        directions[name] = load_axis(Path(path), args.layer).astype(np.float64)

    report = {}
    for name, v in directions.items():
        st = stats(v, geo)
        report[name] = {k: {"value": val, "null_pct": percentile(val, null_stats[k]), "null_abs_pct": percentile(abs(val), np.abs(null_stats[k]))} for k, val in st.items()}
        report[name]["massive_coords"] = [float(v[d]) for d in massive_order[:5]]

    result = {
        "layer": args.layer, "n_responses": n, "hidden": hidden, "roles": sorted({x.split("/")[0] for x in names}),
        "mean_norm": float(np.linalg.norm(mean)), "median_abs_mean": med_abs, "massive_dims": massive,
        "pc_explained_top10": [float(x) for x in explained[:10]], "pc_explained_top50": float(np.sum(explained[:50])),
        "null": {k: {"mean": float(v.mean()), "sd": float(v.std()), "p95": float(np.percentile(v, 95)), "p99": float(np.percentile(v, 99)), "max": float(v.max())} for k, v in null_stats.items()},
        "null_count": args.null_count, "directions": report,
    }
    (args.out / "random_direction_geometry.json").write_text(json.dumps(result, indent=1))

    lines = [f"# Random directions in the layer-{args.layer} activation geometry",
             f"{n} response means ({', '.join(result['roles'])}), hidden {hidden}; mean-activation norm {result['mean_norm']:.1f}; "
             f"median |mean_i| {med_abs:.3f}; top-50 PCs explain {100 * result['pc_explained_top50']:.0f}% of response-mean variance "
             f"(top 10: {', '.join(f'{100 * x:.0f}' for x in explained[:10])}%). Null: {args.null_count} isotropic unit directions from the same generator.", "",
             "## Massive dimensions (largest |mean|)", "", "| rank | dim | mean | sd | |mean| / median |mean| |", "|---|---|---|---|---|"]
    for i, m in enumerate(massive[:10], 1):
        lines.append(f"| {i} | {m['dim']} | {m['mean']:+.2f} | {m['sd']:.2f} | {m['ratio_to_median_abs_mean']:.0f}× |")
    keys = list(null_stats)
    lines += ["", "## Directions (value, percentile in the null; |value| percentile in brackets where sign is arbitrary)", "",
              "| statistic | null mean ± sd | null p99 | " + " | ".join(directions) + " |", "|---|---|---|" + "---|" * len(directions)]
    for k in keys:
        nz = result["null"][k]
        cells = []
        for name in directions:
            r = report[name][k]
            cells.append(f"{r['value']:+.4f} (p{r['null_pct']:.0f}" + (f", |·| p{r['null_abs_pct']:.0f})" if k == "cos_mean" else ")"))
        lines.append(f"| {k} | {nz['mean']:.4f} ± {nz['sd']:.4f} | {nz['p99']:.4f} | " + " | ".join(cells) + " |")
    lines += ["", "Coordinates on the five most massive dims: " + "; ".join(f"{name}: " + ", ".join(f"{c:+.4f}" for c in report[name]["massive_coords"]) for name in directions)]
    (args.out / "random_direction_geometry.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
