"""Split a seeded random steering direction into its component inside the top-k principal components of a layer's
response-mean activations and the orthogonal remainder, and save each as a steering artifact (``{"model", "axis"}``
with the (unit) component at every layer, so ``load_unit_axes`` accepts it) plus the energy split and cosines.

    python scripts/make_split_direction.py --act-dir <act_response_only> --layer 30 --seed 1729 --pcs 50
        --model google/gemma-4-31B-it --out <dir> [--axis gi_v2=<persona_contrast.pt>]

Writes ``rand<seed>_pc<k>.pt`` (the in-subspace component, unit), ``rand<seed>_rem<k>.pt`` (the remainder, unit) and
``split.json`` (energy fraction in the subspace, norms, cosines with the full direction and with any --axis).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from random_direction_geometry import load_axis, load_responses  # noqa: E402

from selfconcept.correlation.random_vectors import random_directions  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--act-dir", type=Path, required=True)
    ap.add_argument("--roles", nargs="*", default=None)
    ap.add_argument("--layer", type=int, default=30)
    ap.add_argument("--seed", type=int, default=1729)
    ap.add_argument("--pcs", type=int, default=50)
    ap.add_argument("--model", default="google/gemma-4-31B-it")
    ap.add_argument("--num-layers", type=int, default=60)
    ap.add_argument("--axis", action="append", default=[], help="name=path of a direction artifact to report cosines with")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    responses, _ = load_responses(args.act_dir, args.roles, args.layer)
    mean = responses.mean(axis=0)
    _, singular, vt = np.linalg.svd(responses - mean, full_matrices=False)
    basis = vt[: args.pcs]  # (k, hidden), orthonormal rows
    v = random_directions(responses.shape[1], 1, args.seed, args.layer)[0].astype(np.float64)
    inside = basis.T @ (basis @ v)
    remainder = v - inside
    parts = {f"rand{args.seed}_pc{args.pcs}": inside, f"rand{args.seed}_rem{args.pcs}": remainder}
    report = {"seed": args.seed, "layer": args.layer, "pcs": args.pcs, "responses": int(responses.shape[0]),
              "variance_explained_by_pcs": float((singular[: args.pcs] ** 2).sum() / (singular ** 2).sum()),
              "energy_in_pcs": float(inside @ inside), "norm_in_pcs": float(np.linalg.norm(inside)),
              "norm_remainder": float(np.linalg.norm(remainder)), "parts": {}}
    axes = {}
    for spec in args.axis:
        name, path = spec.split("=", 1)
        axes[name] = load_axis(Path(path), args.layer).astype(np.float64)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, part in parts.items():
        unit = part / np.linalg.norm(part)
        entry = {"norm_in_full": float(np.linalg.norm(part)), "cos_with_full": float(unit @ v),
                 "cos_with_mean_activation": float(unit @ mean / np.linalg.norm(mean))}
        for axis_name, axis in axes.items():
            entry[f"cos_with_{axis_name}"] = float(unit @ axis)
        axis_tensor = torch.as_tensor(np.tile(unit.astype(np.float32), (args.num_layers, 1)))
        torch.save({"model": args.model, "axis": axis_tensor, "note": f"random direction {args.seed} layer {args.layer}: {name} "
                    f"(unit; use only at --steer-layer {args.layer})"}, args.out / f"{name}.pt")
        report["parts"][name] = entry
    (args.out / "split.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
