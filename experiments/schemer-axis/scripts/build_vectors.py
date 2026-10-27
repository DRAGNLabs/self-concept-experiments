"""Build the schemer-axis candidate directions from per-role vectors (PLAN.md, "Vector definitions").

Reads {role}.pt files written by the assistant-axis pipeline step 4 ({"vector": (layers, hidden), "type", "role"})
and the model's Assistant Axis, and writes one artifact per candidate in the AA format accepted by
selfconcept.assistant_axis.projection.load_unit_axes: {"axis": (layers, hidden) tensor, "model": ..., plus provenance}.

    python scripts/build_vectors.py --vectors-dir <root>/vectors_with_cot --model <hf model id> --out <dir>
        [--assistant-axis <axis.pt>] [--schemer schemer deceiver cheater] [--honest straight_shooter rule_follower]
        [--default default] [--layers 17]

persona_raw      = mean(schemer roles) - default
persona_contrast = mean(schemer roles) - mean(honest roles)        (skipped if an honest role vector is missing)
persona_orth     = persona_raw with the AA component removed per layer          (only with --assistant-axis)
persona_contrast_orth = persona_contrast with the AA component removed per layer (only with --assistant-axis)
<role>_minus_default for every schemer role.
Also prints, per candidate, the norm at the layers of interest and, with an Assistant Axis, the cosine with it.
``--model`` is stored in every artifact and checked against the model at steering time (load_unit_axes).
"""
import argparse
import hashlib
import json
from pathlib import Path
from typing import Literal

import torch

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_role(vectors_dir: Path, role: str) -> torch.Tensor | None:
    path = vectors_dir / f"{role}.pt"
    if not path.exists():
        return None
    data = torch.load(path, map_location="cpu", weights_only=False)
    return data["vector"].float()


def load_question_indices(vectors_dir: Path, role: str) -> list[int] | Literal["unrecorded"]:
    """Ids of the questions the role vector was fitted on; role files from before this was recorded give "unrecorded"."""
    data = torch.load(vectors_dir / f"{role}.pt", map_location="cpu", weights_only=False)
    return data.get("question_indices", "unrecorded")


def load_axis(path: Path) -> torch.Tensor:
    data = torch.load(path, map_location="cpu", weights_only=True)
    return (data["axis"] if isinstance(data, dict) else data).float()


def unit(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True)


def project_out(direction: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    reference = unit(reference)
    return direction - reference * (direction * reference).sum(dim=-1, keepdim=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vectors-dir", type=Path, required=True)
    parser.add_argument("--assistant-axis", type=Path, help="Assistant Axis artifact; omit for models without one (no *_orth candidates, no cosine)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--schemer", nargs="+", default=["schemer", "deceiver", "cheater"])
    parser.add_argument("--honest", nargs="+", default=["straight_shooter", "rule_follower"])
    parser.add_argument("--default", default="default")
    parser.add_argument("--costume", default="neutral_fic", help="Ordinary fictional-character role; without its vector persona_contrast_no_costume is not built")
    parser.add_argument("--layers", nargs="+", type=int, default=[17], help="Layers to report in the summary")
    parser.add_argument("--model", required=True, help="HF model id the role vectors came from, stored in the artifacts")
    args = parser.parse_args()

    roles = {role: load_role(args.vectors_dir, role) for role in [*args.schemer, *args.honest, args.default]}
    missing = sorted(role for role, vector in roles.items() if vector is None)
    schemer = [roles[r] for r in args.schemer if roles[r] is not None]
    honest = [roles[r] for r in args.honest if roles[r] is not None]
    default = roles[args.default]
    if not schemer or default is None or missing:
        raise SystemExit(f"Need the default vector and at least one schemer role; missing {missing}")
    costume_role = load_role(args.vectors_dir, args.costume)
    if costume_role is not None:
        roles[args.costume] = costume_role
    aa = load_axis(args.assistant_axis) if args.assistant_axis is not None else None
    if aa is not None and aa.shape != default.shape:
        raise SystemExit(f"Assistant axis shape {tuple(aa.shape)} differs from role vectors {tuple(default.shape)}")

    schemer_mean = torch.stack(schemer).mean(0)
    honest_mean = torch.stack(honest).mean(0)
    candidates = {
        "persona_raw": schemer_mean - default,
        "honest_raw": honest_mean - default,
    }
    if len(honest) == len(args.honest):
        candidates["persona_contrast"] = schemer_mean - honest_mean
    if aa is not None:
        candidates["persona_orth"] = project_out(candidates["persona_raw"], aa)
        if "persona_contrast" in candidates:
            candidates["persona_contrast_orth"] = project_out(candidates["persona_contrast"], aa)
    for role in args.schemer:
        if roles[role] is not None:
            candidates[f"{role}_minus_default"] = roles[role] - default
    if "persona_contrast" in candidates:
        candidates["persona_contrast_honest_projected_out"] = project_out(candidates["persona_contrast"], candidates["honest_raw"])
        if costume_role is not None:
            candidates["persona_contrast_no_costume"] = project_out(candidates["persona_contrast"], costume_role - default)

    args.out.mkdir(parents=True, exist_ok=True)
    provenance = {"vectors_dir": str(args.vectors_dir),
                  "assistant_axis": str(args.assistant_axis) if aa is not None else None,
                  "assistant_axis_sha256": sha256(args.assistant_axis) if aa is not None else None,
                  "role_sha256": {role: sha256(args.vectors_dir / f"{role}.pt") for role, v in roles.items() if v is not None},
                  "question_indices_by_role": {role: load_question_indices(args.vectors_dir, role) for role, v in roles.items() if v is not None},
                  "schemer_roles": [r for r in args.schemer if roles[r] is not None],
                  "honest_roles": [r for r in args.honest if roles[r] is not None], "missing_roles": missing}
    summary = {}
    for name, axis in candidates.items():
        torch.save({"axis": axis, "model": args.model, "name": name, **provenance}, args.out / f"{name}.pt")
        summary[name] = {str(layer): {"norm": round(axis[layer].norm().item(), 4)} for layer in args.layers}
        if aa is not None:
            cos = (unit(axis) * unit(aa)).sum(-1)
            for layer in args.layers:
                summary[name][str(layer)]["cosine_with_aa"] = round(cos[layer].item(), 4)
            summary[name]["max_abs_cosine_with_aa_layer"] = int(cos.abs().argmax())
    if "persona_contrast" in candidates:
        cos_raw_contrast = (unit(candidates["persona_raw"]) * unit(candidates["persona_contrast"])).sum(-1)
        summary["cosine_persona_raw_vs_contrast"] = {str(layer): round(cos_raw_contrast[layer].item(), 4) for layer in args.layers}
    summary["default_norm"] = {str(layer): round(default[layer].norm().item(), 4) for layer in args.layers}
    summary["missing_roles"] = missing
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
