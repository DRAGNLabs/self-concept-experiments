"""Phase 1 validation checks for the schemer-axis role vectors (PLAN.md, "Phase 1: extraction and validation").

Reads the per-response activations ({role}.pt: {"pos_p{i}_q{j}": (layers, hidden)}) and judge scores
({role}.json: key -> 0..3) written by the assistant-axis pipeline, and reports, without touching reward-hacking data:

1. score-3 counts per role (keep threshold --min-count);
2. split-half reliability: cosine at --layer between candidate vectors built from even- and odd-numbered questions;
3. held-out separation: vectors built from even questions, AUROC of the projection for schemer-role vs honest-role
   (and vs default) responses to odd questions, and vice versa;
4. cosine with the Assistant Axis per candidate at --layer.

    python scripts/validate_vectors.py --activations-dir <root>/act_with_cot --scores-dir <root>/scores
        --assistant-axis <axis.pt> --out <dir> [--min-count 50] [--layer 17]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata

SCHEMER, HONEST, DEFAULT = ["schemer", "deceiver", "cheater"], ["straight_shooter", "rule_follower"], "default"


def load_role(activations_dir: Path, scores_dir: Path, role: str, min_score: int) -> dict[str, torch.Tensor]:
    path = activations_dir / f"{role}.pt"
    if not path.exists():
        return {}
    activations = torch.load(path, map_location="cpu", weights_only=False)
    if role == DEFAULT:
        return {k: v.float() for k, v in activations.items()}
    scores_path = scores_dir / f"{role}.json"
    scores = json.loads(scores_path.read_text()) if scores_path.exists() else {}
    return {k: v.float() for k, v in activations.items() if scores.get(k, -1) >= min_score}


def question_index(key: str) -> int:
    return int(key.rsplit("_q", 1)[1])


def half(items: dict[str, torch.Tensor], parity: int) -> dict[str, torch.Tensor]:
    return {k: v for k, v in items.items() if question_index(k) % 2 == parity}


def mean_vector(groups: list[dict[str, torch.Tensor]]) -> torch.Tensor | None:
    role_means = [torch.stack(list(g.values())).mean(0) for g in groups if g]
    return torch.stack(role_means).mean(0) if role_means else None


def candidate(schemer: list[dict], honest: list[dict], default: dict, name: str) -> torch.Tensor | None:
    s = mean_vector(schemer)
    if s is None:
        return None
    if name == "persona_raw":
        d = mean_vector([default])
        return None if d is None else s - d
    if name == "persona_contrast":
        h = mean_vector(honest)
        return None if h is None or len([g for g in honest if g]) < len(honest) else s - h
    raise ValueError(name)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(a @ b / (a.norm() * b.norm()))


def auroc(positive: np.ndarray, negative: np.ndarray) -> float | None:
    if not len(positive) or not len(negative):
        return None
    ranks = rankdata(np.concatenate([positive, negative]))[:len(positive)]
    return float((ranks.sum() - len(positive) * (len(positive) + 1) / 2) / (len(positive) * len(negative)))


def projections(groups: list[dict[str, torch.Tensor]], direction: torch.Tensor, layer: int) -> np.ndarray:
    rows = [v[layer] for g in groups for v in g.values()]
    return np.asarray([float(r @ direction) for r in rows]) if rows else np.empty(0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--activations-dir", type=Path, required=True)
    parser.add_argument("--scores-dir", type=Path, required=True)
    parser.add_argument("--assistant-axis", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-count", type=int, default=50)
    parser.add_argument("--min-score", type=int, default=3)
    parser.add_argument("--layer", type=int, default=17)
    args = parser.parse_args()
    layer = args.layer
    roles = {role: load_role(args.activations_dir, args.scores_dir, role, args.min_score) for role in [*SCHEMER, *HONEST, DEFAULT]}
    all_scores = {}
    for role in [*SCHEMER, *HONEST]:
        p = args.scores_dir / f"{role}.json"
        if p.exists():
            values = list(json.loads(p.read_text()).values())
            all_scores[role] = {str(s): values.count(s) for s in range(4)}
    counts = {role: len(acts) for role, acts in roles.items()}
    usable = {role: counts[role] >= args.min_count for role in [*SCHEMER, *HONEST]}
    aa = torch.load(args.assistant_axis, map_location="cpu", weights_only=True)
    aa = (aa["axis"] if isinstance(aa, dict) else aa).float()
    schemer, honest, default = [roles[r] for r in SCHEMER], [roles[r] for r in HONEST], roles[DEFAULT]
    report = {"layer": layer, "min_count": args.min_count, "min_score": args.min_score, "kept_counts": counts,
              "judge_score_histogram": all_scores, "usable_at_min_count": usable, "candidates": {}}
    for name in ("persona_raw", "persona_contrast"):
        full = candidate(schemer, honest, default, name)
        entry = {"buildable": full is not None}
        if full is not None:
            halves = [candidate([half(g, p) for g in schemer], [half(g, p) for g in honest], half(default, p), name) for p in (0, 1)]
            entry["split_half_cosine"] = cosine(halves[0][layer], halves[1][layer]) if all(h is not None for h in halves) else None
            entry["cosine_with_aa"] = cosine(full[layer], aa[layer])
            entry["norm"] = float(full[layer].norm())
            held_out = {}
            for fit, test in ((0, 1), (1, 0)):
                direction = halves[fit]
                if direction is None:
                    continue
                u = direction[layer] / direction[layer].norm()
                pos = projections([half(g, test) for g in schemer], u, layer)
                held_out[f"fit_{fit}"] = {
                    "schemer_vs_honest": auroc(pos, projections([half(g, test) for g in honest], u, layer)),
                    "schemer_vs_default": auroc(pos, projections([half(default, test)], u, layer)),
                    "honest_vs_default": auroc(projections([half(g, test) for g in honest], u, layer), projections([half(default, test)], u, layer))}
            entry["held_out_auroc"] = held_out
        report["candidates"][name] = entry
    # Per-role: does each role's own mean separate from default on held-out questions? (role identity sanity)
    report["per_role_vs_default_held_out_auroc"] = {}
    for role in [*SCHEMER, *HONEST]:
        g = roles[role]
        if not g or not default:
            continue
        vals = {}
        for fit, test in ((0, 1), (1, 0)):
            d = mean_vector([half(g, fit)])
            dd = mean_vector([half(default, fit)])
            if d is None or dd is None:
                continue
            u = (d - dd)[layer]
            u = u / u.norm()
            vals[f"fit_{fit}"] = auroc(projections([half(g, test)], u, layer), projections([half(default, test)], u, layer))
        report["per_role_vs_default_held_out_auroc"][role] = vals
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "validation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
