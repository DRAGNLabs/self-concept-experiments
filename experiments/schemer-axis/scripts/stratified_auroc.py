"""Question-stratified AUROCs of the raw and cleaned ("no-costume") 17-role contrasts at one layer.

Koby's costume check (2026-10-23) scores each response on the contrast and computes the AUROC within each question,
then averages over questions, so differences between questions do not enter. This reproduces that on our cached
activations, for the raw contrast (mean schemer-role vector - mean honest-role vector) and the cleaned one (the
acting arrow mean(costume role) - mean(default) projected out):

- deception: schemer-type vs honest-type role responses;
- costume: the neutral costume role vs the plain assistant (0.5 = blind to the costume);
- honest vs default.

Both in-sample (contrast from every question) and held-out (contrast fitted on even questions, scored on odd ones,
and the reverse, averaged). Character roles keep judge score 3; the costume role and default keep every response.

    python scripts/stratified_auroc.py --activations-dir <root>/act_response_only --scores-dir <root>/scores --layer 32 \
        --schemer ... --honest ... --out <file.json>
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from validate_vectors import auroc, load_role, question_index


def role_mean(groups: list[dict[str, torch.Tensor]], layer: int) -> torch.Tensor:
    return torch.stack([torch.stack([v[layer] for v in g.values()]).mean(0) for g in groups]).mean(0)


def directions(schemer, honest, default, costume, layer):
    raw = role_mean(schemer, layer) - role_mean(honest, layer)
    arrow = role_mean([costume], layer) - role_mean([default], layer)
    unit = arrow / arrow.norm()
    return {"raw": raw, "clean": raw - (raw @ unit) * unit}


def stratified(pos: list[dict], neg: list[dict], u: torch.Tensor, layer: int, questions: set[int]) -> float:
    values = []
    for q in sorted(questions):
        p = np.array([float(v[layer] @ u) for g in pos for k, v in g.items() if question_index(k) == q])
        n = np.array([float(v[layer] @ u) for g in neg for k, v in g.items() if question_index(k) == q])
        a = auroc(p, n)
        if a is not None:
            values.append(a)
    return float(np.mean(values))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--activations-dir", type=Path, required=True)
    parser.add_argument("--scores-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--schemer", nargs="+", required=True)
    parser.add_argument("--honest", nargs="+", required=True)
    parser.add_argument("--default", default="default")
    parser.add_argument("--costume", default="neutral_fic")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    L = args.layer
    load = lambda role, min_score: load_role(args.activations_dir, args.scores_dir, role, min_score)
    schemer, honest = [load(r, 3) for r in args.schemer], [load(r, 3) for r in args.honest]
    default, costume = load(args.default, 0), load(args.costume, 0)
    split = lambda g, parity: {k: v for k, v in g.items() if question_index(k) % 2 == parity}
    all_q = {question_index(k) for k in default}
    report = {"layer": L, "in_sample": {}, "held_out": {}}
    full = directions(schemer, honest, default, costume, L)
    halves = [directions([split(g, p) for g in schemer], [split(g, p) for g in honest], split(default, p), split(costume, p), L)
              for p in (0, 1)]
    for name in ("raw", "clean"):
        u = full[name] / full[name].norm()
        report["in_sample"][name] = {
            "deception": stratified(schemer, honest, u, L, all_q),
            "costume": stratified([costume], [default], u, L, all_q),
            "honest_vs_default": stratified(honest, [default], u, L, all_q)}
        held = {"deception": [], "costume": [], "honest_vs_default": []}
        for fit, test in ((0, 1), (1, 0)):
            u = halves[fit][name] / halves[fit][name].norm()
            tq = {q for q in all_q if q % 2 == test}
            held["deception"].append(stratified(schemer, honest, u, L, tq))
            held["costume"].append(stratified([costume], [default], u, L, tq))
            held["honest_vs_default"].append(stratified(honest, [default], u, L, tq))
        report["held_out"][name] = {k: float(np.mean(v)) for k, v in held.items()}
    report["cosine_raw_clean"] = float(full["raw"] @ full["clean"] / (full["raw"].norm() * full["clean"].norm()))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
