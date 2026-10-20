"""Compare binding vectors (Δ = pair 2 − pair 1) across tasks, layer by layer.

If two tasks bind with the same mechanism, their Δs should point the same way (cosine ≈ 1)
and have similar norms. Reads the `<stem>_deltas.pt` files the metric writes.

Usage (from the repo root):
    python experiments/binding_ids/scripts/compare_deltas.py experiments/binding_ids/results/olmo2-13b \
        rooms_people rooms_animals rooms_objects rooms_i_you rooms_animal_object
"""

import argparse
import itertools
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("stems", nargs="+", help="result file stems, e.g. rooms_animals")
    parser.add_argument("--layer-step", type=int, default=4, help="print every n-th layer")
    args = parser.parse_args()

    deltas = {s: torch.load(args.results_dir / f"{s}_deltas.pt") for s in args.stems}
    names = [s.removeprefix("rooms_") for s in args.stems]
    for side in ("entity", "attribute"):
        # (n_layers, hidden): entities use the entity token (position 0), not the token after it
        vectors = {n: deltas[s][side][:, 0] for n, s in zip(names, args.stems)}
        n_layers = next(iter(vectors.values())).shape[0]
        layers = list(range(0, n_layers, args.layer_step))
        if layers[-1] != n_layers - 1:
            layers.append(n_layers - 1)

        print(f"\n## {side} Δ: cosine similarity by layer")
        print("| pair | " + " | ".join(f"L{i}" for i in layers) + " | mean |")
        print("|---|" + "---|" * (len(layers) + 1))
        for a, b in itertools.combinations(names, 2):
            cos = torch.cosine_similarity(vectors[a], vectors[b], dim=-1)
            print(f"| {a} / {b} | " + " | ".join(f"{cos[i]:.2f}" for i in layers) + f" | {cos.mean():.2f} |")

        print(f"\n## {side} Δ: norm by layer")
        print("| task | " + " | ".join(f"L{i}" for i in layers) + " |")
        print("|---|" + "---|" * len(layers))
        for n, v in vectors.items():
            norms = v.norm(dim=-1)
            print(f"| {n} | " + " | ".join(f"{norms[i]:.1f}" for i in layers) + " |")


if __name__ == "__main__":
    main()
