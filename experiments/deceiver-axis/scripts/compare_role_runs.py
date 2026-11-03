"""Compare two role-extraction runs of the same model and roles: judge score distributions, and per-role vector cosine
(raw, and centered on the mean over roles, which removes the shared component that keeps raw cosines near 1).

    python scripts/compare_role_runs.py <old_root> <new_root> [--layer 32]
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from jaxtyping import Float
from torch import Tensor

VECTOR_VARIANTS = ("vectors_response_only", "vectors_with_cot")


def score_counts(root: Path) -> Counter[int]:
    return Counter(score for path in (root / "scores").glob("*.json") for score in json.loads(path.read_text()).values())


def vectors_by_role(vectors_dir: Path, layer: int) -> dict[str, Float[Tensor, "hidden"]]:
    return {
        path.stem: torch.load(path, weights_only=False)["vector"][layer].float()
        for path in sorted(vectors_dir.glob("*.pt"))
    }


def centered(vector_by_role: dict[str, Float[Tensor, "hidden"]]) -> dict[str, Float[Tensor, "hidden"]]:
    mean_vector = torch.stack(list(vector_by_role.values())).mean(dim=0)
    return {role: vector - mean_vector for role, vector in vector_by_role.items()}


def cosine_by_role(old: dict[str, Float[Tensor, "hidden"]], new: dict[str, Float[Tensor, "hidden"]]) -> dict[str, float]:
    return {role: torch.nn.functional.cosine_similarity(old[role], new[role], dim=0).item() for role in sorted(old.keys() & new.keys())}


def print_cosines(title: str, cosine_by_role_name: dict[str, float]) -> None:
    print(f"  {title}: min {min(cosine_by_role_name.values()):.4f}  mean {sum(cosine_by_role_name.values()) / len(cosine_by_role_name):.4f}")
    for role, cosine in sorted(cosine_by_role_name.items(), key=lambda item: item[1])[:5]:
        print(f"    {role:34s} {cosine:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("old_root", type=Path)
    parser.add_argument("new_root", type=Path)
    parser.add_argument("--layer", type=int, default=32)
    args = parser.parse_args()

    for name, root in (("old", args.old_root), ("new", args.new_root)):
        counts = score_counts(root)
        print(f"scores {name}: " + "  ".join(f"{score}={counts[score]}" for score in range(4)) + f"  total={counts.total()}")

    for variant in VECTOR_VARIANTS:
        old_vectors = vectors_by_role(args.old_root / variant, args.layer)
        new_vectors = vectors_by_role(args.new_root / variant, args.layer)
        print(f"{variant} layer {args.layer}: {len(old_vectors.keys() & new_vectors.keys())} shared roles"
              f" (old only {sorted(old_vectors.keys() - new_vectors.keys())}, new only {sorted(new_vectors.keys() - old_vectors.keys())})")
        print_cosines("raw cosine (5 lowest)", cosine_by_role(old_vectors, new_vectors))
        print_cosines("centered cosine (5 lowest)", cosine_by_role(centered(old_vectors), centered(new_vectors)))


if __name__ == "__main__":
    main()
