"""Logit lens of a schemer-axis artifact (CPU): which output tokens each side of the axis promotes.

Decodes a unit direction v at a layer through the model's final RMSNorm gain and unembedding, logits = W_U (g * v),
and lists the tokens at both ends. Next to the axis itself (deceptive mean - honest mean) it decodes the two halves,
deceptive mean - default and honest mean - default, the Assistant Axis named in the artifact (if it exists), and
random Gaussian directions as the null. Tokens are ranked two ways: by logit (the plain logit lens) and by cosine
between v and the token's gain-scaled unembedding row, which removes the bias toward tokens with large rows.

    python scripts/logit_lens.py --axis <root>/axis_response_only/persona_contrast.pt --layers 17 23 29 35 --out <dir>
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import TypedDict

from huggingface_hub import hf_hub_download
from jaxtyping import Float
from safetensors import safe_open
import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ScoredToken(TypedDict):
    token: str
    token_id: int
    score: float
    z: float


class SideTokens(TypedDict):
    positive: list[ScoredToken]
    negative: list[ScoredToken]


class DirectionLens(TypedDict):
    layer: int
    direction: str
    tokens_by_ranking: dict[str, SideTokens]


class Unembedding(TypedDict):
    gain_scaled_weight: Float[torch.Tensor, "vocab hidden"]
    row_norms: Float[torch.Tensor, "vocab"]


def load_unembedding(model_name: str) -> Unembedding:
    index_path = hf_hub_download(model_name, "model.safetensors.index.json")
    shard_by_tensor: dict[str, str] = json.loads(Path(index_path).read_text())["weight_map"]

    def load_tensor(name: str) -> torch.Tensor:
        with safe_open(hf_hub_download(model_name, shard_by_tensor[name]), "pt") as shard:
            return shard.get_tensor(name).float()

    gain_scaled_weight = load_tensor("lm_head.weight") * load_tensor("model.norm.weight")
    return {"gain_scaled_weight": gain_scaled_weight, "row_norms": gain_scaled_weight.norm(dim=-1)}


def mean_role_vector(vectors_dir: Path, roles: list[str]) -> Float[torch.Tensor, "layer hidden"]:
    return torch.stack([torch.load(vectors_dir / f"{role}.pt", map_location="cpu", weights_only=False)["vector"].float() for role in roles]).mean(dim=0)


def directions_from_artifact(axis_path: Path) -> dict[str, Float[torch.Tensor, "layer hidden"]]:
    artifact = torch.load(axis_path, map_location="cpu", weights_only=False)
    vectors_dir = Path(artifact["vectors_dir"])
    default = mean_role_vector(vectors_dir, ["default"])
    directions_by_name = {
        artifact["name"]: artifact["axis"].float(),
        "deceptive_minus_default": mean_role_vector(vectors_dir, artifact["schemer_roles"]) - default,
        "honest_minus_default": mean_role_vector(vectors_dir, artifact["honest_roles"]) - default,
    }
    assistant_axis_path = Path(artifact.get("assistant_axis") or "")
    if assistant_axis_path.is_file():
        assistant_axis = torch.load(assistant_axis_path, map_location="cpu", weights_only=False)
        directions_by_name["assistant_axis"] = (assistant_axis["axis"] if isinstance(assistant_axis, dict) else assistant_axis).float()
    return directions_by_name


def scores_by_ranking(unembedding: Unembedding, unit_direction: Float[torch.Tensor, "hidden"]) -> dict[str, Float[torch.Tensor, "vocab"]]:
    logits = unembedding["gain_scaled_weight"] @ unit_direction
    return {"logit": logits, "cosine": logits / unembedding["row_norms"]}


def extreme_tokens(scores: Float[torch.Tensor, "vocab"], tokenizer: PreTrainedTokenizerBase, top_k: int) -> SideTokens:
    z_scores = (scores - scores.mean()) / scores.std()

    def scored(token_ids: list[int]) -> list[ScoredToken]:
        return [{"token": tokenizer.decode([token_id]), "token_id": token_id, "score": scores[token_id].item(), "z": z_scores[token_id].item()} for token_id in token_ids]

    return {"positive": scored(torch.topk(scores, top_k).indices.tolist()), "negative": scored(torch.topk(-scores, top_k).indices.tolist())}


def lens_direction(unembedding: Unembedding, tokenizer: PreTrainedTokenizerBase, direction: Float[torch.Tensor, "hidden"], layer: int, name: str, top_k: int) -> DirectionLens:
    unit_direction = direction / direction.norm()
    return {"layer": layer, "direction": name,
            "tokens_by_ranking": {ranking: extreme_tokens(scores, tokenizer, top_k) for ranking, scores in scores_by_ranking(unembedding, unit_direction).items()}}


def random_extreme_z(unembedding: Unembedding, hidden_size: int, count: int, seed: int) -> dict[str, list[float]]:
    """|z| of the most extreme token on each side of isotropic random directions (two values per direction): the
    null for how peaked one side of a lens is."""
    generator = torch.Generator().manual_seed(seed)
    random_directions = torch.randn(count, hidden_size, generator=generator)
    z_by_ranking: dict[str, list[float]] = {"logit": [], "cosine": []}
    for random_direction in random_directions:
        for ranking, scores in scores_by_ranking(unembedding, random_direction / random_direction.norm()).items():
            z_scores = (scores - scores.mean()) / scores.std()
            z_by_ranking[ranking] += [z_scores.max().item(), -z_scores.min().item()]
    return z_by_ranking


def markdown_report(lenses: list[DirectionLens], random_z_by_ranking: dict[str, list[float]], tokens_shown: int) -> str:
    lines = ["# Logit lens", "", "Null (isotropic random directions), |z| of the most extreme token on one side:"]
    lines += [f"- {ranking}: median {torch.tensor(z_values).median().item():.2f}, 95th percentile {torch.tensor(z_values).quantile(0.95).item():.2f}, "
              f"max {max(z_values):.2f} over {len(z_values)} sides" for ranking, z_values in random_z_by_ranking.items()]
    for lens in lenses:
        for ranking, side_tokens in lens["tokens_by_ranking"].items():
            lines += ["", f"## {lens['direction']}, layer {lens['layer']}, ranked by {ranking}", ""]
            for side in ("positive", "negative"):
                shown = side_tokens[side][:tokens_shown]
                lines.append(f"**{side}** (top z {shown[0]['z']:+.1f}): " + ", ".join(f"`{json.dumps(token["token"], ensure_ascii=False)[1:-1]}`" for token in shown))
                lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--axis", type=Path, required=True, help="Artifact from build_vectors.py (persona_contrast.pt)")
    parser.add_argument("--layers", nargs="+", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--tokens-shown", type=int, default=40, help="Tokens per side in the markdown report")
    parser.add_argument("--random-directions", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    model_name = torch.load(args.axis, map_location="cpu", weights_only=False)["model"]
    unembedding = load_unembedding(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    directions_by_name = directions_from_artifact(args.axis)
    lenses = [lens_direction(unembedding, tokenizer, direction[layer], layer, name, args.top_k)
              for layer in args.layers for name, direction in directions_by_name.items()]
    random_z_by_ranking = random_extreme_z(unembedding, unembedding["gain_scaled_weight"].shape[1], args.random_directions, args.seed)
    cosine_by_pair = {f"{first} vs {second}, layer {layer}": torch.nn.functional.cosine_similarity(directions_by_name[first][layer], directions_by_name[second][layer], dim=0).item()
                      for layer in args.layers for first in directions_by_name for second in directions_by_name if first < second}

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "logit_lens.json").write_text(json.dumps({"axis": str(args.axis), "model": model_name, "lenses": lenses,
                                                          "random_extreme_z": random_z_by_ranking, "cosine_by_pair": cosine_by_pair}, indent=1))
    (args.out / "logit_lens.md").write_text(markdown_report(lenses, random_z_by_ranking, args.tokens_shown))
    logger.info("Wrote %s", args.out)


if __name__ == "__main__":
    main()
