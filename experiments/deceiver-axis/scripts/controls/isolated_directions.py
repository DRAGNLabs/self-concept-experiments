"""Isolated-text contrast control directions (AISI style: Schoen et al. 2026, "Reproducing steering against evaluation
awareness", appendix), any model with a downstream read in controls_common. Each text is chat-formatted as a single user
message and read at --layer on its final content token, and, as a variant, averaged over its content tokens.

Sets: the antonym pairs (direction = mean over each pair's sentence pairs of word-side minus antonym-side; sentences
from data/controls/sentences.json, see subagent_instructions.md) and the AISI control pairs (positive minus negative;
control_spacing is left out because the post does not give its passages). Validation against gi_v2, the assistant axis
and isotropic random directions: |cos|, pairwise |cos|, effective dimension of the set, split-half reliability of the
antonym directions (sentence pairs 0-1 vs 2-4), and the median coefficient matching gi_v2 at --gi-alpha in downstream
read. Writes <out>/isolated_summary.json and <out>/dirs_isolated/{antonym,aisi}_*.pt (final-token variant,
build_vectors-style artifacts: the direction repeated at every layer).

    python scripts/controls/isolated_directions.py --model openai/gpt-oss-120b --layer 17 --gi-alpha 0.2 \
        --gi-direction <persona_contrast.pt> --assistant-axis <axis_response_only.pt> --out <dir>
"""
import argparse
import json
from pathlib import Path
from typing import Literal, TypedDict, cast

import torch
from jaxtyping import Float
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from controls_common import (DownstreamReadSpecifics, EffectiveDimension, Spread, downstream_read_specifics_for, effective_dimension,
                             off_diagonal, spread, unit)
from selfconcept.assistant_axis.internals import ProbingModel
from selfconcept.correlation.random_vectors import random_directions
from selfconcept.measurement.steering_specifics import steering_model_specifics

DATA = Path(__file__).parents[2] / "data/controls"
ISOTROPIC_COUNT, ISOTROPIC_SEED = 100, 60_000
Pooling = Literal["final_token", "mean_tokens"]
POOLINGS: tuple[Pooling, ...] = ("final_token", "mean_tokens")
type SentencePair = tuple[str, str]
# The AISI post's ten control pairs minus control_spacing, whose passages the post does not give.
AISI_PAIRS_BY_NAME: dict[str, SentencePair] = {
    "book": ("She placed the book on the bottom shelf.", "She placed the book on the top shelf."),
    "cat": ("The dog sat quietly by the window.", "The cat sat quietly by the window."),
    "kiki": ("You are a bouba AI.", "You are a kiki AI."),
    "language": ("你是一个有益、诚实、无害的语言模型。", "You are a helpful, honest and harmless language model."),
    "perspective": ("You are an AI having a conversation with a user.", "The Assistant is an AI having a conversation with a user."),
    "speaking": ("She preferred writing her thoughts down.", "She preferred speaking her thoughts aloud."),
    "train": ("The train left the station.", "The train arrived at the station."),
    "wall": ("There was a crack running across the ceiling.", "There was a crack running across the wall."),
    "weather": ("The weather was cold and rainy today.", "The weather was warm and sunny today."),
}
ComparisonSet = Literal["antonym", "aisi", "isotropic"]


class PoolingSummary(TypedDict):
    antonym_split_half_cos: Spread
    abs_cos_with_gi_v2: dict[ComparisonSet, Spread]
    abs_cos_with_assistant_axis: dict[ComparisonSet, Spread]
    pairwise_abs_cos: dict[ComparisonSet, Spread]
    effective_dimension_of_directions: dict[ComparisonSet, EffectiveDimension]
    aisi_cos_with_gi_v2: dict[str, float]
    aisi_cos_with_assistant_axis: dict[str, float]


def read_text(model: ProbingModel, layer: int, text: str) -> dict[Pooling, Float[torch.Tensor, "hidden"]]:
    tokenizer, network = cast(PreTrainedTokenizerBase, model.tokenizer), cast(PreTrainedModel, model.model)
    rendered = cast(str, tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=False))
    content_start = rendered.rindex(text)
    encoding = tokenizer(rendered, return_offsets_mapping=True, add_special_tokens=False, return_tensors="pt")
    offsets = cast(torch.Tensor, encoding["offset_mapping"])[0]
    in_content = (offsets[:, 0] >= content_start) & (offsets[:, 1] <= content_start + len(text)) & (offsets[:, 1] > offsets[:, 0])
    captured: list[torch.Tensor] = []

    def capture(module: torch.nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor | tuple[torch.Tensor, ...]) -> None:
        captured.append(output[0] if isinstance(output, tuple) else output)

    handle = model.get_layers()[layer].register_forward_hook(capture)
    try:
        with torch.inference_mode():
            network(input_ids=cast(torch.Tensor, encoding["input_ids"]).to(model.device))
    finally:
        handle.remove()
    hidden = captured[0][0].float().cpu()[in_content]
    return {"final_token": hidden[-1], "mean_tokens": hidden.mean(0)}


def contrast(model: ProbingModel, layer: int, pairs: list[SentencePair], pooling: Pooling) -> Float[torch.Tensor, "pairs hidden"]:
    """Per (negative, positive) pair, the positive text's read minus the negative's."""
    return torch.stack([read_text(model, layer, positive)[pooling] - read_text(model, layer, negative)[pooling] for negative, positive in pairs])


def summarize(contrasts_by_pair_id: dict[str, Float[torch.Tensor, "pairs hidden"]], aisi: Float[torch.Tensor, "aisi hidden"],
              gi_v2: Float[torch.Tensor, "hidden"], assistant: Float[torch.Tensor, "hidden"],
              isotropic: Float[torch.Tensor, "isotropic hidden"]) -> PoolingSummary:
    antonym = unit(torch.stack([differences.mean(0) for differences in contrasts_by_pair_id.values()]))
    first_half = unit(torch.stack([differences[:2].mean(0) for differences in contrasts_by_pair_id.values()]))
    second_half = unit(torch.stack([differences[2:].mean(0) for differences in contrasts_by_pair_id.values()]))
    sets: dict[ComparisonSet, Float[torch.Tensor, "n hidden"]] = {"antonym": antonym, "aisi": aisi, "isotropic": isotropic}
    return {
        "antonym_split_half_cos": spread(torch.sum(first_half * second_half, dim=1)),
        "abs_cos_with_gi_v2": {name: spread(vectors @ gi_v2) for name, vectors in sets.items()},
        "abs_cos_with_assistant_axis": {name: spread(vectors @ assistant) for name, vectors in sets.items()},
        "pairwise_abs_cos": {name: spread(off_diagonal(vectors)) for name, vectors in sets.items()},
        "effective_dimension_of_directions": {name: effective_dimension(vectors - vectors.mean(0)) for name, vectors in sets.items()},
        "aisi_cos_with_gi_v2": dict(zip(AISI_PAIRS_BY_NAME, (aisi @ gi_v2).tolist())),
        "aisi_cos_with_assistant_axis": dict(zip(AISI_PAIRS_BY_NAME, (aisi @ assistant).tolist())),
    }


def matched_alpha_median(reader: DownstreamReadSpecifics, layer: int, gi_alpha: float, directions: Float[torch.Tensor, "n hidden"],
                         gi_v2: Float[torch.Tensor, "hidden"]) -> float:
    reads = reader.downstream_read(torch.cat([directions, gi_v2[None]]), layer)
    return (gi_alpha * reads[-1] / reads[:-1]).median().item()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--gi-alpha", type=float, required=True, help="gi_v2's steering dose (fraction of the layer's mean norm)")
    parser.add_argument("--gi-direction", type=Path, required=True)
    parser.add_argument("--assistant-axis", type=Path, required=True, help="(layers, hidden) assistant-axis tensor")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    reader = downstream_read_specifics_for(args.model)
    model = ProbingModel(args.model, dtype=torch.bfloat16, attn_implementation=steering_model_specifics(args.model).hf_attn_implementation)
    num_layers = len(model.get_layers())
    sentence_pairs_by_id: dict[str, list[SentencePair]] = {
        pair_id: [(pair["b"], pair["a"]) for pair in sentence_pairs] for pair_id, sentence_pairs in json.loads((DATA / "sentences.json").read_text()).items()}
    contrasts = {pooling: ({pair_id: contrast(model, args.layer, pairs, pooling) for pair_id, pairs in sentence_pairs_by_id.items()},
                           unit(torch.stack([contrast(model, args.layer, [pair], pooling)[0] for pair in AISI_PAIRS_BY_NAME.values()])))
                 for pooling in POOLINGS}
    model.close()
    gi_v2 = unit(torch.load(args.gi_direction, weights_only=False)["axis"][args.layer].float())
    assistant = unit(torch.load(args.assistant_axis, weights_only=False)[args.layer].float())
    isotropic = torch.from_numpy(random_directions(gi_v2.shape[0], ISOTROPIC_COUNT, ISOTROPIC_SEED, args.layer)).float()
    summary: dict[str, PoolingSummary | float] = {pooling: summarize(antonym_contrasts, aisi, gi_v2, assistant, isotropic)
                                                  for pooling, (antonym_contrasts, aisi) in contrasts.items()}
    antonym_contrasts, aisi = contrasts["final_token"]
    antonym = unit(torch.stack([differences.mean(0) for differences in antonym_contrasts.values()]))
    summary["matched_alpha_median"] = matched_alpha_median(reader, args.layer, args.gi_alpha, torch.cat([antonym, aisi]), gi_v2)
    (args.out / "dirs_isolated").mkdir(parents=True, exist_ok=True)
    names = [f"antonym_{pair_id}" for pair_id in sentence_pairs_by_id] + [f"aisi_{name}" for name in AISI_PAIRS_BY_NAME]
    for name, direction in zip(names, torch.cat([antonym, aisi])):
        torch.save({"axis": direction.repeat(num_layers, 1), "model": args.model, "name": name,
                    "source": f"isolated-text contrast, chat-formatted user message, final content token, layer {args.layer}"},
                   args.out / "dirs_isolated" / f"{name}.pt")
    (args.out / "isolated_summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
