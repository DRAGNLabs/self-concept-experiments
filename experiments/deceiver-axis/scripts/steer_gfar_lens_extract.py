"""Downstream effect of gi_v2 steering on the gfar projection, per response token, on finished unsteered episodes.

Each turn of each unsteered episode is teacher-forced twice, once as generated and once with gi_v2 added at the steer
layer (coefficient alpha x the layer's mean residual norm, every position, as in the steered runs), and at every
layer the projection of each reasoning / answer token's residual onto the unit gfar direction is stored. Chart:
steer_gfar_lens_chart.py.

    SOO_CHAT_KWARGS='{"reasoning_effort":"low"}' python scripts/steer_gfar_lens_extract.py --model openai/gpt-oss-120b \
        --data <data dir> --tag main --run <unsteered arm dir with seedN/> --seeds 0 1 \
        --gi-v2 <persona_contrast.pt> --gfar <directions_headline.pt>:goodfire_all_response_pl_all_families \
        --steer-layer 17 --alpha 0.2 --layer-norms <layer_norms.pt> --out <dir>

Writes <out>/seed<N>.pt per seed; existing files are skipped.
"""
import argparse
import json
import logging
import time
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import torch
from jaxtyping import Float, Int8
from torch import Tensor, nn
from transformers import AutoConfig, AutoTokenizer, PreTrainedTokenizerBase

import make_dossiers
from episode_rendering import ANSWER_SEGMENT, REASONING_SEGMENT, Family, episode_turns, load_reasoning_sidecar
from selfconcept.common.chat import chat_template_kwargs
from selfconcept.common.loading import load_causal_lm
from selfconcept.measurement.intervene import _hooks, add_transform
from selfconcept.measurement.steering_specifics import steering_model_specifics
from selfconcept.measurement.unit_axes import load_unit_axes
from selfconcept.soo.activations import get_decoder_layers

logger = logging.getLogger("steer_gfar_lens")
MIN_TOKENS_BEFORE_GIVING_UP = 4096
FP16_SAFE_ABS = 6e4

type Transform = Callable[[Tensor], Tensor]
type Projections = Float[Tensor, "layers response_tokens"]


class SeedLens(TypedDict):
    example_id: list[str]
    verdict_key: list[str]
    token_offsets: np.ndarray
    segment: Int8[np.ndarray, "response_tokens"]
    unsteered: Projections
    steered: Projections
    turn_lengths: list[list[int]]
    oom_truncated: list[list[tuple[int, int]]]
    missing_reasoning: list[list[int]]
    model: str
    family: Family
    chat_kwargs: dict[str, Any]
    tag: str
    run: str
    seed: int
    steer_layer: int
    alpha: float
    coefficient: float
    gi_v2: str
    gfar: str
    max_length: int
    seconds: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--attn-implementation", default=None, help="default: the family's steering_model_specifics kernel")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--tag", required=True, help="records file stem: <tag>_impossible_original.jsonl")
    parser.add_argument("--run", type=Path, required=True, help="unsteered arm directory holding seed<N>/")
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--gi-v2", type=Path, required=True, help="persona_contrast.pt")
    parser.add_argument("--gfar", required=True, help="path:key of the (layers, hidden) gfar artifact")
    parser.add_argument("--steer-layer", type=int, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--layer-norms", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=40960, help="per-turn token cap (the tail is dropped)")
    parser.add_argument("--n", type=int, default=None, help="episodes per seed (smoke)")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def model_dims(model_name: str) -> tuple[int, int]:
    config = AutoConfig.from_pretrained(model_name)
    text_config = getattr(config, "text_config", None) or config
    return int(text_config.num_hidden_layers), int(text_config.hidden_size)


def load_gfar_units(spec: str, num_layers: int, hidden: int) -> Float[Tensor, "layers hidden"]:
    path, key = spec.rsplit(":", 1)
    matrix = torch.as_tensor(torch.load(path, map_location="cpu", weights_only=False)[key]).float()
    if matrix.shape != (num_layers, hidden):
        raise ValueError(f"{spec}: shape {tuple(matrix.shape)}, expected {(num_layers, hidden)}")
    return matrix / matrix.norm(dim=-1, keepdim=True)


def steering_coefficient(layer_norms_path: Path, layer: int, alpha: float) -> float:
    norms = torch.load(layer_norms_path, map_location="cpu", weights_only=False)
    layer_norms = (norms["layer_norms"] if isinstance(norms, dict) else norms).float()
    return alpha * float(layer_norms[layer])


def response_projections(model: nn.Module, runner: nn.Module, runner_kwargs: dict[str, Any], ids: list[int],
                         response_mask: torch.Tensor, gfar_units: Float[Tensor, "layers hidden"],
                         steering_by_layer: dict[int, Transform]) -> Projections:
    decoder_layers = get_decoder_layers(model)
    projections_by_layer: dict[int, Tensor] = {}

    def record(layer: int):
        def hook(_module, _inputs, output) -> None:
            states = output[0] if isinstance(output, tuple) else output
            hidden_states = states[0]
            projections_by_layer[layer] = (hidden_states[response_mask.to(hidden_states.device)].float() @ gfar_units[layer].to(hidden_states.device)).cpu()
        return hook

    # Steering hooks are registered before the recording hooks so the steer layer records its steered output.
    transforms_by_layer: dict[int, object] = dict(steering_by_layer)
    with _hooks(model, transforms_by_layer) if transforms_by_layer else nullcontext():
        handles = [decoder_layers[layer].register_forward_hook(record(layer)) for layer in range(len(decoder_layers))]
        try:
            with torch.inference_mode():
                runner(input_ids=torch.tensor([ids], device=response_mask.device), **runner_kwargs)
        finally:
            for handle in handles:
                handle.remove()
    return torch.stack([projections_by_layer[layer] for layer in range(len(decoder_layers))])


def paired_turn_projections(model: nn.Module, runner: nn.Module, runner_kwargs: dict[str, Any], ids: list[int],
                            segments: Int8[np.ndarray, "tokens"], gfar_units: Float[Tensor, "layers hidden"],
                            steering_by_layer: dict[int, Transform]) -> tuple[Projections, Projections, Int8[np.ndarray, "response_tokens"], int]:
    """(unsteered, steered, response segments, tokens kept); a turn the GPU cannot hold is cut to its first 80% and retried."""
    kept = len(ids)
    while True:
        is_response = (segments[:kept] == REASONING_SEGMENT) | (segments[:kept] == ANSWER_SEGMENT)
        response_mask = torch.as_tensor(is_response, device=gfar_units.device)
        try:
            unsteered = response_projections(model, runner, runner_kwargs, ids[:kept], response_mask, gfar_units, {})
            steered = response_projections(model, runner, runner_kwargs, ids[:kept], response_mask, gfar_units, steering_by_layer)
            return unsteered, steered, segments[:kept][is_response], kept
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            if kept < MIN_TOKENS_BEFORE_GIVING_UP:
                raise
            kept = int(kept * 0.8)
            logger.warning("out of memory: turn retried on its first %d of %d tokens", kept, len(ids))


def compact(projections: Projections) -> Projections:
    return projections.half() if float(projections.abs().max()) < FP16_SAFE_ABS else projections


def seed_lens(args: argparse.Namespace, seed: int, tokenizer: PreTrainedTokenizerBase, family: Family, chat_kwargs: dict[str, Any],
              model: nn.Module, runner: nn.Module, runner_kwargs: dict[str, Any], gfar_units: Float[Tensor, "layers hidden"],
              steering_by_layer: dict[int, Transform], coefficient: float) -> SeedLens:
    run_dir = args.run / f"seed{seed}"
    records = [json.loads(line) for line in (run_dir / f"{args.tag}_impossible_original.jsonl").open()][: args.n]
    reasoning_by_turn_by_example = load_reasoning_sidecar(run_dir, args.tag)
    unsteered_parts, steered_parts, segment_parts, token_offsets = [], [], [], [0]
    turn_lengths, oom_truncated, missing_reasoning = [], [], []
    started = time.time()
    for episode_index, record in enumerate(records):
        example = make_dossiers.find_example(args.data, args.tag, record["example_id"], run_dir)
        turns, missing, full_lengths = episode_turns(tokenizer, family, example, record, reasoning_by_turn_by_example.get(record["example_id"], {}),
                                                     args.max_length, chat_kwargs)
        truncations = []
        episode_response_tokens = 0
        for ids, segments, turn in turns:
            unsteered, steered, response_segments, kept = paired_turn_projections(model, runner, runner_kwargs, ids, segments, gfar_units, steering_by_layer)
            if kept < len(ids):
                truncations.append((turn, kept))
            unsteered_parts.append(unsteered)
            steered_parts.append(steered)
            segment_parts.append(response_segments)
            episode_response_tokens += len(response_segments)
        token_offsets.append(token_offsets[-1] + episode_response_tokens)
        turn_lengths.append(full_lengths)
        oom_truncated.append(truncations)
        missing_reasoning.append(missing)
        logger.info("seed %d: %d/%d episodes, %.0f s", seed, episode_index + 1, len(records), time.time() - started)
    return {
        "example_id": [record["example_id"] for record in records],
        "verdict_key": [f"{args.run.name}/seed{seed}/{record['example_id']}" for record in records],
        "token_offsets": np.asarray(token_offsets, dtype=np.int64),
        "segment": np.concatenate(segment_parts),
        "unsteered": compact(torch.cat(unsteered_parts, dim=1)),
        "steered": compact(torch.cat(steered_parts, dim=1)),
        "turn_lengths": turn_lengths, "oom_truncated": oom_truncated, "missing_reasoning": missing_reasoning,
        "model": args.model, "family": family, "chat_kwargs": chat_kwargs, "tag": args.tag, "run": str(args.run), "seed": seed,
        "steer_layer": args.steer_layer, "alpha": args.alpha, "coefficient": coefficient, "gi_v2": str(args.gi_v2), "gfar": args.gfar,
        "max_length": args.max_length, "seconds": time.time() - started,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args = parse_args()
    specifics = steering_model_specifics(args.model)
    family: Family = specifics.family  # type: ignore[assignment]
    chat_kwargs = chat_template_kwargs()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    num_layers, hidden = model_dims(args.model)
    device = torch.device("cuda:0")
    gfar_units = load_gfar_units(args.gfar, num_layers, hidden).to(device)
    gi_v2_unit = torch.as_tensor(load_unit_axes(args.gi_v2, [args.steer_layer], num_layers, hidden, args.model)[args.steer_layer]["direction"]).to(device)
    coefficient = steering_coefficient(args.layer_norms, args.steer_layer, args.alpha)
    steering_by_layer = {args.steer_layer: add_transform(gi_v2_unit, coefficient)}
    logger.info("gi_v2 at layer %d, alpha %.3f, coefficient %.1f; cos(gi_v2, gfar) there %.4f",
                args.steer_layer, args.alpha, coefficient, float(gi_v2_unit @ gfar_units[args.steer_layer]))

    load_kwargs: dict[str, Any] = {"dtype": torch.bfloat16, "device_map": "auto", "local_files_only": True}
    if attention := (args.attn_implementation or specifics.hf_attn_implementation):
        load_kwargs["attn_implementation"] = attention
    model = load_causal_lm(args.model, **load_kwargs).eval()
    # The base model without the LM head: only the residual stream is read, and logits over a long turn would not fit.
    base = getattr(model, model.base_model_prefix, None)
    runner, runner_kwargs = ((base, {"use_cache": False}) if isinstance(base, nn.Module) and base is not model
                             else (model, {"use_cache": False, "logits_to_keep": 1}))

    args.out.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        out_path = args.out / f"seed{seed}.pt"
        if out_path.exists():
            logger.info("%s exists, skipped", out_path)
            continue
        lens = seed_lens(args, seed, tokenizer, family, chat_kwargs, model, runner, runner_kwargs, gfar_units, steering_by_layer, coefficient)
        temporary = out_path.with_suffix(".pt.tmp")
        torch.save(lens, temporary)
        temporary.rename(out_path)
        logger.info("wrote %s: %d episodes, %d response tokens, %.0f s, out-of-memory retries %d", out_path, len(lens["example_id"]),
                    len(lens["segment"]), lens["seconds"], sum(len(truncations) for truncations in lens["oom_truncated"]))


if __name__ == "__main__":
    main()
