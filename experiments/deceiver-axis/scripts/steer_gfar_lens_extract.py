"""Downstream effect of gi_v2 steering on the gfar projection, per response token, on finished unsteered episodes.

Each turn of each unsteered episode is teacher-forced once as generated and once per steering arm, and at every layer
the projection of each reasoning / answer token's residual onto the unit gfar direction is stored. Arms add, at the
steer layer and every position, coefficient (alpha x the layer's mean residual norm) times
- steered: the unit gi_v2 direction, as in the steered runs;
- steered_gfar_component: only gi_v2's component along the unit gfar direction at the steer layer, (gi_v2 . gfar) gfar;
- --null-direction NAME=PATH: a control direction's unit vector at the steer layer (same coefficient), stored under
  null_projections[NAME]; with --null-mode gfar_component only its component along gfar at the steer layer.
  --token-stride K keeps every K-th response token of each turn (for many arms).
Chart: steer_gfar_lens_chart.py.

    SOO_CHAT_KWARGS='{"reasoning_effort":"low"}' python scripts/steer_gfar_lens_extract.py --model openai/gpt-oss-120b \
        --data <data dir> --tag main --run <unsteered arm dir with seedN/> --seeds 0 1 \
        --gi-v2 <persona_contrast.pt> --gfar <directions_headline.pt>:goodfire_all_response_pl_all_families \
        --steer-layer 17 --alpha 0.2 --layer-norms <layer_norms.pt> --arms steered steered_gfar_component --out <dir>

Writes <out>/seed<N>.pt per seed; existing files are skipped.
"""
import argparse
import json
import logging
import time
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Literal, NamedTuple, NotRequired, TypedDict

import numpy as np
import torch
from jaxtyping import Float, Int8
from torch import Tensor, nn
from transformers import AutoConfig, AutoTokenizer

import make_dossiers
from selfconcept.arm_search.rh.dossier import episode_conversation
from selfconcept.arm_search.rh.episodes import reasoning_by_turn_by_example_id
from selfconcept.codebench.harness import CheckRecord, ExampleRecord, WorkspaceRecord, is_scored_record
from selfconcept.common.chat import chat_template_kwargs
from selfconcept.common.hf_strong_types import HFTokenizer
from selfconcept.common.teacher_forcing.completion_markup import CompletionMarkupSpecifics
from selfconcept.common.teacher_forcing.registry import completion_markup_specifics_for
from selfconcept.common.teacher_forcing.rendering import RenderedTurn, Segment, episode_turns
from selfconcept.common.loading import load_causal_lm
from selfconcept.measurement.intervene import _hooks, add_transform
from selfconcept.measurement.steering_specifics import steering_model_specifics
from selfconcept.measurement.unit_axes import load_unit_axes
from selfconcept.soo.activations import get_decoder_layers

logger = logging.getLogger("steer_gfar_lens")
MIN_TOKENS_BEFORE_GIVING_UP = 4096
FP16_SAFE_ABS = 6e4

type Transform = Callable[[Tensor], Tensor]
Arm = Literal["steered", "steered_gfar_component"]
ARMS: tuple[Arm, ...] = ("steered", "steered_gfar_component")
NullMode = Literal["full", "gfar_component"]
type Projections = Float[Tensor, "layers response_tokens"]


class SeedLens(TypedDict):
    example_id: list[str]
    verdict_key: list[str]
    token_offsets: np.ndarray
    segment: Int8[np.ndarray, "response_tokens"]
    arms: list[Arm]
    unsteered: Projections
    steered: NotRequired[Projections]
    steered_gfar_component: NotRequired[Projections]
    null_projections: dict[str, Projections]
    null_mode: NullMode
    token_stride: int
    turn_lengths: list[list[int]]
    oom_truncated: list[list[tuple[int, int]]]
    missing_reasoning: list[list[int]]
    model: str
    family: str
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
    parser.add_argument("--arms", nargs="*", choices=ARMS, default=["steered"])
    parser.add_argument("--null-direction", action="append", default=[], help="NAME=PATH of a (layers, hidden) {axis, model} artifact")
    parser.add_argument("--null-mode", choices=("full", "gfar_component"), default="full")
    parser.add_argument("--token-stride", type=int, default=1, help="keep every K-th response token of each turn")
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


class LensForward(NamedTuple):
    model: nn.Module
    runner: nn.Module  # the model, or its base without the LM head
    runner_kwargs: dict[str, Any]
    gfar_units: Float[Tensor, "layers hidden"]
    steering_by_layer_by_arm: dict[str, dict[int, Transform]]
    token_stride: int


class TurnProjections(NamedTuple):
    unsteered: Projections
    steered_by_arm: dict[str, Projections]
    response_segments: Int8[np.ndarray, "response_tokens"]
    kept_tokens: int


class EpisodeProjections(TypedDict):
    unsteered: list[Projections]
    steered_by_arm: dict[str, list[Projections]]
    response_segments: list[Int8[np.ndarray, "response_tokens"]]
    turn_lengths: list[int]
    oom_truncated: list[tuple[int, int]]
    missing_reasoning: list[int]


def response_projections(forward: LensForward, ids: list[int], response_mask: torch.Tensor, steering_by_layer: dict[int, Transform]) -> Projections:
    decoder_layers = get_decoder_layers(forward.model)
    projections_by_layer: dict[int, Tensor] = {}

    def record(layer: int):
        def hook(_module, _inputs, output) -> None:
            states = output[0] if isinstance(output, tuple) else output
            hidden_states = states[0]
            projections_by_layer[layer] = (hidden_states[response_mask.to(hidden_states.device)].float() @ forward.gfar_units[layer].to(hidden_states.device)).cpu()
        return hook

    # Steering hooks are registered before the recording hooks so the steer layer records its steered output.
    transforms_by_layer: dict[int, object] = dict(steering_by_layer)
    with _hooks(forward.model, transforms_by_layer) if transforms_by_layer else nullcontext():
        handles = [decoder_layers[layer].register_forward_hook(record(layer)) for layer in range(len(decoder_layers))]
        try:
            with torch.inference_mode():
                forward.runner(input_ids=torch.tensor([ids], device=response_mask.device), **forward.runner_kwargs)
        finally:
            for handle in handles:
                handle.remove()
    return torch.stack([projections_by_layer[layer] for layer in range(len(decoder_layers))])


def gfar_component(unit: Float[Tensor, "hidden"], gfar_unit: Float[Tensor, "hidden"]) -> Float[Tensor, "hidden"]:
    return (unit @ gfar_unit) * gfar_unit


def arm_vector(arm: Arm, gi_v2_unit: Float[Tensor, "hidden"], gfar_unit: Float[Tensor, "hidden"]) -> Float[Tensor, "hidden"]:
    return gi_v2_unit if arm == "steered" else gfar_component(gi_v2_unit, gfar_unit)


def strided_response_mask(segments: Int8[np.ndarray, "tokens"], token_stride: int) -> np.ndarray:
    is_response = (segments == Segment.REASONING) | (segments == Segment.ANSWER)
    is_response[np.flatnonzero(is_response)[np.arange(int(is_response.sum())) % token_stride != 0]] = False
    return is_response


def turn_projections(forward: LensForward, ids: list[int], segments: Int8[np.ndarray, "tokens"]) -> TurnProjections:
    """A turn the GPU cannot hold is cut to its first 80% and retried."""
    kept = len(ids)
    while True:
        is_response = strided_response_mask(segments[:kept], forward.token_stride)
        response_mask = torch.as_tensor(is_response, device=forward.gfar_units.device)
        try:
            unsteered = response_projections(forward, ids[:kept], response_mask, {})
            steered_by_arm = {arm: response_projections(forward, ids[:kept], response_mask, steering_by_layer)
                              for arm, steering_by_layer in forward.steering_by_layer_by_arm.items()}
            return TurnProjections(unsteered, steered_by_arm, segments[:kept][is_response], kept)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            if kept < MIN_TOKENS_BEFORE_GIVING_UP:
                raise
            kept = int(kept * 0.8)
            logger.warning("out of memory: turn retried on its first %d of %d tokens", kept, len(ids))


def episode_projections(forward: LensForward, turns: list[RenderedTurn]) -> EpisodeProjections:
    projections_by_turn = [turn_projections(forward, turn.ids, turn.segments) for turn in turns]
    return {
        "unsteered": [projections.unsteered for projections in projections_by_turn],
        "steered_by_arm": {arm: [projections.steered_by_arm[arm] for projections in projections_by_turn] for arm in forward.steering_by_layer_by_arm},
        "response_segments": [projections.response_segments for projections in projections_by_turn],
        "turn_lengths": [turn.uncapped_length for turn in turns],
        "oom_truncated": [(turn.turn, projections.kept_tokens) for turn, projections in zip(turns, projections_by_turn)
                          if projections.kept_tokens < len(turn.ids)],
        "missing_reasoning": [turn.turn for turn in turns if turn.reasoning_missing],
    }


def compact(projections: Projections) -> Projections:
    return projections.half() if float(projections.abs().max()) < FP16_SAFE_ABS else projections


def seed_reasoning_by_turn_by_example_id(run_dir: Path, tag: str) -> dict[str, dict[int, str]]:
    """Merged over the run's per-scenario reasoning sidecars."""
    merged: dict[str, dict[int, str]] = {}
    for path in sorted(run_dir.glob(f"{tag}_*_reasoning.jsonl")):
        for example_id, reasoning_by_turn in reasoning_by_turn_by_example_id(path).items():
            merged.setdefault(example_id, {}).update(reasoning_by_turn)
    return merged


def rendered_episode(args: argparse.Namespace, run_dir: Path, specifics: CompletionMarkupSpecifics, tokenizer: HFTokenizer,
                     record: CheckRecord | WorkspaceRecord, reasoning_by_turn: dict[int, str], chat_kwargs: dict[str, Any]) -> list[RenderedTurn]:
    example = make_dossiers.find_example(args.data, args.tag, record["example_id"], run_dir)
    steps = episode_conversation(example, record, record.get("feedback") or "impossiblebench")
    answers = [attempt.get("completion") or "" for attempt in record["attempts"]]
    return episode_turns(specifics, tokenizer, steps, answers, reasoning_by_turn, args.max_length, chat_kwargs)


def seed_lens(args: argparse.Namespace, seed: int, specifics: CompletionMarkupSpecifics, tokenizer: HFTokenizer, family: str,
              chat_kwargs: dict[str, Any], forward: LensForward, coefficient: float) -> SeedLens:
    run_dir = args.run / f"seed{seed}"
    all_records: list[ExampleRecord] = [json.loads(line) for line in (run_dir / f"{args.tag}_impossible_original.jsonl").open()]
    records = [record for record in all_records if is_scored_record(record)][: args.n]
    reasoning_by_turn_by_example = seed_reasoning_by_turn_by_example_id(run_dir, args.tag)
    started = time.time()
    episodes: list[EpisodeProjections] = []
    for record in records:
        turns = rendered_episode(args, run_dir, specifics, tokenizer, record, reasoning_by_turn_by_example.get(record["example_id"], {}), chat_kwargs)
        episodes.append(episode_projections(forward, turns))
        logger.info("seed %d: %d/%d episodes, %.0f s", seed, len(episodes), len(records), time.time() - started)
    response_tokens_by_episode = [sum(len(segments) for segments in episode["response_segments"]) for episode in episodes]
    lens: SeedLens = {
        "arms": list(args.arms),
        "null_projections": {}, "null_mode": args.null_mode, "token_stride": args.token_stride,
        "example_id": [record["example_id"] for record in records],
        "verdict_key": [f"{args.run.name}/seed{seed}/{record['example_id']}" for record in records],
        "token_offsets": np.concatenate([[0], np.cumsum(response_tokens_by_episode)]).astype(np.int64),
        "segment": np.concatenate([segments for episode in episodes for segments in episode["response_segments"]]),
        "unsteered": compact(torch.cat([projections for episode in episodes for projections in episode["unsteered"]], dim=1)),
        "turn_lengths": [episode["turn_lengths"] for episode in episodes],
        "oom_truncated": [episode["oom_truncated"] for episode in episodes],
        "missing_reasoning": [episode["missing_reasoning"] for episode in episodes],
        "model": args.model, "family": family, "chat_kwargs": chat_kwargs, "tag": args.tag, "run": str(args.run), "seed": seed,
        "steer_layer": args.steer_layer, "alpha": args.alpha, "coefficient": coefficient, "gi_v2": str(args.gi_v2), "gfar": args.gfar,
        "max_length": args.max_length, "seconds": time.time() - started,
    }
    for arm in forward.steering_by_layer_by_arm:
        projections = compact(torch.cat([part for episode in episodes for part in episode["steered_by_arm"][arm]], dim=1))
        if arm == "steered" or arm == "steered_gfar_component":
            lens[arm] = projections
        else:
            lens["null_projections"][arm] = projections
    return lens


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args = parse_args()
    specifics = steering_model_specifics(args.model)
    markup_specifics = completion_markup_specifics_for(args.model)
    chat_kwargs = chat_template_kwargs()
    tokenizer: HFTokenizer = AutoTokenizer.from_pretrained(args.model)
    num_layers, hidden = model_dims(args.model)
    device = torch.device("cuda:0")
    gfar_units = load_gfar_units(args.gfar, num_layers, hidden).to(device)
    gi_v2_unit = torch.as_tensor(load_unit_axes(args.gi_v2, [args.steer_layer], num_layers, hidden, args.model)[args.steer_layer]["direction"]).to(device)
    coefficient = steering_coefficient(args.layer_norms, args.steer_layer, args.alpha)
    steering_by_layer_by_arm: dict[str, dict[int, Transform]] = {
        arm: {args.steer_layer: add_transform(arm_vector(arm, gi_v2_unit, gfar_units[args.steer_layer]), coefficient)} for arm in args.arms}
    for spec in args.null_direction:
        name, path = spec.split("=", 1)
        null_unit = torch.as_tensor(load_unit_axes(Path(path), [args.steer_layer], num_layers, hidden, args.model)[args.steer_layer]["direction"]).to(device)
        null_vector = null_unit if args.null_mode == "full" else gfar_component(null_unit, gfar_units[args.steer_layer])
        steering_by_layer_by_arm[name] = {args.steer_layer: add_transform(null_vector, coefficient)}
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
    forward = LensForward(model, runner, runner_kwargs, gfar_units, steering_by_layer_by_arm, args.token_stride)

    args.out.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        out_path = args.out / f"seed{seed}.pt"
        if out_path.exists():
            logger.info("%s exists, skipped", out_path)
            continue
        lens = seed_lens(args, seed, markup_specifics, tokenizer, specifics.family, chat_kwargs, forward, coefficient)
        temporary = out_path.with_suffix(".pt.tmp")
        torch.save(lens, temporary)
        temporary.rename(out_path)
        logger.info("wrote %s: %d episodes, %d response tokens, %.0f s, out-of-memory retries %d", out_path, len(lens["example_id"]),
                    len(lens["segment"]), lens["seconds"], sum(len(truncations) for truncations in lens["oom_truncated"]))


if __name__ == "__main__":
    main()
