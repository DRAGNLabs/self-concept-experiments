"""Reward-hacking harness on the HF backend with an optional residual-stream intervention (PLAN.md, Phase 3).

Runs selfconcept.codebench.harness on gpt-oss (Harmony) through HF generate(), so forward hooks can shift or
cap the residual stream along a direction while the model works on the clean-protocol coding tasks. Records,
reasoning sidecars and summaries have the same layout as the vLLM runs (``{out}/{tag}_{scenario}.jsonl``,
``{tag}_{scenario}_reasoning.jsonl``), plus ``{tag}_{scenario}_projection.jsonl`` with per-turn projections of
the residual stream onto the direction at the scored layers.

    python scripts/steered_rh.py --model openai/gpt-oss-120b --revision <sha> --data <dir> --scenarios ib_solvhard_s0
        --out <dir> --tag <tag> --temperature 1.0 --sample-seed 0 --impossible-prompt critical_no_carve
        --max-attempts 5 --state-attempt-budget --feedback neutral --max-new-tokens 16384
        --direction <axis.pt> --layer-norms <layer_norms.pt>
        [--steer-mode add --steer-layer 17 --steer-alpha 0.1]
        [--steer-mode cap --cap-layers 12 13 ... --cap-thresholds <json: {layer: threshold}>]
        [--random-direction-seed N]   (replace the direction by an isotropic random one, matched by unit norm)
        [--measure-norms]             (only: token-mean residual norms at every layer over the turn-0 prompts)

Set SOO_CHAT_KWARGS='{"reasoning_effort":"low"}' for the recipe's low effort.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from selfconcept.assistant_axis.projection import load_unit_axes
from selfconcept.codebench import harness
from selfconcept.codebench.hf_harmony import hf_harmony_generate
from selfconcept.common.chat import chat_template_kwargs
from selfconcept.common.loading import load_causal_lm
from selfconcept.correlation.random_vectors import random_directions
from selfconcept.measurement.capture import capture
from selfconcept.measurement.intervene import add_direction, cap_direction
from selfconcept.soo.activations import get_decoder_layers

ATTENTION_BY_FAMILY = {"gpt-oss": "kernels-community/vllm-flash-attn3"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--attn-implementation", default=None,
                        help="Defaults to the gpt-oss flash-attention kernel for gpt-oss models, else the model default")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--direction", type=Path, help="(layers, hidden) axis artifact; required unless --measure-norms")
    parser.add_argument("--layer-norms", type=Path, help="layer_norms.pt (LayerNormResult) for alpha units")
    parser.add_argument("--steer-mode", choices=["none", "add", "cap"], default="none")
    parser.add_argument("--steer-layer", type=int, default=17)
    parser.add_argument("--steer-alpha", type=float, default=0.0, help="add: coefficient in units of the layer's mean norm")
    parser.add_argument("--cap-layers", nargs="+", type=int)
    parser.add_argument("--cap-thresholds", type=Path, help="JSON {layer: threshold} in projection units")
    parser.add_argument("--random-direction-seed", type=int, help="Control: isotropic random unit direction instead of --direction")
    parser.add_argument("--projection-layers", nargs="+", type=int, default=[17])
    parser.add_argument("--measure-norms", action="store_true")
    harness.add_run_args(parser)
    args = parser.parse_args()
    if not args.measure_norms and args.direction is None:
        parser.error("--direction is required unless --measure-norms")
    if args.steer_mode == "add" and (args.layer_norms is None or args.steer_alpha == 0):
        parser.error("--steer-mode add needs --layer-norms and a nonzero --steer-alpha")
    if args.steer_mode == "cap" and (not args.cap_layers or args.cap_thresholds is None):
        parser.error("--steer-mode cap needs --cap-layers and --cap-thresholds")
    return args


def load_model(args):
    family_attention = ATTENTION_BY_FAMILY.get("gpt-oss") if "gpt-oss" in args.model else None
    kwargs = {"dtype": torch.bfloat16, "device_map": "auto"}
    if args.revision:
        kwargs["revision"] = args.revision
    if attention := (args.attn_implementation or family_attention):
        kwargs["attn_implementation"] = attention
    if torch.cuda.is_available():
        kwargs["local_files_only"] = True
    model = load_causal_lm(args.model, **kwargs)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision or "main")
    return model, tokenizer


def measure_norms(args, model, tokenizer) -> None:
    """Token-mean L2 norm of every decoder layer's output over the turn-0 task prompts of the scenarios."""
    layers = list(range(len(get_decoder_layers(model))))
    sums, count = np.zeros(len(layers)), 0
    device = next(model.get_input_embeddings().parameters()).device
    for scenario in args.scenarios:
        for example in harness.load_examples(scenario, args.data, args.n):
            instruction = harness.instruction_by_impossible_prompt[args.impossible_prompt]
            message = harness.check_task_message(example, args.max_attempts if args.state_attempt_budget else None,
                                                 instruction, args.feedback)
            ids = tokenizer.apply_chat_template([{"role": "user", "content": message}], add_generation_prompt=True,
                                                **chat_template_kwargs(), tokenize=True, return_dict=False)
            inputs = torch.tensor([ids], device=device)
            with torch.inference_mode(), capture(model, layers, lambda layer, residual: residual.norm(dim=-1)) as norms:
                model(input_ids=inputs, attention_mask=torch.ones_like(inputs), use_cache=False, logits_to_keep=1)
            for i, layer in enumerate(layers):
                sums[i] += float(norms[layer][0].sum())
            count += len(ids)
    result = {"layer_norms": torch.tensor(sums / count, dtype=torch.float32), "num_layers": len(layers),
              "num_tokens": count, "num_conversations": 0, "num_skipped": 0}
    args.out.mkdir(parents=True, exist_ok=True)
    torch.save(result, args.out / "layer_norms.pt")
    (args.out / "layer_norms.json").write_text(json.dumps({"layer_norms": result["layer_norms"].tolist(),
                                                           "num_tokens": count, "scenarios": args.scenarios,
                                                           "chat_kwargs": chat_template_kwargs()}, indent=2))
    print(json.dumps({"num_tokens": count, "layer_norms": [round(x, 2) for x in result["layer_norms"].tolist()]}))


def main() -> None:
    args = parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1") if torch.cuda.is_available() else None
    model, tokenizer = load_model(args)
    if args.measure_norms:
        measure_norms(args, model, tokenizer)
        return
    num_layers = len(get_decoder_layers(model))
    hidden = getattr(model.config, "text_config", model.config).hidden_size
    needed_layers = sorted({*([args.steer_layer] if args.steer_mode == "add" else []), *args.projection_layers, *(args.cap_layers or [])})
    unit_axes = load_unit_axes(args.direction, needed_layers, num_layers, hidden, args.model if "gpt-oss" in args.model else None)
    directions = {layer: torch.as_tensor(unit_axes[layer]["direction"]) for layer in needed_layers}
    if args.random_direction_seed is not None:
        directions = {layer: torch.as_tensor(random_directions(hidden, 1, args.random_direction_seed, layer)[0])
                      for layer in needed_layers}
    interventions = []
    steering = {"mode": args.steer_mode, "direction": str(args.direction), "random_direction_seed": args.random_direction_seed}
    if args.steer_mode == "add":
        norms = torch.load(args.layer_norms, map_location="cpu", weights_only=False)
        layer_norm = float((norms["layer_norms"] if isinstance(norms, dict) else norms)[args.steer_layer])
        coefficient = args.steer_alpha * layer_norm
        interventions.append(lambda: add_direction(model, args.steer_layer, directions[args.steer_layer], coefficient))
        steering |= {"layer": args.steer_layer, "alpha": args.steer_alpha, "layer_norm": layer_norm, "coefficient": coefficient}
    elif args.steer_mode == "cap":
        thresholds = {int(k): float(v) for k, v in json.loads(args.cap_thresholds.read_text()).items()}
        thresholds = {layer: thresholds[layer] for layer in args.cap_layers}
        interventions.append(lambda: cap_direction(model, thresholds, {layer: directions[layer] for layer in thresholds}))
        steering |= {"cap_layers": args.cap_layers, "cap_thresholds": thresholds}
    meta = {"model": args.model, "revision": args.revision, "backend": "hf-harmony", "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature, "sample_seed": args.sample_seed if args.temperature > 0 else None,
            "chat_kwargs": chat_template_kwargs(), "steering": steering, "projection_layers": args.projection_layers}
    for scenario in args.scenarios:
        stem = f"{args.tag}_{scenario}"
        generate = hf_harmony_generate(
            model, tokenizer, args.max_new_tokens, args.out / f"{stem}_reasoning.jsonl", args.temperature, args.sample_seed,
            interventions=interventions, projection_layers=args.projection_layers,
            directions_by_layer={layer: directions[layer] for layer in args.projection_layers},
            projection_log_path=args.out / f"{stem}_projection.jsonl")
        harness.run_scenario(harness.load_examples(scenario, args.data, args.n), generate, max_attempts=args.max_attempts,
                             out=args.out, tag=args.tag, meta=meta, state_attempt_budget=args.state_attempt_budget,
                             impossible_prompt=args.impossible_prompt, feedback=args.feedback)
    print("=== steered_rh done ===", file=sys.stderr)


if __name__ == "__main__":
    main()
