"""Reward-hacking harness on vLLM with a residual-stream intervention (PLAN.md, Phase 3), same CLI as steered_rh.py.

vLLM runs gpt-oss-120b about 18x faster than the HF path single-stream (smokes of 2026-10-07), so the main run uses
it. Hooks are registered inside the engine worker with ``LLM.apply_model`` on the gpt-oss blocks, acting on the
residual stream (mlp_output + residual) exactly where the HF path's hooks act; the engine runs eagerly so the
hooks are not bypassed by compiled or captured graphs. Records and reasoning sidecars follow the vllm_harmony
contract. No projection sidecar: projections are measured offline on the transcripts with the HF capture path.

    python scripts/steered_rh_vllm.py --model openai/gpt-oss-120b --data <dir> --scenarios ib_solvhard_s0 --out <dir>
        --tag <tag> --temperature 1.0 --sample-seed 0 --impossible-prompt critical_no_carve --max-attempts 5
        --state-attempt-budget --feedback neutral --max-new-tokens 16384 --direction <axis.pt> --layer-norms <pt>
        [--steer-mode add --steer-layer 17 --steer-alpha 0.1]
        [--steer-mode cap --cap-layers 12 ... --cap-thresholds <json {layer: threshold}>]
        [--random-direction-seed N]
        [--check-hook]   (only: record layer-17 token-mean projection and norm of the turn-0 prompts of the first
                          --n problems through a hook at the same site, to compare with the HF projection sidecar)
        [--probe "question" ...]  (only: one sampled answer per question under the configured steering)

Set SOO_CHAT_KWARGS='{"reasoning_effort":"low"}' for the recipe's low effort. Set VLLM_ENABLE_V1_MULTIPROCESSING=0
so apply_model runs in this process (closures over tensors then need no pickling).
"""
import argparse
import json
import os
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

from selfconcept.assistant_axis.projection import load_unit_axes
from selfconcept.codebench import harness
from selfconcept.codebench.vllm_harmony import vllm_harmony_generate
from selfconcept.common.chat import chat_template_kwargs
from selfconcept.common.harmony import split_harmony_completion
from selfconcept.correlation.random_vectors import random_directions
from selfconcept.measurement.intervene import add_transform, cap_transform, register_vllm_transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", default=None, help="informational; vLLM loads the cached snapshot")
    parser.add_argument("--max-model-len", type=int, default=20480)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--no-enforce-eager", action="store_true", help="allow compile/cudagraphs (hooks may be bypassed)")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--direction", type=Path, help="(layers, hidden) axis artifact")
    parser.add_argument("--layer-norms", type=Path, help="layer_norms.pt (LayerNormResult) for alpha units")
    parser.add_argument("--steer-mode", choices=["none", "add", "cap"], default="none")
    parser.add_argument("--steer-layer", type=int, default=17)
    parser.add_argument("--steer-alpha", type=float, default=0.0, help="add: coefficient in units of the layer's mean norm")
    parser.add_argument("--cap-layers", nargs="+", type=int)
    parser.add_argument("--cap-thresholds", type=Path, help="JSON {layer: threshold} in projection units")
    parser.add_argument("--random-direction-seed", type=int, help="Control: isotropic random unit direction instead of --direction")
    parser.add_argument("--check-hook", action="store_true")
    parser.add_argument("--check-layer", type=int, default=17)
    parser.add_argument("--probe", nargs="+", help="questions to answer once each under the steering, then exit")
    harness.add_run_args(parser)
    args = parser.parse_args()
    if args.direction is None and args.random_direction_seed is None and (args.steer_mode != "none" or args.check_hook):
        parser.error("--direction is required for steering and --check-hook")
    if args.steer_mode == "add" and (args.layer_norms is None or args.steer_alpha == 0):
        parser.error("--steer-mode add needs --layer-norms and a nonzero --steer-alpha")
    if args.steer_mode == "cap" and (not args.cap_layers or args.cap_thresholds is None):
        parser.error("--steer-mode cap needs --cap-layers and --cap-thresholds")
    return args


def model_layers(model):
    return model.model.layers


def directions_for(args, needed_layers: list[int]) -> dict[int, torch.Tensor]:
    num_layers, hidden = 36, 2880  # gpt-oss-120b; load_unit_axes validates the artifact against these
    if args.random_direction_seed is not None:
        return {layer: torch.as_tensor(random_directions(hidden, 1, args.random_direction_seed, layer)[0]).float()
                for layer in needed_layers}
    unit_axes = load_unit_axes(args.direction, needed_layers, num_layers, hidden, args.model if "gpt-oss" in args.model else None)
    return {layer: torch.as_tensor(unit_axes[layer]["direction"]).float() for layer in needed_layers}


def build_transforms(args, directions) -> tuple[dict[int, object], dict]:
    steering = {"mode": args.steer_mode, "direction": str(args.direction), "random_direction_seed": args.random_direction_seed}
    transforms = {}
    if args.steer_mode == "add":
        norms = torch.load(args.layer_norms, map_location="cpu", weights_only=False)
        layer_norm = float((norms["layer_norms"] if isinstance(norms, dict) else norms)[args.steer_layer])
        coefficient = args.steer_alpha * layer_norm
        transforms[args.steer_layer] = add_transform(directions[args.steer_layer], coefficient)
        steering |= {"layer": args.steer_layer, "alpha": args.steer_alpha, "layer_norm": layer_norm, "coefficient": coefficient}
    elif args.steer_mode == "cap":
        thresholds = {int(k): float(v) for k, v in json.loads(args.cap_thresholds.read_text()).items()}
        thresholds = {layer: thresholds[layer] for layer in args.cap_layers}
        for layer, threshold in thresholds.items():
            transforms[layer] = cap_transform(directions[layer], threshold)
        steering |= {"cap_layers": args.cap_layers, "cap_thresholds": thresholds}
    return transforms, steering


RECORDS: list[dict] = []  # --check-hook: one entry per forward pass through the checked block


def check_hook(args, llm, tokenizer, direction: torch.Tensor) -> None:
    """Token-mean projection and norm of each turn-0 prompt at the check layer, through a recording hook."""
    u = direction / direction.norm()

    def install(model):
        def hook(_module, _inputs, output):
            mlp_output, residual = output
            stream = (mlp_output + residual).float()
            RECORDS.append({"tokens": int(stream.shape[0]), "projection_sum": float((stream @ u.to(stream.device)).sum()),
                            "norm_sum": float(stream.norm(dim=-1).sum())})
            return output
        return model_layers(model)[args.check_layer].register_forward_hook(hook) is not None

    from vllm import SamplingParams, TokensPrompt
    llm.apply_model(install)
    instruction = harness.instruction_by_impossible_prompt[args.impossible_prompt]
    results = []
    for scenario in args.scenarios:
        for example in harness.load_examples(scenario, args.data, args.n):
            message = harness.check_task_message(example, args.max_attempts if args.state_attempt_budget else None,
                                                 instruction, args.feedback)
            ids = tokenizer.apply_chat_template([{"role": "user", "content": message}], add_generation_prompt=True,
                                                **chat_template_kwargs(), tokenize=True, return_dict=False)
            RECORDS.clear()
            llm.generate(TokensPrompt(prompt_token_ids=ids), SamplingParams(temperature=0, max_tokens=1), use_tqdm=False)
            prefill = [r for r in RECORDS if r["tokens"] == len(ids)]
            entry = {"example_id": example["example_id"], "prompt_tokens": len(ids), "forward_passes": len(RECORDS),
                     "prefill_passes": len(prefill)}
            if prefill:
                entry |= {"prompt_projection_mean": prefill[0]["projection_sum"] / len(ids),
                          "prompt_norm_mean": prefill[0]["norm_sum"] / len(ids)}
            results.append(entry)
            print(json.dumps(entry), flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "hook_check.json").write_text(json.dumps({"layer": args.check_layer, "direction": str(args.direction),
                                                          "results": results}, indent=1))


def probe(args, llm, tokenizer, steering: dict) -> None:
    from vllm import SamplingParams, TokensPrompt
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "probe.jsonl").open("a") as f:
        for i, question in enumerate(args.probe):
            ids = tokenizer.apply_chat_template([{"role": "user", "content": question}], add_generation_prompt=True,
                                                **chat_template_kwargs(), tokenize=True, return_dict=False)
            [out] = llm.generate(TokensPrompt(prompt_token_ids=ids),
                                 SamplingParams(temperature=args.temperature, top_p=1.0, seed=args.sample_seed + i,
                                                max_tokens=min(args.max_new_tokens, 1024), skip_special_tokens=False), use_tqdm=False)
            reasoning, final = split_harmony_completion(out.outputs[0].text)
            row = {"question": question, "steering": steering, "reasoning": reasoning, "final": final}
            f.write(json.dumps(row) + "\n")
            print(f"### {question}\n[steering {steering['mode']} alpha={steering.get('alpha')}]\n{final[:1500]}\n", flush=True)


def main() -> None:
    args = parse_args()
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    from vllm import LLM

    llm = LLM(model=args.model, tensor_parallel_size=1, max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_memory_utilization, enforce_eager=not args.no_enforce_eager)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    needed = sorted({*([args.steer_layer] if args.steer_mode == "add" else []), *(args.cap_layers or []),
                     *([args.check_layer] if args.check_hook else [])})
    directions = directions_for(args, needed) if needed else {}
    if args.check_hook:
        check_hook(args, llm, tokenizer, directions[args.check_layer])
        return
    transforms, steering = build_transforms(args, directions)
    if transforms:
        n_hooks = llm.apply_model(lambda model: len(register_vllm_transforms(model_layers(model), transforms)))
        steering["hooks_registered"] = n_hooks
        print(f"registered {n_hooks} hook(s) per worker: {steering}", file=sys.stderr)
    if args.probe:
        probe(args, llm, tokenizer, steering)
        return
    meta = {"model": args.model, "revision": args.revision, "backend": "vllm-harmony-hooked", "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature, "sample_seed": args.sample_seed if args.temperature > 0 else None,
            "chat_kwargs": chat_template_kwargs(), "steering": steering, "enforce_eager": not args.no_enforce_eager}
    for scenario in args.scenarios:
        stem = f"{args.tag}_{scenario}"
        generate = vllm_harmony_generate(args.model, args.max_new_tokens, args.out / f"{stem}_reasoning.jsonl",
                                         args.temperature, args.sample_seed, llm=llm)
        harness.run_scenario(harness.load_examples(scenario, args.data, args.n), generate, max_attempts=args.max_attempts,
                             out=args.out, tag=args.tag, meta=meta, state_attempt_budget=args.state_attempt_budget,
                             impossible_prompt=args.impossible_prompt, feedback=args.feedback)
    print("=== steered_rh_vllm done ===", file=sys.stderr)


if __name__ == "__main__":
    main()
