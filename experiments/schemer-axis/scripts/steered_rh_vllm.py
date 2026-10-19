"""Reward-hacking harness on vLLM with a residual-stream intervention (PLAN.md, Phase 3), same CLI as steered_rh.py.

vLLM runs gpt-oss-120b far faster than the HF path, so the main run uses it. The intervention is a class-level patch
of the model's decoder-block class made before the engine is built (``intervene.patch_vllm_blocks`` for gpt-oss's
``TransformerBlock``, whose forward returns ``(mlp_output, residual)``; ``intervene.patch_vllm_stream_layers`` for
Gemma 4's ``Gemma4DecoderLayer``, whose forward returns the residual stream itself), acting on the residual stream
exactly where the HF path's hooks act; because the patch precedes torch.compile and CUDA-graph capture, it is part
of the compiled model and runs at full speed. The engine runs in this process (VLLM_ENABLE_V1_MULTIPROCESSING=0) so
the patch is visible to it. ``--eager`` disables compile and graphs (about 9x slower, smoke of 2026-10-08) and is
kept for cross-checks. Records and reasoning sidecars follow the vllm_harmony contract (the completion is split
into reasoning and final by the model family's parser). No projection sidecar: projections are measured offline
with the HF capture path.

    python scripts/steered_rh_vllm.py --model openai/gpt-oss-120b --data <dir> --scenarios ib_solvhard_s0 --out <dir>
        --tag <tag> --temperature 1.0 --sample-seed 0 --impossible-prompt critical_no_carve --max-attempts 5
        --state-attempt-budget --feedback neutral --max-new-tokens 16384 --direction <axis.pt> --layer-norms <pt>
        [--steer-mode add --steer-layer 17 --steer-alpha 0.1]
        [--steer-mode cap --cap-layers 12 ... --cap-thresholds <json {layer: threshold}>]
        [--random-direction-seed N]
        [--measure-norms]  (only: token-mean residual norm at every layer over the turn-0 prompts of the scenarios,
                            through eager hooks at the intervention site -> <out>/layer_norms.pt, the file
                            --layer-norms reads; same contract as steered_rh.py --measure-norms)
        [--check-hook]   (only: record layer-17 token-mean projection and norm of the turn-0 prompts of the first
                          --n problems through a hook at the same site, to compare with the HF projection sidecar)
        [--probe "question" ...]  (only: one sampled answer per question under the configured steering)

Model families: ``gpt-oss`` (Harmony; set SOO_CHAT_KWARGS='{"reasoning_effort":"low"}' for the recipe's low effort)
and ``gemma4`` (google/gemma-4-*; set SOO_CHAT_KWARGS='{"enable_thinking": true}' so the thought channel is on, as
in the Gemma 4 role extraction). Set VLLM_ENABLE_V1_MULTIPROCESSING=0 so apply_model runs in this process (closures
over tensors then need no pickling).
"""
import argparse
import json
import os
import sys
from pathlib import Path

import torch
from transformers import AutoConfig, AutoTokenizer

from selfconcept.assistant_axis.internals.model_specifics.gemma4 import split_gemma4_completion
from selfconcept.assistant_axis.projection import load_unit_axes
from selfconcept.codebench import harness
from selfconcept.codebench.vllm_harmony import vllm_harmony_generate
from selfconcept.common.chat import chat_template_kwargs
from selfconcept.common.harmony import split_harmony_completion
from selfconcept.correlation.random_vectors import random_directions
from selfconcept.measurement.intervene import add_transform, cap_transform, patch_vllm_blocks, patch_vllm_stream_layers


def split_gemma4(raw: str) -> tuple[str, str]:
    reasoning, content = split_gemma4_completion(raw)
    return reasoning or "", content


# Per model family: the vLLM decoder-block class to patch (module:attr), the patcher matching its forward contract,
# how to read the residual stream off a forward hook's output, and the completion parser.
FAMILIES = {
    "gpt-oss": {"block": ("vllm.model_executor.models.gpt_oss", "TransformerBlock"), "patch": patch_vllm_blocks,
                "stream": lambda output: output[0] + output[1], "split": split_harmony_completion},
    "gemma4": {"block": ("vllm.model_executor.models.gemma4", "Gemma4DecoderLayer"), "patch": patch_vllm_stream_layers,
               "stream": lambda output: output[0], "split": split_gemma4},
}


def family_of(model: str) -> str:
    name = model.lower()
    if "gpt-oss" in name or "gpt_oss" in name:
        return "gpt-oss"
    if "gemma-4" in name or "gemma4" in name:
        return "gemma4"
    raise SystemExit(f"no vLLM steering recipe for model {model!r} (families: {sorted(FAMILIES)})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", default=None, help="informational; vLLM loads the cached snapshot")
    parser.add_argument("--max-model-len", type=int, default=20480)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--engine-kwargs", type=json.loads, default={},
                        help='JSON of extra vllm.LLM kwargs, e.g. {"limit_mm_per_prompt": {"image": 0}} to skip the multimodal profiling budget on an 80 GB card')
    parser.add_argument("--eager", action="store_true", help="enforce_eager: no torch.compile / CUDA graphs (slow)")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=1.0, help="nucleus sampling (recipe: 1.0; Gemma 4 generation config: 0.95)")
    parser.add_argument("--top-k", type=int, default=-1, help="top-k sampling (recipe: off; Gemma 4 generation config: 64)")
    parser.add_argument("--direction", type=Path, help="(layers, hidden) axis artifact")
    parser.add_argument("--layer-norms", type=Path, help="layer_norms.pt (LayerNormResult) for alpha units")
    parser.add_argument("--steer-mode", choices=["none", "add", "cap"], default="none")
    parser.add_argument("--steer-layer", type=int, default=17)
    parser.add_argument("--steer-alpha", type=float, default=0.0, help="add: coefficient in units of the layer's mean norm")
    parser.add_argument("--cap-layers", nargs="+", type=int)
    parser.add_argument("--cap-thresholds", type=Path, help="JSON {layer: threshold} in projection units")
    parser.add_argument("--random-direction-seed", type=int, help="Control: isotropic random unit direction instead of --direction")
    parser.add_argument("--measure-norms", action="store_true")
    parser.add_argument("--check-hook", action="store_true")
    parser.add_argument("--check-layer", type=int, default=17)
    parser.add_argument("--probe", nargs="+", help="questions to answer once each under the steering, then exit")
    parser.add_argument("--score", type=Path, help="JSONL of {question, final}: write prompt logprobs of each text under the steering, then exit")
    harness.add_run_args(parser)
    args = parser.parse_args()
    if args.direction is None and args.random_direction_seed is None and (args.steer_mode != "none" or args.check_hook):
        parser.error("--direction is required for steering and --check-hook")
    if args.steer_mode == "add" and (args.layer_norms is None or args.steer_alpha == 0):
        parser.error("--steer-mode add needs --layer-norms and a nonzero --steer-alpha")
    if args.steer_mode == "cap" and (not args.cap_layers or args.cap_thresholds is None):
        parser.error("--steer-mode cap needs --cap-layers and --cap-thresholds")
    if args.score is not None and family_of(args.model) != "gpt-oss":
        parser.error("--score renders Harmony prompts and is gpt-oss only")
    return args


def model_dims(model_name: str) -> tuple[int, int]:
    """(num_layers, hidden) of the text tower from the model config (multimodal configs keep them in text_config)."""
    config = AutoConfig.from_pretrained(model_name)
    text = getattr(config, "text_config", None) or config
    return int(text.num_hidden_layers), int(text.hidden_size)


def model_layers(model):
    """The decoder layers of a loaded vLLM model: gpt-oss at model.model.layers, Gemma 4 (unified multimodal wrapper)
    at model.language_model.model.layers."""
    for path in ("model.layers", "language_model.model.layers"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
        except AttributeError:
            continue
        return obj
    raise AttributeError(f"no decoder layers found on {type(model).__name__}")


def directions_for(args, needed_layers: list[int], device) -> dict[int, torch.Tensor]:
    """Unit directions on ``device`` (the transforms must not copy host tensors inside a captured graph)."""
    num_layers, hidden = model_dims(args.model)  # load_unit_axes validates the artifact against these
    if args.random_direction_seed is not None:
        return {layer: torch.as_tensor(random_directions(hidden, 1, args.random_direction_seed, layer)[0]).float().to(device)
                for layer in needed_layers}
    unit_axes = load_unit_axes(args.direction, needed_layers, num_layers, hidden, args.model)
    return {layer: torch.as_tensor(unit_axes[layer]["direction"]).float().to(device) for layer in needed_layers}


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


RECORDS: list[dict] = []  # hook modes: one entry per forward pass through a hooked block


def turn0_prompts(args, tokenizer):
    """(example_id, prompt token ids) of the turn-0 task prompt of every selected problem."""
    instruction = harness.instruction_by_impossible_prompt[args.impossible_prompt]
    for scenario in args.scenarios:
        for example in harness.load_examples(scenario, args.data, args.n):
            message = harness.check_task_message(example, args.max_attempts if args.state_attempt_budget else None,
                                                 instruction, args.feedback)
            ids = tokenizer.apply_chat_template([{"role": "user", "content": message}], add_generation_prompt=True,
                                                **chat_template_kwargs(), tokenize=True, return_dict=False)
            yield example["example_id"], list(ids)


def prefill_records(llm, ids: list[int]) -> list[dict]:
    """Run one prompt through the engine for a single token and return the hook records of that request. Prefix
    caching is off in hook modes, so the records cover every prompt token (possibly across prefill chunks)."""
    from vllm import SamplingParams, TokensPrompt
    RECORDS.clear()
    llm.generate(TokensPrompt(prompt_token_ids=ids), SamplingParams(temperature=0, max_tokens=1), use_tqdm=False)
    return list(RECORDS)


def measure_norms(args, llm, tokenizer, stream_of) -> None:
    """Token-mean L2 norm of every decoder layer's output over the turn-0 task prompts (steered_rh.py contract)."""
    def install(model):
        layers = model_layers(model)
        for index, layer in enumerate(layers):
            def hook(_module, _inputs, output, index=index):
                stream = stream_of(output).float()
                RECORDS.append({"layer": index, "tokens": int(stream.shape[0]), "norm_sum": float(stream.norm(dim=-1).sum())})
                return output
            layer.register_forward_hook(hook)
        return len(layers)

    result = llm.apply_model(install)  # one return value per worker
    num_layers = int(result[0] if isinstance(result, list) else result)
    sums, count, skipped = torch.zeros(num_layers, dtype=torch.float64), 0, 0
    for example_id, ids in turn0_prompts(args, tokenizer):
        records = prefill_records(llm, ids)
        tokens = {layer: sum(r["tokens"] for r in records if r["layer"] == layer) for layer in range(num_layers)}
        if any(tokens[layer] != len(ids) for layer in range(num_layers)):
            print(f"{example_id}: hook saw {sorted(set(tokens.values()))} tokens per layer, prompt has {len(ids)}; skipped", file=sys.stderr)
            skipped += 1
            continue
        for r in records:
            sums[r["layer"]] += r["norm_sum"]
        count += len(ids)
    result = {"layer_norms": (sums / count).to(torch.float32), "num_layers": num_layers, "num_tokens": count,
              "num_conversations": 0, "num_skipped": skipped}
    args.out.mkdir(parents=True, exist_ok=True)
    torch.save(result, args.out / "layer_norms.pt")
    (args.out / "layer_norms.json").write_text(json.dumps({"layer_norms": result["layer_norms"].tolist(), "num_tokens": count,
                                                           "num_skipped": skipped, "scenarios": args.scenarios, "model": args.model,
                                                           "chat_kwargs": chat_template_kwargs(), "backend": "vllm-eager-hook"}, indent=2))
    print(json.dumps({"num_tokens": count, "num_skipped": skipped,
                      "layer_norms": [round(x, 2) for x in result["layer_norms"].tolist()]}))


def check_hook(args, llm, tokenizer, direction: torch.Tensor, stream_of) -> None:
    """Token-mean projection and norm of each turn-0 prompt at the check layer, through a recording hook."""
    u = direction / direction.norm()

    def install(model):
        def hook(_module, _inputs, output):
            stream = stream_of(output).float()
            RECORDS.append({"layer": args.check_layer, "tokens": int(stream.shape[0]),
                            "projection_sum": float((stream @ u.to(stream.device)).sum()), "norm_sum": float(stream.norm(dim=-1).sum())})
            return output
        return model_layers(model)[args.check_layer].register_forward_hook(hook) is not None

    llm.apply_model(install)
    results = []
    for example_id, ids in turn0_prompts(args, tokenizer):
        records = prefill_records(llm, ids)
        tokens = sum(r["tokens"] for r in records)
        entry = {"example_id": example_id, "prompt_tokens": len(ids), "forward_passes": len(records), "hooked_tokens": tokens}
        if tokens == len(ids):
            entry |= {"prompt_projection_mean": sum(r["projection_sum"] for r in records) / len(ids),
                      "prompt_norm_mean": sum(r["norm_sum"] for r in records) / len(ids)}
        results.append(entry)
        print(json.dumps(entry), flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "hook_check.json").write_text(json.dumps({"layer": args.check_layer, "direction": str(args.direction),
                                                          "results": results}, indent=1))


def probe(args, llm, tokenizer, steering: dict, split_completion) -> None:
    from vllm import SamplingParams, TokensPrompt
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "probe.jsonl").open("a") as f:
        for i, question in enumerate(args.probe):
            ids = tokenizer.apply_chat_template([{"role": "user", "content": question}], add_generation_prompt=True,
                                                **chat_template_kwargs(), tokenize=True, return_dict=False)
            [out] = llm.generate(TokensPrompt(prompt_token_ids=ids),
                                 SamplingParams(temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
                                                seed=args.sample_seed + i if args.temperature > 0 else None,
                                                max_tokens=min(args.max_new_tokens, 2048), skip_special_tokens=False), use_tqdm=False)
            reasoning, final = split_completion(out.outputs[0].text)
            row = {"question": question, "steering": steering, "eager": args.eager, "temperature": args.temperature,
                   "reasoning": reasoning, "final": final, "truncated": out.outputs[0].finish_reason == "length"}
            f.write(json.dumps(row) + "\n")
            print(f"### {question}\n[steering {steering['mode']} alpha={steering.get('alpha')} eager={args.eager} T={args.temperature}]"
                  f"\n{final[:1500]}\n", flush=True)


def score(args, llm, tokenizer, steering: dict) -> None:
    """Prompt log-probabilities of fixed texts under the configured steering (gpt-oss / Harmony only).

    Input: a JSONL of {"question": ..., "final": ...}; each row is scored as the Harmony prompt for the question
    followed by a final-channel message holding the text. Writes score.jsonl with one row per text holding the
    per-token log-probabilities of the appended text, so engines (eager / compiled, steered / unsteered) can be
    compared on identical tokens. vLLM skips the prefix cache for prompt_logprobs requests.
    """
    from vllm import SamplingParams, TokensPrompt
    args.out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in Path(args.score).read_text().splitlines() if line.strip()]
    with (args.out / "score.jsonl").open("w") as f:
        for row in rows:
            prompt_ids = tokenizer.apply_chat_template([{"role": "user", "content": row["question"]}],
                                                       add_generation_prompt=True, **chat_template_kwargs(),
                                                       tokenize=True, return_dict=False)
            text_ids = tokenizer.encode(f"<|channel|>final<|message|>{row['final']}<|return|>", add_special_tokens=False)
            ids = list(prompt_ids) + list(text_ids)
            [out] = llm.generate(TokensPrompt(prompt_token_ids=ids),
                                 SamplingParams(temperature=0, max_tokens=1, prompt_logprobs=0), use_tqdm=False)
            logprobs = []
            for pos, (token, entry) in enumerate(zip(ids, out.prompt_logprobs)):
                if pos < len(prompt_ids) or entry is None:
                    continue
                logprobs.append(entry[token].logprob)
            if len(logprobs) != len(text_ids):
                raise RuntimeError(f"scored {len(logprobs)} tokens, expected {len(text_ids)}")
            result = {"question": row["question"], "steering": steering, "eager": args.eager,
                      "prompt_tokens": len(prompt_ids), "text_tokens": len(text_ids),
                      "sum_logprob": float(sum(logprobs)), "logprobs": logprobs}
            f.write(json.dumps(result) + "\n")
            print(f"scored {len(text_ids)} tokens: sum logprob {result['sum_logprob']:.2f} | {row['question'][:50]}", flush=True)


def main() -> None:
    args = parse_args()
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    # vLLM keys its torch.compile cache on the model config, not on the patched block forward, so a cached
    # graph from a differently-steered run gets served to this one (smokes 14028603 and 14029300 crashed that way).
    os.environ["VLLM_DISABLE_COMPILE_CACHE"] = "1"
    import importlib
    from vllm import LLM

    family = family_of(args.model)
    recipe = FAMILIES[family]
    module_name, class_name = recipe["block"]
    block_cls = getattr(importlib.import_module(module_name), class_name)
    hook_mode = args.check_hook or args.measure_norms
    eager = args.eager or hook_mode  # the recording hooks have Python side effects and cannot be captured
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    needed = sorted({*([args.steer_layer] if args.steer_mode == "add" else []), *(args.cap_layers or []),
                     *([args.check_layer] if args.check_hook else [])})
    directions = directions_for(args, needed, device) if needed else {}
    transforms, steering = build_transforms(args, directions)
    steering["family"] = family
    if transforms and not hook_mode:
        recipe["patch"](block_cls, transforms)  # before the engine builds, compiles and captures the model
        steering["patched_layers"] = sorted(transforms)
        print(f"patched {class_name}.forward for layers {sorted(transforms)}: {steering}", file=sys.stderr)
    llm = LLM(model=args.model, tensor_parallel_size=1, max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_memory_utilization, enforce_eager=eager,
              **({"enable_prefix_caching": False} if hook_mode else {}),  # hooks must see every prompt token
              **args.engine_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if args.measure_norms:
        measure_norms(args, llm, tokenizer, recipe["stream"])
        return
    if args.check_hook:
        check_hook(args, llm, tokenizer, directions[args.check_layer], recipe["stream"])
        return
    if args.probe:
        probe(args, llm, tokenizer, steering, recipe["split"])
        return
    if args.score:
        score(args, llm, tokenizer, steering)
        return
    meta = {"model": args.model, "revision": args.revision, "backend": f"vllm-{family}-patched", "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k,
            "sample_seed": args.sample_seed if args.temperature > 0 else None,
            "chat_kwargs": chat_template_kwargs(), "steering": steering, "enforce_eager": eager}
    for scenario in args.scenarios:
        stem = f"{args.tag}_{scenario}"
        generate = vllm_harmony_generate(args.model, args.max_new_tokens, args.out / f"{stem}_reasoning.jsonl",
                                         args.temperature, args.sample_seed, llm=llm, split_completion=recipe["split"],
                                         top_p=args.top_p, top_k=args.top_k)
        harness.run_scenario(harness.load_examples(scenario, args.data, args.n), generate, max_attempts=args.max_attempts,
                             out=args.out, tag=args.tag, meta=meta, state_attempt_budget=args.state_attempt_budget,
                             impossible_prompt=args.impossible_prompt, feedback=args.feedback)
    print("=== steered_rh_vllm done ===", file=sys.stderr)


if __name__ == "__main__":
    main()
