"""EXPLORATORY (lightly reviewed when merged 2026-11-20): kept for reference, not a validated measurement.

Resample the decision turn of unsteered hack episodes with the abstain-flag instruction, under one arm's steering
(vLLM, patched decoder block as steered_rh_vllm.py; one engine per arm, so one job per arm).

Prefixes: --prefixes unsteered HACK episodes with a rubric-v3 judge detail, one per problem (problems and seeds drawn
with a fixed rng, so every arm gets the same prefixes). The prompt is the episode up to the judge's first-consideration
turn, rendered as abstain_flag_logprobs.flagged_turns does (INSTRUCTION appended to the task message; earlier
assistant turns as their visible replies), ending in the generation prompt. Per prefix, --k samples at the rollouts'
sampling settings: stage 1 generates until the model opens its reply to the user (Muse "to=user<|message|>"), stage 2
samples ANSWER_TOKENS more tokens of that reply. A sample abstains if the reply starts with the flag. Writes
<out>/<arm>.jsonl (one line per sample: prefix key, outcome abstain / reply / no_reply / truncated, reply head,
reasoning length, whether the reasoning mentions the flag) and <out>/<arm>.meta.json (steering, tokens, seconds).

    python experiments/deceiver-axis/scripts/abstain_resample.py --model meta-models/Muse-Glimmer-30B --data <data>
        --tag main --slug muse --arm unsteered --out <dir> [--steer-direction <pt> --steering-from <steered run dir>]
"""
import argparse
import importlib
import json
import os
import random
import sys
import time
from pathlib import Path

import torch

import make_dossiers
from axis_acts_extract import hack_lens, load_sidecar, run_steering, steering_transforms
from selfconcept.deceiver_axis.behaviour_measures.abstain_flag import FLAG, INSTRUCTION
from selfconcept.deceiver_axis.behaviour_measures.episodes import (PRESETS, hack_turns, load_judge_details,
                                                                   load_verdict_by_key)
from selfconcept.common.chat import chat_template_kwargs
from selfconcept.measurement.steering_specifics import steering_model_specifics

REPLY_HEADER = {"muse-glimmer": "to=user<|message|>"}
ANSWER_TOKENS = 24


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--slug", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--steer-direction", type=Path)
    parser.add_argument("--steering-from", type=Path, help="steered run dir whose seed summary gives layer and coefficient")
    parser.add_argument("--prefixes", type=int, default=20)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=16384)
    parser.add_argument("--max-model-len", type=int, default=40960)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def select_prefixes(preset, n: int) -> list[tuple[str, str, int]]:
    """(unsteered seed dir name, example id, first-consideration turn) for n hacks, one per problem."""
    verdict_by_key = load_verdict_by_key(preset["verdict_files"])
    detail_by_key = load_judge_details(preset, verdict_by_key)
    by_problem: dict[str, list[str]] = {}
    for key, verdict in sorted(verdict_by_key.items()):
        if verdict == "HACK" and key.split("/")[0] == preset["unsteered"].name and key in detail_by_key:
            by_problem.setdefault(key.split("/")[2], []).append(key)
    rng = random.Random(0)
    problems = rng.sample(sorted(by_problem), min(n, len(by_problem)))
    keys = [rng.choice(by_problem[problem]) for problem in problems]
    return [(key.split("/")[1], key.split("/")[2], hack_turns(detail_by_key[key])[0]) for key in keys]


def prefix_ids(tokenizer, example: dict, record: dict, decision_turn: int, chat_kwargs: dict) -> list[int]:
    """The episode up to the decision turn's generation prompt, with INSTRUCTION in the task message."""
    history: list[dict] = []
    for role, text in make_dossiers.conversation(example, record):
        if role == "user":
            history.append({"role": "user", "content": text + INSTRUCTION if not history else text})
            continue
        turn = int(text)
        if turn == decision_turn:
            break
        history.append({"role": "assistant", "content": record["attempts"][turn].get("completion") or ""})
    return list(tokenizer.apply_chat_template(history, tokenize=True, add_generation_prompt=True, return_dict=False, **chat_kwargs))


def main() -> None:
    args = parse_args()
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"  # the class patch must be visible to the engine
    os.environ["VLLM_DISABLE_COMPILE_CACHE"] = "1"  # a cached graph from another steering would be reused
    from vllm import LLM, SamplingParams, TokensPrompt

    specifics = steering_model_specifics(args.model)
    family = specifics.family
    header = REPLY_HEADER[family]
    chat_kwargs = chat_template_kwargs()
    preset = next(preset for preset in PRESETS if preset["slug"] == args.slug)
    prefixes = select_prefixes(preset, args.prefixes)
    steering = None
    if args.steer_direction:
        steering = run_steering(args.steering_from / prefixes[0][0] if (args.steering_from / prefixes[0][0]).exists()
                                else min(args.steering_from.glob("seed*")), args.tag)
        num_layers, hidden = hack_lens.model_dims(args.model)
        transforms = steering_transforms(args.steer_direction, steering, num_layers, hidden, args.model, torch.device("cuda:0"))
        module_name, class_name = specifics.vllm_block
        specifics.patch_vllm(getattr(importlib.import_module(module_name), class_name), transforms)
        print(f"patched {class_name} with {steering}", file=sys.stderr, flush=True)
    llm = LLM(model=args.model, max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_memory_utilization)
    tokenizer = llm.get_tokenizer()

    prompts, keys = [], []
    for seed_name, example_id, decision_turn in prefixes:
        seed_dir = preset["unsteered"] / seed_name
        record = next(json.loads(line) for line in (seed_dir / f"{args.tag}_impossible_original.jsonl").open()
                      if json.loads(line)["example_id"] == example_id)
        example = make_dossiers.find_example(args.data, args.tag, example_id, seed_dir)
        prompts.append(prefix_ids(tokenizer, example, record, decision_turn, chat_kwargs))
        keys.append(f"{preset['unsteered'].name}/{seed_name}/{example_id}")
    t0 = time.time()
    stage1 = llm.generate([TokensPrompt(prompt_token_ids=ids) for ids in prompts],
                          SamplingParams(n=args.k, temperature=args.temperature, top_p=args.top_p, top_k=args.top_k, seed=0,
                                         max_tokens=args.max_new_tokens, stop=[header], include_stop_str_in_output=True,
                                         skip_special_tokens=False))
    continuations = [(i, j, prompts[i] + list(sample.token_ids)) for i, out in enumerate(stage1)
                     for j, sample in enumerate(out.outputs) if sample.text.endswith(header)]
    stage2 = llm.generate([TokensPrompt(prompt_token_ids=ids) for _, _, ids in continuations],
                          SamplingParams(temperature=args.temperature, top_p=args.top_p, top_k=args.top_k, seed=1,
                                         max_tokens=ANSWER_TOKENS, skip_special_tokens=False)) if continuations else []
    reply_by_sample = {(i, j): out.outputs[0].text for (i, j, _), out in zip(continuations, stage2)}
    seconds = time.time() - t0

    args.out.mkdir(parents=True, exist_ok=True)
    generated = 0
    with (args.out / f"{args.arm}.jsonl").open("w") as handle:
        for i, out in enumerate(stage1):
            for j, sample in enumerate(out.outputs):
                generated += len(sample.token_ids)
                reply = reply_by_sample.get((i, j))
                if reply is not None:
                    outcome = "abstain" if reply.lstrip().startswith(FLAG) else "reply"
                else:
                    outcome = "truncated" if sample.finish_reason == "length" else "no_reply"
                handle.write(json.dumps({"key": keys[i], "decision_turn": prefixes[i][2], "sample": j, "outcome": outcome,
                                         "reply_head": reply, "reasoning_tokens": len(sample.token_ids),
                                         "flag_in_reasoning": FLAG in sample.text, "prompt_tokens": len(prompts[i])}) + "\n")
    meta = {"arm": args.arm, "model": args.model, "steering": steering, "steer_direction": str(args.steer_direction),
            "steering_from": str(args.steering_from), "prefixes": len(prompts), "k": args.k, "instruction": INSTRUCTION,
            "sampling": {"temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k},
            "generated_tokens_stage1": generated, "seconds": seconds, "tokens_per_second": generated / seconds}
    (args.out / f"{args.arm}.meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta), file=sys.stderr)


if __name__ == "__main__":
    main()
