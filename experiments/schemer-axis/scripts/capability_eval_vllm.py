"""General-capability benchmarks under the Phase 3 residual-stream interventions (PLAN.md, "General-capability check").

Same steering CLI and engine construction as steered_rh_vllm.py (class patch before the engine builds, so the
intervention is part of the compiled graph), but instead of the reward-hacking harness it runs batched single-turn
benchmark prompts and scores them:

    gsm8k      openai/gsm8k test (1319), free-form numeric answer, exact match on the number after "Answer:"
    mmlu_pro   TIGER-Lab/MMLU-Pro test, a fixed stratified subsample (--mmlu-per-category per category, seed 0),
               letter answer A-J
    humaneval  openai/openai_humaneval (164), the function in a fenced block, executed against the test suite in
               the codebench sandbox (pass@1 at the run's sampling)

Prompts go through the model's chat template with SOO_CHAT_KWARGS (gpt-oss: reasoning_effort), the completion is
split into reasoning and final by the family's parser, and only the final channel is scored. Sampling as in the
experiment (temperature, --sample-seed; the per-item seed is turn_sampling_seed(seed, "<task>/<id>", 0)).

Output: <out>/<task>.jsonl (one record per item: id, prompt, reasoning, final, truncated, pred, gold, correct,
reasoning_chars) and <out>/summary.json. Rerunning resumes: items already in the jsonl are skipped.

    python scripts/capability_eval_vllm.py --model openai/gpt-oss-120b --out <dir> --tasks gsm8k mmlu_pro humaneval
        [--direction <axis.pt> --layer-norms <pt> --steer-mode add --steer-layer 17 --steer-alpha -0.2]
        [--steer-mode cap --cap-layers 12 ... --cap-thresholds <json>]
"""
import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from steered_rh_vllm import FAMILIES, build_transforms, directions_for, family_of  # noqa: E402

from selfconcept.codebench import sandbox  # noqa: E402
from selfconcept.common.chat import chat_template_kwargs  # noqa: E402
from selfconcept.common.sampling import turn_sampling_seed  # noqa: E402

TASKS = ("gsm8k", "mmlu_pro", "humaneval")
LETTERS = "ABCDEFGHIJ"
FENCE = re.compile(r"```[a-zA-Z0-9_+-]*\n(.*?)```", re.DOTALL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", default=list(TASKS), choices=TASKS)
    parser.add_argument("--n", type=int, help="first n items of every task (smoke)")
    parser.add_argument("--mmlu-per-category", type=int, default=100)
    parser.add_argument("--max-model-len", type=int, default=20480)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=256, help="prompts per llm.generate call (records are written per batch)")
    parser.add_argument("--exec-timeout", type=float, default=20.0, help="seconds per HumanEval test run")
    # steering, same names as steered_rh_vllm.py so arms.tsv lines apply unchanged
    parser.add_argument("--direction", type=Path)
    parser.add_argument("--layer-norms", type=Path)
    parser.add_argument("--steer-mode", choices=["none", "add", "cap"], default="none")
    parser.add_argument("--steer-layer", type=int, default=17)
    parser.add_argument("--steer-alpha", type=float, default=0.0)
    parser.add_argument("--cap-layers", nargs="+", type=int)
    parser.add_argument("--cap-thresholds", type=Path)
    parser.add_argument("--random-direction-seed", type=int)
    args = parser.parse_args()
    if args.steer_mode == "add" and (args.layer_norms is None or args.steer_alpha == 0):
        parser.error("--steer-mode add needs --layer-norms and a nonzero --steer-alpha")
    if args.steer_mode == "cap" and (not args.cap_layers or args.cap_thresholds is None):
        parser.error("--steer-mode cap needs --cap-layers and --cap-thresholds")
    if args.steer_mode != "none" and args.direction is None and args.random_direction_seed is None:
        parser.error("--direction is required for steering")
    return args


# ----------------------------------------------------------------------------------------------------------------- items

def load_items(task: str, args) -> list[dict]:
    """[{id, prompt, gold, ...}] in a fixed order."""
    from datasets import load_dataset

    if task == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="test")
        items = [{"id": f"gsm8k/{i}", "gold": row["answer"].split("####")[-1].strip().replace(",", ""),
                  "prompt": (row["question"].strip() + "\n\nSolve the problem step by step. "
                             "On the last line write the final numeric answer as 'Answer: <number>'.")}
                 for i, row in enumerate(ds)]
    elif task == "mmlu_pro":
        ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
        by_category: dict[str, list[int]] = {}
        for i, category in enumerate(ds["category"]):
            by_category.setdefault(category, []).append(i)
        rng = random.Random(0)
        chosen = sorted(j for category in sorted(by_category)
                        for j in rng.sample(by_category[category], min(args.mmlu_per_category, len(by_category[category]))))
        items = []
        for j in chosen:
            row = ds[j]
            options = "\n".join(f"{LETTERS[k]}. {option}" for k, option in enumerate(row["options"]))
            items.append({"id": f"mmlu_pro/{row['question_id']}", "gold": row["answer"], "category": row["category"],
                          "prompt": (row["question"].strip() + "\n\n" + options + "\n\nThink it through, then on the last "
                                     "line write the letter of the correct option as 'Answer: <letter>'.")})
    elif task == "humaneval":
        ds = load_dataset("openai/openai_humaneval", split="test")
        items = [{"id": row["task_id"], "gold": None, "test": row["test"], "entry_point": row["entry_point"],
                  "starter": row["prompt"],
                  "prompt": ("Complete the following Python function. Reply with the complete function, including "
                             "its signature and any imports it needs, in a single fenced ```python code block.\n\n"
                             "```python\n" + row["prompt"].rstrip() + "\n```")}
                 for row in ds]
    else:
        raise ValueError(task)
    return items[: args.n] if args.n else items


# --------------------------------------------------------------------------------------------------------------- scoring

def last_answer_line(final: str) -> str | None:
    matches = re.findall(r"[Aa]nswer\s*[:：]\s*\**\s*([^\n]*)", final)
    return matches[-1].strip() if matches else None


def normalize_number(text: str) -> str | None:
    text = text.replace(",", "").replace("$", "").strip()
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None
    value = float(match.group())
    return str(int(value)) if value == int(value) else str(value)


def score_gsm8k(item: dict, final: str) -> tuple[str | None, bool]:
    tail = last_answer_line(final)
    pred = normalize_number(tail) if tail else None
    if pred is None:  # fall back to the last number in the final message
        numbers = re.findall(r"-?\d[\d,]*(?:\.\d+)?", final)
        pred = normalize_number(numbers[-1]) if numbers else None
    return pred, pred is not None and pred == normalize_number(item["gold"])


def score_mmlu_pro(item: dict, final: str) -> tuple[str | None, bool]:
    tail = last_answer_line(final)
    pred = None
    if tail:
        match = re.match(r"\(?([A-J])\)?(?![A-Za-z])", tail.strip())
        pred = match.group(1) if match else None
    if pred is None:
        letters = re.findall(r"(?<![A-Za-z])\(?([A-J])\)?(?![A-Za-z])", final)
        pred = letters[-1] if letters else None
    return pred, pred == item["gold"]


def score_humaneval(item: dict, final: str, timeout: float) -> tuple[str | None, bool]:
    blocks = FENCE.findall(final)
    code = None
    for block in reversed(blocks):
        if re.search(rf"def\s+{re.escape(item['entry_point'])}\s*\(", block):
            code = block
            break
    if code is None and blocks:  # body-only reply: put it under the starter
        code = item["starter"] + blocks[-1]
    if code is None:
        return None, False
    program = code + "\n\n" + item["test"] + f"\n\ncheck({item['entry_point']})\n"
    result = sandbox.run_python({"solution.py": program}, ["solution.py"], timeout=timeout)
    return code, bool(result.success)


def score(task: str, item: dict, final: str, args) -> tuple[str | None, bool]:
    if task == "gsm8k":
        return score_gsm8k(item, final)
    if task == "mmlu_pro":
        return score_mmlu_pro(item, final)
    return score_humaneval(item, final, args.exec_timeout)


# ------------------------------------------------------------------------------------------------------------ generation

def generate_batch(llm, tokenizer, items: list[dict], task: str, args, split_completion) -> list[dict]:
    from vllm import SamplingParams, TokensPrompt

    prompts, params = [], []
    for item in items:
        ids = tokenizer.apply_chat_template([{"role": "user", "content": item["prompt"]}], add_generation_prompt=True,
                                            **chat_template_kwargs(), tokenize=True, return_dict=False)
        prompts.append(TokensPrompt(prompt_token_ids=list(ids)))
        params.append(SamplingParams(
            temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
            seed=turn_sampling_seed(args.sample_seed, item["id"], 0) if args.temperature > 0 else None,
            max_tokens=min(args.max_new_tokens, args.max_model_len - len(ids)), skip_special_tokens=False))
    outputs = llm.generate(prompts, params, use_tqdm=True)
    records = []
    for item, output in zip(items, outputs):
        completion = output.outputs[0]
        reasoning, final = split_completion(completion.text)
        pred, correct = score(task, item, final, args)
        records.append({"id": item["id"], "category": item.get("category"), "prompt": item["prompt"],
                        "reasoning": reasoning, "final": final, "truncated": completion.finish_reason == "length",
                        "completion_tokens": len(completion.token_ids), "reasoning_chars": len(reasoning),
                        "pred": pred, "gold": item["gold"], "correct": bool(correct)})
    return records


def run_task(task: str, llm, tokenizer, args, split_completion) -> dict:
    path = args.out / f"{task}.jsonl"
    done: dict[str, dict] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                done[record["id"]] = record
    items = load_items(task, args)
    todo = [item for item in items if item["id"] not in done]
    print(f"{task}: {len(items)} items, {len(done)} done, {len(todo)} to run", file=sys.stderr)
    with path.open("a") as out:
        for start in range(0, len(todo), args.batch_size):
            batch = todo[start:start + args.batch_size]
            for record in generate_batch(llm, tokenizer, batch, task, args, split_completion):
                out.write(json.dumps(record) + "\n")
                done[record["id"]] = record
            out.flush()
    records = [done[item["id"]] for item in items]
    n = len(records)
    summary = {"n": n, "accuracy": sum(r["correct"] for r in records) / n, "truncated": sum(r["truncated"] for r in records) / n,
               "no_answer": sum(r["pred"] is None for r in records) / n,
               "mean_completion_tokens": sum(r["completion_tokens"] for r in records) / n}
    if task == "mmlu_pro":
        by_category: dict[str, list[bool]] = {}
        for r in records:
            by_category.setdefault(r["category"], []).append(r["correct"])
        summary["by_category"] = {c: sum(v) / len(v) for c, v in sorted(by_category.items())}
    print(f"{task}: {json.dumps({k: v for k, v in summary.items() if k != 'by_category'})}", file=sys.stderr)
    return summary


def main() -> None:
    args = parse_args()
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["VLLM_DISABLE_COMPILE_CACHE"] = "1"
    import importlib
    from vllm import LLM

    family = family_of(args.model)
    recipe = FAMILIES[family]
    module_name, class_name = recipe["block"]
    block_cls = getattr(importlib.import_module(module_name), class_name)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    needed = sorted({*([args.steer_layer] if args.steer_mode == "add" else []), *(args.cap_layers or [])})
    directions = directions_for(args, needed, device) if needed else {}
    transforms, steering = build_transforms(args, directions)
    steering["family"] = family
    if transforms:
        recipe["patch"](block_cls, transforms)
        steering["patched_layers"] = sorted(transforms)
        print(f"patched {class_name}.forward for layers {sorted(transforms)}: {steering}", file=sys.stderr)
    llm = LLM(model=args.model, tensor_parallel_size=1, max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_memory_utilization)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    args.out.mkdir(parents=True, exist_ok=True)
    meta = {"model": args.model, "backend": f"vllm-{family}-patched", "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k, "sample_seed": args.sample_seed,
            "chat_kwargs": chat_template_kwargs(), "steering": steering, "mmlu_per_category": args.mmlu_per_category, "n": args.n}
    summary_path = args.out / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    summary["meta"] = meta
    for task in args.tasks:
        summary[task] = run_task(task, llm, tokenizer, args, recipe["split"])
        summary_path.write_text(json.dumps(summary, indent=1))
    print("=== capability_eval_vllm done ===", file=sys.stderr)


if __name__ == "__main__":
    main()
