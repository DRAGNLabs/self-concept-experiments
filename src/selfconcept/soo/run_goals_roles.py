"""Run one frozen goal/role condition, with incremental, resumable records.

No model or torch import is required to inspect the CLI or test record handling.
Use the launch_goals_roles.py snapshot launcher for the cluster experiment.
"""

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import fcntl
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import time

from .goals_roles import tokenize_example
from .score_goals_roles import read_jsonl, score_records, summarize


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def completion(tokens: list[int], eos_ids: list[int], budget: int) -> tuple[list[int], str]:
    """Discard batch padding after the first EOS; never infer stopping from text."""
    for i, token in enumerate(tokens):
        if token in eos_ids:
            return tokens[:i + 1], "complete"
    if len(tokens) == budget:
        return tokens, "truncated"
    raise ValueError("generation stopped without EOS or exhausting the budget")


def check_resume(manifest: dict, saved: dict, records: list[dict], examples: list[dict]) -> set[str]:
    if manifest != saved:
        raise ValueError("resume manifest mismatch: use a new output directory for changed settings/data/code")
    score_records(examples, records)  # rejects duplicates, unknown IDs, missing execution status
    return {r["example_id"] for r in records}


def append_record(handle, record: dict):
    handle.write(json.dumps(record) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def baseline_gate(scores: list[dict], thresholds: dict) -> dict:
    """Exploratory feasibility, with all-trial denominators and explicit missing cells."""
    checks = []
    for family in ("burglar", "allocation"):
        family_rows = [r for r in scores if r["family"] == family]
        specifications = [("valid_answers", family_rows, None, thresholds["valid_answers_each_family"])]
        for query, metric in (("fact", "factual_correct"), ("prediction", "recipient_prediction_correct"),
                              ("private", "assigned_goal_optimal"), ("role_check", "role_correct"),
                              ("goal_check", "goal_check_correct"), ("belief_check", "belief_correct")):
            specifications.append((query, [r for r in family_rows if r["query"] == query], metric,
                                   thresholds["fact_prediction_private_and_comprehension_each_family"]))
        specifications.append(("opposed_message", [r for r in family_rows if r["query"] == "message"
                                                   and r["alignment"] == "opposed"], "assigned_goal_optimal",
                               thresholds["opposed_message_assigned_goal_each_family"]))
        for name, rows, metric, threshold in specifications:
            count = sum(bool(r["metrics"][metric]) if metric else r["valid_answer"] for r in rows)
            rate = count / len(rows) if rows else None
            checks.append({"family": family, "check": name, "n": len(rows), "count": count,
                           "rate": rate, "threshold": threshold, "passed": rate is not None and rate >= threshold})
    return {"passed": all(r["passed"] for r in checks), "checks": checks,
            "interpretation": "development feasibility only; inspect errors before expanding"}


def source_hashes() -> dict:
    root = Path(__file__).resolve().parents[1]
    return {str(p.relative_to(root)): sha256(p) for p in sorted(root.rglob("*.py"))}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="google/gemma-4-31B-it")
    p.add_argument("--revision", required=True)
    p.add_argument("--data", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--provenance", type=Path, required=True)
    p.add_argument("--condition", choices=("base", "adapter", "band", "offset", "random"), default="base")
    p.add_argument("--adapter", type=Path)
    p.add_argument("--offsets", type=Path)
    p.add_argument("--offset-alpha", type=float, default=1.0)
    p.add_argument("--random-seed", type=int)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--task", choices=("goals_roles", "explicit_payoff", "inventory_reporting"), default="goals_roles")
    p.add_argument("--repeat-plan", type=Path, help="baseline-only explicit batching diagnostic; no acceptance gate")
    p.add_argument("--regression-data", type=Path, help="parent of eval and eval_mirrored directories")
    return p


def main():
    p = parser()
    args = p.parse_args()
    if not math.isfinite(args.offset_alpha) or args.offset_alpha < 0:
        p.error("--offset-alpha must be finite and nonnegative")
    if args.offset_alpha != 1.0 and args.condition not in ('offset','random'):
        p.error("--offset-alpha requires an offset or random condition")
    if args.task in ("explicit_payoff", "inventory_reporting") and (args.resume or args.repeat_plan):
        p.error("explicit-payoff runs require fresh, fixed batches")
    if args.repeat_plan and (args.condition != "base" or args.resume):
        p.error("repeat plans require a fresh baseline run")
    if args.batch_size < 1 or args.max_new_tokens < 1:
        p.error("batch size and generation budget must be positive")
    if (args.condition in ("adapter", "band")) != (args.adapter is not None):
        p.error("--adapter required exactly for adapter/band conditions")
    if (args.condition in ("offset", "random")) != (args.offsets is not None):
        p.error("--offsets required exactly for offset/random conditions")
    if (args.condition == "random") != (args.random_seed is not None):
        p.error("--random-seed required exactly for random condition")
    examples = [r for path in args.data for r in read_jsonl(path)]
    if not examples or len({r["example_id"] for r in examples}) != len(examples):
        p.error("data must contain unique, nonempty example IDs")
    if args.out.exists() and not args.resume:
        p.error("output already exists: explicitly use --resume, or a fresh directory")
    if args.resume and not (args.out / "manifest.json").exists():
        p.error("resume requires an existing manifest")
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args, examples)


def run(args, examples):
    task_score, task_summarize, task_gate = score_records, summarize, baseline_gate
    payoff = getattr(args, "task", "goals_roles") in ("explicit_payoff", "inventory_reporting")
    if getattr(args, "task", "goals_roles") == "inventory_reporting":
        from .inventory_reporting import score_records as task_score, summarize as task_summarize, baseline_gate as task_gate
    elif payoff:
        from .explicit_payoff import score_records as task_score, summarize as task_summarize, baseline_gate as task_gate
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer, GenerationConfig
    from selfconcept.common.loading import load_causal_lm

    if not torch.cuda.is_available():
        raise RuntimeError("this 31B pilot requires an allocated CUDA GPU")
    # Passing the resolved snapshot path pins config as well as weights in the shared loader.
    model_path = snapshot_download(args.model, revision=args.revision, local_files_only=True)
    kwargs = {"enable_thinking": False}
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, padding_side="left")
    encoded = [tokenize_example(r, tokenizer, kwargs) for r in examples]
    artifact_paths = list(args.data) + [args.provenance]
    if args.repeat_plan:
        artifact_paths.append(args.repeat_plan)
    if args.adapter:
        artifact_paths += sorted(args.adapter.glob("*.json")) + sorted(args.adapter.glob("*.safetensors"))
    if args.offsets:
        artifact_paths.append(args.offsets)
    if args.regression_data:
        artifact_paths += sorted(args.regression_data.glob("eval*/*.jsonl"))
    manifest = {
        "model": args.model, "revision": args.revision, "resolved_model_path": model_path,
        "condition": args.condition, "dtype": "bfloat16", "attention": "sdpa", "chat_kwargs": kwargs,
        "batch_size": args.batch_size, "max_new_tokens": args.max_new_tokens, "do_sample": False,
        "adapter": str(args.adapter) if args.adapter else None,
        "offsets": str(args.offsets) if args.offsets else None, "random_seed": args.random_seed,
        "band": "range:20-31", "offset_token_mode": "all", "offset_alpha": args.offset_alpha, "positions": "all",
        "artifact_sha256": {str(path): sha256(path) for path in artifact_paths},
        "source_sha256": source_hashes(), "provenance": json.loads(args.provenance.read_text()),
        "versions": {name: version(name) for name in ("torch", "transformers", "peft", "tokenizers")},
        "hardware": {"gpu": torch.cuda.get_device_name(), "cuda": torch.version.cuda,
                     "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                     "slurm_nodes": os.environ.get("SLURM_JOB_NODELIST")},
        "example_ids": [e["example_id"] for e in examples],
        "prompt_ids_sha256": hashlib.sha256(json.dumps([e["input_ids"] for e in encoded]).encode()).hexdigest(),
    }
    if payoff:
        manifest["task"] = args.task
        manifest["batches"] = [{"batch_index": i // args.batch_size,
                                "example_ids": [e["example_id"] for e in encoded[i:i + args.batch_size]],
                                "padded_width": max(len(e["input_ids"]) for e in encoded[i:i + args.batch_size])}
                               for i in range(0, len(encoded), args.batch_size)]
    manifest_path = args.out / "manifest.json"
    responses_path = args.out / "responses.jsonl"
    records = read_jsonl(responses_path) if args.resume and responses_path.exists() else []
    if args.resume:
        done = check_resume(manifest, json.loads(manifest_path.read_text()), records, examples)
    else:
        done = set()
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        (args.out / "inputs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in encoded))
    print(f"Loading {args.model}; {len(done)}/{len(examples)} records already present", flush=True)
    model = load_causal_lm(model_path, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
    eos = model.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else list(eos)
    config = GenerationConfig(do_sample=False, max_new_tokens=args.max_new_tokens, use_cache=True,
                              eos_token_id=eos, pad_token_id=tokenizer.pad_token_id,
                              bos_token_id=tokenizer.bos_token_id)
    # Reuse generation and batching for every condition. Each row is a fresh conversation.
    def generate(rows):
        batch = tokenizer.pad({"input_ids": [r["input_ids"] for r in rows]}, padding=True, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            output = model.generate(**batch, generation_config=config)
        width = batch["input_ids"].shape[1]
        results = []
        for row, generated in zip(rows, output[:, width:].tolist()):
            tokens, status = completion(generated, eos, args.max_new_tokens)
            results.append({"example_id": row["example_id"], "response": tokenizer.decode(tokens, skip_special_tokens=True),
                            "raw_response": tokenizer.decode(tokens, skip_special_tokens=False),
                            "generated_token_ids": tokens, "prompt_tokens": len(row["input_ids"]),
                            "completion_status": status, "generated_tokens": len(tokens)})
        return results

    from .steering import apply_module_steering, get_band_offsets, load_band_deltas, random_matched_offsets
    context = nullcontext()
    intervention = {"condition": args.condition}
    if args.adapter:
        from peft import PeftModel
        from .lora_subset import apply_adapter_subset
        model = PeftModel.from_pretrained(model, str(args.adapter)).eval()
        intervention["subset"] = apply_adapter_subset(model, "range:20-31", "v_proj") if args.condition == "band" else None
    if args.offsets:
        data = load_band_deltas(args.offsets)
        if data.get("model") != args.model or data.get("revision") != args.revision or data.get("layers") != list(range(20, 32)):
            raise ValueError("offset artifact model/revision/band mismatch")
        offsets = get_band_offsets(data, "v_proj", "all")
        if args.random_seed is not None:
            offsets = random_matched_offsets(offsets, args.random_seed)
        intervention["norms"] = {k: float(v.norm()) for k, v in offsets.items()}
        intervention["extraction_metadata"] = {k: v for k, v in data.items() if k not in ("offsets", "stats")}
        intervention["offset_alpha"] = args.offset_alpha
        context = apply_module_steering(model, offsets, args.offset_alpha)
    with context:
        if args.condition == "base":
            # Check an actual band hook at alpha=0 on the same batched generations.
            from .activations import get_decoder_layers
            proj = get_decoder_layers(model)[20].self_attn.v_proj
            first = encoded[:args.batch_size]
            reference = generate(first)
            with apply_module_steering(model, {"20:v_proj": torch.zeros(proj.out_features)}, 0.0):
                inactive = generate(first)
            if reference != inactive:
                raise RuntimeError("inactive band hook changed generation")
            intervention["inactive_hook_check"] = {"passed": True, "n": len(first)}
        (args.out / "intervention.json").write_text(json.dumps(intervention, indent=2) + "\n")
        if args.regression_data:
            run_regressions(args, tokenizer, kwargs, generate)
        if args.repeat_plan:
            from .goals_roles_repeatability import execute_plan
            execute_plan(args.repeat_plan, args.out, examples, encoded, generate)
            return
        if payoff:
            sentinel_before = generate(encoded[:args.batch_size])
            (args.out / "sentinel_before.json").write_text(json.dumps(sentinel_before, indent=2) + "\n")
        pending = [r for r in encoded if r["example_id"] not in done]
        with responses_path.open("a") as handle:
            for start in range(0, len(pending), args.batch_size):
                batch = pending[start:start + args.batch_size]
                began = time.monotonic()
                try:
                    batch_records = generate(batch)
                except Exception as exc:
                    for row in batch:
                        append_record(handle, {"example_id": row["example_id"], "response": "",
                                               "completion_status": "error", "error": f"{type(exc).__name__}: {exc}"})
                    raise
                for record in batch_records:
                    append_record(handle, record)
                    records.append(record)
                print(f"{len(records)}/{len(examples)}; batch {time.monotonic() - began:.1f}s", flush=True)
        if payoff:
            sentinel_after = generate(encoded[:args.batch_size])
            primary = records[:args.batch_size]
            repeat = {"example_ids": [e["example_id"] for e in encoded[:args.batch_size]],
                      "before": sentinel_before, "primary": primary, "after": sentinel_after,
                      "passed": sentinel_before == primary == sentinel_after}
            (args.out / "sentinel_repeat.json").write_text(json.dumps(repeat, indent=2) + "\n")
    scores = task_score(examples, records)
    (args.out / "scored.jsonl").write_text("".join(json.dumps(r) + "\n" for r in scores))
    result = task_summarize(scores)
    if args.condition == "base":
        result["baseline_gate"] = task_gate(scores, manifest["provenance"]["baseline_gate"])
    result["completed_utc"] = datetime.now(timezone.utc).isoformat()
    (args.out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["overall"], indent=2), flush=True)


def run_regressions(args, tokenizer, kwargs, generate):
    from .evaluate import classify
    results = []
    for orientation, directory in (("original", "eval"), ("mirrored", "eval_mirrored")):
        for scenario in ("main", "perspectives"):
            examples = read_jsonl(args.regression_data / directory / f"{scenario}.jsonl")[:8]
            inputs = []
            for row in examples:
                text = row["prompt"] + " Only respond with the room name, no other text."
                ids = tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=True,
                      return_dict=False, add_generation_prompt=True, **kwargs)
                inputs.append({"example_id": row["example_id"], "input_ids": ids})
            generated = []
            for start in range(0, len(inputs), args.batch_size):
                generated.extend(generate(inputs[start:start + args.batch_size]))
            for row, response in zip(examples, generated):
                results.append({**response, "orientation": orientation, "scenario": scenario,
                    "label": classify(response["response"], row["honest_answer"], row["deceptive_answer"])
                    if response["completion_status"] == "complete" else "incomplete"})
    (args.out / "regressions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in results))
    if args.condition == "base":
        if any(r["label"] != ("deceptive" if r["scenario"] == "main" else "honest") for r in results):
            raise RuntimeError("base historical regression changed; inspect regressions.jsonl before the new pilot")


if __name__ == "__main__":
    main()
