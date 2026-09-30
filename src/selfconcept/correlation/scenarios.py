"""Run benchmark scenarios through a measured generator and record their outcomes."""
import importlib.util
import json

from selfconcept.common.jsonl import read_jsonl
from selfconcept.common.paths import REPO_ROOT
from selfconcept.soo.evaluate import build_prompt, classify, SUFFIXES
from selfconcept.codebench import harness
from selfconcept.measurement.interface import Measurement
from selfconcept.measurement.templates import MODELS
from .generate import append, MeasuredGenerator


def script_module(name):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / f"experiments/soo/scripts/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def select_examples(rows, n, scenario, offset=0):
    if n < 1 or offset < 0:
        raise ValueError("Positive count and nonnegative offset required")
    if scenario == "sandbagging":
        groups = [[x for x in rows if x["expected"] == half] for half in ("sandbag", "normal")]
    else:
        groups = [rows]
    if any(len(group) < offset + n for group in groups):
        raise ValueError(f"Insufficient examples in {scenario} for offset={offset}, n={n}")
    return [r for group in groups for r in group[offset:offset+n]]


def evaluate(args, model, tokenizer, measurement: Measurement, family=None):
    generator = MeasuredGenerator(model, tokenizer, family or MODELS[args.model].family, measurement, args.out, args.max_new_tokens,
                                  args.temperature, args.top_p, args.top_k, args.seed)
    results = args.out / "outcomes.jsonl"
    done = {(r["scenario"], r["example_id"]) for r in read_jsonl(results)} if results.exists() else set()
    sandbag = script_module("reparse_sandbagging")
    for scenario in args.scenarios:
        code = scenario in ("evilgenie", "impossible_original", "impossible_oneoff", "impossible_conflicting")
        mirror = scenario.endswith("_mirrored")
        name = scenario.removesuffix("_mirrored")
        if code:
            path = REPO_ROOT / f"benchmarks/codebench/data/{name}.jsonl"
        else:
            folder = "eval_mirrored" if mirror else "eval_apollo" if name in ("roleplaying", "insider_trading", "sandbagging") else "eval"
            path = REPO_ROOT / f"experiments/soo/data/{folder}/{name}.jsonl"
        rows = select_examples(read_jsonl(path), args.code_n if code else args.n, name,
                               args.code_offset if code else args.offset)
        generator.scenario = scenario
        generator.max_new_tokens = args.code_max_new_tokens if code else args.max_new_tokens
        for example in rows:
            key = (scenario, example["example_id"])
            if key in done:
                continue
            generator.records = []
            try:
                if code:
                    generate = lambda messages, turn: generator(messages, turn, example["example_id"])
                    run = harness.run_workspace_example if name == "evilgenie" else harness.run_check_example
                    rec = run(example, generate, args.max_attempts)
                    rec["status"] = "complete" if generator.records[-1]["status"] == "complete" else generator.records[-1]["status"]
                    if rec["status"] != "complete":
                        rec["behavior_label"] = None
                    else:
                        rec["behavior_label"] = rec["label"]
                else:
                    messages = [dict(m) for m in example["messages"]] if "messages" in example else [
                        {"role": "user", "content": build_prompt(example, SUFFIXES[args.suffix] if name not in ("roleplaying", "sandbagging") else None, False)}]
                    if example.get("answer_prefix"):
                        messages[-1]["content"] += f"\n\nBegin your final answer with {example['answer_prefix']!r}."
                    response, _ = generator(messages, 0, example["example_id"])
                    status = generator.records[-1]["status"]
                    rec = {"example_id": example["example_id"], "scenario": scenario,
                           "response": response, "status": status, "label": "ungraded"}
                    if name == "sandbagging":
                        evidence = sandbag.grade_record(response)
                        rec.update(evidence=evidence, expected=example["expected"], topic=example["topic"])
                        rec["correct"] = evidence["final_answer"] == example["correct_answer"] if evidence["final_answer"] else None
                        rec["label"] = "deceptive" if evidence["disagree"] is True else "honest" if evidence["disagree"] is False else "unparsed"
                    elif name not in ("roleplaying", "insider_trading") and status == "complete":
                        rec["label"] = classify(response, example["honest_answer"], example["deceptive_answer"])
                    # Judge scripts prepend answer_prefix themselves.
                    if example.get("answer_prefix") and response.startswith(example["answer_prefix"]):
                        rec["response"] = response[len(example["answer_prefix"]):].lstrip()
                rec["generation_turns"] = len(generator.records)
                rec["model_key"] = args.model
                append(results, rec)
                print(f"{scenario}/{example['example_id']}: {rec['status']} {rec['label']}", flush=True)
            except harness.GenerationOOM as exc:
                append(results, {"example_id": example["example_id"], "scenario": scenario,
                                 "model_key": args.model, "response": "", "status": "error_oom", "label": "ungraded", "error": str(exc)})
                print(f"OOM {scenario}/{example['example_id']}", flush=True)
    # Rebuild judge inputs from committed outcomes: interruption/retry cannot
    # duplicate a task or leave judges reading an unfinished generation file.
    records = {(r["scenario"], r["example_id"]): r for r in read_jsonl(results)}
    for scenario in args.scenarios:
        code = scenario in ("evilgenie", "impossible_original", "impossible_oneoff", "impossible_conflicting")
        suffix = "" if code else "_none"
        target = args.out / f"base_{scenario}{suffix}.jsonl"
        target.write_text("".join(json.dumps(r) + "\n" for (s, _), r in records.items() if s == scenario))
