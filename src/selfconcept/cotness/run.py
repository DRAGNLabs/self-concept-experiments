"""Train model-specific role probes and run the existing deception harnesses."""
import argparse
import hashlib
import importlib.util
import json
import os
import shutil
from pathlib import Path
import sys

import numpy as np
import sklearn
import torch
import transformers
from transformers import AutoTokenizer

from selfconcept.common.loading import load_causal_lm
from selfconcept.common.paths import REPO_ROOT
from selfconcept.soo.activations import get_decoder_layers
from selfconcept.soo.evaluate import build_prompt, classify, SUFFIXES
from selfconcept.codebench import harness
from . import probe
from .roles import MODELS, clean_final, content_indices, render_generation, response_spans, template_kwargs


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def append(path, record):
    with path.open("a") as f:
        f.write(json.dumps(record, allow_nan=False) + "\n")
        f.flush()


def script_module(name):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / f"experiments/soo/scripts/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def score_summary(values, indices):
    selected = values[[i for i in indices if i < len(values)]]
    return {"n": len(selected), "mean": float(selected.mean()) if len(selected) else None,
            "p90": float(np.quantile(selected, .9)) if len(selected) else None}


def token_offsets(tokenizer, ids, text):
    # Single-token decoding is exact for ordinary text, but not split UTF-8.
    pieces = [tokenizer.decode([i], skip_special_tokens=False, clean_up_tokenization_spaces=False) for i in ids]
    if "".join(pieces) == text:
        ends = np.cumsum([len(p) for p in pieces]).tolist()
        return list(zip([0] + ends[:-1], ends))
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    if enc["input_ids"] == ids:
        return enc["offset_mapping"]
    return None  # Never assign an activation to the wrong token.


def generation_seed(seed, scenario, example_id, turn):
    return int.from_bytes(hashlib.sha256(f"{seed}:{scenario}:{example_id}:{turn}".encode()).digest()[:4], "big")


class MeasuredGenerator:
    def __init__(self, model, tokenizer, family, probes, out, max_new_tokens,
                 temperature=0.0, top_p=0.95, top_k=64, seed=1729):
        self.model, self.tokenizer, self.family = model, tokenizer, family
        self.probes, self.out, self.max_new_tokens = probes, out, max_new_tokens
        self.records = []
        self.scenario = ""
        self.sampling = ({"do_sample": True, "temperature": temperature, "top_p": top_p, "top_k": top_k}
                         if temperature > 0 else {"do_sample": False})
        self.seed = seed

    def __call__(self, messages, turn, example_id):
        tok = self.tokenizer
        prompt = render_generation(tok, self.family, messages)
        enc = tok(prompt, add_special_tokens=False, return_tensors="pt").to(self.model.get_input_embeddings().weight.device)
        context_limit = getattr(getattr(self.model.config, "text_config", self.model.config), "max_position_embeddings", None)
        if context_limit and enc.input_ids.shape[1] + self.max_new_tokens > context_limit:
            raise ValueError("Prompt plus generation budget exceeds model context; refusing silent truncation")
        try:
            seed = generation_seed(self.seed, self.scenario, example_id, turn)
            torch.manual_seed(seed)
            with torch.inference_mode(), probe.capture(self.model, list(self.probes), self.probes) as captured:
                output = self.model.generate(**enc, max_new_tokens=self.max_new_tokens, **self.sampling,
                                             pad_token_id=tok.eos_token_id, use_cache=True)
        except torch.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            raise harness.GenerationOOM(str(exc)[:200]) from None
        n_prompt = enc.input_ids.shape[1]
        ids = output[0].tolist()
        new_ids = ids[n_prompt:]
        raw = tok.decode(new_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        spans = response_spans(prompt, raw, self.family)
        final = clean_final(raw, spans, tok)
        eos = self.model.generation_config.eos_token_id
        eos = eos if isinstance(eos, list) else [eos]
        truncated = len(new_ids) >= self.max_new_tokens and new_ids[-1] not in eos
        status = "truncated" if truncated else "complete" if final else "no_final"
        prompt_enc = tok(prompt, add_special_tokens=False, return_offsets_mapping=True)
        # Probe only content from the last externally supplied message, excluding
        # assistant history, system instructions and generation-role delimiters.
        # Native templates may trim user content (notably coding feedback's
        # leading newline). Match the shared non-whitespace content exactly.
        last = messages[-1]["content"].strip()
        start = prompt.rfind(last)
        if start < 0 or prompt_enc.input_ids != ids[:n_prompt]:
            raise ValueError("Cannot exactly locate the last message in the rendered prompt")
        p_indices = content_indices(prompt_enc.offset_mapping, [(start, start+len(last))], tok.all_special_ids, ids[:n_prompt])
        offsets = token_offsets(tok, new_ids, raw)
        groups = {"prompt": p_indices}
        if offsets is not None:
            groups.update({role: [n_prompt+i for i in content_indices(offsets, segments, tok.all_special_ids, new_ids)]
                           for role, segments in spans.items()})
        scores, arrays = {}, {"input_ids": np.asarray(ids, dtype=np.int32), "prompt_length": np.array(n_prompt)}
        for layer, chunks in captured.items():
            values = torch.cat(chunks).numpy()
            if len(values) != len(ids)-1:
                raise ValueError(f"Unexpected cached generation alignment: {len(values)} vs {len(ids)-1}")
            scores[str(layer)] = {name: score_summary(values, idx) for name, idx in groups.items()}
            arrays[f"cotness_layer_{layer}"] = values
        for name, indices in groups.items():
            arrays[f"indices_{name}"] = np.asarray([i for i in indices if i < len(ids)-1], dtype=np.int32)
        trace_key = hashlib.sha256(f"{self.scenario}:{example_id}:{turn}".encode()).hexdigest()[:24]
        trace = self.out / "traces" / f"{trace_key}.npz"
        trace.parent.mkdir(exist_ok=True)
        np.savez_compressed(trace, **arrays)
        rec = {"example_id": example_id, "scenario": self.scenario, "turn": turn,
               "response": final, "raw_response": raw, "rendered_prompt": prompt,
               "status": status, "truncated": truncated, "generated_tokens": len(new_ids),
               "sampling": self.sampling, "seed": seed,
               "prompt_tokens": n_prompt, "spans": spans, "scores": scores,
               "generated_alignment": "exact" if offsets is not None else "unavailable",
               "trace": str(trace.relative_to(self.out))}
        self.records.append(rec)
        append(self.out / "generations.jsonl", rec)
        # A partial final answer/code is never a completed behavioral submission.
        return final if status == "complete" else "", truncated


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


def evaluate(args, model, tokenizer, fitted):
    generator = MeasuredGenerator(model, tokenizer, MODELS[args.model].family, fitted, args.out, args.max_new_tokens,
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, default=REPO_ROOT / "experiments/cotness/data/neutral.jsonl")
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--code-n", type=int, default=4)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--code-offset", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--probe-source", type=Path, help="Frozen validated probe directory with source_manifest.json")
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--code-max-new-tokens", type=int, default=16384)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--suffix", choices=SUFFIXES, default="i_would")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--scenarios", nargs="+", default=["main", "main_mirrored", "treasure_hunt", "treasure_hunt_mirrored",
                        "perspectives", "perspectives_mirrored", "roleplaying", "insider_trading", "sandbagging",
                        "impossible_original", "impossible_conflicting", "impossible_oneoff", "evilgenie"])
    args = parser.parse_args()
    if min(args.n, args.code_n, args.max_new_tokens, args.code_max_new_tokens, args.max_attempts) < 1:
        parser.error("Counts and budgets must be positive")
    if min(args.offset, args.code_offset, args.temperature, args.top_k) < 0 or not 0 < args.top_p <= 1:
        parser.error("Invalid offset or sampling configuration")
    os.environ["HF_HUB_OFFLINE"] = "1"
    torch.manual_seed(1729)
    spec = MODELS[args.model]
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {"args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "model": spec.__dict__, "chat_kwargs": template_kwargs(spec.family),
                "corpus_sha256": hashlib.sha256(args.corpus.read_bytes()).hexdigest(),
                "torch": torch.__version__, "transformers": transformers.__version__,
                "sklearn": sklearn.__version__, "numpy": np.__version__,
                "python": sys.version, "stage": "started"}
    manifest_path = args.out / "manifest.json"
    if args.probe_source:
        source = json.loads((args.probe_source / "source_manifest.json").read_text())
        if source["model"] != manifest["model"] or source["corpus_sha256"] != manifest["corpus_sha256"]:
            raise ValueError("Frozen probe model/corpus differs from evaluation")
        if not json.loads((args.probe_source / "validation.json").read_text())["usable"]:
            raise ValueError("Cannot reuse a probe that failed validation")
        manifest["probe_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in args.probe_source.iterdir() if p.is_file()}
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if any(previous.get(k) != manifest.get(k) for k in ("args", "model", "chat_kwargs", "corpus_sha256", "probe_sha256")):
            raise ValueError("Refusing to mix different configurations in an output directory")
    manifest_path.write_text(json.dumps(manifest, indent=2))
    tok = AutoTokenizer.from_pretrained(spec.model, revision=spec.revision, local_files_only=True)
    model = load_causal_lm(spec.model, revision=spec.revision, local_files_only=True, dtype=torch.bfloat16,
                           device_map="auto")
    model.eval()
    count = len(get_decoder_layers(model))
    layers = [int(count * fraction)-1 for fraction in (.25, .5, .75)]
    directory = args.out / "probes"
    if args.probe_source:
        shutil.copytree(args.probe_source, directory, dirs_exist_ok=True)
    if (directory / "validation.json").exists():
        fitted, report = probe.load(directory)
    else:
        fitted, report = probe.train(model, tok, spec.family, args.corpus, directory, layers)
    manifest["probe_usable"] = report["usable"]
    manifest["stage"] = "probe_ready" if report["usable"] else "probe_failed_validation"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    if not report["usable"]:
        print("Midpoint probe failed validation; no deception correlations will be interpreted.", flush=True)
        return
    if not args.train_only:
        evaluate(args, model, tok, fitted)
    manifest["stage"] = "complete"
    manifest_path.write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
