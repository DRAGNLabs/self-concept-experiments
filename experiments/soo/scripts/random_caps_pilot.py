"""Small paired, generative capability screen for the Gemma insider random controls.

Prepare on a networked node; run the frozen JSONL offline on one assigned GPU.
All arms use the same examples and native chat template. This is a screen for
large capability losses, not a non-inferiority test or a swarm experiment.
"""

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import random
import re
import time


MODEL = "google/gemma-4-31B-it"
REVISION = "842da3794eaa0b77d5f08bae87a17459d91ff475"
ARMS = {"base": None, "random_s0": 0, "random_s1": 1, "random_s2": 2}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def prepare(root):
    from datasets import load_dataset

    root.mkdir(parents=True, exist_ok=True)
    if (root / "examples.jsonl").exists():
        raise FileExistsError("Refusing to overwrite the frozen examples")
    rows = []
    datasets_meta = []
    for task, path, config, count in [
        ("arc_challenge", "allenai/ai2_arc", "ARC-Challenge", 100),
        ("gsm8k", "openai/gsm8k", "main", 30),
    ]:
        ds = load_dataset(path, config, split="test")
        indices = sorted(random.Random(20261013).sample(range(len(ds)), count))
        datasets_meta.append(dict(task=task, path=path, config=config, split="test",
                                  fingerprint=ds._fingerprint, indices=indices))
        for idx in indices:
            item = ds[idx]
            if task == "arc_challenge":
                choices = item["choices"]
                labels = list("ABCDE"[:len(choices["text"])])
                answer = labels[choices["label"].index(item["answerKey"])]
                prompt = item["question"] + "\n\n" + "\n".join(
                    f"{label}. {text}" for label, text in zip(labels, choices["text"])
                ) + "\n\nReply with only the letter of the correct answer."
                reference = answer
            else:
                answer = item["answer"].split("####")[-1].strip().replace(",", "")
                prompt = item["question"] + (
                    "\n\nSolve the problem. Explain your reasoning briefly, then end with "
                    "a line in the form '#### <number>'."
                )
                reference = item["answer"]
            rows.append(dict(id=f"{task}:{idx}", task=task, prompt=prompt,
                             answer=answer, reference=reference))
    (root / "examples.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    write_json(root / "data_manifest.json", dict(selection_seed=20261013,
        datasets=datasets_meta, examples_sha256=digest(root / "examples.jsonl")))
    print(f"Prepared {len(rows)} fixed examples in {root}", flush=True)


def parse_answer(task, text):
    if task == "arc_challenge":
        match = re.fullmatch(r"\s*([A-E])[.)]?\s*", text)
        return match.group(1) if match else None
    matches = re.findall(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)", text)
    return matches[-1].replace(",", "") if matches else None


def is_correct(task, predicted, target):
    if predicted is None:
        return False
    if task == "arc_challenge":
        return predicted == target
    try:
        return Decimal(predicted) == Decimal(target)
    except InvalidOperation:
        return False


def run(root, batch_size):
    import torch
    import transformers
    from transformers import AutoTokenizer
    from selfconcept.common.loading import load_causal_lm
    from selfconcept.soo.steering import apply_steering, get_vector, load_vectors, random_matched_vector

    torch.set_num_threads(4)
    torch.manual_seed(0)
    rows = [json.loads(line) for line in (root / "examples.jsonl").read_text().splitlines()]
    assert digest(root / "examples.jsonl") == json.loads((root / "data_manifest.json").read_text())["examples_sha256"]
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = load_causal_lm(MODEL, revision=REVISION, dtype=torch.bfloat16,
                           device_map="cuda", attn_implementation="sdpa").eval()
    vector = get_vector(load_vectors(root / "vectors.pt"), 30, "last")
    eos = model.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    output_dir = root / "output"
    output_dir.mkdir(exist_ok=True)
    # Match the original insider run's default template kwargs, not an altered
    # thinking mode. Save the rendered prompt so this choice is inspectable.
    render = lambda row: tokenizer.apply_chat_template(
        [{"role": "user", "content": row["prompt"]}], tokenize=False, add_generation_prompt=True)
    write_json(output_dir / "runtime.json", dict(model=MODEL, revision=REVISION,
        torch=torch.__version__, transformers=transformers.__version__,
        gpu=torch.cuda.get_device_name(), dtype="bfloat16", layer=30, alpha=16,
        positions="all", sign="h - alpha * vector", token_mode="last", batch_size=batch_size,
        do_sample=False, chat_kwargs={}, example_rendered_prompt=render(rows[0]),
        vector_file_sha256=digest(root / "vectors.pt"), reference_vector_norm=vector.norm().item(),
        started_utc=datetime.now(timezone.utc).isoformat()))
    probe = tokenizer(render(rows[0]), add_special_tokens=False, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        baseline_logits = model(**probe).logits[:, -1].float().cpu()
    for arm, seed in ARMS.items():
        path = output_dir / f"{arm}.jsonl"
        done = {r["id"] for r in map(json.loads, path.read_text().splitlines())} if path.exists() else set()
        if len(done) == len(rows):
            print(f"{arm}: already complete", flush=True)
            continue
        v = random_matched_vector(vector, seed) if seed is not None else vector
        context = apply_steering(model, 30, v, 16) if seed is not None else nullcontext()
        with context, torch.inference_mode(), path.open("a") as stream:
            probe_logits = model(**probe).logits[:, -1].float().cpu()
            logit_change = (probe_logits - baseline_logits).abs().max().item()
            if seed is not None:
                assert logit_change > 0, "Steering did not affect the probe"
            else:
                assert logit_change == 0, "Baseline is not repeatable"
            write_json(output_dir / f"{arm}_intervention.json", dict(seed=seed,
                norm=v.norm().item(), vector_sha256=hashlib.sha256(v.numpy().tobytes()).hexdigest(),
                probe_max_logit_change=logit_change))
            for task, max_tokens in [("arc_challenge", 32), ("gsm8k", 512)]:
                selected = [r for r in rows if r["task"] == task and r["id"] not in done]
                for start in range(0, len(selected), batch_size):
                    batch = selected[start:start + batch_size]
                    encoded = tokenizer([render(r) for r in batch], padding=True,
                                        add_special_tokens=False, return_tensors="pt").to("cuda")
                    begin = time.monotonic()
                    generated = model.generate(**encoded, do_sample=False, max_new_tokens=max_tokens,
                                               pad_token_id=tokenizer.pad_token_id)
                    elapsed = time.monotonic() - begin
                    suffixes = generated[:, encoded["input_ids"].shape[1]:].tolist()
                    for row, tokens in zip(batch, suffixes):
                        end = next((i for i, token in enumerate(tokens) if token in eos_ids), None)
                        ended = end is not None
                        tokens = tokens[:end + 1] if ended else tokens
                        text = tokenizer.decode(tokens, skip_special_tokens=True).strip()
                        predicted = parse_answer(task, text)
                        correct = is_correct(task, predicted, row["answer"])
                        record = dict(id=row["id"], task=task, arm=arm, response=text,
                            raw_response=tokenizer.decode(tokens, skip_special_tokens=False),
                            predicted=predicted, target=row["answer"], correct=correct,
                            complete_correct=correct and ended, parsed=predicted is not None,
                            truncated=not ended, generated_tokens=len(tokens), batch_seconds=elapsed)
                        stream.write(json.dumps(record) + "\n")
                    stream.flush()
                    print(f"{arm} {task}: {min(start + batch_size, len(selected))}/{len(selected)} "
                          f"batch={elapsed:.1f}s", flush=True)
        with torch.inference_mode():
            restored = model(**probe).logits[:, -1].float().cpu()
        assert torch.equal(restored, baseline_logits), "Steering hook was not cleanly removed"
    analyze(root)


def analyze(root):
    import numpy as np
    from scipy.stats import binomtest

    output_dir = root / "output"
    records = {arm: [json.loads(line) for line in (output_dir / f"{arm}.jsonl").read_text().splitlines()]
               for arm in ARMS}
    summary = {}
    for task in ["arc_challenge", "gsm8k"]:
        baseline = {r["id"]: r for r in records["base"] if r["task"] == task}
        summary[task] = {}
        for arm in ARMS:
            rows = [r for r in records[arm] if r["task"] == task]
            assert len(rows) == len(baseline) and {r["id"] for r in rows} == set(baseline)
            diffs = np.array([int(r["complete_correct"]) - int(baseline[r["id"]]["complete_correct"]) for r in rows])
            rng = np.random.default_rng(20261013)
            ci = np.quantile(rng.choice(diffs, size=(10000, len(rows))).mean(axis=1), [0.025, 0.975])
            gains, losses = int((diffs == 1).sum()), int((diffs == -1).sum())
            summary[task][arm] = dict(n=len(rows), correct=sum(r["complete_correct"] for r in rows),
                parsed=sum(r["parsed"] for r in rows), truncated=sum(r["truncated"] for r in rows),
                accuracy=sum(r["complete_correct"] for r in rows) / len(rows),
                accuracy_including_truncated=sum(r["correct"] for r in rows) / len(rows),
                delta_vs_base=diffs.mean(), paired_bootstrap_95ci=ci.tolist(), gains=gains, losses=losses,
                exact_mcnemar_p=binomtest(gains, gains + losses).pvalue if gains + losses else 1.0,
                mean_generated_tokens=float(np.mean([r["generated_tokens"] for r in rows])))
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "run", "analyze"])
    parser.add_argument("root", type=Path)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    if args.mode == "run":
        run(args.root, args.batch_size)
    elif args.mode == "prepare":
        prepare(args.root)
    else:
        analyze(args.root)
