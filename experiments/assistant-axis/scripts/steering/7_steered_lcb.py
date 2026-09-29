"""Generate LiveCodeBench solutions under assistant-axis steering and grade them.

Does the assistant axis, extracted from general conversational prompts, do anything on
coding tasks? For a fixed sample of the LiveCodeBench problems bundled with EvilGenie
(benchmarks/codebench/data/evilgenie.jsonl), ask the model to solve each problem in plain
LiveCodeBench format -- one user message, one answer, no workspace, no test files, no
feedback or retries -- while adding ``coefficient * layer_norm`` times the unit axis
direction to one decoder layer's output at every position (the Section 3.2.1 setup of
6_steered_traces.py). The final code block is then graded silently against every test the
problem has (EvilGenie's visible plus holdout cases) with EvilGenie's test runner; the model
never sees a test result.

One run = one coefficient. Every coefficient uses the same problems, the same batches and
the same per-batch sampling seeds, so conditions differ only in the steering. Writes one
record per problem to ``{output_dir}/coef_{coefficient:+.2f}.jsonl`` (full completion with
its thinking, extracted code, grade) and a summary JSON next to it. Re-running skips problems
already on disk.

Usage (from experiments/assistant-axis):
    python scripts/steering/7_steered_lcb.py --config configs/steering/7_steered_lcb.yaml \
        --run.coefficient -0.25 --run.axis_path AXIS --run.norms_path NORMS --run.output_dir OUT
"""

from __future__ import annotations

import json
import logging
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from jaxtyping import Float
from torch import Tensor

from selfconcept.assistant_axis.internals.model import ProbingModel
from selfconcept.assistant_axis.models import get_config
from selfconcept.assistant_axis.steering import apply_steering
from selfconcept.codebench import harness
from selfconcept.codebench.sandbox import run_python

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DTYPE_MAP: dict[str, torch.dtype] = {"bfloat16": torch.bfloat16, "float16": torch.float16}

# LiveCodeBench's generic code-generation prompt (lcb_runner/prompts/code_generation.py).
LCB_SYSTEM = (
    "You are an expert Python programmer. You will be given a question (problem specification) and will "
    "generate a correct Python program that matches the specification and passes all tests."
)
LCB_WITH_STARTER = (
    "You will use the following starter code to write the solution to the problem and enclose your code within delimiters."
)
LCB_STDIN = (
    "Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample "
    "inputs). Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads the "
    "inputs, runs the algorithm and writes output to STDOUT."
)
# LiveCodeBench's grader prepends this to every solution (lcb_runner/evaluation/testing_util.py import_string),
# so starter-code annotations like List[int] resolve; EvilGenie's runner does not, so it is added here.
LCB_IMPORTS = (
    "from string import *\nfrom re import *\nfrom datetime import *\nfrom collections import *\nfrom heapq import *\n"
    "from bisect import *\nfrom copy import *\nfrom math import *\nfrom random import *\nfrom statistics import *\n"
    "from itertools import *\nfrom functools import *\nfrom operator import *\nfrom io import *\nfrom sys import *\n"
    "from json import *\nfrom builtins import *\nfrom typing import *\nimport string\nimport re\nimport datetime\n"
    "import collections\nimport heapq\nimport bisect\nimport copy\nimport math\nimport random\nimport statistics\n"
    "import itertools\nimport functools\nimport operator\nimport io\nimport sys\nimport json\nsys.setrecursionlimit(50000)\n"
)
THINK_CLOSE = "</think>"
FENCE = re.compile(r"```(?:[A-Za-z0-9_+-]*)\n(.*?)```", re.DOTALL)
RESULTS_LINE = re.compile(r"Results: (\d+)/(\d+) passed")


@dataclass(frozen=True)
class RunConfig:
    """Model, axis, problem sample, sampling, and steering strength for one run."""

    model: str = "allenai/Olmo-3.1-32B-Think"
    axis_path: Path = Path("axis_response_only.pt")
    norms_path: Path = Path("layer_norms.pt")
    output_dir: Path = Path("steered_lcb")
    target_layer: int | None = None
    coefficient: float = 0.0
    n_problems: int = 40
    sample_seed: int = 0
    max_new_tokens: int = 16384
    batch_size: int = 12
    do_sample: bool = True
    temperature: float = 0.6
    top_p: float = 0.95
    seed: int = 0
    dtype: str = "bfloat16"


def load_axis(path: Path) -> Float[Tensor, "layers hidden"]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    return (data["axis"] if isinstance(data, dict) else data).float()


def load_layer_norms(path: Path) -> Float[Tensor, "layers"]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    return (data["layer_norms"] if isinstance(data, dict) else data).float()


def select_problems(examples: list[dict], n: int, seed: int) -> list[dict]:
    """A fixed random sample, in sample order (the same for every coefficient)."""
    order = random.Random(seed).sample(range(len(examples)), min(n, len(examples)))
    return [examples[i] for i in order]


def lcb_prompt(example: dict) -> str:
    """LiveCodeBench's generic prompt, with its system sentence leading the user turn (the chat
    template keeps the model's own default system prompt)."""
    text = f"{LCB_SYSTEM}\n\n### Question:\n{example['description']}\n\n"
    if example["is_function"]:
        text += f"### Format: {LCB_WITH_STARTER}\n```python\n{example['starter_code']}\n```\n\n"
    else:
        text += f"### Format: {LCB_STDIN}\n```python\n# YOUR CODE HERE\n```\n\n"
    return text + "### Answer: (use the provided format with backticks)\n\n"


def split_thinking(completion: str) -> tuple[str, str, bool]:
    """(thinking, answer, closed). The generation prompt already opens <think>."""
    if THINK_CLOSE in completion:
        thinking, answer = completion.split(THINK_CLOSE, 1)
        return thinking, answer, True
    return completion, "", False


def extract_code(answer: str) -> str | None:
    """Last fenced block of the post-thinking answer (LiveCodeBench takes the last block)."""
    blocks = FENCE.findall(answer)
    return blocks[-1] if blocks else None


def grade(example: dict, code: str | None) -> dict:
    """Run EvilGenie's own test runner over every test case (visible + holdout), LCB import header first."""
    if code is None:
        return {"passed": False, "n_passed": 0, "n_total": None, "test_output_tail": "no code block"}
    problem_type = "function" if example["is_function"] else "stdio"
    tests = example["visible_tests"] + example["holdout_tests"]
    files = {
        "solution.py": LCB_IMPORTS + code,
        "test.py": harness.workspace_files(example)["test.py"],
        "test_cases.json": harness.test_cases_json(tests, problem_type),
    }
    result = run_python(files, ["test.py"], timeout=harness.EVILGENIE_TEST_TIMEOUT)
    m = RESULTS_LINE.search(result.stdout)
    return {
        "passed": result.success,
        "n_passed": int(m.group(1)) if m else 0,
        "n_total": len(tests),
        "timed_out": result.timed_out,
        "test_output_tail": (result.stdout + ("\n" + result.stderr if result.stderr else ""))[-1500:],
    }


def stop_ids(probing_model: ProbingModel) -> list[int]:
    tok = probing_model.tokenizer
    ids = {tok.eos_token_id, probing_model.model.generation_config.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")}
    flat = set()
    for i in ids:
        flat.update(i if isinstance(i, list) else [i])
    return sorted(i for i in flat if isinstance(i, int) and i >= 0)


def generate_batch(probing_model: ProbingModel, prompts: list[str], run: RunConfig, seed: int, eos: list[int]) -> list[dict]:
    tok = probing_model.tokenizer
    texts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True) for p in prompts]
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(probing_model.device)
    torch.manual_seed(seed)
    sampling = {"do_sample": True, "temperature": run.temperature, "top_p": run.top_p} if run.do_sample else {"do_sample": False}
    with torch.no_grad():
        out = probing_model.model.generate(
            **enc, max_new_tokens=run.max_new_tokens, eos_token_id=eos, pad_token_id=tok.pad_token_id, **sampling
        )
    width = enc["input_ids"].shape[1]
    results = []
    for row in out[:, width:].tolist():
        cut = next((i for i, t in enumerate(row) if t in eos), None)
        new = row if cut is None else row[:cut]
        results.append({"completion": tok.decode(new, skip_special_tokens=False), "n_new_tokens": len(new), "finished": cut is not None})
    return results


def generate_with_backoff(probing_model, prompts, run, seed, eos) -> list[dict]:
    """Generate one batch; on CUDA OOM split it in half (each half keeps a derived seed)."""
    try:
        return generate_batch(probing_model, prompts, run, seed, eos)
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        if len(prompts) == 1:
            raise
        mid = len(prompts) // 2
        logger.warning("OOM at batch %d; splitting", len(prompts))
        return (generate_with_backoff(probing_model, prompts[:mid], run, seed * 2 + 1, eos)
                + generate_with_backoff(probing_model, prompts[mid:], run, seed * 2 + 2, eos))


def summarize(records: list[dict]) -> dict:
    n = len(records)
    think = [r["n_new_tokens"] for r in records]
    return {
        "n": n,
        "pass_rate": sum(r["passed"] for r in records) / n if n else None,
        "has_code_rate": sum(r["code"] is not None for r in records) / n if n else None,
        "think_closed_rate": sum(r["think_closed"] for r in records) / n if n else None,
        "truncated_rate": sum(r["truncated"] for r in records) / n if n else None,
        "mean_new_tokens": sum(think) / n if n else None,
        "mean_test_fraction": sum(r["n_passed"] / r["n_total"] for r in records if r["n_total"]) / n if n else None,
    }


def main(run: RunConfig = RunConfig()) -> None:
    run.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = run.output_dir / f"coef_{run.coefficient:+.2f}.jsonl"
    done = {}
    if out_path.exists():
        done = {r["example_id"]: r for r in map(json.loads, out_path.read_text().splitlines()) if r}
    problems = select_problems(harness.load_examples("evilgenie"), run.n_problems, run.sample_seed)
    (run.output_dir / "problems.json").write_text(json.dumps([p["example_id"] for p in problems], indent=1) + "\n")
    todo = [p for p in problems if p["example_id"] not in done]
    logger.info("%d problems, %d already done, coefficient %+.2f", len(problems), len(done), run.coefficient)

    probing_model = ProbingModel(run.model, dtype=DTYPE_MAP[run.dtype])
    layers = probing_model.get_layers()
    target_layer = run.target_layer if run.target_layer is not None else get_config(run.model)["target_layer"]
    axis = load_axis(run.axis_path.expanduser())
    layer_norms = load_layer_norms(run.norms_path.expanduser())
    assert axis.shape[0] == len(layers) == layer_norms.shape[0], (axis.shape, len(layers), layer_norms.shape)
    direction = axis[target_layer]
    unit = (direction / (direction.norm() + 1e-8)).to(probing_model.device)
    layer_norm = float(layer_norms[target_layer])
    scale = run.coefficient * layer_norm
    eos = stop_ids(probing_model)
    logger.info("layer %d, layer_norm %.3f, steering norm %.3f, stop ids %s", target_layer, layer_norm, scale, eos)

    meta = {"run": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(run).items()},
            "target_layer": target_layer, "layer_norm": layer_norm, "steering_norm": scale,
            "axis_layer_norm_raw": float(direction.norm()), "stop_ids": eos}
    # Batches are fixed over the full sample so every coefficient sees the same batch composition and seeds.
    batches = [problems[i:i + run.batch_size] for i in range(0, len(problems), run.batch_size)]
    for b, batch in enumerate(batches):
        pending = [p for p in batch if p["example_id"] not in done]
        if not pending:
            continue
        prompts = [lcb_prompt(p) for p in batch]
        logger.info("batch %d/%d (%d problems)", b + 1, len(batches), len(batch))
        if scale == 0.0:
            outs = generate_with_backoff(probing_model, prompts, run, run.seed + b, eos)
        else:
            with apply_steering(probing_model, target_layer, unit, coefficient=scale):
                outs = generate_with_backoff(probing_model, prompts, run, run.seed + b, eos)
        with out_path.open("a") as f:
            for example, prompt, gen in zip(batch, prompts, outs):
                if example["example_id"] in done:
                    continue
                thinking, answer, closed = split_thinking(gen["completion"])
                code = extract_code(answer)
                record = {
                    "example_id": example["example_id"], "title": example["title"], "is_function": example["is_function"],
                    "coefficient": run.coefficient, "target_layer": target_layer, "steering_norm": scale,
                    "prompt": prompt, "completion": gen["completion"], "thinking": thinking, "answer": answer,
                    "think_closed": closed, "n_new_tokens": gen["n_new_tokens"], "truncated": not gen["finished"],
                    "code": code, **grade(example, code),
                }
                done[example["example_id"]] = record
                f.write(json.dumps(record) + "\n")
        logger.info("batch %d: %s", b + 1, json.dumps(summarize([done[p["example_id"]] for p in batch])))

    records = [done[p["example_id"]] for p in problems]
    summary = {**meta, **summarize(records)}
    (run.output_dir / f"coef_{run.coefficient:+.2f}_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    logger.info("summary: %s", json.dumps(summarize(records)))


if __name__ == "__main__":
    from jsonargparse import auto_cli

    auto_cli(main)
