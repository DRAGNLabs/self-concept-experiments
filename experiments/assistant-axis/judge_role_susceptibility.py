# Derived from safety-research/assistant-axis (https://github.com/safety-research/assistant-axis),
# MIT licensed. See the NOTICE file at the repository root for the full license text.
"""Apply the role-susceptibility judge (§3.2.1) to generated trace files.

Reads trace JSONL (schema: role_susceptibility_traces.md), scores each (role, request,
response) with a local vLLM judge, and appends one labelled record per trace to the output
JSONL. Restartable: traces whose input already appears in the output are skipped.

Usage (from experiments/assistant-axis):
    python judge_role_susceptibility.py --traces_path <dir-or-file.jsonl> --output_path judgements.jsonl
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import jsonlines
from tqdm import tqdm

from selfconcept.assistant_axis.hf_generation import Answer, UnclosedReasoning, parse_response
from selfconcept.assistant_axis.role_susceptibility_judge import (
    RoleResponse,
    build_role_judge_messages,
    judge_role_responses,
)
from selfconcept.common.paths import scratch_dir
from selfconcept.transcript_generation.vllm_generation import VLLMGenerator

if TYPE_CHECKING:
    from vllm.config.model import ModelDType

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

_REQUIRED_FIELDS = ("role", "request", "response")
DEGENERATE_LABEL = "degenerate_cot"
DEGENERATE_NOTE = "[auto] response never closed </think>"


@dataclass(frozen=True)
class RunConfig:
    """Judge model, data locations, and sampling parameters for one judging run."""

    traces_path: Path = scratch_dir("assistant-axis") / "role_susceptibility" / "traces"
    output_path: Path = scratch_dir("assistant-axis") / "role_susceptibility" / "judgements.jsonl"
    judge_model: str = "Qwen/Qwen2.5-7B-Instruct"
    max_tokens: int = 256
    dtype: str = "auto"
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    max_model_len: int = 4096
    dry_run: bool = False


def trace_files(traces_path: Path) -> list[Path]:
    """Resolve traces_path to the JSONL files to judge (a single file, or *.jsonl in a directory)."""
    if traces_path.is_dir():
        return sorted(traces_path.glob("*.jsonl"))
    return [traces_path]


def input_key(record: dict) -> str:
    """A stable dedup key for a trace: the full input record minus the judge's own output fields."""
    payload = {k: v for k, v in record.items() if k not in ("label", "judge_raw")}
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def load_traces(files: list[Path]) -> list[dict]:
    """Load and validate trace records; records missing a required field are skipped with a warning."""
    traces = []
    for file in files:
        with jsonlines.open(file, "r") as reader:
            for record in reader:
                if all(isinstance(record.get(f), str) for f in _REQUIRED_FIELDS):
                    traces.append(record)
                else:
                    logger.warning("Skipping malformed trace in %s: %s", file.name, record)
    return traces


def already_judged_keys(output_path: Path) -> set[str]:
    """Input keys already present in the output file, so a rerun only judges new traces."""
    if not output_path.exists():
        return set()
    with jsonlines.open(output_path, "r") as reader:
        return {input_key(record) for record in reader}


def main(run: RunConfig = RunConfig()) -> None:
    """Judge every unscored trace under ``traces_path`` and append the labelled records."""
    files = trace_files(run.traces_path)
    logger.info("Reading traces from %d file(s) under %s", len(files), run.traces_path)

    traces = load_traces(files)
    judged_keys = already_judged_keys(run.output_path)
    pending = [t for t in traces if input_key(t) not in judged_keys]
    
    parsed = [(t, parse_response(t["response"])) for t in pending]
    degenerate = [t for t, result in parsed if isinstance(result, UnclosedReasoning)]
    judgeable = [t for t, result in parsed if isinstance(result, Answer)]
    logger.info("%d traces, %d already judged, %d pending (%d degenerate, %d to judge)",
                len(traces), len(traces) - len(pending), len(pending), len(degenerate), len(judgeable))

    if run.dry_run:
        if judgeable:
            sample = judgeable[0]
            messages = build_role_judge_messages(sample["role"], sample["request"], sample["response"])
            logger.info("\n%s\nSAMPLE JUDGE PROMPT (%s):\n%s\n%s\n%s",
                        "=" * 60, run.judge_model, "-" * 60, messages[1]["content"], "=" * 60)
        return

    if not pending:
        logger.info("Nothing to judge.")
        return

    judged: list[tuple[dict, str | None, str]] = [(t, DEGENERATE_LABEL, DEGENERATE_NOTE) for t in degenerate]
    if judgeable:
        responses = [RoleResponse(role=t["role"], request=t["request"], response=t["response"]) for t in judgeable]
        # Greedy, deterministic judging (temperature=0).
        judge = VLLMGenerator(
            model_name=run.judge_model,
            max_model_len=run.max_model_len,
            tensor_parallel_size=run.tensor_parallel_size,
            gpu_memory_utilization=run.gpu_memory_utilization,
            temperature=0.0,
            max_tokens=run.max_tokens,
            top_p=1.0,
            dtype=cast("ModelDType", run.dtype),
        )
        judgements = judge_role_responses(responses, judge)
        judged += [(t, j.label, j.judge_raw) for t, j in zip(judgeable, judgements)]

    run.output_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(run.output_path, "a") as writer:
        for trace, label, judge_raw in tqdm(judged, desc="Writing judgements"):
            writer.write({**trace, "label": label, "judge_raw": judge_raw})

    labelled = sum(1 for _, label, _ in judged if label is not None)
    logger.info("Wrote %d judgements (%d labelled incl. %d degenerate, %d unparseable) to %s",
                len(judged), labelled, len(degenerate), len(judged) - labelled, run.output_path)


if __name__ == "__main__":
    from jsonargparse import auto_cli

    auto_cli(main)
