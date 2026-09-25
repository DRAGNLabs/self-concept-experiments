"""Generate role-susceptibility steering traces (Section 3.2.1).

For each system prompt (a role description) and each introspective behavioral question
(Appendix D.1.2), generate the target model's answer unsteered and under additive steering
along the assistant axis at one middle layer. The steering vector is the unit axis direction
at that layer, scaled to ``coefficient * layer_norm`` -- the per-layer mean post-MLP residual
norm from ``activation_norms.measure_layer_norms`` -- so a coefficient is a fraction of the
layer's natural activation magnitude.

Writes one JSONL file per (role, prompt variant) in the ``role_susceptibility_traces.md``
schema; ``steering_coefficient`` 0 is the unsteered baseline. Re-running skips roles that
already have an output file.

Usage (from experiments/assistant-axis):
    python scripts/steering/6_steered_traces.py --config configs/steering/6_steered_traces.yaml
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple, TypedDict

import jsonlines
import torch
from jaxtyping import Float
from torch import Tensor
from tqdm import tqdm

from selfconcept.assistant_axis.hf_generation import generate_response, serialize_response
from selfconcept.assistant_axis.internals.model import ProbingModel
from selfconcept.assistant_axis.models import get_config
from selfconcept.assistant_axis.steering import apply_steering
from selfconcept.common.hf_utils import build_conversation
from selfconcept.common.paths import scratch_dir

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DTYPE_MAP: dict[str, torch.dtype] = {
    "auto": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "half": torch.float16,
}


@dataclass(frozen=True)
class RunConfig:
    """Model, axis, data, sampling, and steering-strength parameters for one trace run."""

    model: str = "google/gemma-2-27b-it"
    axis_path: Path = scratch_dir("assistant-axis") / "axis.pt"
    norms_path: Path = scratch_dir("assistant-axis") / "layer_norms.pt"
    questions_file: Path = Path("data/behavioral_questions.jsonl")
    system_prompts_file: Path = Path("data/steer_prompts.jsonl")
    output_dir: Path = scratch_dir("assistant-axis") / "steered"
    target_layer: int | None = None
    coefficients: list[float] = field(default_factory=lambda: [-8.0, -4.0, 4.0, 8.0])
    include_baseline: bool = True
    max_new_tokens: int = 512
    enable_thinking: bool = False
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
    dtype: str = "bfloat16"


class SystemPrompt(NamedTuple):
    role_id: str
    prompt_index: int
    role: str


class Question(NamedTuple):
    question_index: int
    text: str


class TraceRecord(TypedDict):
    role: str
    request: str
    response: str
    steering_coefficient: float
    target_model: str
    role_id: str
    question_index: int
    prompt_index: int


def load_axis(path: Path) -> Float[Tensor, "layers hidden"]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    axis = data["axis"] if isinstance(data, dict) else data
    return axis.float()


def load_layer_norms(path: Path) -> Float[Tensor, "layers"]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    norms = data["layer_norms"] if isinstance(data, dict) else data
    return norms.float()


def load_questions(path: Path) -> list[Question]:
    with jsonlines.open(path) as reader:
        return [Question(entry["id"], entry["question"]) for entry in reader]


def load_system_prompts(path: Path) -> list[SystemPrompt]:
    """One SystemPrompt per line, numbering repeated role_ids as successive prompt variants."""
    prompt_count_by_role_id: dict[str, int] = {}
    system_prompts: list[SystemPrompt] = []
    with jsonlines.open(path) as reader:
        for entry in reader:
            role_id = entry["role_id"]
            prompt_index = prompt_count_by_role_id.get(role_id, 0)
            prompt_count_by_role_id[role_id] = prompt_index + 1
            system_prompts.append(SystemPrompt(role_id, prompt_index, entry["role"]))
    return system_prompts


def resolved_coefficients(coefficients: list[float], include_baseline: bool) -> list[float]:
    if include_baseline and 0.0 not in coefficients:
        return [0.0, *coefficients]
    return coefficients


def generate_role_traces(
    probing_model: ProbingModel,
    system_prompt: SystemPrompt,
    questions: list[Question],
    unit_direction: Float[Tensor, "hidden"],
    target_layer: int,
    layer_norm: float,
    coefficients: list[float],
    run: RunConfig,
) -> list[TraceRecord]:
    assert probing_model.tokenizer is not None
    records: list[TraceRecord] = []
    for question in questions:
        conversation = build_conversation(question.text, system_prompt.role, probing_model.tokenizer)
        for coefficient in coefficients:
            if coefficient == 0.0:
                result = generate_response(
                    probing_model,
                    conversation,
                    max_new_tokens=run.max_new_tokens,
                    temperature=run.temperature,
                    top_p=run.top_p,
                    do_sample=run.do_sample,
                    enable_thinking=run.enable_thinking,
                )
            else:
                with apply_steering(
                    probing_model, target_layer, unit_direction, coefficient=coefficient * layer_norm
                ):
                    result = generate_response(
                        probing_model,
                        conversation,
                        max_new_tokens=run.max_new_tokens,
                        temperature=run.temperature,
                        top_p=run.top_p,
                        do_sample=run.do_sample,
                        enable_thinking=run.enable_thinking,
                    )
            records.append(
                TraceRecord(
                    role=system_prompt.role,
                    request=question.text,
                    response=serialize_response(result),
                    steering_coefficient=coefficient,
                    target_model=run.model,
                    role_id=system_prompt.role_id,
                    question_index=question.question_index,
                    prompt_index=system_prompt.prompt_index,
                )
            )
    return records


def main(run: RunConfig = RunConfig()) -> None:
    """Generate steered + baseline traces for every (role, prompt variant) and behavioral question."""
    run.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading model: %s", run.model)
    probing_model = ProbingModel(run.model, dtype=DTYPE_MAP[run.dtype])
    num_layers = len(probing_model.get_layers())
    target_layer = run.target_layer if run.target_layer is not None else get_config(run.model)["target_layer"]

    axis = load_axis(run.axis_path.expanduser()).to(probing_model.device)
    layer_norms = load_layer_norms(run.norms_path.expanduser())
    assert axis.shape[0] == num_layers, f"axis has {axis.shape[0]} rows but model has {num_layers} layers"
    assert layer_norms.shape[0] == num_layers, (
        f"layer_norms has {layer_norms.shape[0]} entries but model has {num_layers} layers"
    )

    direction = axis[target_layer]
    unit_direction = direction / (direction.norm() + 1e-8)
    layer_norm = float(layer_norms[target_layer])
    logger.info("Steering at layer %d, layer_norm=%.3f", target_layer, layer_norm)

    questions = load_questions(run.questions_file)
    system_prompts = load_system_prompts(run.system_prompts_file)
    coefficients = resolved_coefficients(run.coefficients, run.include_baseline)
    logger.info(
        "%d prompts x %d questions x %d coefficients", len(system_prompts), len(questions), len(coefficients)
    )

    for system_prompt in tqdm(system_prompts, desc="Generating traces"):
        output_file = run.output_dir / f"{system_prompt.role_id}_p{system_prompt.prompt_index}.jsonl"
        if output_file.exists():
            logger.info("Skipping %s (already exists)", output_file.name)
            continue
        records = generate_role_traces(
            probing_model, system_prompt, questions, unit_direction, target_layer, layer_norm, coefficients, run
        )
        with jsonlines.open(output_file, "w") as writer:
            writer.write_all(records)
        logger.info("Wrote %d traces to %s", len(records), output_file.name)

    logger.info("Done!")


if __name__ == "__main__":
    from jsonargparse import auto_cli

    auto_cli(main)
