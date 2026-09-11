# Derived from safety-research/assistant-axis (https://github.com/safety-research/assistant-axis),
# MIT licensed. See the NOTICE file at the repository root for the full license text.
"""Generate model responses for all roles using vLLM batch inference.

Loads each role's system-prompt variants and generates responses across the shared
question set, writing one JSONL file per role. Re-running skips roles that already have
an output file. A single vLLM engine spans ``tensor_parallel_size`` GPUs; roles are
processed sequentially.

Usage (from experiments/assistant-axis):
    python -m selfconcept.assistant_axis  # (library only)
    python pipeline/1_generate.py --config configs/1_generate.yaml
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

from selfconcept.assistant_axis.generation import RoleResponseGenerator
from selfconcept.common.paths import scratch_dir

if TYPE_CHECKING:
    from vllm.config.model import ModelDType

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunConfig:
    """Model, data, sampling, and output parameters for one generation run."""

    model: str = "google/gemma-2-27b-it"
    roles_dir: Path = Path("data/roles/instructions")
    questions_file: Path = Path("data/extraction_questions.jsonl")
    output_dir: Path = scratch_dir("assistant-axis") / "responses"
    max_model_len: int = 2048
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    question_count: int = 240
    temperature: float = 0.7
    max_tokens: int = 512
    top_p: float = 0.9
    dtype: str = "auto"
    thinking: bool = False
    roles: list[str] | None = None


def main(run: RunConfig = RunConfig()) -> None:
    """Generate and save role responses for the configured model."""
    generator = RoleResponseGenerator(
        model_name=run.model,
        roles_dir=run.roles_dir,
        output_dir=run.output_dir,
        questions_file=run.questions_file,
        max_model_len=run.max_model_len,
        tensor_parallel_size=run.tensor_parallel_size,
        gpu_memory_utilization=run.gpu_memory_utilization,
        question_count=run.question_count,
        temperature=run.temperature,
        max_tokens=run.max_tokens,
        top_p=run.top_p,
        dtype=cast("ModelDType", run.dtype),
        enable_thinking=run.thinking,
    )
    generator.process_all_roles(skip_existing=True, roles=run.roles)
    logger.info("Done!")


if __name__ == "__main__":
    from jsonargparse import auto_cli

    auto_cli(main)
