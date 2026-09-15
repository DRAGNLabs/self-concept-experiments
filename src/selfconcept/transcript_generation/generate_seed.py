"""Generate one persona-drift transcript for a single conversation seed.

The seed (a :class:`ConversationSeed`), the executor placement, and the run parameters are read
from a ``--config`` YAML file (jsonargparse maps its nested keys onto the typed ``main``
signature; run ``--help`` for the full option list). The seed is run through the local
auditor-vs-target loop in :mod:`selfconcept.transcript_generation.conversation`.

Usage:
    python -m selfconcept.transcript_generation.generate_seed \\
        --config configs/delusion_seed.yaml [--run.turns 4] [--run.output results/foo.json]
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

from selfconcept.transcript_generation.conversation import generate_conversations
from selfconcept.transcript_generation.types import ConversationSeed
from selfconcept.transcript_generation.vllm_generation import (
    ExecutorOptions,
    InProcessExecutor,
    build_vllm_engines,
)

if TYPE_CHECKING:
    from vllm.config.model import ModelDType

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunConfig:
    """Model, sampling, and output parameters for one generation run."""

    target_model: str = "meta-llama/Llama-3.3-70B-Instruct"
    auditor_model: str = "Qwen/Qwen3-32B"
    turns: int = 30
    output: Path = Path("results/transcript.json")
    target_temperature: float = 0.7
    target_max_tokens: int = 512
    auditor_temperature: float = 0.8
    auditor_max_tokens: int = 4096
    max_model_len: int | None = None
    dtype: str = "bfloat16"
    gpu_memory_utilization: float | None = None


def main(
    seed: ConversationSeed,
    run: RunConfig = RunConfig(),
    executor_options: ExecutorOptions = InProcessExecutor(tensor_parallel_size=1),
) -> None:
    """Generate and write one transcript for ``seed``."""
    gmu = run.gpu_memory_utilization
    if gmu is None:
        gmu = 0.45 if isinstance(executor_options, InProcessExecutor) else 0.9

    target, auditor, cleanup = build_vllm_engines(
        target_model=run.target_model,
        auditor_model=run.auditor_model,
        gpu_memory_utilization=gmu,
        max_model_len=run.max_model_len,
        target_temperature=run.target_temperature,
        target_max_tokens=run.target_max_tokens,
        auditor_temperature=run.auditor_temperature,
        auditor_max_tokens=run.auditor_max_tokens,
        executor_options=executor_options,
        dtype=cast("ModelDType", run.dtype),
    )

    try:
        [transcript] = generate_conversations(target, auditor, [seed], turns=run.turns)
    finally:
        cleanup()

    run.output.parent.mkdir(parents=True, exist_ok=True)
    run.output.write_text(json.dumps(transcript, indent=2, ensure_ascii=False))
    logger.info("Wrote transcript (%d turns) to %s", transcript["turns"], run.output)


if __name__ == "__main__":
    from jsonargparse import auto_cli

    auto_cli(main)
