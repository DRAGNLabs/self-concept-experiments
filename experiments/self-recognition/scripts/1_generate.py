"""Generate each model's answers to the shared prompt set (both splits), K samples per prompt.

Resumes from the model's transcripts file: only missing (prompt_id, sample_index) pairs are generated.

Usage (from experiments/self-recognition):
    python scripts/1_generate.py --config configs/1_generate.yaml --config configs/1_generate_qwen3-32b.yaml
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

from selfconcept.common.resumable_jsonl import process_unrecorded_items
from selfconcept.self_recognition.generation import (
    GenerationItem,
    generation_items,
    transcript_chunk_processor,
    transcript_quality_counts,
)
from selfconcept.self_recognition.schemas import (
    DEFAULT_RUN_DIR,
    TranscriptRecord,
    read_prompts,
    read_transcripts,
    transcripts_path,
)
from selfconcept.transcript_generation.vllm_generation import VLLMGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunConfig:
    model: str
    temperature: float
    top_p: float
    top_k: int
    prompts_file: Path = Path("data/prompts.jsonl")
    run_dir: Path = DEFAULT_RUN_DIR
    samples_per_prompt: int = 4
    prompt_limit: int | None = None
    max_tokens: int = 4096
    max_model_len: int = 6144
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    chat_template_kwargs: dict[str, str | bool] = field(default_factory=dict)
    enable_thinking: bool = False
    chunk_size: int = 1000


def generation_item_key(item: GenerationItem) -> tuple[int, int]:
    return item.prompt["id"], item.sample_index


def transcript_record_key(record: TranscriptRecord) -> tuple[int, int]:
    return record["prompt_id"], record["sample_index"]


def main(run: RunConfig) -> None:
    logger.info("%s", run)
    generator = VLLMGenerator(
        model_name=run.model,
        max_model_len=run.max_model_len,
        tensor_parallel_size=run.tensor_parallel_size,
        gpu_memory_utilization=run.gpu_memory_utilization,
        temperature=run.temperature,
        top_p=run.top_p,
        top_k=run.top_k,
        max_tokens=run.max_tokens,
        chat_template_kwargs=run.chat_template_kwargs,
        enable_thinking=run.enable_thinking,
    )
    output_path = transcripts_path(run.run_dir, run.model)
    process_unrecorded_items(
        generation_items(read_prompts(run.prompts_file)[: run.prompt_limit], run.samples_per_prompt),
        output_path,
        item_key=generation_item_key,
        record_key=transcript_record_key,
        process_chunk=transcript_chunk_processor(generator, run.model),
        chunk_size=run.chunk_size,
    )
    logger.info("%s: %s", output_path, transcript_quality_counts(read_transcripts(output_path)))


if __name__ == "__main__":
    from jsonargparse import auto_cli

    auto_cli(main)
