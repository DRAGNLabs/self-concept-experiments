"""Write the shared self-recognition prompt set with its seeded train/holdout split.

Usage (from experiments/self-recognition):
    python scripts/0_make_prompts.py
"""

import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict, cast

import jsonlines

from selfconcept.common.jsonl import read_jsonl
from selfconcept.self_recognition.prompts import prompt_split
from selfconcept.self_recognition.schemas import PromptRecord

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class SourceQuestion(TypedDict):
    question: str
    id: int


@dataclass(frozen=True)
class RunConfig:
    source_file: Path = Path("../assistant-axis/data/extraction_questions.jsonl")
    output_file: Path = Path("data/prompts.jsonl")
    holdout_fraction: float = 0.2
    seed: int = 0


def prompt_record(source_question: SourceQuestion, holdout_fraction: float, seed: int) -> PromptRecord:
    return {
        "id": source_question["id"],
        "question": source_question["question"],
        "split": prompt_split(source_question["id"], holdout_fraction, seed),
    }


def main(run: RunConfig = RunConfig()) -> None:
    source_questions = cast(list[SourceQuestion], read_jsonl(run.source_file))
    prompt_records = [prompt_record(question, run.holdout_fraction, run.seed) for question in source_questions]
    run.output_file.parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(run.output_file, "w") as writer:
        writer.write_all(prompt_records)
    prompt_count_by_split = Counter(record["split"] for record in prompt_records)
    logger.info("%s: %d train, %d holdout", run.output_file, prompt_count_by_split["train"], prompt_count_by_split["holdout"])


if __name__ == "__main__":
    from jsonargparse import auto_cli

    auto_cli(main)
