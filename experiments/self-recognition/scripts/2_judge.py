"""Ask the target model which of two responses (its own and another model's, both orders) it wrote.

Scores each presentation from the next-token probabilities of the labels "1" and "2". Resumes from the
target's judgements file. --run.compliance_check gate (default) first runs the instruction-compliance check
and exits before judging if it fails; true runs only the check.

Usage (from experiments/self-recognition):
    python scripts/2_judge.py --config configs/2_judge.yaml --run.target_model Qwen/Qwen3-32B
"""

import json
import logging
import math
import re
import statistics
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import batched, zip_longest
from pathlib import Path
from typing import Literal, TypedDict, cast

from selfconcept.common.generation.model_specifics import get_generation_specifics
from selfconcept.common.hf_strong_types import HFTokenizer, configure_call
from selfconcept.common.resumable_jsonl import process_unrecorded_items
from selfconcept.self_recognition.judging import (
    JudgeTask,
    answer_label_token_ids,
    answer_probabilities,
    chosen_position,
    judge_task_to_conversation,
    create_judge_tasks,
    judge_prompt_text,
    judgement_chunk_processor,
    read_transcripts_by_model,
)
from selfconcept.self_recognition.schemas import (
    DEFAULT_RUN_DIR,
    AnswerProbabilities,
    CandidatePosition,
    JudgementRecord,
    Split,
    judgements_path,
    model_slug,
    prompts_path,
    read_judgements,
    read_prompts,
)
from selfconcept.transcript_generation.vllm_generation import VLLMGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

COMPLIANCE_PAIR_COUNT = 8
# Pairs alternate presentation orders, so these are pairs 0-3 with the target first, second, first, second.
FREE_GENERATION_PRESENTATION_INDICES = (0, 3, 4, 7)
FREE_GENERATION_MAX_TOKENS = 64
FREE_GENERATION_SEED = 0
FREE_GENERATION_TEMPERATURE_BY_MODEL_SUBSTRING = {"gpt-oss": 1.0}  # gpt-oss loops when greedy
LOGGED_TOP_TOKEN_COUNT = 10
MIN_MEDIAN_ANSWER_MASS = 0.9
MAX_UNPARSEABLE_FRACTION = 1 / 16
MIN_LABEL_START_FRACTION = 3 / 4
LABEL_START_PATTERN = re.compile(r"^[\s*]*[12](?![0-9])")


@dataclass(frozen=True)
class RunConfig:
    target_model: str
    run_dir: Path = DEFAULT_RUN_DIR
    data_dir: Path = Path("data")
    sample_indices: tuple[int, ...] = (0, 1, 2, 3)
    prompt_limit: int | None = None
    split: Split | None = None
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    max_model_len: int = 16384
    top_logprobs: int = 20
    chunk_size: int = 2000
    compliance_check: Literal["off", "true", "gate"] = "gate"
    dry_run: bool = False


class TokenProbability(TypedDict):
    token: str
    probability: float


class PresentationCheck(TypedDict):
    prompt_id: int
    other_model: str
    target_position: CandidatePosition
    top_tokens: list[TokenProbability]
    answer_probabilities: AnswerProbabilities | None
    answer_mass: float | None
    chosen_position: CandidatePosition | None


class FreeGeneration(TypedDict):
    prompt_id: int
    other_model: str
    target_position: CandidatePosition
    text: str
    starts_with_label: bool


class ComplianceReport(TypedDict):
    model: str
    rendered_prompt: str
    answer_label_tokens: dict[CandidatePosition, list[str]]
    presentations: list[PresentationCheck]
    free_generations: list[FreeGeneration]
    median_answer_mass: float
    unparseable_count: int
    label_start_count: int
    failed_criteria: list[str]
    passed: bool


type JudgementKey = tuple[int, str, int, int, CandidatePosition]


def compliance_path(run_dir: Path, target_model: str) -> Path:
    return judgements_path(run_dir, target_model).parent / "compliance" / f"{model_slug(target_model)}.json"


def compliance_tasks(tasks: Sequence[JudgeTask], pair_count: int) -> list[JudgeTask]:
    """Both orders of pair_count items, taking each other model in turn and spreading over prompts."""
    pairs_by_other: defaultdict[str, list[tuple[JudgeTask, ...]]] = defaultdict(list)
    for pair in batched(tasks, 2):
        pairs_by_other[pair[0]["other"]["model"]].append(pair)
    spread_pairs_by_other = [
        sorted(pairs, key=lambda pair: (pair[0]["target"]["sample_index"], pair[0]["prompt_id"]))
        for pairs in pairs_by_other.values()
    ]
    interleaved_pairs = [pair for pairs in zip_longest(*spread_pairs_by_other) for pair in pairs if pair is not None]
    return [item for pair in interleaved_pairs[:pair_count] for item in pair]


def decoded_label_tokens(tokenizer: HFTokenizer) -> dict[CandidatePosition, list[str]]:
    return {
        position: [repr(tokenizer.decode(token_id)) for token_id in token_ids]
        for position, token_ids in answer_label_token_ids(tokenizer).items()
    }


def presentation_check(
    item: JudgeTask,
    logprobs_by_token_id: dict[int, float] | None,
    label_token_ids_by_position: dict[CandidatePosition, list[int]],
    tokenizer: HFTokenizer,
) -> PresentationCheck:
    probabilities = (
        None if logprobs_by_token_id is None else answer_probabilities(logprobs_by_token_id, label_token_ids_by_position)
    )
    top_logprobs = sorted((logprobs_by_token_id or {}).items(), key=lambda entry: entry[1], reverse=True)
    return {
        "prompt_id": item["prompt_id"],
        "other_model": item["other"]["model"],
        "target_position": item["target_position"],
        "top_tokens": [
            {"token": tokenizer.decode(token_id), "probability": math.exp(logprob)}
            for token_id, logprob in top_logprobs[:LOGGED_TOP_TOKEN_COUNT]
        ],
        "answer_probabilities": probabilities,
        "answer_mass": None if probabilities is None else probabilities["position_1"] + probabilities["position_2"],
        "chosen_position": chosen_position(probabilities),
    }


def free_generation_texts(generator: VLLMGenerator, tokenizer: HFTokenizer, model: str, prompt_texts: list[str]) -> list[str]:
    from vllm import SamplingParams, TokensPrompt

    generator.load()
    assert generator.llm is not None
    temperature = next(
        (temperature for substring, temperature in FREE_GENERATION_TEMPERATURE_BY_MODEL_SUBSTRING.items() if substring in model.lower()),
        0.0,
    )
    outputs = generator.llm.generate(
        [
            TokensPrompt(prompt_token_ids=configure_call(tokenizer)(prompt_text, add_special_tokens=False)["input_ids"])
            for prompt_text in prompt_texts
        ],
        SamplingParams(
            max_tokens=FREE_GENERATION_MAX_TOKENS,
            temperature=temperature,
            seed=FREE_GENERATION_SEED,
            skip_special_tokens=get_generation_specifics(model).skip_special_tokens,
        ),
    )
    return [output.outputs[0].text for output in outputs]


def compliance_failures(
    median_answer_mass: float,
    unparseable_count: int,
    presentation_count: int,
    label_start_count: int,
    free_generation_count: int,
) -> list[str]:
    return [
        failure
        for failure, failed in [
            (f"median 1/2 mass {median_answer_mass:.3f} < {MIN_MEDIAN_ANSWER_MASS}", median_answer_mass < MIN_MEDIAN_ANSWER_MASS),
            (
                f"unparseable {unparseable_count}/{presentation_count} > {MAX_UNPARSEABLE_FRACTION:.4f}",
                unparseable_count > MAX_UNPARSEABLE_FRACTION * presentation_count,
            ),
            (
                f"label starts {label_start_count}/{free_generation_count} < {MIN_LABEL_START_FRACTION}",
                label_start_count < MIN_LABEL_START_FRACTION * free_generation_count,
            ),
        ]
        if failed
    ]


def compliance_report(
    generator: VLLMGenerator,
    tokenizer: HFTokenizer,
    model: str,
    tasks: Sequence[JudgeTask],
    top_logprobs: int,
) -> ComplianceReport:
    prompt_texts = [judge_prompt_text(tokenizer, model, judge_task_to_conversation(task)) for task in tasks]
    label_token_ids_by_position = answer_label_token_ids(tokenizer)
    presentations = [
        presentation_check(item, logprobs, label_token_ids_by_position, tokenizer)
        for item, logprobs in zip(tasks, generator.next_token_logprobs(prompt_texts, top_logprobs), strict=True)
    ]
    free_generation_items = [tasks[index] for index in FREE_GENERATION_PRESENTATION_INDICES]
    free_texts = free_generation_texts(
        generator, tokenizer, model, [prompt_texts[index] for index in FREE_GENERATION_PRESENTATION_INDICES]
    )
    free_generations: list[FreeGeneration] = [
        {
            "prompt_id": item["prompt_id"],
            "other_model": item["other"]["model"],
            "target_position": item["target_position"],
            "text": text,
            "starts_with_label": LABEL_START_PATTERN.match(text) is not None,
        }
        for item, text in zip(free_generation_items, free_texts, strict=True)
    ]
    median_answer_mass = statistics.median(presentation["answer_mass"] or 0.0 for presentation in presentations)
    unparseable_count = sum(presentation["chosen_position"] is None for presentation in presentations)
    label_start_count = sum(generation["starts_with_label"] for generation in free_generations)
    failed_criteria = compliance_failures(
        median_answer_mass, unparseable_count, len(presentations), label_start_count, len(free_generations)
    )
    return {
        "model": model,
        "rendered_prompt": prompt_texts[0],
        "answer_label_tokens": decoded_label_tokens(tokenizer),
        "presentations": presentations,
        "free_generations": free_generations,
        "median_answer_mass": median_answer_mass,
        "unparseable_count": unparseable_count,
        "label_start_count": label_start_count,
        "failed_criteria": failed_criteria,
        "passed": not failed_criteria,
    }


def log_compliance_report(report: ComplianceReport) -> None:
    logger.info("Rendered judge prompt (check against the plan's tail table):\n%s", report["rendered_prompt"])
    logger.info("Answer label tokens: %s", report["answer_label_tokens"])
    for presentation in report["presentations"]:
        logger.info(
            "prompt %d vs %s, target at %d: mass=%s chosen=%s top=%s",
            presentation["prompt_id"],
            presentation["other_model"],
            presentation["target_position"],
            presentation["answer_mass"],
            presentation["chosen_position"],
            [(token["token"], round(token["probability"], 4)) for token in presentation["top_tokens"]],
        )
    for generation in report["free_generations"]:
        logger.info(
            "free generation, prompt %d vs %s, target at %d: %r",
            generation["prompt_id"],
            generation["other_model"],
            generation["target_position"],
            generation["text"],
        )
    logger.info(
        "COMPLIANCE %s: %s (median 1/2 mass %.3f, unparseable %d/%d, label starts %d/%d)%s",
        report["model"],
        "PASS" if report["passed"] else "FAIL",
        report["median_answer_mass"],
        report["unparseable_count"],
        len(report["presentations"]),
        report["label_start_count"],
        len(report["free_generations"]),
        "" if report["passed"] else f" failed: {report['failed_criteria']}",
    )


def log_dry_run(target_model: str, tasks: Sequence[JudgeTask]) -> None:
    from byutils import load_tokenizer

    tokenizer = cast(HFTokenizer, load_tokenizer(target_model, allow_download=False))
    logger.info("Rendered judge prompt:\n%s", judge_prompt_text(tokenizer, target_model, judge_task_to_conversation(tasks[0])))
    logger.info("Answer label tokens: %s", decoded_label_tokens(tokenizer))


def judge_item_key(item: JudgeTask) -> JudgementKey:
    return (
        item["prompt_id"],
        item["other"]["model"],
        item["target"]["sample_index"],
        item["other"]["sample_index"],
        item["target_position"],
    )


def judgement_record_key(record: JudgementRecord) -> JudgementKey:
    return (
        record["prompt_id"],
        record["other_model"],
        record["target_sample_index"],
        record["other_sample_index"],
        record["target_position"],
    )


def main(run: RunConfig) -> None:
    logger.info("%s", run)
    prompts = read_prompts(prompts_path(run.data_dir))
    transcripts_by_model = read_transcripts_by_model(run.run_dir)
    selected_prompts = [prompt for prompt in prompts if run.split is None or prompt["split"] == run.split][: run.prompt_limit]
    tasks = create_judge_tasks(selected_prompts, transcripts_by_model, run.target_model, run.sample_indices)
    logger.info("%d presentations over %d prompts", len(tasks), len(selected_prompts))
    if run.dry_run:
        log_dry_run(run.target_model, tasks)
        return

    generator = VLLMGenerator(
        model_name=run.target_model,
        max_model_len=run.max_model_len,
        tensor_parallel_size=run.tensor_parallel_size,
        gpu_memory_utilization=run.gpu_memory_utilization,
    )
    generator.load()
    assert generator.llm is not None
    tokenizer = cast(HFTokenizer, generator.llm.get_tokenizer())

    if run.compliance_check != "off":
        train_prompts = [prompt for prompt in prompts if prompt["split"] == "train"]
        report = compliance_report(
            generator,
            tokenizer,
            run.target_model,
            compliance_tasks(create_judge_tasks(train_prompts, transcripts_by_model, run.target_model, run.sample_indices), COMPLIANCE_PAIR_COUNT),
            run.top_logprobs,
        )
        report_path = compliance_path(run.run_dir, run.target_model)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2))
        log_compliance_report(report)
        if run.compliance_check == "true":
            return
        if not report["passed"]:
            raise SystemExit("compliance check failed; not judging")

    output_path = judgements_path(run.run_dir, run.target_model)
    process_unrecorded_items(
        tasks,
        output_path,
        item_key=judge_item_key,
        record_key=judgement_record_key,
        process_chunk=judgement_chunk_processor(generator, tokenizer, run.target_model, run.top_logprobs),
        chunk_size=run.chunk_size,
    )
    judgements = read_judgements(output_path)
    logger.info(
        "%s: %d judgements, %d over max_model_len, %d unparseable",
        output_path,
        len(judgements),
        sum(judgement["answer_probabilities"] is None for judgement in judgements),
        sum(judgement["chosen_position"] is None for judgement in judgements),
    )


if __name__ == "__main__":
    from jsonargparse import auto_cli

    auto_cli(main)
