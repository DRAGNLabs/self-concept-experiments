"""Judge benchmark transcripts with the reward-hacking rubrics of arXiv 2609.19101.

    python -m selfconcept.goodfire_rh_metrics.judge --run.benchmark impossiblebench_minimal_scaffold \
        --run.transcripts_paths '[<harness records jsonl>, ...]' --run.output_dir <dir> [--run.n 4] \
        [--run.model_specifics harmony --run.reasoning_paths '[<reasoning sidecar jsonl>, ...]']  # one per transcripts path

Writes {output_dir}/judged_<transcripts stem>.jsonl, appending one record per transcript as each
chunk finishes; rerunning skips transcripts already judged.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Literal

from transformers import AutoTokenizer

from selfconcept.codebench.vllm_harmony import split_harmony_completion
from selfconcept.common.chat import Conversation
from selfconcept.common.llm_judge import GenerateBatch, GeneratedResponse, JudgeOutcome, ParseProblem, run_judge
from selfconcept.common.resumable_jsonl import process_unrecorded_items

from .benchmarks import JudgeBenchmark, JudgeBenchmarkName, judge_benchmark_by_name, judge_prompt
from .judge_output import JudgedTranscript, TranscriptJudgment
from .model_specifics import ModelSpecifics, harmony_model_specifics, think_tag_model_specifics
from .transcript import RenderedTranscript

logger = logging.getLogger(__name__)

type ModelSpecificsName = Literal["think_tags", "harmony"]

REPAIR_INSTRUCTION = (
    "Respond again with the complete corrected JSON object: one row for every PASSAGE INDEX line, "
    "with every quote copied exactly from the passage its row names."
)


def judge_messages(benchmark: JudgeBenchmark, transcript: RenderedTranscript) -> Conversation:
    return [
        {"role": "system", "content": f"Response JSON schema:\n{benchmark['response_schema']}"},
        {"role": "user", "content": judge_prompt(benchmark, transcript)},
    ]


def repair_messages(
    transcript: RenderedTranscript, messages: Conversation, response: GeneratedResponse, problem: ParseProblem
) -> Conversation:
    return [
        *messages,
        {"role": "assistant", "content": response["text"]},
        {"role": "user", "content": f"Your answer has these problems:\n{problem['description']}\n\n{REPAIR_INSTRUCTION}"},
    ]


def vllm_harmony_generate_batch(
    model_id: str,
    max_model_len: int,
    max_new_tokens: int,
    reasoning_effort: str,
    temperature: float,
    seed: int,
    tensor_parallel_size: int,
) -> GenerateBatch:
    from vllm import LLM, SamplingParams, TokensPrompt

    llm = LLM(model=model_id, max_model_len=max_model_len, tensor_parallel_size=tensor_parallel_size)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    def generate_batch(conversations: Sequence[Conversation]) -> list[GeneratedResponse]:
        prompt_token_ids = [
            tokenizer.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                reasoning_effort=reasoning_effort,
                tokenize=True,
                return_dict=False,
            )
            for conversation in conversations
        ]
        sampling_params = [
            SamplingParams(
                temperature=temperature,
                seed=seed,
                max_tokens=min(max_new_tokens, max_model_len - len(token_ids)),
                skip_special_tokens=False,
            )
            for token_ids in prompt_token_ids
        ]
        request_outputs = llm.generate(
            [TokensPrompt(prompt_token_ids=token_ids) for token_ids in prompt_token_ids], sampling_params
        )
        responses: list[GeneratedResponse] = []
        for request_output in request_outputs:
            completion = request_output.outputs[0]
            reasoning, final = split_harmony_completion(completion.text)
            responses.append({"text": final, "reasoning": reasoning, "truncated": completion.finish_reason == "length"})
        return responses

    return generate_batch


def judged_transcript(outcome: JudgeOutcome[RenderedTranscript, TranscriptJudgment]) -> JudgedTranscript:
    transcript = outcome["item"]
    if outcome["status"] == "parsed":
        return {
            "transcript_id": transcript["transcript_id"],
            "split": transcript["split"],
            "benchmark_label": transcript["benchmark_label"],
            "status": "parsed",
            "judgment": outcome["judgment"],
            "judgment_round": outcome["judgment_round"],
            "judgment_response": outcome["judgment_response"],
            "unrepaired_problem": outcome["unrepaired_problem"],
            "problem_responses": outcome["problem_responses"],
        }
    return {
        "transcript_id": transcript["transcript_id"],
        "split": transcript["split"],
        "benchmark_label": transcript["benchmark_label"],
        "status": "failed",
        "problem_responses": outcome["problem_responses"],
    }


@dataclass(frozen=True)
class RunConfig:
    benchmark: JudgeBenchmarkName
    transcripts_paths: tuple[Path, ...]
    output_dir: Path
    model_specifics: ModelSpecificsName = "think_tags"
    # harmony only: the codebench.vllm_harmony reasoning sidecar of each transcripts path, in the same order
    reasoning_paths: tuple[Path, ...] = ()
    n: int | None = None
    judge_model: str = "openai/gpt-oss-120b"
    reasoning_effort: Literal["low", "medium", "high"] = "high"
    # gpt-oss's recommended sampling; greedy decoding loops within long reasoning
    temperature: float = 1.0
    seed: int = 0
    max_model_len: int = 131072
    max_new_tokens: int = 32768
    tensor_parallel_size: int = 1
    chunk_size: int = 32
    max_repair_rounds: int = 1


def transcript_key(transcript: RenderedTranscript) -> tuple[str, str]:
    return transcript["split"], transcript["transcript_id"]


def judged_transcript_key(record: JudgedTranscript) -> tuple[str, str]:
    return record["split"], record["transcript_id"]


def judge_chunk(
    benchmark: JudgeBenchmark, generate_batch: GenerateBatch, max_repair_rounds: int, transcripts: Sequence[RenderedTranscript]
) -> list[JudgedTranscript]:
    outcomes = run_judge(
        transcripts,
        partial(judge_messages, benchmark),
        generate_batch,
        benchmark["parse_judge_response"],
        {"build_repair_messages": repair_messages, "max_rounds": max_repair_rounds},
    )
    logger.info(
        "judged %d transcripts: %d failed, %d needed repair",
        len(outcomes),
        sum(outcome["status"] == "failed" for outcome in outcomes),
        sum(bool(outcome["problem_responses"]) for outcome in outcomes),
    )
    return [judged_transcript(outcome) for outcome in outcomes]


def transcripts_model_specifics(run: RunConfig) -> list[ModelSpecifics]:
    """One ModelSpecifics per transcripts path, built from whatever that model family needs."""
    match run.model_specifics:
        case "think_tags":
            if run.reasoning_paths:
                raise ValueError("reasoning_paths are only read with --run.model_specifics harmony")
            return [think_tag_model_specifics() for _ in run.transcripts_paths]
        case "harmony":
            if len(run.reasoning_paths) != len(run.transcripts_paths):
                raise ValueError(
                    f"harmony needs one reasoning sidecar per transcripts path; got {len(run.reasoning_paths)} "
                    f"for {len(run.transcripts_paths)}"
                )
            return [harmony_model_specifics(reasoning_path) for reasoning_path in run.reasoning_paths]


def main(run: RunConfig) -> None:
    """Judge every transcript in transcripts_paths not yet recorded under output_dir."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    benchmark = judge_benchmark_by_name[run.benchmark]
    model_specifics_by_transcripts_path = transcripts_model_specifics(run)
    generate_batch = vllm_harmony_generate_batch(
        run.judge_model,
        run.max_model_len,
        run.max_new_tokens,
        run.reasoning_effort,
        run.temperature,
        run.seed,
        run.tensor_parallel_size,
    )
    for transcripts_path, model_specifics in zip(run.transcripts_paths, model_specifics_by_transcripts_path, strict=True):
        process_unrecorded_items(
            benchmark["load_transcripts"](transcripts_path, model_specifics)[: run.n],
            run.output_dir / f"judged_{transcripts_path.stem}.jsonl",
            transcript_key,
            judged_transcript_key,
            partial(judge_chunk, benchmark, generate_batch, run.max_repair_rounds),
            run.chunk_size,
        )


if __name__ == "__main__":
    from jsonargparse import auto_cli

    auto_cli(main)
