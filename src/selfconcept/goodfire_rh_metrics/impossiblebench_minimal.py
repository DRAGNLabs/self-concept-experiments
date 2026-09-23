import json
from pathlib import Path
from typing import NotRequired, TypedDict, cast

from selfconcept.codebench.harness import (
    TOOL_RESULT_MAX_CHARS,
    CheckExample,
    check_feedback_error,
    check_retry_message,
    check_task_message,
    load_examples,
)

from .transcript import ContextBlock, RenderedTranscript, Step, render_transcript

THINK_CLOSE = "</think>"
NO_SYSTEM_PROMPT = "(none)"


class CheckAttempt(TypedDict):
    completion: str
    truncated: bool
    answer: str | None
    flagged: NotRequired[bool]
    success: NotRequired[bool]
    returncode: NotRequired[int]
    timed_out: NotRequired[bool]
    stdout: NotRequired[str]
    stderr: NotRequired[str]


class CheckRecord(TypedDict):
    example_id: str
    scenario: str
    label: str
    attempts: list[CheckAttempt]


def split_think_completion(completion: str) -> tuple[str, str]:
    """-> (reasoning, content). The chat template opens <think>, so completions
    carry only the close tag; without one the generation never left reasoning."""
    reasoning, think_close, content = completion.rpartition(THINK_CLOSE)
    if not think_close:
        return completion.strip(), ""
    return reasoning.strip(), content.strip()


def feedback_block(attempt: CheckAttempt) -> ContextBlock:
    stored_stdout, stored_stderr = attempt.get("stdout", ""), attempt.get("stderr", "")
    stored_tail_was_cut = len(stored_stderr or stored_stdout) >= TOOL_RESULT_MAX_CHARS
    marker = (
        "=== FEEDBACK (error reconstructed from the stored last 3000 characters) ==="
        if stored_tail_was_cut
        else "=== FEEDBACK ==="
    )
    return {"marker": marker, "text": check_retry_message(check_feedback_error(stored_stdout, stored_stderr))}


WHOLE_REPLY_EXECUTED_NOTE = "(the reply contained no fenced code block, so the harness executed the entire reply text)"


def executed_submission_block(step_number: int, attempt: CheckAttempt) -> ContextBlock:
    exit_description = "timed out" if attempt.get("timed_out") else f"exit code {attempt.get('returncode')}"
    executed_whole_reply = attempt["answer"] == attempt["completion"]
    return {
        "marker": f"=== EXECUTED SUBMISSION (step {step_number}; {exit_description}) ===",
        "text": WHOLE_REPLY_EXECUTED_NOTE if executed_whole_reply else attempt["answer"] or "",
    }


def attempt_step(step_number: int, attempt: CheckAttempt, has_next_attempt: bool) -> Step:
    reasoning, content = split_think_completion(attempt["completion"])
    ran_submission = attempt["answer"] is not None
    context_blocks = [
        *([executed_submission_block(step_number, attempt)] if ran_submission else []),
        *([feedback_block(attempt)] if has_next_attempt else []),
    ]
    return {
        "reasoning": reasoning,
        "content": content,
        "tool_calls": [],
        "truncated": attempt["truncated"],
        "context_blocks_after": context_blocks,
    }


def exit_event(attempts: list[CheckAttempt]) -> str:
    final_attempt = attempts[-1]
    if final_attempt.get("flagged"):
        return "flagged for human intervention"
    if final_attempt.get("success"):
        return f"passed on submission {len(attempts)}"
    return f"{len(attempts)} failed submissions"


def render_check_record(record: CheckRecord, example: CheckExample) -> RenderedTranscript:
    attempts = record["attempts"]
    steps = [
        attempt_step(step_number, attempt, has_next_attempt=step_number < len(attempts))
        for step_number, attempt in enumerate(attempts, start=1)
    ]
    return render_transcript(
        transcript_id=record["example_id"],
        split=record["scenario"],
        benchmark_label=record["label"],
        header_fields={
            "benchmark": "Impossible-LiveCodeBench, minimal scaffold (no tools, at most 3 submissions)",
            "example_id": record["example_id"],
            "entry_point": example["entry_point"],
        },
        system_prompt=NO_SYSTEM_PROMPT,
        task=check_task_message(example),
        steps=steps,
        exit_event=exit_event(attempts),
    )


def load_check_transcripts(records_path: Path) -> list[RenderedTranscript]:
    """Harness records with no attempts (error_oom) have nothing to judge and are skipped."""
    records: list[CheckRecord] = [json.loads(line) for line in records_path.open()]
    judgeable_records = [record for record in records if record["attempts"]]
    example_by_scenario_and_id = {
        (scenario, example["example_id"]): cast(CheckExample, example)
        for scenario in {record["scenario"] for record in judgeable_records}
        for example in load_examples(scenario)
    }
    return [
        render_check_record(record, example_by_scenario_and_id[(record["scenario"], record["example_id"])])
        for record in judgeable_records
    ]
