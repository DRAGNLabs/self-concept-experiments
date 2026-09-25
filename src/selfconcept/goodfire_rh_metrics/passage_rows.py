"""Parsing for the per-passage judge response the benchmarks' rubrics share: one row per PASSAGE
INDEX line, each flag drawn from the benchmark's own category set. A benchmark declares its own
flag/row/response TypedDicts (their names become the JSON schema's $defs names the judge sees),
using step_number_from_index_form on the row's step, and they satisfy the read-only shapes below."""

from collections.abc import Sequence
from typing import Literal, ReadOnly, TypedDict

from pydantic import TypeAdapter, ValidationError

from selfconcept.common.llm_judge import GeneratedResponse, ParseResult

from .judge_output import FlagStatus, PassageFlag, TranscriptJudgment, json_object_text, quote_in_passage, unverified_quote_problems
from .transcript import Passage, RenderedTranscript

MAX_VALIDATION_ERROR_CHARS = 2000


class PassageRowFlagShape(TypedDict):
    category: ReadOnly[str]
    status: ReadOnly[FlagStatus]
    quote: ReadOnly[str]
    why: ReadOnly[str]


class PassageRowShape(TypedDict):
    n: ReadOnly[int]
    step: ReadOnly[int]
    channel: ReadOnly[Literal["reasoning", "content"]]
    label: ReadOnly[Literal["positive", "negative"]]
    flags: ReadOnly[Sequence[PassageRowFlagShape]]


class PassageRowsResponseShape(TypedDict):
    rows: ReadOnly[Sequence[PassageRowShape]]
    rationale: ReadOnly[str]


def step_number_from_index_form(step: object) -> object:
    """The judge sometimes copies the PASSAGE INDEX's "step<N>" into the integer step field."""
    return step.removeprefix("step") if isinstance(step, str) else step


def row_problems(rows: Sequence[PassageRowShape], passages: list[Passage]) -> list[str]:
    expected_addresses = [f"P{passage['n']} | step{passage['step']} | {passage['channel']}" for passage in passages]
    row_addresses = [f"P{row['n']} | step{row['step']} | {row['channel']}" for row in rows]
    address_problems = (
        []
        if row_addresses == expected_addresses
        else [
            "The rows must match the PASSAGE INDEX exactly, one per line, in order. "
            f"Expected: {expected_addresses}. Got: {row_addresses}."
        ]
    )
    label_problems = [
        f"P{row['n']}: a positive row needs at least one flag and a negative row none, "
        f"but it is {row['label']} with {len(row['flags'])} flags."
        for row in rows
        if (row["label"] == "positive") != bool(row["flags"])
    ]
    return address_problems + label_problems


def parse_passage_rows_response[ResponseT: PassageRowsResponseShape](
    response_adapter: TypeAdapter[ResponseT], transcript: RenderedTranscript, response: GeneratedResponse
) -> ParseResult[TranscriptJudgment]:
    try:
        judge_response = response_adapter.validate_json(json_object_text(response["text"]))
    except ValidationError as error:
        return {
            "status": "problem",
            "description": f"The answer is not a JSON object matching the schema: {str(error)[:MAX_VALIDATION_ERROR_CHARS]}",
        }
    structural_problems = row_problems(judge_response["rows"], transcript["passages"])
    if structural_problems:
        return {"status": "problem", "description": "\n".join(structural_problems)}
    passage_by_n = {passage["n"]: passage for passage in transcript["passages"]}
    flags = [
        PassageFlag(
            step=row["step"],
            channel=row["channel"],
            category=flag["category"],
            status=flag["status"],
            quote=flag["quote"],
            why=flag["why"],
            quote_verified=quote_in_passage(flag["quote"], passage_by_n[row["n"]]),
        )
        for row in judge_response["rows"]
        for flag in row["flags"]
    ]
    judgment: TranscriptJudgment = {"flags": flags, "rationale": judge_response["rationale"]}
    quote_problems = unverified_quote_problems(flags)
    if quote_problems:
        return {"status": "repairable", "judgment": judgment, "description": "\n".join(quote_problems)}
    return {"status": "parsed", "judgment": judgment}
