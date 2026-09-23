from typing import Literal, TypedDict

from selfconcept.common.llm_judge import GeneratedResponse, ProblemResponse

from .transcript import Channel, Passage

type FlagStatus = Literal["enacted", "attempted", "considered"]


class PassageFlag(TypedDict):
    step: int
    channel: Channel
    category: str
    status: FlagStatus
    quote: str
    why: str
    quote_verified: bool


class TranscriptJudgment(TypedDict):
    flags: list[PassageFlag]
    rationale: str


def json_object_text(response_text: str) -> str:
    return response_text[response_text.find("{") : response_text.rfind("}") + 1]


def collapse_whitespace(text: str) -> str:
    return " ".join(text.split())


def quote_in_passage(quote: str, passage: Passage) -> bool:
    collapsed_quote = collapse_whitespace(quote)
    return bool(collapsed_quote) and collapsed_quote in collapse_whitespace(passage["text"])


def unverified_quote_problems(flags: list[PassageFlag]) -> list[str]:
    return [
        f"step {flag['step']} {flag['channel']}: quote not found verbatim in that passage: {flag['quote']!r}"
        for flag in flags
        if not flag["quote_verified"]
    ]


class JudgedTranscriptParsed(TypedDict):
    transcript_id: str
    split: str
    benchmark_label: str
    status: Literal["parsed"]
    judgment: TranscriptJudgment
    final_response: GeneratedResponse
    problem_responses: list[ProblemResponse]


class JudgedTranscriptFailed(TypedDict):
    transcript_id: str
    split: str
    benchmark_label: str
    status: Literal["failed"]
    problem_responses: list[ProblemResponse]


type JudgedTranscript = JudgedTranscriptParsed | JudgedTranscriptFailed
