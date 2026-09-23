from collections.abc import Callable, Sequence
from typing import Literal, TypedDict

from selfconcept.common.chat import Conversation


class GeneratedResponse(TypedDict):
    text: str
    reasoning: str | None
    truncated: bool


class ParsedJudgment[JudgmentT](TypedDict):
    status: Literal["parsed"]
    judgment: JudgmentT


class ParseProblem(TypedDict):
    status: Literal["problem"]
    description: str


type ParseResult[JudgmentT] = ParsedJudgment[JudgmentT] | ParseProblem

type BuildJudgeMessages[ItemT] = Callable[[ItemT], Conversation]
type GenerateBatch = Callable[[Sequence[Conversation]], list[GeneratedResponse]]
type ParseJudgeResponse[ItemT, JudgmentT] = Callable[[ItemT, GeneratedResponse], ParseResult[JudgmentT]]
type BuildRepairMessages[ItemT] = Callable[[ItemT, Conversation, GeneratedResponse, ParseProblem], Conversation]


class RepairPolicy[ItemT](TypedDict):
    build_repair_messages: BuildRepairMessages[ItemT]
    max_rounds: int


class ProblemResponse(TypedDict):
    response: GeneratedResponse
    problem: ParseProblem


class JudgeParsed[ItemT, JudgmentT](TypedDict):
    status: Literal["parsed"]
    item: ItemT
    judgment: JudgmentT
    final_response: GeneratedResponse
    problem_responses: list[ProblemResponse]


class JudgeFailed[ItemT](TypedDict):
    status: Literal["failed"]
    item: ItemT
    problem_responses: list[ProblemResponse]


type JudgeOutcome[ItemT, JudgmentT] = JudgeParsed[ItemT, JudgmentT] | JudgeFailed[ItemT]


class _PendingJudgment[ItemT](TypedDict):
    index: int
    item: ItemT
    messages: Conversation
    problem_responses: list[ProblemResponse]


def run_judge[ItemT, JudgmentT](
    items: Sequence[ItemT],
    build_judge_messages: BuildJudgeMessages[ItemT],
    generate_batch: GenerateBatch,
    parse_judge_response: ParseJudgeResponse[ItemT, JudgmentT],
    repair_policy: RepairPolicy[ItemT] | None = None,
) -> list[JudgeOutcome[ItemT, JudgmentT]]:
    """One outcome per item, in item order."""
    initial_pending: list[_PendingJudgment[ItemT]] = [
        {"index": index, "item": item, "messages": build_judge_messages(item), "problem_responses": []}
        for index, item in enumerate(items)
    ]
    repair_rounds_left = repair_policy["max_rounds"] if repair_policy else 0
    indexed_outcomes = _judge_rounds(
        initial_pending, generate_batch, parse_judge_response, repair_policy, repair_rounds_left
    )
    return [outcome for _, outcome in sorted(indexed_outcomes, key=lambda indexed: indexed[0])]


def _judge_rounds[ItemT, JudgmentT](
    pending: list[_PendingJudgment[ItemT]],
    generate_batch: GenerateBatch,
    parse_judge_response: ParseJudgeResponse[ItemT, JudgmentT],
    repair_policy: RepairPolicy[ItemT] | None,
    repair_rounds_left: int,
) -> list[tuple[int, JudgeOutcome[ItemT, JudgmentT]]]:
    if not pending:
        return []
    responses = generate_batch([judgment["messages"] for judgment in pending])
    resolved: list[tuple[int, JudgeOutcome[ItemT, JudgmentT]]] = []
    retry: list[_PendingJudgment[ItemT]] = []
    for judgment, response in zip(pending, responses, strict=True):
        parse_result = parse_judge_response(judgment["item"], response)
        if parse_result["status"] == "parsed":
            parsed_outcome: JudgeParsed[ItemT, JudgmentT] = {
                "status": "parsed",
                "item": judgment["item"],
                "judgment": parse_result["judgment"],
                "final_response": response,
                "problem_responses": judgment["problem_responses"],
            }
            resolved.append((judgment["index"], parsed_outcome))
            continue
        problem_responses = [*judgment["problem_responses"], ProblemResponse(response=response, problem=parse_result)]
        if repair_policy is None or repair_rounds_left == 0:
            failed_outcome: JudgeFailed[ItemT] = {
                "status": "failed",
                "item": judgment["item"],
                "problem_responses": problem_responses,
            }
            resolved.append((judgment["index"], failed_outcome))
            continue
        repair_messages = repair_policy["build_repair_messages"](
            judgment["item"], judgment["messages"], response, parse_result
        )
        retry.append({**judgment, "messages": repair_messages, "problem_responses": problem_responses})
    return resolved + _judge_rounds(
        retry, generate_batch, parse_judge_response, repair_policy, repair_rounds_left - 1
    )
