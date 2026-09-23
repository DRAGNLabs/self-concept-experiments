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


class RepairableJudgment[JudgmentT](TypedDict):
    """Usable as the final judgment, but sent back for repair while rounds remain."""

    status: Literal["repairable"]
    judgment: JudgmentT
    description: str


class ParseProblem(TypedDict):
    status: Literal["problem"]
    description: str


type ParseResult[JudgmentT] = ParsedJudgment[JudgmentT] | RepairableJudgment[JudgmentT] | ParseProblem

type BuildJudgeMessages[ItemT] = Callable[[ItemT], Conversation]
type GenerateBatch = Callable[[Sequence[Conversation]], list[GeneratedResponse]]
type ParseJudgeResponse[ItemT, JudgmentT] = Callable[[ItemT, GeneratedResponse], ParseResult[JudgmentT]]
type BuildRepairMessages[ItemT] = Callable[[ItemT, Conversation, GeneratedResponse, ParseProblem], Conversation]


class RepairPolicy[ItemT](TypedDict):
    build_repair_messages: BuildRepairMessages[ItemT]
    max_rounds: int


class ProblemResponse(TypedDict):
    round: int
    response: GeneratedResponse
    problem: ParseProblem


class JudgeParsed[ItemT, JudgmentT](TypedDict):
    status: Literal["parsed"]
    item: ItemT
    judgment: JudgmentT
    judgment_round: int
    judgment_response: GeneratedResponse
    unrepaired_problem: ParseProblem | None
    problem_responses: list[ProblemResponse]


class JudgeFailed[ItemT](TypedDict):
    status: Literal["failed"]
    item: ItemT
    problem_responses: list[ProblemResponse]


type JudgeOutcome[ItemT, JudgmentT] = JudgeParsed[ItemT, JudgmentT] | JudgeFailed[ItemT]


class _RepairableCandidate[JudgmentT](TypedDict):
    round: int
    judgment: JudgmentT
    response: GeneratedResponse
    problem: ParseProblem


class _PendingJudgment[ItemT, JudgmentT](TypedDict):
    index: int
    item: ItemT
    messages: Conversation
    round: int
    rejected_responses: list[ProblemResponse]
    fallback: _RepairableCandidate[JudgmentT] | None


def run_judge[ItemT, JudgmentT](
    items: Sequence[ItemT],
    build_judge_messages: BuildJudgeMessages[ItemT],
    generate_batch: GenerateBatch,
    parse_judge_response: ParseJudgeResponse[ItemT, JudgmentT],
    repair_policy: RepairPolicy[ItemT] | None = None,
) -> list[JudgeOutcome[ItemT, JudgmentT]]:
    """One outcome per item, in item order. When repair rounds run out, the item falls back to
    its latest repairable judgment, so a repair attempt can never lose a usable judgment."""
    initial_pending: list[_PendingJudgment[ItemT, JudgmentT]] = [
        {
            "index": index,
            "item": item,
            "messages": build_judge_messages(item),
            "round": 0,
            "rejected_responses": [],
            "fallback": None,
        }
        for index, item in enumerate(items)
    ]
    repair_rounds_left = repair_policy["max_rounds"] if repair_policy else 0
    indexed_outcomes = _judge_rounds(
        initial_pending, generate_batch, parse_judge_response, repair_policy, repair_rounds_left
    )
    return [outcome for _, outcome in sorted(indexed_outcomes, key=lambda indexed: indexed[0])]


def _as_problem_response(candidate: _RepairableCandidate) -> ProblemResponse:
    return {"round": candidate["round"], "response": candidate["response"], "problem": candidate["problem"]}


def _in_round_order(problem_responses: list[ProblemResponse]) -> list[ProblemResponse]:
    return sorted(problem_responses, key=lambda problem_response: problem_response["round"])


def _parsed_outcome[ItemT, JudgmentT](
    item: ItemT,
    judgment: JudgmentT,
    judgment_round: int,
    judgment_response: GeneratedResponse,
    unrepaired_problem: ParseProblem | None,
    rejected_responses: list[ProblemResponse],
) -> JudgeParsed[ItemT, JudgmentT]:
    return {
        "status": "parsed",
        "item": item,
        "judgment": judgment,
        "judgment_round": judgment_round,
        "judgment_response": judgment_response,
        "unrepaired_problem": unrepaired_problem,
        "problem_responses": _in_round_order(rejected_responses),
    }


def _judge_rounds[ItemT, JudgmentT](
    pending: list[_PendingJudgment[ItemT, JudgmentT]],
    generate_batch: GenerateBatch,
    parse_judge_response: ParseJudgeResponse[ItemT, JudgmentT],
    repair_policy: RepairPolicy[ItemT] | None,
    repair_rounds_left: int,
) -> list[tuple[int, JudgeOutcome[ItemT, JudgmentT]]]:
    if not pending:
        return []
    responses = generate_batch([judgment["messages"] for judgment in pending])
    resolved: list[tuple[int, JudgeOutcome[ItemT, JudgmentT]]] = []
    retry: list[_PendingJudgment[ItemT, JudgmentT]] = []
    for judgment, response in zip(pending, responses, strict=True):
        item, round_number, fallback = judgment["item"], judgment["round"], judgment["fallback"]
        superseded_fallback = [_as_problem_response(fallback)] if fallback else []
        parse_result = parse_judge_response(item, response)
        if parse_result["status"] == "parsed":
            outcome = _parsed_outcome(
                item,
                parse_result["judgment"],
                round_number,
                response,
                None,
                judgment["rejected_responses"] + superseded_fallback,
            )
            resolved.append((judgment["index"], outcome))
            continue
        problem = ParseProblem(status="problem", description=parse_result["description"])
        can_repair = repair_policy is not None and repair_rounds_left > 0
        if parse_result["status"] == "repairable":
            candidate: _RepairableCandidate[JudgmentT] = {
                "round": round_number,
                "judgment": parse_result["judgment"],
                "response": response,
                "problem": problem,
            }
            rejected_responses = judgment["rejected_responses"] + superseded_fallback
            next_fallback: _RepairableCandidate[JudgmentT] | None = candidate
        else:
            rejected_responses = [
                *judgment["rejected_responses"],
                ProblemResponse(round=round_number, response=response, problem=problem),
            ]
            next_fallback = fallback
        if repair_policy is None or not can_repair:
            final_outcome: JudgeOutcome[ItemT, JudgmentT] = (
                _parsed_outcome(
                    item,
                    next_fallback["judgment"],
                    next_fallback["round"],
                    next_fallback["response"],
                    next_fallback["problem"],
                    rejected_responses,
                )
                if next_fallback
                else {"status": "failed", "item": item, "problem_responses": _in_round_order(rejected_responses)}
            )
            resolved.append((judgment["index"], final_outcome))
            continue
        retry.append(
            {
                **judgment,
                "messages": repair_policy["build_repair_messages"](item, judgment["messages"], response, problem),
                "round": round_number + 1,
                "rejected_responses": rejected_responses,
                "fallback": next_fallback,
            }
        )
    return resolved + _judge_rounds(
        retry, generate_batch, parse_judge_response, repair_policy, repair_rounds_left - 1
    )
