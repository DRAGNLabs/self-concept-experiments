import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

THINK_CLOSE = "</think>"

type ReasoningByExampleTurn = dict[tuple[str, int], str]


class ModelSpecifics(Protocol):
    def split_attempt(self, example_id: str, turn: int, completion: str) -> tuple[str, str]:
        """-> (reasoning, content) of the example's attempt at 0-based turn."""
        ...


def split_think_completion(completion: str) -> tuple[str, str]:
    """-> (reasoning, content). The chat template opens <think>, so completions
    carry only the close tag; without one the generation never left reasoning."""
    reasoning, think_close, content = completion.rpartition(THINK_CLOSE)
    if not think_close:
        return completion.strip(), ""
    return reasoning.strip(), content.strip()


class ThinkTagModelSpecifics:
    """Templates that open <think> in the generation prompt (OLMo 3): the completion holds the reasoning."""

    def split_attempt(self, example_id: str, turn: int, completion: str) -> tuple[str, str]:
        return split_think_completion(completion)


@dataclass(frozen=True)
class HarmonyModelSpecifics:
    """gpt-oss via codebench.vllm_harmony: the completion is the final channel only, and the analysis
    channel is in the run's reasoning sidecar."""

    reasoning_by_example_turn: ReasoningByExampleTurn

    def split_attempt(self, example_id: str, turn: int, completion: str) -> tuple[str, str]:
        return self.reasoning_by_example_turn[(example_id, turn)].strip(), completion.strip()


def think_tag_model_specifics() -> ModelSpecifics:
    return ThinkTagModelSpecifics()


def harmony_model_specifics(reasoning_path: Path) -> ModelSpecifics:
    """A resumed run regenerates the example it was interrupted in, so a later sidecar line for the
    same (example_id, turn) replaces the earlier one."""
    reasoning_by_example_turn: ReasoningByExampleTurn = {}
    for line in reasoning_path.open():
        entry = json.loads(line)
        reasoning_by_example_turn[(entry["example_id"], entry["turn"])] = entry["reasoning"]
    return HarmonyModelSpecifics(reasoning_by_example_turn)
