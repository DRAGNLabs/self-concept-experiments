import json
from pathlib import Path
from typing import Any, TypedDict, cast

from selfconcept.arm_search.interface import EpisodeResult
from selfconcept.arm_search.records import PlanningOutcome, RoundPlan, RoundRecord, RoundResult, SearchConfig
from selfconcept.common.jsonl import append_jsonl, read_jsonl_if_present

CONFIG_FILE_NAME = "config.json"
EPISODES_FILE_NAME = "episodes.jsonl"
PLANS_FILE_NAME = "plans.jsonl"
RESULTS_FILE_NAME = "results.jsonl"
PLANNING_OUTCOME_FILE_NAME = "planning_outcome.json"


class StoredRoundResult(TypedDict):
    round_index: int
    result: RoundResult


def _as_json_value(config: SearchConfig) -> Any:
    return json.loads(json.dumps(config, default=str))


def initialize_state(config: SearchConfig) -> None:
    """Create the state dir, or check that an existing one was made with the same config."""
    config_path = config["state_dir"] / CONFIG_FILE_NAME
    if config_path.exists():
        if json.loads(config_path.read_text()) != _as_json_value(config):
            raise ValueError(f"config differs from the one stored in {config_path}")
        return
    config["state_dir"].mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(_as_json_value(config), indent=2))


def load_episodes(state_dir: Path) -> list[EpisodeResult]:
    return cast(list[EpisodeResult], read_jsonl_if_present(state_dir / EPISODES_FILE_NAME))


def append_episodes(state_dir: Path, episodes: list[EpisodeResult]) -> None:
    for episode in episodes:
        append_jsonl(state_dir / EPISODES_FILE_NAME, episode)


def append_round_plan(state_dir: Path, plan: RoundPlan) -> None:
    append_jsonl(state_dir / PLANS_FILE_NAME, plan)


def append_round_result(state_dir: Path, round_index: int, result: RoundResult) -> None:
    append_jsonl(state_dir / RESULTS_FILE_NAME, StoredRoundResult(round_index=round_index, result=result))


def load_rounds(state_dir: Path) -> list[RoundRecord]:
    plans = cast(list[RoundPlan], read_jsonl_if_present(state_dir / PLANS_FILE_NAME))
    stored_results = cast(list[StoredRoundResult], read_jsonl_if_present(state_dir / RESULTS_FILE_NAME))
    result_by_round_index = {stored["round_index"]: stored["result"] for stored in stored_results}
    return [RoundRecord(plan=plan, result=result_by_round_index.get(plan["round_index"])) for plan in plans]


def write_planning_outcome(state_dir: Path, outcome: PlanningOutcome) -> None:
    outcome_path = state_dir / PLANNING_OUTCOME_FILE_NAME
    temporary_path = outcome_path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(outcome, indent=2))
    temporary_path.replace(outcome_path)


def load_planning_outcome(state_dir: Path) -> PlanningOutcome:
    return cast(PlanningOutcome, json.loads((state_dir / PLANNING_OUTCOME_FILE_NAME).read_text()))
