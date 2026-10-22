"""Judge dossiers: the conversation exactly as the model saw it, rebuilt with the harness's own message functions, plus
every turn's full reasoning and the held-out result (the format rubric v3 and the rh_hillclimb verdicts use).

Harness outputs inside retry messages are rebuilt from the stored stdout/stderr, which records keep as their last 3000
characters, so a very long test output may differ from the middle-truncated text the model saw.
"""

import json
from pathlib import Path
from typing import Literal, cast

from selfconcept.arm_search.rh.conditions import note_path
from selfconcept.arm_search.rh.episodes import LabelledEpisode, reasoning_by_turn_by_example_id, record_key
from selfconcept.arm_search.rh.round_files import RunSpec, pool_data_dir, reasoning_path, records_path
from selfconcept.codebench import harness

type Role = Literal["user", "model_turn"]


def _remaining(record: dict, completed_turns: int) -> harness.AttemptsRemaining | None:
    return harness.stated_attempts_remaining(record["attempt_budget_stated"], completed_turns, record["max_attempts"])


def _check_conversation(example: dict, record: dict, feedback: harness.Feedback) -> list[tuple[Role, str]]:
    budget = record["max_attempts"] if record["attempt_budget_stated"] else None
    instruction = harness.instruction_by_impossible_prompt[record.get("impossible_prompt") or "minimal"]
    task = harness.check_task_message(cast(harness.CheckExample, example), budget, instruction, feedback)
    messages: list[tuple[Role, str]] = [("user", task)]
    for turn, attempt in enumerate(record["attempts"]):
        messages.append(("model_turn", str(turn)))
        if attempt.get("answer") is None or attempt.get("success"):
            continue
        error = harness.check_feedback_error(attempt.get("stdout") or "", attempt.get("stderr") or "")
        remaining = _remaining(record, turn + 1)
        messages.append(
            (
                "user",
                harness.neutral_retry_message(error, remaining)
                if harness.is_neutral(feedback)
                else harness.check_retry_message(error, remaining, instruction),
            )
        )
    return messages


def _workspace_conversation(example: dict, record: dict, feedback: harness.Feedback) -> list[tuple[Role, str]]:
    budget = record["max_attempts"] if record["attempt_budget_stated"] else None
    instruction = harness.workspace_instruction_by_impossible_prompt[record.get("impossible_prompt") or "minimal"]
    workspace_example = cast(harness.WorkspaceExample, example)
    task = harness.workspace_task_message(
        workspace_example, harness.workspace_files(workspace_example), budget, feedback, instruction
    )
    messages: list[tuple[Role, str]] = [("user", task)]
    for turn, attempt in enumerate(record["attempts"]):
        messages.append(("model_turn", str(turn)))
        remaining = _remaining(record, turn + 1)
        if attempt.get("success") or not attempt.get("wrote") and feedback == "neutral":
            continue
        if not attempt.get("wrote") and feedback == "neutral_continue":
            messages.append(("user", harness.neutral_no_files_message(remaining)))
            continue
        output = harness.workspace_test_output(attempt.get("stdout") or "", attempt.get("stderr") or "")
        messages.append(
            (
                "user",
                harness.neutral_workspace_retry_message(output, remaining)
                if harness.is_neutral(feedback)
                else harness.workspace_retry_message(output, remaining),
            )
        )
    return messages


def _after_episode(example: dict, record: dict) -> str:
    if example["kind"] == "check":
        return (
            f"passed shown tests: {record['passed']}  passed original correct suite: {record['passed_original']}\n"
            "Full correct suite (shown tests are a subset):\n```python\n" + example["original_test"] + "\n```"
        )
    return (
        f"passed visible: {record['passed_visible']}  passed hidden holdout: {record['passed_holdout']}  "
        f"test files modified: {record['test_modified']}\nholdout runner output:\n```\n"
        + record["holdout_output"]
        + "\n```"
    )


def dossier_text(
    dossier_name: str,
    spec: RunSpec,
    example: dict,
    record: dict,
    reasoning_by_turn: dict[int, str],
    repo_root: Path,
    model_id: str,
) -> str:
    feedback: harness.Feedback = record.get("feedback") or "impossiblebench"
    note = note_path(spec["condition"]["note"], "check" if example["kind"] == "check" else "workspace")
    conversation = (
        _check_conversation(example, record, feedback)
        if example["kind"] == "check"
        else _workspace_conversation(example, record, feedback)
    )
    parts = [
        f"# {dossier_name}",
        f"model: {model_id}",
        f"run: {spec['run_name']}  scenario: {record['scenario']}  harness label: {record['label']}  "
        f"attempts used: {record['n_attempts']}/{record['max_attempts']}  stop_reason: {record.get('stop_reason')}",
        "## SYSTEM MESSAGE\n" + ((repo_root / note).read_text().strip() if note else "(none)"),
    ]
    for role, text in conversation:
        if role == "user":
            parts.append("## USER / HARNESS MESSAGE\n" + text)
            continue
        turn = int(text)
        parts.append(
            f"## MODEL TURN {turn}: REASONING (hidden thinking, not shown to the user)\n"
            + reasoning_by_turn.get(turn, "(missing)")
        )
        parts.append(f"## MODEL TURN {turn}: REPLY\n" + record["attempts"][turn]["completion"])
    parts.append("## AFTER THE EPISODE (never shown to the model)\n" + _after_episode(example, record))
    return "\n\n".join(parts)



def write_dossiers(round_path: Path, episodes: list[LabelledEpisode], repo_root: Path, model_id: str) -> list[Path]:
    """One <record key>.md per episode under the round's dossiers dir."""
    dossier_dir = round_path / "dossiers"
    dossier_dir.mkdir(parents=True, exist_ok=True)
    run_names = {episode["spec"]["run_name"]: episode["spec"] for episode in episodes}
    record_by_key = {
        record_key(spec, record["example_id"]): record
        for spec in run_names.values()
        for record in map(json.loads, records_path(round_path, spec).open())
    }
    example_by_key = {
        record_key(spec, row["example_id"]): row
        for spec in run_names.values()
        for row in map(json.loads, (pool_data_dir(round_path, spec) / f"{spec['scenario']}.jsonl").open())
    }
    reasoning_by_run = {
        run_name: reasoning_by_turn_by_example_id(reasoning_path(round_path, spec))
        for run_name, spec in run_names.items()
    }
    paths = []
    for episode in episodes:
        path = dossier_dir / f"{episode['record_key']}.md"
        path.write_text(
            dossier_text(
                path.name,
                episode["spec"],
                example_by_key[episode["record_key"]],
                record_by_key[episode["record_key"]],
                reasoning_by_run[episode["spec"]["run_name"]].get(episode["example_id"], {}),
                repo_root,
                model_id,
            )
        )
        paths.append(path)
    return paths
