"""One round's files: problem subsets, worklists for vllm_server_workers.sbatch, harness outputs, dossiers and verdicts.

Every setting of a run is in the round manifest, since harness records lack effort, note and seed.
"""

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Final, TypedDict, cast

from selfconcept.arm_search.interface import BatchUnit
from selfconcept.arm_search.rh.conditions import (
    FEEDBACK_BY_WORDING,
    Condition,
    harness_impossible_prompt,
    note_path,
    parse_condition_id,
)
from selfconcept.arm_search.rh.model_specifics import RhModelSpecifics
from selfconcept.arm_search.rh.pool import PoolProblem, PoolScenario, load_pool_rows

MAX_NEW_TOKENS_BY_SCENARIO: Final[dict[PoolScenario, int]] = {
    "impossible_original": 16384,
    "evilgenie": 32768,
    "evilgenie_lcbpro": 32768,
}
MANIFEST_FILE_NAME: Final = "manifest.json"


class RunSpec(TypedDict):
    """One harness run: one condition, seed and scenario over a subset of problems."""

    run_name: str
    condition: Condition
    seed: int
    scenario: PoolScenario
    example_ids: list[str]


class RoundManifest(TypedDict):
    round_index: int
    run_specs: list[RunSpec]


def round_dir(state_dir: Path, round_index: int) -> Path:
    return state_dir / "rounds" / f"round_{round_index:04d}"


def run_dir(round_path: Path, spec: RunSpec) -> Path:
    return round_path / "runs" / spec["run_name"]


def pool_data_dir(round_path: Path, spec: RunSpec) -> Path:
    return round_path / "pools" / spec["run_name"]


def records_path(round_path: Path, spec: RunSpec) -> Path:
    return run_dir(round_path, spec) / f"{spec['scenario']}_{spec['scenario']}.jsonl"


def reasoning_path(round_path: Path, spec: RunSpec) -> Path:
    return run_dir(round_path, spec) / f"{spec['scenario']}_{spec['scenario']}_reasoning.jsonl"


def run_specs(units: list[BatchUnit], problem_by_example_id: dict[str, PoolProblem]) -> list[RunSpec]:
    """Units with the same condition, seed and scenario share a run."""
    example_ids_by_run: dict[tuple[str, int, PoolScenario], list[str]] = defaultdict(list)
    for unit in units:
        example_id, unit_condition_id = unit["arm"]["arm_id"].split("|")
        scenario = problem_by_example_id[example_id]["scenario"]
        for seed in unit["seeds"]:
            example_ids_by_run[(unit_condition_id, seed, scenario)].append(example_id)
    return [
        RunSpec(
            run_name=f"{run_condition_id}__seed{seed}__{scenario}",
            condition=parse_condition_id(run_condition_id),
            seed=seed,
            scenario=scenario,
            example_ids=sorted(set(example_ids)),
        )
        for (run_condition_id, seed, scenario), example_ids in sorted(example_ids_by_run.items())
    ]


def worklist_line(round_path: Path, spec: RunSpec, repo_root: Path, model: RhModelSpecifics) -> str:
    """The sbatch splits lines on whitespace without unquoting, so no argument may contain spaces."""
    condition = spec["condition"]
    task_format = "check" if spec["scenario"] == "impossible_original" else "workspace"
    note = note_path(condition["note"], task_format)
    arguments = [
        f"SOO_CHAT_KWARGS={json.dumps(model.chat_kwargs(condition['effort']), separators=(',', ':'))}",
        *["--data", str(pool_data_dir(round_path, spec)), "--scenarios", spec["scenario"]],
        *["--out", str(run_dir(round_path, spec)), "--tag", spec["scenario"]],
        *["--temperature", str(model.temperature), "--sample-seed", str(spec["seed"])],
        *["--impossible-prompt", harness_impossible_prompt(condition["prompt_tier"])],
        *["--feedback", FEEDBACK_BY_WORDING[condition["wording"]]],
        *["--max-attempts", str(condition["max_attempts"])],
        *(["--state-attempt-budget"] if condition["budget_stated"] else []),
        *["--max-new-tokens", str(MAX_NEW_TOKENS_BY_SCENARIO[spec["scenario"]])],
        *(["--system-prompt-file", str(repo_root / note)] if note else []),
    ]
    return " ".join(arguments)


def worklist_shards(specs: list[RunSpec], episodes_per_shard: int, max_runs_per_shard: int) -> list[list[RunSpec]]:
    """Greedy balance of episode counts over as few shards as keep each near episodes_per_shard; every run in a shard
    is a concurrent worker process."""
    shard_count = max(
        1,
        math.ceil(sum(len(spec["example_ids"]) for spec in specs) / episodes_per_shard),
        math.ceil(len(specs) / max_runs_per_shard),
    )
    shards: list[list[RunSpec]] = [[] for _ in range(shard_count)]
    for spec in sorted(specs, key=lambda spec: -len(spec["example_ids"])):
        min(shards, key=lambda shard: sum(len(member["example_ids"]) for member in shard)).append(spec)
    return [shard for shard in shards if shard]


def write_round_inputs(
    round_path: Path,
    manifest: RoundManifest,
    episodes_per_shard: int,
    max_runs_per_shard: int,
    repo_root: Path,
    model: RhModelSpecifics,
) -> list[Path]:
    """Writes the manifest, each run's problem subset and the worklist shards; returns the worklist paths."""
    round_path.mkdir(parents=True, exist_ok=True)
    (round_path / MANIFEST_FILE_NAME).write_text(json.dumps(manifest, indent=1))
    scenarios: set[PoolScenario] = {spec["scenario"] for spec in manifest["run_specs"]}
    row_by_example_id_by_scenario = {
        scenario: {row["example_id"]: row for row in load_pool_rows(scenario)} for scenario in scenarios
    }
    for spec in manifest["run_specs"]:
        data_dir = pool_data_dir(round_path, spec)
        data_dir.mkdir(parents=True, exist_ok=True)
        row_by_example_id = row_by_example_id_by_scenario[spec["scenario"]]
        (data_dir / f"{spec['scenario']}.jsonl").write_text(
            "".join(json.dumps(row_by_example_id[example_id]) + "\n" for example_id in spec["example_ids"])
        )
    worklist_paths = []
    for index, shard in enumerate(worklist_shards(manifest["run_specs"], episodes_per_shard, max_runs_per_shard)):
        path = round_path / f"worklist_{index}.txt"
        path.write_text("".join(worklist_line(round_path, spec, repo_root, model) + "\n" for spec in shard))
        worklist_paths.append(path)
    return worklist_paths


def load_manifest(round_path: Path) -> RoundManifest:
    return cast(RoundManifest, json.loads((round_path / MANIFEST_FILE_NAME).read_text()))
