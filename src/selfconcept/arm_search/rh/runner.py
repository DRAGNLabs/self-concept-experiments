"""Runs one round on the cluster: generation jobs (one per worklist shard, in parallel, each on whatever GPU option
gpu_pick ranks first), then the GLM judge job over the round's dossiers. No job uses standby: a round waits for its
slowest job, and a preempted one goes back to the end of the queue. Every step resumes: the harness and the judge
skip what their outputs already hold, so rerunning a round only finishes it.
"""

import json
import logging
import os
import subprocess
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Final, TypedDict, cast

from byutils.gpu_pick import JobRequest, ranked_gpu_candidates, suggested_sbatch_flags

from selfconcept.arm_search.interface import BatchUnit, EpisodeResult
from selfconcept.arm_search.rh.dossier import write_dossiers
from selfconcept.arm_search.rh.episodes import episode_result, labelled_episodes, needs_judging
from selfconcept.arm_search.rh.model_specifics import RhModelSpecifics
from selfconcept.arm_search.rh.pool import PoolProblem
from selfconcept.arm_search.rh.round_files import (
    MANIFEST_FILE_NAME,
    RoundManifest,
    SamplingSettings,
    load_manifest,
    records_path,
    round_dir,
    run_dir,
    run_specs,
    write_round_inputs,
)
from selfconcept.codebench.rh_judge import JudgedDossier, is_strong_hack

logger = logging.getLogger(__name__)

GENERATION_SBATCH: Final = Path("experiments/codebench/slurm/vllm_server_workers.sbatch")
JUDGE_SBATCH: Final = Path("experiments/codebench/slurm/rh_judge.sbatch")
FALLBACK_GENERATION_FLAGS: Final = ["-p", "cs", "--qos", "cs", "--gpus", "a100:2"]
JUDGE_ARGUMENTS: Final = [
    *["--run.judge_model", "zai-org/GLM-4.5-Air-FP8", "--run.chat_format", "think_tags"],
    *["--run.temperature", "0.6", "--run.top_p", "0.95"],
]
FALLBACK_JUDGE_FLAGS: Final = ["--gpus", "h200:2", "--time", "02:00:00"]
# The judge was validated on H200s only (GLM-4.5-Air FP8 also needs native FP8, so never Ampere or older)
NON_JUDGE_GPU_TYPES: Final = frozenset({"p100", "v100", "a100", "l40s", "h100", "b200"})
JUDGE_VRAM_GB: Final = 160
JUDGE_HOURS: Final = 2.0
# Later attempts finish the runs of a job that died (node failure, cancelled at start); the harness skips recorded
# examples
GENERATION_ATTEMPTS: Final = 3
TENSOR_PARALLEL_SIZES: Final = frozenset({1, 2, 4, 8})
# Long enough for a submitted job to show in the queue state gpu_pick reads
SUBMISSION_GAP_SECONDS: Final = 15


class RunnerSettings(TypedDict):
    """`generation_hours` is each generation job's time limit."""

    repo_root: Path
    episodes_per_shard: int
    max_runs_per_shard: int
    generation_hours: float
    excluded_qos_names: list[str]


def picked_sbatch_flags(request: JobRequest, fallback_flags: list[str]) -> list[str]:
    """Only GPU counts the models' attention heads split over evenly (vLLM tensor parallelism)."""
    try:
        ranked_candidates = [
            candidate for candidate in ranked_gpu_candidates(request) if candidate.gpu_count in TENSOR_PARALLEL_SIZES
        ]
    except subprocess.CalledProcessError as error:
        logger.warning("gpu_pick failed (%s); using the fallback", error)
        ranked_candidates = []
    return suggested_sbatch_flags(ranked_candidates, request) if ranked_candidates else fallback_flags


def generation_sbatch_flags(settings: RunnerSettings, model: RhModelSpecifics) -> list[str]:
    request = JobRequest(
        vram_gb=model.generation_vram_gb,
        hours=settings["generation_hours"],
        cpus=24,
        mem_gb=200,
        requeue_ok=False,
        fast_turnaround=False,
        cpu_arch="x86_64",
        cuda=(13, 0),
        excluded_gpu_types=frozenset({"p100", "v100"}),
        excluded_qos_names=frozenset(settings["excluded_qos_names"]),
    )
    return picked_sbatch_flags(
        request, [*FALLBACK_GENERATION_FLAGS, "--time", f"{int(settings['generation_hours'])}:00:00"]
    )


def judge_sbatch_flags(settings: RunnerSettings) -> list[str]:
    request = JobRequest(
        vram_gb=JUDGE_VRAM_GB,
        hours=JUDGE_HOURS,
        cpus=8,
        mem_gb=128,
        requeue_ok=False,
        fast_turnaround=False,
        cpu_arch="x86_64",
        cuda=(13, 0),
        excluded_gpu_types=NON_JUDGE_GPU_TYPES,
        excluded_qos_names=frozenset(settings["excluded_qos_names"]),
    )
    return picked_sbatch_flags(request, FALLBACK_JUDGE_FLAGS)


def _wait_for_jobs(
    build_commands: list[Callable[[], list[str]]], cwd: Path, extra_environment: dict[str, str] | None = None
) -> None:
    """Submits each command (an `sbatch --wait`) in turn, building it only once the previous one is queued so that a
    GPU choice sees the jobs before it, then waits for all of them."""
    environment = {**os.environ, **(extra_environment or {})}
    processes = []
    for build_command in build_commands:
        processes.append(subprocess.Popen(build_command(), cwd=cwd, env=environment))
        time.sleep(SUBMISSION_GAP_SECONDS)
    for process in processes:
        exit_code = process.wait()
        if exit_code != 0:
            logger.warning("job exited %d: %s", exit_code, " ".join(map(str, process.args)))


def _generation_command(worklist_path: Path, settings: RunnerSettings, model: RhModelSpecifics) -> list[str]:
    return [
        "sbatch",
        "--wait",
        *generation_sbatch_flags(settings, model),
        f"--export=ALL,WORKLIST={worklist_path}",
        str(GENERATION_SBATCH),
    ]


def generation_environment(model: RhModelSpecifics) -> dict[str, str]:
    """Through the submitting environment rather than --export, which splits values on commas (JSON overrides)."""
    return {"MODEL": model.model_id, "BACKEND": model.backend, "VLLM_EXTRA_ARGS": " ".join(model.vllm_serve_args())}


def run_generation(worklist_paths: list[Path], settings: RunnerSettings, model: RhModelSpecifics) -> None:
    logger.info("submitting %d generation jobs", len(worklist_paths))
    _wait_for_jobs(
        [partial(_generation_command, worklist_path, settings, model) for worklist_path in worklist_paths],
        settings["repo_root"],
        generation_environment(model),
    )


def _recorded_example_ids(path: Path) -> set[str]:
    return {json.loads(line)["example_id"] for line in path.open()} if path.exists() else set()


def incomplete_worklists(round_path: Path, manifest: RoundManifest, worklist_paths: list[Path]) -> list[Path]:
    """Worklists with a run that has not recorded all its examples (a run's line names its --out dir)."""
    incomplete_run_dirs = {
        str(run_dir(round_path, spec))
        for spec in manifest["run_specs"]
        if not set(spec["example_ids"]) <= _recorded_example_ids(records_path(round_path, spec))
    }
    return [
        path
        for path in worklist_paths
        if any(f"--out {run_directory} " in path.read_text() for run_directory in incomplete_run_dirs)
    ]


def run_judge(dossier_paths: list[Path], round_path: Path, settings: RunnerSettings) -> Path:
    verdicts_path = round_path / "verdicts.jsonl"
    judged_names = (
        {json.loads(line)["dossier_name"] for line in verdicts_path.open()} if verdicts_path.exists() else set()
    )
    if {path.name for path in dossier_paths} <= judged_names:
        return verdicts_path
    dossier_list = round_path / "dossiers.list"
    dossier_list.write_text("".join(f"{path}\n" for path in dossier_paths))
    logger.info("judging %d dossiers", len(dossier_paths))
    _wait_for_jobs(
        [
            lambda: [
                "sbatch",
                "--wait",
                *judge_sbatch_flags(settings),
                str(JUDGE_SBATCH),
                str(dossier_list),
                str(verdicts_path),
                *JUDGE_ARGUMENTS,
            ]
        ],
        settings["repo_root"],
    )
    return verdicts_path


def strong_record_keys(verdicts_path: Path) -> set[str]:
    """Dossiers the judge failed on, or that were too long to judge, count as not STRONG."""
    records = [cast(JudgedDossier, json.loads(line)) for line in verdicts_path.open()] if verdicts_path.exists() else []
    unjudged = [record["dossier_name"] for record in records if record["status"] != "parsed"]
    if unjudged:
        logger.warning("%d dossiers unjudged: %s", len(unjudged), ", ".join(unjudged))
    return {
        record["dossier_name"].removesuffix(".md")
        for record in records
        if record["status"] == "parsed" and is_strong_hack(record["judgment"]["verdict"])
    }


def run_round(
    units: list[BatchUnit],
    round_index: int,
    state_dir: Path,
    problem_by_example_id: dict[str, PoolProblem],
    settings: RunnerSettings,
    model: RhModelSpecifics,
    sampling: SamplingSettings,
) -> list[EpisodeResult]:
    round_path = round_dir(state_dir, round_index)
    if (round_path / MANIFEST_FILE_NAME).exists():
        manifest = load_manifest(round_path)
        worklist_paths = sorted(round_path.glob("worklist_*.txt"))
    else:
        manifest = RoundManifest(round_index=round_index, run_specs=run_specs(units, problem_by_example_id))
        worklist_paths = write_round_inputs(
            round_path,
            manifest,
            settings["episodes_per_shard"],
            settings["max_runs_per_shard"],
            settings["repo_root"],
            model,
            sampling,
        )
    for _ in range(GENERATION_ATTEMPTS):
        if pending_worklists := incomplete_worklists(round_path, manifest, worklist_paths):
            run_generation(pending_worklists, settings, model)
    episodes = labelled_episodes(round_path, manifest)
    judged_episodes = [episode for episode in episodes if needs_judging(episode)]
    dossier_paths = write_dossiers(round_path, judged_episodes, settings["repo_root"], model.model_id)
    verdicts_path = run_judge(dossier_paths, round_path, settings)
    strong_keys = strong_record_keys(verdicts_path)
    logger.info(
        "round %d: %d episodes, %d cheats, %d judged, %d STRONG",
        round_index,
        len(episodes),
        sum(episode["cheats"] for episode in episodes),
        len(judged_episodes),
        len(strong_keys),
    )
    return [episode_result(episode, strong_keys) for episode in episodes]
