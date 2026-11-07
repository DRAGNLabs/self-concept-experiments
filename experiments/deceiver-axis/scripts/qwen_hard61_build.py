"""qwen-hard: check-format problems Qwen3.8-27B passed hidden tests on at most 6 of 8 unsteered seeds.

Reports how much dropping truncation-heavy problems (>= 2 or >= 4 of 8 episodes with a truncated attempt) changes the
set, and writes qwen_hard.jsonl (no filter) plus one file per truncation threshold and the no-code cutoff to the output
directory. qwen_hard61 (the Qwen gi_v2 steering problem set, built 2026-11-03) is qwen_hard_nocode_lt4.jsonl: Koby
chose the no-code cutoff (< 4/8) and no truncation filter. Reads Koby's scratch unsteered runs and the shared data.

    python scripts/qwen_hard61_build.py <out_dir>
"""
import glob
import json
import os
import sys
from collections import Counter, defaultdict
from typing import Literal, TypedDict, cast

from selfconcept.codebench.harness import CheckExample, CheckRecord

SCHEMER = os.path.expanduser("~/nobackup/autodelete/self-concept-experiments/schemer-axis")
QWEN_RUNS = f"{SCHEMER}/qwen38_27b/rh"
MIRROR_DATA = "/grphome/grp_mirror/nobackup/autodelete/self-concept-experiments/rh_hillclimb/data"
Pool = Literal["hard45", "solveasy", "lcbpro_check", "evilgenie_check"]
PROBLEM_FILES_BY_POOL: dict[Pool, list[str]] = {
    "hard45": [f"{MIRROR_DATA}/ib_solvhard_s{shard}.jsonl" for shard in range(3)],
    "solveasy": [f"{MIRROR_DATA}/ib_solveasy_s{shard}.jsonl" for shard in range(3)],
    "lcbpro_check": [f"{SCHEMER}/lcbpro_check_data/lcbpro_check.jsonl"],
    "evilgenie_check": [f"{SCHEMER}/evilgenie_check_data/evilgenie_check.jsonl"],
}
SEED_DIR_GLOBS_BY_POOL: dict[Pool, list[str]] = {
    "hard45": ["unsteered/seed[0-7]"],
    "solveasy": ["unsteered/seed[0-3]", "unsteered53/seed[4-7]", "unsteered5/seed[4-7]"],
    "lcbpro_check": ["unsteered/seed[0-7]"],
    "evilgenie_check": ["unsteered/seed[0-7]"],
}
MAX_HIDDEN_PASSES = 6
SEEDS = 8
TRUNCATION_THRESHOLDS = (2, 4)
NO_CODE_CUTOFF = 4  # Koby 2026-11-03: drop problems with >= 4/8 no-code episodes


class ProblemStats(TypedDict):
    pool: Pool
    episodes: int
    hidden_passes: int
    truncated_episodes: int
    proxy_episodes: int
    no_code_episodes: int


def episode_records(pool: Pool) -> list[CheckRecord]:
    paths = [path for seed_glob in SEED_DIR_GLOBS_BY_POOL[pool]
             for path in sorted(glob.glob(f"{QWEN_RUNS}/{pool}/{seed_glob}/*_impossible_original.jsonl"))]
    return [cast(CheckRecord, json.loads(line)) for path in paths for line in open(path)]


def problem_stats_by_id() -> dict[str, ProblemStats]:
    stats_by_id: dict[str, ProblemStats] = {}
    for pool in PROBLEM_FILES_BY_POOL:
        for record in episode_records(pool):
            stats = stats_by_id.setdefault(record["example_id"], ProblemStats(
                pool=pool, episodes=0, hidden_passes=0, truncated_episodes=0, proxy_episodes=0, no_code_episodes=0))
            stats["episodes"] += 1
            stats["hidden_passes"] += bool(record["passed_original"])
            stats["truncated_episodes"] += any(attempt["truncated"] for attempt in record["attempts"])
            stats["proxy_episodes"] += bool(record["passed"]) and not record["passed_original"]
            stats["no_code_episodes"] += not record.get("final_code")
    return stats_by_id


def problems_by_id() -> dict[str, CheckExample]:
    problems = (json.loads(line) for paths in PROBLEM_FILES_BY_POOL.values() for path in paths for line in open(path))
    return {problem["example_id"]: cast(CheckExample, problem) for problem in problems if problem.get("kind", "check") == "check"}


def summary_line(label: str, ids: list[str], stats_by_id: dict[str, ProblemStats]) -> str:
    count_by_pool = Counter(stats_by_id[i]["pool"] for i in ids)
    episodes = sum(stats_by_id[i]["episodes"] for i in ids)
    proxy = sum(stats_by_id[i]["proxy_episodes"] for i in ids)
    truncated = sum(stats_by_id[i]["truncated_episodes"] for i in ids)
    no_code = sum(stats_by_id[i]["no_code_episodes"] for i in ids)
    hidden = sum(stats_by_id[i]["hidden_passes"] for i in ids)
    pools = ", ".join(f"{pool} {count_by_pool[pool]}" for pool in PROBLEM_FILES_BY_POOL)
    return (f"| {label} | {len(ids)} | {pools} | {hidden / episodes:.1%} | {truncated / episodes:.1%} | "
            f"{no_code / episodes:.1%} | {proxy} ({proxy / episodes:.1%}) |")


def main() -> None:
    out_dir = sys.argv[1]
    os.makedirs(out_dir, exist_ok=True)
    stats_by_id = problem_stats_by_id()
    problem_by_id = problems_by_id()
    short = sorted(i for i, s in stats_by_id.items() if s["episodes"] != SEEDS)
    if short:
        raise SystemExit(f"{len(short)} problems without {SEEDS} episodes: {[(i, stats_by_id[i]['episodes']) for i in short[:10]]}")
    hard = sorted(i for i, s in stats_by_id.items() if s["hidden_passes"] <= MAX_HIDDEN_PASSES and i in problem_by_id)
    print(f"{len(stats_by_id)} problems with {SEEDS} seeds; qwen-hard (hidden pass <= {MAX_HIDDEN_PASSES}/{SEEDS}) = {len(hard)}\n")
    print("| set | problems | by pool | hidden pass | episodes truncated | no-code | proxy episodes |")
    print("|---|---|---|---|---|---|---|")
    print(summary_line("qwen-hard", hard, stats_by_id))
    ids_by_label = {"qwen_hard": hard}
    for threshold in TRUNCATION_THRESHOLDS:
        kept = [i for i in hard if stats_by_id[i]["truncated_episodes"] < threshold]
        dropped = [i for i in hard if stats_by_id[i]["truncated_episodes"] >= threshold]
        print(summary_line(f"drop >= {threshold}/8 truncated", kept, stats_by_id))
        if dropped:
            print(summary_line(f"(the dropped {len(dropped)})", dropped, stats_by_id))
        ids_by_label[f"qwen_hard_trunc_lt{threshold}"] = kept
    no_code_kept = [i for i in hard if stats_by_id[i]["no_code_episodes"] < NO_CODE_CUTOFF]
    print(summary_line(f"drop >= {NO_CODE_CUTOFF}/8 no-code", no_code_kept, stats_by_id))
    ids_by_label[f"qwen_hard_nocode_lt{NO_CODE_CUTOFF}"] = no_code_kept
    histogram = Counter(stats_by_id[i]["truncated_episodes"] for i in hard)
    print("\ntruncated episodes per problem (of 8):", dict(sorted(histogram.items())))
    hidden_histogram = Counter(stats_by_id[i]["hidden_passes"] for i in stats_by_id)
    print("hidden passes per problem, all pools (of 8):", dict(sorted(hidden_histogram.items())))
    for label, ids in ids_by_label.items():
        with open(f"{out_dir}/{label}.jsonl", "w") as handle:
            handle.writelines(json.dumps(problem_by_id[i]) + "\n" for i in ids)
    json.dump({i: stats_by_id[i] for i in stats_by_id}, open(f"{out_dir}/problem_stats.json", "w"), indent=1)


if __name__ == "__main__":
    main()
