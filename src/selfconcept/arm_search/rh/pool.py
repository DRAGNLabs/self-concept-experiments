"""The problem pool: every problem of the ImpossibleBench split with a hidden suite, EvilGenie and LCB-Pro, as written
by `python -m selfconcept.codebench.convert` (run it once first). A search narrows it with its config's `example_ids`
(e.g. to the problems a model's calibration round kept).
"""

import json
from typing import Final, Literal, TypedDict, get_args

from selfconcept.arm_search.rh.conditions import TaskFormat
from selfconcept.common.paths import benchmark_dir

PoolScenario = Literal["impossible_original", "evilgenie", "evilgenie_lcbpro"]
Benchmark = Literal["impossiblebench", "evilgenie", "lcbpro"]

DATA_DIR: Final = benchmark_dir("codebench") / "data"
DATA_FILE_NAME_BY_SCENARIO: Final[dict[PoolScenario, str]] = {
    "impossible_original": "impossible_original_heldout.jsonl",
    "evilgenie": "evilgenie.jsonl",
    "evilgenie_lcbpro": "evilgenie_lcbpro.jsonl",
}
BENCHMARK_BY_SCENARIO: Final[dict[PoolScenario, Benchmark]] = {
    "impossible_original": "impossiblebench",
    "evilgenie": "evilgenie",
    "evilgenie_lcbpro": "lcbpro",
}
BENCHMARKS: Final[list[Benchmark]] = list(get_args(Benchmark))


class PoolProblem(TypedDict):
    example_id: str
    scenario: PoolScenario
    benchmark: Benchmark
    task_format: TaskFormat


def load_pool_rows(scenario: PoolScenario) -> list[dict]:
    return [json.loads(line) for line in (DATA_DIR / DATA_FILE_NAME_BY_SCENARIO[scenario]).open()]


def load_pool() -> list[PoolProblem]:
    return [
        PoolProblem(
            example_id=row["example_id"],
            scenario=scenario,
            benchmark=BENCHMARK_BY_SCENARIO[scenario],
            task_format=row["kind"],
        )
        for scenario in get_args(PoolScenario)
        for row in load_pool_rows(scenario)
    ]

