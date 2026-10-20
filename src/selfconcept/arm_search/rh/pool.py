"""The problem pool: every problem of the ImpossibleBench split with a hidden suite, EvilGenie and LCB-Pro, as written
by `python -m selfconcept.codebench.convert` (run it once first). A search can narrow it with its config's
`example_ids`, and split each benchmark's problems into two origins by a calibration round's signal.
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
NO_SIGNAL_SUFFIX: Final = "_no_signal"
SIGNAL_SPLIT_ORIGINS: Final[list[str]] = [*BENCHMARKS, *(f"{benchmark}{NO_SIGNAL_SUFFIX}" for benchmark in BENCHMARKS)]


class PoolProblem(TypedDict):
    """`origin` is the benchmark, or `<benchmark>_no_signal` for a problem without calibration signal."""

    example_id: str
    scenario: PoolScenario
    benchmark: Benchmark
    origin: str
    task_format: TaskFormat


def pool_origins(problems: list[PoolProblem]) -> list[str]:
    """The ladder model's origins in prior order: the benchmarks, then their no-signal halves if the pool is split."""
    return SIGNAL_SPLIT_ORIGINS if any(problem["origin"] not in BENCHMARKS for problem in problems) else [*BENCHMARKS]


def load_pool_rows(scenario: PoolScenario) -> list[dict]:
    return [json.loads(line) for line in (DATA_DIR / DATA_FILE_NAME_BY_SCENARIO[scenario]).open()]


def load_pool() -> list[PoolProblem]:
    return [
        PoolProblem(
            example_id=row["example_id"],
            scenario=scenario,
            benchmark=BENCHMARK_BY_SCENARIO[scenario],
            origin=BENCHMARK_BY_SCENARIO[scenario],
            task_format=row["kind"],
        )
        for scenario in get_args(PoolScenario)
        for row in load_pool_rows(scenario)
    ]

