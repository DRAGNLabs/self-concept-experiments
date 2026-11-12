"""Capping vs negative steering along gi_v2, per model: problem-stratified judged hack rate (Qwen: hidden-test fail,
not judged), hidden pass (the capability measure, 2026-11-09 amendment) and, secondary, hidden pass among honest
episodes (episodes not judged HACK; BORDERLINE and unjudged count as honest, as in the analyses). Honest hidden pass is
the stratified ratio mean_p(honest passes_p) / mean_p(honest episodes_p), with per-problem shares over seeds. Differences vs unsteered and cap minus steering, with 95% CIs from a paired
bootstrap over problems. Runs per model come from a setups JSON (data/cap_vs_steer_setups.json): arm directories, the
episode-file tag, the verdicts file (null = hack is the hidden-test-fail proxy), and which arms are caps and which are
negative steering doses.

    python scripts/cap_vs_steer.py --setups data/cap_vs_steer_setups.json --out <cap_vs_steer.json>
"""
import argparse
import collections
import json
from pathlib import Path
from typing import Callable, TypedDict

import numpy as np
from jaxtyping import Float

BOOTSTRAPS = 10_000
IsHack = Callable[[str, dict], bool]


class ModelSetup(TypedDict):
    directories_by_arm: dict[str, Path]
    tag: str
    is_hack: IsHack
    caps: list[str]
    steered: list[str]


class ModelSetupFile(TypedDict):
    directories_by_arm: dict[str, str]
    tag: str
    verdicts: str | None
    caps: list[str]
    steered: list[str]


class Interval(TypedDict):
    estimate: float
    low: float
    high: float


class ProblemShares(TypedDict):
    hack: Float[np.ndarray, "problems"]
    hidden_pass: Float[np.ndarray, "problems"]
    honest: Float[np.ndarray, "problems"]
    honest_pass: Float[np.ndarray, "problems"]


def judged(verdicts_path: Path) -> IsHack:
    verdicts_by_key = json.loads(verdicts_path.read_text())
    return lambda key, record: verdicts_by_key.get(key) == "HACK"


def proxy(key: str, record: dict) -> bool:
    return bool(record["passed"] and not record["passed_original"])


def problem_shares(arm: str, directory: Path, tag: str, is_hack: IsHack, problems: list[str]) -> ProblemShares:
    outcomes_by_problem = collections.defaultdict(list)
    for path in sorted(directory.glob(f"seed*/{tag}_impossible_original.jsonl")):
        for record in map(json.loads, path.open()):
            hack = is_hack(f"{arm}/{path.parent.name}/{record['example_id']}", record)
            hidden_pass = bool(record["passed_original"])
            outcomes_by_problem[record["example_id"]].append((hack, hidden_pass, not hack, not hack and hidden_pass))
    shares = np.array([np.mean(outcomes_by_problem[p], axis=0) for p in problems])
    return {"hack": shares[:, 0], "hidden_pass": shares[:, 1], "honest": shares[:, 2], "honest_pass": shares[:, 3]}


def metrics(shares: ProblemShares, picks: np.ndarray) -> dict[str, np.ndarray]:
    return {"hack": shares["hack"][picks].mean(-1), "hidden_pass": shares["hidden_pass"][picks].mean(-1),
            "honest_hidden_pass": shares["honest_pass"][picks].mean(-1) / shares["honest"][picks].mean(-1)}


def interval(draws: np.ndarray, point: float) -> Interval:
    return {"estimate": round(100 * point, 1), "low": round(100 * float(np.percentile(draws, 2.5)), 1), "high": round(100 * float(np.percentile(draws, 97.5)), 1)}


def compare(setup: ModelSetup) -> dict:
    directories_by_arm, tag = setup["directories_by_arm"], setup["tag"]
    problems = sorted({json.loads(line)["example_id"] for line in next(directories_by_arm["unsteered"].glob(f"seed*/{tag}_impossible_original.jsonl")).open()})
    shares_by_arm = {arm: problem_shares(arm, directory, tag, setup["is_hack"], problems) for arm, directory in directories_by_arm.items()}
    every = np.arange(len(problems))[None, :]
    picks = np.random.default_rng(0).integers(0, len(problems), (BOOTSTRAPS, len(problems)))
    point = {arm: {name: float(value[0]) for name, value in metrics(shares, every).items()} for arm, shares in shares_by_arm.items()}
    draws = {arm: metrics(shares, picks) for arm, shares in shares_by_arm.items()}

    def difference(first: str, second: str) -> dict[str, Interval]:
        return {name: interval(draws[first][name] - draws[second][name], point[first][name] - point[second][name]) for name in point[first]}
    return {
        "problems": len(problems),
        "levels": {arm: {name: round(100 * value, 1) for name, value in values.items()} for arm, values in point.items()},
        "vs_unsteered": {arm: difference(arm, "unsteered") for arm in directories_by_arm if arm != "unsteered"},
        "cap_minus_steered": {f"{cap} - {arm}": difference(cap, arm) for cap in setup["caps"] for arm in setup["steered"]},
    }


def model_setup(entry: ModelSetupFile) -> ModelSetup:
    return {"directories_by_arm": {arm: Path(directory) for arm, directory in entry["directories_by_arm"].items()}, "tag": entry["tag"],
            "is_hack": judged(Path(entry["verdicts"])) if entry["verdicts"] is not None else proxy,
            "caps": entry["caps"], "steered": entry["steered"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--setups", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    entries: dict[str, ModelSetupFile] = json.loads(args.setups.read_text())
    args.out.write_text(json.dumps({model: compare(model_setup(entry)) for model, entry in entries.items()}, indent=1))


if __name__ == "__main__":
    main()
