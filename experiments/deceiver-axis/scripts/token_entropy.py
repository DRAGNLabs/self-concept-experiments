"""Token entropy of gi_v2 - and gfar - against unsteered, per model (behaviour_measures.episodes.PRESETS, main dose), each
text tokenized with the model's own tokenizer. (CPU)

Per episode, the unigram entropy (bits) of the token mix in a fixed-length window, so length differences between arms
do not drive it: the first --reasoning-window tokens of turn-0 reasoning and the first --reply-window tokens of the
turn-0 visible reply (episodes shorter than the window are left out of that measure and counted). Also the entropy of
all reasoning tokens of the episode, with the mean episode length beside it (length-confounded: longer text has
higher unigram entropy). Arm statistic = mean over problems of the per-problem mean; change vs unsteered and the
gi_v2 - gfar gap with problem-bootstrap 95% intervals (the same draws for every arm). Writes <out>.json and prints a
table.

    python experiments/deceiver-axis/scripts/token_entropy.py --out <stem> --workers 16
"""

import argparse
import json
import math
from collections import Counter
from multiprocessing import Pool
from pathlib import Path
from typing import Literal, TypedDict

import numpy as np
from byutils import load_tokenizer

from selfconcept.deceiver_axis.behaviour_measures.episodes import (ARMS, MODEL_ID_BY_SLUG, PRESETS, ArmName,
                                                                   ModelPreset, main_dose_arm_dirs)
from selfconcept.deceiver_axis.behaviour_measures.turn_lengths import (MEASURES, Measure, MeasureSummary, SteeredValue,
                                                                       SummaryByModel)

_tokenizer_by_model_id = {}


class EpisodeEntropy(TypedDict):
    problem: str
    reasoning_window: float | None
    reply_window: float | None
    reasoning_all: float | None
    reasoning_tokens: float


def entropy_bits(token_ids: list[int]) -> float:
    counts = Counter(token_ids)
    return -sum(count / len(token_ids) * math.log2(count / len(token_ids)) for count in counts.values())


def window_entropy(token_ids: list[int], window: int) -> float | None:
    return entropy_bits(token_ids[:window]) if len(token_ids) >= window else None


def seed_dir_entropies(task: tuple[str, str, str, int, int]) -> list[EpisodeEntropy]:
    model_id, seed_dir, record_file, reasoning_window, reply_window = task
    if model_id not in _tokenizer_by_model_id:
        _tokenizer_by_model_id[model_id] = load_tokenizer(model_id)
    tokenizer = _tokenizer_by_model_id[model_id]
    reasoning_by_problem_turn: dict[str, dict[int, str]] = {}
    for path in sorted(Path(seed_dir).glob("*_reasoning.jsonl")):
        for row in map(json.loads, path.read_text().splitlines()):
            reasoning_by_problem_turn.setdefault(row["example_id"], {})[row["turn"]] = row["reasoning"] or ""
    entropies: list[EpisodeEntropy] = []
    for record in map(json.loads, (Path(seed_dir) / record_file).read_text().splitlines()):
        turns = reasoning_by_problem_turn.get(record["example_id"], {})
        reasoning_ids_by_turn = {turn: tokenizer.encode(text, add_special_tokens=False) for turn, text in turns.items()}
        all_reasoning_ids = [token for turn in sorted(reasoning_ids_by_turn) for token in reasoning_ids_by_turn[turn]]
        reply_ids = tokenizer.encode(record["attempts"][0]["completion"] or "", add_special_tokens=False)
        entropies.append({"problem": record["example_id"],
                          "reasoning_window": window_entropy(reasoning_ids_by_turn.get(0, []), reasoning_window),
                          "reply_window": window_entropy(reply_ids, reply_window),
                          "reasoning_all": entropy_bits(all_reasoning_ids) if all_reasoning_ids else None,
                          "reasoning_tokens": float(len(all_reasoning_ids))})
    return entropies


def problem_mean_draws(entropies: list[EpisodeEntropy], measure: Measure, problem_ids: list[str],
                       resampled: np.ndarray) -> tuple[float, np.ndarray, int]:
    """Mean over problems of the per-problem mean, its value in each bootstrap draw, and the number of episodes."""
    values_by_problem: dict[str, list[float]] = {}
    for entropy in entropies:
        value = entropy[measure]
        if value is not None:
            values_by_problem.setdefault(entropy["problem"], []).append(value)
    problem_means = np.array([np.mean(values_by_problem[problem]) if problem in values_by_problem else np.nan
                              for problem in problem_ids])
    return (float(np.nanmean(problem_means)), np.nanmean(problem_means[resampled], axis=1),
            sum(map(len, values_by_problem.values())))


def model_entropies(preset: ModelPreset, workers: int, reasoning_window: int, reply_window: int
                    ) -> dict[ArmName, list[EpisodeEntropy]]:
    model_id = MODEL_ID_BY_SLUG[preset["slug"]]
    tasks: list[tuple[ArmName, tuple[str, str, str, int, int]]] = [
        (arm, (model_id, str(seed_dir), preset["record_file"], reasoning_window, reply_window))
        for arm, arm_dir in main_dose_arm_dirs(preset).items() for seed_dir in sorted(arm_dir.glob("seed*"))]
    with Pool(workers) as pool:
        results = pool.map(seed_dir_entropies, [task for _, task in tasks])
    entropies_by_arm: dict[ArmName, list[EpisodeEntropy]] = {arm: [] for arm in ARMS}
    for (arm, _), entropies in zip(tasks, results):
        entropies_by_arm[arm].extend(entropies)
    return entropies_by_arm


def interval(draws: np.ndarray) -> list[float]:
    return [float(bound) for bound in np.nanpercentile(draws, [2.5, 97.5])]


def measure_summary(entropies_by_arm: dict[ArmName, list[EpisodeEntropy]], measure: Measure, problem_ids: list[str],
                    resampled: np.ndarray) -> MeasureSummary:
    """Unsteered value, each steered arm's value and change with its interval, and the gi_v2 - gfar gap."""
    stats = {arm: problem_mean_draws(entropies_by_arm[arm], measure, problem_ids, resampled) for arm in ARMS}
    changes = {arm: stats[arm][1] - stats["unsteered"][1] for arm in ("gi_v2", "gfar")}
    def steered(arm: Literal["gi_v2", "gfar"]) -> SteeredValue:
        return {"value": stats[arm][0], "change": stats[arm][0] - stats["unsteered"][0], "ci95": interval(changes[arm])}

    return {"unsteered": stats["unsteered"][0], "gi_v2": steered("gi_v2"), "gfar": steered("gfar"),
            "gap": {"value": (stats["gi_v2"][0] - stats["gfar"][0]), "ci95": interval(changes["gi_v2"] - changes["gfar"])},
            "n_episodes": {arm: stats[arm][2] for arm in ARMS}}


def summary_line(model: str, measure: Measure, row: MeasureSummary) -> str:
    return (f"{model:13s} {measure:17s} unsteered {row['unsteered']:8.3f} | "
            + " | ".join(f"{arm} {row[arm]['change']:+.3f} [{row[arm]['ci95'][0]:+.3f}, "
                         f"{row[arm]['ci95'][1]:+.3f}]" for arm in ("gi_v2", "gfar"))
            + f" | gap {row['gap']['value']:+.3f} [{row['gap']['ci95'][0]:+.3f}, {row['gap']['ci95'][1]:+.3f}]"
            + f" | n {row['n_episodes']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--reasoning-window", type=int, default=1000)
    parser.add_argument("--reply-window", type=int, default=200)
    parser.add_argument("--n-draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    summary_by_model: SummaryByModel = {}
    for preset in PRESETS:
        entropies_by_arm = model_entropies(preset, args.workers, args.reasoning_window, args.reply_window)
        problem_ids = sorted({entropy["problem"] for entropies in entropies_by_arm.values() for entropy in entropies})
        resampled = rng.integers(len(problem_ids), size=(args.n_draws, len(problem_ids)))
        summary_by_model[preset["label"]] = {measure: measure_summary(entropies_by_arm, measure, problem_ids, resampled)
                                             for measure in MEASURES}
        for measure, row in summary_by_model[preset["label"]].items():
            print(summary_line(preset["label"], measure, row), flush=True)
        args.out.with_suffix(".json").write_text(json.dumps(summary_by_model, indent=1))


if __name__ == "__main__":
    main()
