"""EXPLORATORY (lightly reviewed when merged 2026-11-20): kept for reference, not a validated measurement.

Episode-level token-frequency shifts of gi_v2 - and gfar - against unsteered and against each other, per model
(behaviour_measures.episodes.PRESETS, main dose), in the reasoning (all turns) and the visible replies (all turns), each
tokenized with the model's own tokenizer. Generalizes the Gemma random-1729 token_episode_level.py analysis. (CPU)

Each episode is one unit: per widespread token (in >= --min-episodes episodes and >= --min-problems problems across
the three arms), episode rate = count / episode length; arm statistic = mean over problems of the per-problem mean
episode rate; log2 ratio of the two arms with a problem-bootstrap CI (the same resampled problems for every arm).
Contrasts: gi_v2 / unsteered, gfar / unsteered, gi_v2 / gfar. Writes <out>/<model>_<region>.tsv (every widespread
token) and <out>/report.txt: per contrast the number of tokens whose CI excludes 0 (and the number expected by chance),
and the tokens whose gi_v2 / gfar 99% CI excludes 0, largest shifts first.

    python experiments/deceiver-axis/scripts/token_shift.py --out <dir> --workers 16
"""

import argparse
import json
from collections import Counter
from multiprocessing import Pool
from pathlib import Path
from typing import Literal, TypedDict

import numpy as np
from byutils import load_tokenizer
from jaxtyping import Float

from selfconcept.deceiver_axis.behaviour_measures.episodes import (ARMS, MODEL_ID_BY_SLUG, PRESETS, ArmName,
                                                                   main_dose_arm_dirs)

type Region = Literal["reasoning", "reply"]
type Contrast = tuple[ArmName, ArmName]

REGIONS: list[Region] = ["reasoning", "reply"]
CONTRASTS: list[Contrast] = [("gi_v2", "unsteered"), ("gfar", "unsteered"), ("gi_v2", "gfar")]
EPSILON = 1e-9
N_LISTED = 25

_tokenizer_by_model_id = {}


class EpisodeTokens(TypedDict):
    problem: str
    reasoning: Counter
    reply: Counter


def seed_dir_tokens(task: tuple[str, str, str]) -> list[EpisodeTokens]:
    """Token counts of every episode in one seed directory."""
    model_id, seed_dir, record_file = task
    if model_id not in _tokenizer_by_model_id:
        _tokenizer_by_model_id[model_id] = load_tokenizer(model_id)
    tokenizer = _tokenizer_by_model_id[model_id]
    reasoning_by_problem_turn: dict[str, dict[int, str]] = {}
    for path in sorted(Path(seed_dir).glob("*_reasoning.jsonl")):
        for row in map(json.loads, path.read_text().splitlines()):
            reasoning_by_problem_turn.setdefault(row["example_id"], {})[row["turn"]] = row["reasoning"] or ""
    episodes: list[EpisodeTokens] = []
    for record in map(json.loads, (Path(seed_dir) / record_file).read_text().splitlines()):
        turns = reasoning_by_problem_turn.get(record["example_id"], {})
        reasoning = "\n".join(turns[turn] for turn in sorted(turns))
        reply = "\n".join(attempt["completion"] or "" for attempt in record["attempts"])
        episodes.append({"problem": record["example_id"],
                         "reasoning": Counter(tokenizer.encode(reasoning, add_special_tokens=False)),
                         "reply": Counter(tokenizer.encode(reply, add_special_tokens=False))})
    return episodes


def problem_mean_rates(episodes: list[EpisodeTokens], region: Region, problem_index: dict[str, int],
                       token_column: dict[int, int]) -> Float[np.ndarray, "problems tokens"]:
    rate_sum = np.zeros((len(problem_index), len(token_column)))
    episode_count = np.zeros(len(problem_index))
    for episode in episodes:
        if episode["problem"] not in problem_index:
            continue
        row = problem_index[episode["problem"]]
        episode_count[row] += 1
        length = sum(episode[region].values())
        for token, count in episode[region].items():
            if length and token in token_column:
                rate_sum[row, token_column[token]] += count / length
    return rate_sum / np.maximum(episode_count, 1)[:, None]


def region_report(model_label: str, slug: str, region: Region, episodes_by_arm: dict[ArmName, list[EpisodeTokens]],
                  args: argparse.Namespace, tokenizer) -> list[str]:
    problems = sorted(set.intersection(*({episode["problem"] for episode in episodes_by_arm[arm]} for arm in ARMS)))
    problem_index = {problem: index for index, problem in enumerate(problems)}
    episodes_with_token: Counter = Counter()
    problems_with_token: dict[int, set[str]] = {}
    for arm in ARMS:
        for episode in episodes_by_arm[arm]:
            episodes_with_token.update(episode[region].keys())
            for token in episode[region]:
                problems_with_token.setdefault(token, set()).add(episode["problem"])
    tokens = sorted(token for token, n_episodes in episodes_with_token.items()
                    if n_episodes >= args.min_episodes and len(problems_with_token[token]) >= args.min_problems)
    token_column = {token: column for column, token in enumerate(tokens)}
    rates_by_arm = {arm: problem_mean_rates(episodes_by_arm[arm], region, problem_index, token_column) for arm in ARMS}

    weights = np.random.default_rng(args.seed).multinomial(len(problems), [1 / len(problems)] * len(problems),
                                                           size=args.n_draws).astype(np.float64)

    def log2_ratio(contrast: Contrast, problem_weights: Float[np.ndarray, "draws problems"]) -> np.ndarray:
        numerator, denominator = contrast
        return np.log2((problem_weights @ rates_by_arm[numerator] + EPSILON)
                       / (problem_weights @ rates_by_arm[denominator] + EPSILON))

    point_by_contrast = {contrast: log2_ratio(contrast, np.ones((1, len(problems))))[0] for contrast in CONTRASTS}
    draws_by_contrast = {contrast: log2_ratio(contrast, weights) for contrast in CONTRASTS}
    bounds_by_level = {level: {contrast: np.percentile(draws, [(100 - level) / 2, 100 - (100 - level) / 2], axis=0)
                               for contrast, draws in draws_by_contrast.items()} for level in (95, 99)}

    def contrast_name(contrast: Contrast) -> str:
        return f"{contrast[0]}/{contrast[1]}"

    def token_text(token: int) -> str:
        return json.dumps(tokenizer.decode([token]))

    with open(args.out / f"{slug}_{region}.tsv", "w") as handle:
        handle.write("token\tepisodes\tproblems\t" + "\t".join(
            f"log2_{contrast_name(c)}\tlow95_{contrast_name(c)}\thigh95_{contrast_name(c)}" for c in CONTRASTS) + "\n")
        for column, token in enumerate(tokens):
            handle.write(f"{token_text(token)}\t{episodes_with_token[token]}\t{len(problems_with_token[token])}\t"
                         + "\t".join(f"{point_by_contrast[c][column]:.3f}\t{bounds_by_level[95][c][0][column]:.3f}\t"
                                     f"{bounds_by_level[95][c][1][column]:.3f}" for c in CONTRASTS) + "\n")

    n_tokens_by_arm = {arm: sum(sum(episode[region].values()) for episode in episodes_by_arm[arm]) for arm in ARMS}
    lines = [f"\n===== {model_label} {region}: {len(problems)} problems, {len(tokens)} widespread tokens; tokens per arm "
             + ", ".join(f"{arm} {n / 1e6:.1f}M" for arm, n in n_tokens_by_arm.items())]
    for level in (95, 99):
        excluded = {c: (bounds_by_level[level][c][0] > 0) | (bounds_by_level[level][c][1] < 0) for c in CONTRASTS}
        lines.append(f"-- {level}% CIs: tokens whose CI excludes 0: "
                     + ", ".join(f"{contrast_name(c)} {int(excluded[c].sum())}" for c in CONTRASTS)
                     + f"  (chance ~{len(tokens) * (100 - level) / 100:.0f})")
    gap = ("gi_v2", "gfar")
    low, high = bounds_by_level[99][gap]
    for label, chosen in (("more in gi_v2 - than gfar -", np.where(low > 0)[0]),
                          ("more in gfar - than gi_v2 -", np.where(high < 0)[0])):
        ranked = sorted(chosen, key=lambda column: -abs(point_by_contrast[gap][column]))[:N_LISTED]
        lines.append(f"-- {label} at 99% ({len(chosen)} tokens; largest {len(ranked)}; log2 ratios)")
        lines.extend(f"   {token_text(tokens[column])[:20]:20s} eps {episodes_with_token[tokens[column]]:5d} | "
                     + "  ".join(f"{contrast_name(c)} {point_by_contrast[c][column]:+.2f}" for c in CONTRASTS)
                     for column in ranked)
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--min-episodes", type=int, default=100)
    parser.add_argument("--min-problems", type=int, default=20)
    parser.add_argument("--n-draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    lines = ["Episode-level token-frequency shifts of gi_v2 - and gfar - (log2 ratios, problem-bootstrap CIs)"]
    for preset in PRESETS:
        model_id = MODEL_ID_BY_SLUG[preset["slug"]]
        tasks: list[tuple[ArmName, tuple[str, str, str]]] = [
            (arm, (model_id, str(seed_dir), preset["record_file"]))
            for arm, arm_dir in main_dose_arm_dirs(preset).items() for seed_dir in sorted(arm_dir.glob("seed*"))]
        with Pool(args.workers) as pool:
            results = pool.map(seed_dir_tokens, [task for _, task in tasks])
        episodes_by_arm: dict[ArmName, list[EpisodeTokens]] = {arm: [] for arm in ARMS}
        for (arm, _), episodes in zip(tasks, results):
            episodes_by_arm[arm].extend(episodes)
        tokenizer = load_tokenizer(model_id)
        for region in REGIONS:
            lines.extend(region_report(preset["label"], preset["slug"], region, episodes_by_arm, args, tokenizer))
        (args.out / "report.txt").write_text("\n".join(lines) + "\n")
        print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
