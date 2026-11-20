"""EXPLORATORY (lightly reviewed when merged 2026-11-20): kept for reference, not a validated measurement.

Token-level and episode-level separation of judged hacks by the gi_v2 and gfar projections, per model, arm
(unsteered, gi_v2 -, gfar -; main dose, behaviour_measures.episodes.PRESETS), layer, direction and segment (reasoning,
answer), from axis_acts_extract.py --token-projections caches (<dir>/<model>/<arm dir>__seed<N>.tokens.pt). (CPU)

Token labels: hack = tokens of a HACK episode in its first-considered turn or its hack turn (rubric-v3 judge details);
the other turns of HACK episodes and all of BORDERLINE episodes are left out; tokens of NOT_HACK and unjudged episodes
are non-hack. Token level: per problem and class, histograms over --bins uniform bins whose edges (per layer, direction
and segment) are the token min and max over the model's three arms; from them the pooled ROC curve and AUROC, and the
problem-stratified AUROC (same-problem hack / non-hack token pairs pooled over problems, ties half) with a
problem-bootstrap 95% interval. Bin ties count half, so the AUROCs are exact up to the bin width.
Episode level (whole episodes): per episode and segment, the mean and the 50th, 60th, ..., 100th percentile of each
projection; HACK vs NOT_HACK + unjudged (BORDERLINE left out); problem-stratified AUROC of the mean, median and max with
problem-bootstrap intervals. Writes <out>/<model slug>.pt (token_projection_charts.py reads it).

    python experiments/deceiver-axis/scripts/token_projection_analysis.py --tokens <dir> --out <dir> [--models gptoss muse gemma]
"""

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from selfconcept.deceiver_axis.behaviour_measures.episodes import (ARMS, PRESETS, ArmName, JudgeDetail, ModelPreset,
                                                                   arm_dir_names, hack_turns, load_judge_details,
                                                                   load_verdict_by_key)

SEGMENT_CODES = {"reasoning": 1, "answer": 2}
SEGMENTS = list(SEGMENT_CODES)
EPISODE_QUANTILES = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
EPISODE_STATISTICS = ["mean", *[f"p{round(q * 100)}" for q in EPISODE_QUANTILES]]
AUROC_STATISTICS = {"mean": "mean", "median": "p50", "max": "p100"}
TOKEN_DIR_BY_SLUG = {"gptoss": "gptoss120b", "muse": "muse", "gemma": "gemma4_31b", "qwen": "qwen38_27b"}
TORCH_QUANTILE_LIMIT = 16_000_000
TOKEN_CHUNK = 500_000

type EpisodeLabel = Literal["hack", "not_hack", "borderline"]


def token_files(token_root: Path, preset: ModelPreset) -> dict[ArmName, list[Path]]:
    directory = token_root / TOKEN_DIR_BY_SLUG[preset["slug"]]
    return {arm: sorted(directory.glob(f"{dir_name}__seed*.tokens.pt")) for dir_name, arm in arm_dir_names(preset).items()}


def load_tokens(path: Path) -> dict:
    return torch.load(path, weights_only=False)


def episode_label(verdict: str | None) -> EpisodeLabel:
    return {"HACK": "hack", "BORDERLINE": "borderline"}.get(verdict or "", "not_hack")  # type: ignore[return-value]


def bin_edges(files_by_arm: dict[ArmName, list[Path]]) -> tuple[np.ndarray, np.ndarray, list[str], dict]:
    """Per (segment, layer, direction): token min and max over every file of the model's arms; also the problems seen
    and one file's metadata."""
    low = high = None
    problems: set[str] = set()
    metadata: dict = {}
    for files in files_by_arm.values():
        for path in files:
            tokens = load_tokens(path)
            problems.update(tokens["example_id"])
            metadata = {"record_layers": tokens["record_layers"], "named_directions": tokens["named_directions"]}
            projections = tokens["proj"].float()
            for segment_index, code in enumerate(SEGMENT_CODES.values()):
                selected = projections[torch.as_tensor(tokens["segment"] == code)]
                if not len(selected):
                    continue
                if low is None or high is None:
                    shape = (len(SEGMENTS), *projections.shape[1:])
                    low, high = np.full(shape, np.inf), np.full(shape, -np.inf)
                low[segment_index] = np.minimum(low[segment_index], selected.amin(0).numpy())
                high[segment_index] = np.maximum(high[segment_index], selected.amax(0).numpy())
    assert low is not None and high is not None
    return low, np.where(high > low, high, low + 1e-6), sorted(problems), metadata


def token_classes(tokens: dict, verdict_by_key: dict[str, str], detail_by_key: dict[str, JudgeDetail]) -> np.ndarray:
    """Per token: 1 hack, 0 non-hack, -1 left out."""
    classes = np.zeros(len(tokens["segment"]), np.int8)
    offsets = tokens["episode_offsets"]
    for episode, key in enumerate(tokens["verdict_key"]):
        start, stop = offsets[episode], offsets[episode + 1]
        label = episode_label(verdict_by_key.get(key))
        if label == "borderline":
            classes[start:stop] = -1
        elif label == "hack":
            hack_turn_set = set(hack_turns(detail_by_key[key]))
            classes[start:stop] = np.where(np.isin(tokens["turn"][start:stop], list(hack_turn_set)), 1, -1)
    return classes


def accumulate_histograms(histogram: np.ndarray, tokens: dict, classes: np.ndarray, problem_index: dict[str, int],
                          low: np.ndarray, high: np.ndarray, n_bins: int) -> None:
    """Add one file's tokens to histogram[problem, segment, layer, direction, bin, class] (class 0 non-hack, 1 hack)."""
    offsets = tokens["episode_offsets"]
    token_problem = np.repeat([problem_index[example] for example in tokens["example_id"]], np.diff(offsets))
    projections = tokens["proj"]
    n_problems, _, n_layers, n_directions, _, _ = histogram.shape
    for segment_index, code in enumerate(SEGMENT_CODES.values()):
        keep = (tokens["segment"] == code) & (classes >= 0)
        if not keep.any():
            continue
        kept_positions = np.nonzero(keep)[0]
        cell = (np.arange(n_layers)[:, None] * n_directions + np.arange(n_directions)[None, :])[None]
        counts = np.zeros(n_problems * n_layers * n_directions * n_bins * 2, np.int64)
        for start in range(0, len(kept_positions), TOKEN_CHUNK):
            positions = kept_positions[start:start + TOKEN_CHUNK]
            values = projections[torch.as_tensor(positions)].float().numpy()
            scaled = (values - low[segment_index]) / (high[segment_index] - low[segment_index]) * n_bins
            bins = np.clip(scaled.astype(np.int64), 0, n_bins - 1)
            flat = (((token_problem[positions][:, None, None] * n_layers * n_directions + cell) * n_bins + bins) * 2
                    + classes[positions][:, None, None])
            counts += np.bincount(flat.ravel(), minlength=len(counts))
        histogram[:, segment_index] += counts.reshape(n_problems, n_layers, n_directions, n_bins, 2)


def episode_statistics(tokens: dict) -> np.ndarray:
    """(episodes, segments, statistics, layers, directions): mean and EPISODE_QUANTILES of each projection; NaN when a
    segment has no tokens."""
    offsets = tokens["episode_offsets"]
    projections = tokens["proj"]
    n_episodes = len(offsets) - 1
    statistics = np.full((n_episodes, len(SEGMENTS), len(EPISODE_STATISTICS), *projections.shape[1:]), np.nan,
                         np.float32)
    quantiles = torch.tensor(EPISODE_QUANTILES)
    for episode in range(n_episodes):
        start, stop = offsets[episode], offsets[episode + 1]
        segment = tokens["segment"][start:stop]
        for segment_index, code in enumerate(SEGMENT_CODES.values()):
            values = projections[start:stop][torch.as_tensor(segment == code)].float()
            if not len(values):
                continue
            statistics[episode, segment_index, 0] = values.mean(0).numpy()
            if values.numel() <= TORCH_QUANTILE_LIMIT:
                statistics[episode, segment_index, 1:] = torch.quantile(values, quantiles, dim=0).numpy()
            else:
                statistics[episode, segment_index, 1:] = np.quantile(values.numpy(), EPISODE_QUANTILES, axis=0)
    return statistics


def pairs_from_histograms(hack: np.ndarray, non_hack: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Correctly ranked (hack, non-hack) pairs (ties half) and the number of pairs, over the last (bin) axis."""
    below = np.cumsum(non_hack, axis=-1) - non_hack
    correct = (hack * (below + 0.5 * non_hack)).sum(-1)
    return correct, hack.sum(-1) * non_hack.sum(-1)


def stratified_auroc(correct: np.ndarray, pairs: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Point estimate and 95% bootstrap interval of sum(correct) / sum(pairs) over problems (axis 0); weights
    (draws, problems) are the problem resampling counts."""
    point = correct.sum(0) / np.where(pairs.sum(0) > 0, pairs.sum(0), np.nan)
    flat_correct, flat_pairs = correct.reshape(len(correct), -1), pairs.reshape(len(pairs), -1)
    draws = (weights @ flat_correct) / np.where(weights @ flat_pairs > 0, weights @ flat_pairs, np.nan)
    interval = np.nanpercentile(draws, [2.5, 97.5], axis=0).reshape(2, *correct.shape[1:])
    return point, interval


def episode_pairs(statistic: np.ndarray, labels: list[EpisodeLabel], problems: list[str], problem_index: dict[str, int]
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Per problem: correctly ranked same-problem (hack, non-hack) episode pairs (ties half; NaN scores skipped) and the
    number of pairs; statistic is (episodes, ...)."""
    correct = np.zeros((len(problem_index), *statistic.shape[1:]))
    pairs = np.zeros_like(correct)
    episodes_by_problem: dict[str, list[int]] = defaultdict(list)
    for episode, problem in enumerate(problems):
        episodes_by_problem[problem].append(episode)
    for problem, episodes in episodes_by_problem.items():
        hack = statistic[[episode for episode in episodes if labels[episode] == "hack"]]
        non_hack = statistic[[episode for episode in episodes if labels[episode] == "not_hack"]]
        if not len(hack) or not len(non_hack):
            continue
        higher, lower = hack[:, None], non_hack[None, :]
        valid = ~(np.isnan(higher) | np.isnan(lower))
        correct[problem_index[problem]] = np.where(valid, (higher > lower) + 0.5 * (higher == lower), 0).sum((0, 1))
        pairs[problem_index[problem]] = valid.sum((0, 1))
    return correct, pairs


def analyze_model(preset: ModelPreset, token_root: Path, n_bins: int, n_draws: int, rng: np.random.Generator) -> dict:
    files_by_arm = token_files(token_root, preset)
    verdict_by_key = load_verdict_by_key(preset["verdict_files"])
    detail_by_key = load_judge_details(preset, verdict_by_key)
    low, high, problem_ids, metadata = bin_edges(files_by_arm)
    problem_index = {problem: index for index, problem in enumerate(problem_ids)}
    _, n_layers, n_directions = low.shape
    weights = rng.multinomial(len(problem_ids), [1 / len(problem_ids)] * len(problem_ids), size=n_draws).astype(float)
    result: dict = {"model": preset["label"], "slug": preset["slug"], "record_layers": metadata["record_layers"],
                    "directions": metadata["named_directions"], "segments": SEGMENTS, "episode_statistics": EPISODE_STATISTICS,
                    "bin_low": low, "bin_high": high, "n_bins": n_bins, "problems": problem_ids, "arms": {}}
    for arm, files in files_by_arm.items():
        histogram = np.zeros((len(problem_ids), len(SEGMENTS), n_layers, n_directions, n_bins, 2), np.int64)
        statistics, labels, problems, keys = [], [], [], []
        for path in files:
            tokens = load_tokens(path)
            accumulate_histograms(histogram, tokens, token_classes(tokens, verdict_by_key, detail_by_key), problem_index,
                                  low, high, n_bins)
            statistics.append(episode_statistics(tokens))
            labels += [episode_label(verdict_by_key.get(key)) for key in tokens["verdict_key"]]
            problems += list(tokens["example_id"])
            keys += list(tokens["verdict_key"])
            print(f"{preset['label']} {arm} {path.name}: {len(tokens['segment'])} tokens", flush=True)
        episode_stats = np.concatenate(statistics)
        correct, pairs = pairs_from_histograms(histogram[..., 1], histogram[..., 0])
        token_point, token_interval = stratified_auroc(correct, pairs, weights)
        pooled_correct, pooled_pairs = pairs_from_histograms(histogram[..., 1].sum(0), histogram[..., 0].sum(0))
        episode_auroc = {}
        for name, statistic in AUROC_STATISTICS.items():
            statistic_index = EPISODE_STATISTICS.index(statistic)
            episode_correct, episode_pair_counts = episode_pairs(episode_stats[:, :, statistic_index], labels, problems,
                                                                 problem_index)
            episode_auroc[name] = stratified_auroc(episode_correct, episode_pair_counts, weights)
        result["arms"][arm] = {
            "histogram": histogram.sum(0), "token_auroc_pooled": pooled_correct / np.maximum(pooled_pairs, 1),
            "token_auroc_stratified": token_point, "token_auroc_stratified_ci": token_interval,
            "token_counts": histogram.sum((0, 4)), "episode_auroc_stratified": episode_auroc,
            "episode_stats": episode_stats, "episode_labels": labels, "episode_problems": problems, "episode_keys": keys}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tokens", type=Path, required=True, help="directory with <model>/<arm dir>__seed<N>.tokens.pt")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=["gptoss", "muse", "gemma"])
    parser.add_argument("--bins", type=int, default=2048)
    parser.add_argument("--n-draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    for preset in PRESETS:
        if preset["slug"] in args.models:
            torch.save(analyze_model(preset, args.tokens, args.bins, args.n_draws, rng), args.out / f"{preset['slug']}.pt")


if __name__ == "__main__":
    main()
