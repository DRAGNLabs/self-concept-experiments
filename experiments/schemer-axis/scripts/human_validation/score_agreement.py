"""Agreement between human labelers and the LLM judge on a human-validation stage. Headline numbers: mean pairwise
human-human and mean LLM-judge-human quadratic-weighted Cohen's kappa, with 95% percentile CIs from a bootstrap over
items. Only items every labeler labeled are scored.

    python score_agreement.py --key key.jsonl --labels labels_a.jsonl labels_b.jsonl --levels 0,1,2,3
"""
import argparse
import itertools
import json
from pathlib import Path
from typing import TypedDict, cast

import numpy as np
from jaxtyping import Int
from sklearn.metrics import cohen_kappa_score, confusion_matrix


class KeyRecord(TypedDict):
    item_id: str
    judge_label: str
    model: str


class KappaEstimate(TypedDict):
    mean: float
    ci_low: float
    ci_high: float


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def quadratic_kappa(first_ratings: Int[np.ndarray, " items"], second_ratings: Int[np.ndarray, " items"], level_count: int) -> float:
    return float(cohen_kappa_score(first_ratings, second_ratings, labels=list(range(level_count)), weights="quadratic"))


def mean_pairwise_kappa(rating_pairs: list[tuple[Int[np.ndarray, " items"], Int[np.ndarray, " items"]]], level_count: int) -> float:
    return float(np.mean([quadratic_kappa(first, second, level_count) for first, second in rating_pairs]))


def bootstrap_mean_kappa(
    rating_pairs: list[tuple[Int[np.ndarray, " items"], Int[np.ndarray, " items"]]],
    level_count: int,
    resample_count: int,
    seed: int,
) -> KappaEstimate:
    item_count = len(rating_pairs[0][0])
    generator = np.random.default_rng(seed)
    resampled_means = []
    for _ in range(resample_count):
        item_indices = generator.integers(0, item_count, item_count)
        resampled_means.append(mean_pairwise_kappa([(first[item_indices], second[item_indices]) for first, second in rating_pairs], level_count))
    ci_low, ci_high = np.nanpercentile(resampled_means, [2.5, 97.5])
    return {"mean": mean_pairwise_kappa(rating_pairs, level_count), "ci_low": float(ci_low), "ci_high": float(ci_high)}


def format_estimate(estimate: KappaEstimate) -> str:
    return f"{estimate['mean']:.3f} [{estimate['ci_low']:.3f}, {estimate['ci_high']:.3f}]"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--labels", type=Path, nargs="+", required=True)
    parser.add_argument("--levels", required=True, help="comma-separated, lowest to highest")
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    levels: list[str] = args.levels.split(",")
    level_index_by_label = {label: index for index, label in enumerate(levels)}
    key_record_by_item_id = {record["item_id"]: record for record in cast(list[KeyRecord], read_jsonl(args.key))}
    label_by_item_id_by_labeler: dict[str, dict[str, str]] = {
        path.stem: {record["item_id"]: record["label"] for record in read_jsonl(path)} for path in args.labels
    }
    shared_item_ids = sorted(set(key_record_by_item_id).intersection(*label_by_item_id_by_labeler.values()))
    print(f"{len(shared_item_ids)} items labeled by all {len(label_by_item_id_by_labeler)} labelers ({len(key_record_by_item_id)} in key)")

    judge_ratings = np.array([level_index_by_label[str(key_record_by_item_id[item_id]["judge_label"])] for item_id in shared_item_ids])
    ratings_by_labeler = {
        labeler: np.array([level_index_by_label[label_by_item_id[item_id]] for item_id in shared_item_ids])
        for labeler, label_by_item_id in label_by_item_id_by_labeler.items()
    }

    judge_human_pairs = [(judge_ratings, ratings) for ratings in ratings_by_labeler.values()]
    print(f"LLM-judge-human kappa (mean over labelers): {format_estimate(bootstrap_mean_kappa(judge_human_pairs, len(levels), args.resamples, args.seed))}")
    human_human_pairs = list(itertools.combinations(ratings_by_labeler.values(), 2))
    if human_human_pairs:
        print(f"human-human kappa (mean over pairs):        {format_estimate(bootstrap_mean_kappa(human_human_pairs, len(levels), args.resamples, args.seed))}")

    models = np.array([key_record_by_item_id[item_id]["model"] for item_id in shared_item_ids])
    for labeler, ratings in ratings_by_labeler.items():
        print(f"\n{labeler}: exact agreement with judge {np.mean(ratings == judge_ratings):.3f}")
        print(f"  confusion (rows judge, cols human; levels {levels}):")
        for row in confusion_matrix(judge_ratings, ratings, labels=list(range(len(levels)))):
            print("   " + " ".join(f"{count:4d}" for count in row))
        for model in sorted(set(models)):
            in_model = models == model
            print(f"  {model:8s} n={in_model.sum():3d}  kappa {quadratic_kappa(judge_ratings[in_model], ratings[in_model], len(levels)):.3f}")


if __name__ == "__main__":
    main()
