"""Agreement between human labelers and the LLM judge on a human-validation stage, with every label lumped into
positive (one of --positive-levels) or not. Headline numbers: mean pairwise human-human and mean LLM-judge-human Cohen's
kappa on those lumped labels, with 95% percentile CIs from a bootstrap over items, and the judge's precision: the share
of items it judged positive that a strict majority of labelers (and, stricter, every labeler) also labeled positive,
with a 95% Wilson CI. Only items every labeler labeled are scored.

    python score_agreement.py --key key.jsonl --labels labels_a.jsonl labels_b.jsonl --levels 0,1,2,3 --positive-levels 3
"""
import argparse
import itertools
import json
from pathlib import Path
from typing import TypedDict, cast

import numpy as np
from jaxtyping import Bool, Int
from scipy.stats import binomtest
from sklearn.metrics import cohen_kappa_score, confusion_matrix


class KeyRecord(TypedDict):
    item_id: str
    judge_label: str
    model: str


class Estimate(TypedDict):
    mean: float
    ci_low: float
    ci_high: float


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def binary_kappa(first_positives: Bool[np.ndarray, " items"], second_positives: Bool[np.ndarray, " items"]) -> float:
    return float(cohen_kappa_score(first_positives, second_positives, labels=[False, True]))


def mean_pairwise_kappa(positive_pairs: list[tuple[Bool[np.ndarray, " items"], Bool[np.ndarray, " items"]]]) -> float:
    return float(np.mean([binary_kappa(first, second) for first, second in positive_pairs]))


def bootstrap_mean_kappa(
    positive_pairs: list[tuple[Bool[np.ndarray, " items"], Bool[np.ndarray, " items"]]],
    resample_count: int,
    seed: int,
) -> Estimate:
    item_count = len(positive_pairs[0][0])
    generator = np.random.default_rng(seed)
    resampled_means = []
    for _ in range(resample_count):
        item_indices = generator.integers(0, item_count, item_count)
        resampled_means.append(mean_pairwise_kappa([(first[item_indices], second[item_indices]) for first, second in positive_pairs]))
    ci_low, ci_high = np.nanpercentile(resampled_means, [2.5, 97.5])
    return {"mean": mean_pairwise_kappa(positive_pairs), "ci_low": float(ci_low), "ci_high": float(ci_high)}


def judge_precision(judge_positives: Bool[np.ndarray, " items"], human_positives: Bool[np.ndarray, " items"]) -> Estimate:
    result = binomtest(int(human_positives[judge_positives].sum()), int(judge_positives.sum()))
    interval = result.proportion_ci(method="wilson")
    return {"mean": result.statistic, "ci_low": interval.low, "ci_high": interval.high}


def format_estimate(estimate: Estimate) -> str:
    return f"{estimate['mean']:.3f} [{estimate['ci_low']:.3f}, {estimate['ci_high']:.3f}]"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--labels", type=Path, nargs="+", required=True)
    parser.add_argument("--levels", required=True, help="comma-separated, lowest to highest")
    parser.add_argument("--positive-levels", required=True, help="comma-separated levels lumped into positive; the rest are negative")
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    levels: list[str] = args.levels.split(",")
    level_index_by_label = {label: index for index, label in enumerate(levels)}
    positive_level_indices = [level_index_by_label[label] for label in args.positive_levels.split(",")]
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

    judge_positives = np.isin(judge_ratings, positive_level_indices)
    positives_by_labeler = {labeler: np.isin(ratings, positive_level_indices) for labeler, ratings in ratings_by_labeler.items()}

    judge_human_pairs = [(judge_positives, positives) for positives in positives_by_labeler.values()]
    print(f"LLM-judge-human kappa (mean over labelers): {format_estimate(bootstrap_mean_kappa(judge_human_pairs, args.resamples, args.seed))}")
    human_human_pairs = list(itertools.combinations(positives_by_labeler.values(), 2))
    if human_human_pairs:
        print(f"human-human kappa (mean over pairs):        {format_estimate(bootstrap_mean_kappa(human_human_pairs, args.resamples, args.seed))}")

    models = np.array([key_record_by_item_id[item_id]["model"] for item_id in shared_item_ids])
    positive_vote_counts = np.sum(list(positives_by_labeler.values()), axis=0)
    human_positives_by_rule = {
        "majority": positive_vote_counts > len(positives_by_labeler) / 2,
        "unanimous": positive_vote_counts == len(positives_by_labeler),
    }
    for rule, human_positives in human_positives_by_rule.items():
        print(f"judge precision (judged positive, human {rule} positive): "
              f"{human_positives[judge_positives].sum()}/{judge_positives.sum()} = {format_estimate(judge_precision(judge_positives, human_positives))}")
        for model in sorted(set(models)):
            in_model = models == model
            print(f"  {model:8s} {human_positives[judge_positives & in_model].sum()}/{(judge_positives & in_model).sum()}"
                  f" = {format_estimate(judge_precision(judge_positives[in_model], human_positives[in_model]))}")
    for labeler, ratings in ratings_by_labeler.items():
        positives = positives_by_labeler[labeler]
        print(f"\n{labeler}: positive/negative agreement with judge {np.mean(positives == judge_positives):.3f}, exact {np.mean(ratings == judge_ratings):.3f}")
        print(f"  confusion (rows judge, cols human; levels {levels}):")
        for row in confusion_matrix(judge_ratings, ratings, labels=list(range(len(levels)))):
            print("   " + " ".join(f"{count:4d}" for count in row))
        for model in sorted(set(models)):
            in_model = models == model
            print(f"  {model:8s} n={in_model.sum():3d}  kappa {binary_kappa(judge_positives[in_model], positives[in_model]):.3f}")


if __name__ == "__main__":
    main()
