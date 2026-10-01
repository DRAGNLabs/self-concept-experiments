"""Stratified AUROC: hacks are compared only with non-hacks from the same base problem."""
from collections import defaultdict
from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
from scipy.stats import rankdata

from .analyze import BinaryOutcome


class StratifiedExample(NamedTuple):
    score: float
    outcome: BinaryOutcome
    stratum: str


class StratifiedAuroc(NamedTuple):
    auroc: float | None
    n_pairs: int


class ConfidenceInterval(NamedTuple):
    low: float
    high: float


def mann_whitney_u(positive_scores: list[float], negative_scores: list[float]) -> float:
    """Pairs with the positive score higher, ties counting 1/2: positive rank sum minus its minimum possible value."""
    positive_ranks = rankdata(positive_scores + negative_scores)[:len(positive_scores)]
    return float(positive_ranks.sum()) - len(positive_scores) * (len(positive_scores) + 1) / 2


def stratified_auroc(examples: Sequence[StratifiedExample]) -> StratifiedAuroc:
    """Within-stratum (hack, non-hack) pairs pooled over strata; a higher hack score counts 1, a tie 1/2."""
    scores_by_stratum_and_outcome: defaultdict[tuple[str, BinaryOutcome], list[float]] = defaultdict(list)
    for example in examples:
        scores_by_stratum_and_outcome[example.stratum, example.outcome].append(example.score)
    concordant_pairs, n_pairs = 0., 0
    for stratum in {example.stratum for example in examples}:
        positive_scores = scores_by_stratum_and_outcome[stratum, 1]
        negative_scores = scores_by_stratum_and_outcome[stratum, 0]
        if not positive_scores or not negative_scores:
            continue
        concordant_pairs += mann_whitney_u(positive_scores, negative_scores)
        n_pairs += len(positive_scores) * len(negative_scores)
    return StratifiedAuroc(concordant_pairs / n_pairs if n_pairs else None, n_pairs)


def stratum_bootstrap_interval(examples: Sequence[StratifiedExample], samples: int = 1000,
                               seed: int = 1729) -> ConfidenceInterval | None:
    """95% interval from resampling whole strata; a stratum drawn twice enters as two separate strata."""
    examples_by_stratum: defaultdict[str, list[StratifiedExample]] = defaultdict(list)
    for example in examples:
        examples_by_stratum[example.stratum].append(example)
    strata = sorted(examples_by_stratum)
    rng = np.random.default_rng(seed)
    draws: list[float] = []
    for _ in range(samples):
        drawn_strata = [strata[index] for index in rng.integers(len(strata), size=len(strata))]
        resampled = [example._replace(stratum=f"{draw}:{stratum}")
                     for draw, stratum in enumerate(drawn_strata) for example in examples_by_stratum[stratum]]
        if (auroc := stratified_auroc(resampled).auroc) is not None:
            draws.append(auroc)
    if not draws:
        return None
    low, high = np.quantile(draws, [.025, .975]).tolist()
    return ConfidenceInterval(low, high)
