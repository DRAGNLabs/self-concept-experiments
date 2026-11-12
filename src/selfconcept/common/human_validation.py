"""Shared pieces of the human-validation stages: the labeling-page item, the judged candidates a sampler draws from,
the kept-back key, and the half-positive, model-balanced sample."""
import random
from collections.abc import Sequence
from typing import ReadOnly, TypedDict


class Section(TypedDict):
    title: str
    text: str


class Item(TypedDict):
    item_id: str
    sections: list[Section]


class JudgedCandidate(TypedDict):
    model: str
    judge_label: ReadOnly[str]


class KeyRecord(TypedDict):
    item_id: str
    model: str
    judge_label: ReadOnly[str]


def balanced_sample[CandidateT](candidates_by_model: dict[str, list[CandidateT]], sample_size: int, generator: random.Random) -> list[CandidateT]:
    """Round-robin over models, skipping a model once its pool is empty."""
    pools = {model: generator.sample(candidates, len(candidates)) for model, candidates in sorted(candidates_by_model.items())}
    sample: list[CandidateT] = []
    while len(sample) < sample_size and any(pools.values()):
        for pool in pools.values():
            if pool and len(sample) < sample_size:
                sample.append(pool.pop())
    assert len(sample) == sample_size, f"only {len(sample)} candidates available"
    return sample


def half_positive_sample[CandidateT: JudgedCandidate](
    candidates: Sequence[CandidateT], models: Sequence[str], positive_levels: Sequence[str], per_half: int, generator: random.Random,
) -> list[CandidateT]:
    """per_half candidates judged in positive_levels and per_half judged otherwise, each half model-balanced, shuffled."""
    sample = [
        candidate
        for positive in (True, False)
        for candidate in balanced_sample(
            {model: [c for c in candidates if c["model"] == model and (c["judge_label"] in positive_levels) == positive] for model in models},
            per_half, generator,
        )
    ]
    generator.shuffle(sample)
    return sample
