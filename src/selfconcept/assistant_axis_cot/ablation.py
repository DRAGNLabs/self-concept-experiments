from collections.abc import Sequence
from typing import TypedDict

import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor

from selfconcept.assistant_axis_cot.records import (
    SEQUENCE_REGIONS,
    AxisName,
    SequenceProjectionRecord,
    SequenceProjections,
    SequenceRegion,
)


class RegionProjectionMoments(TypedDict):
    token_count: int
    mean: float
    std: float


class NormOutlierToken(TypedDict):
    token_id: int
    outlier_count: int
    occurrence_count: int
    min_outlying_norm_ratio: float
    max_outlying_norm_ratio: float
    example_positions: list[int]


class RegionMeans(TypedDict):
    """The unprompted transcripts' per-region projection moments: the mean-ablation targets."""

    model: str
    layer: int
    axis_name: AxisName
    unit_direction_by_name: dict[str, Float[Tensor, " hidden"]]
    moments_by_region_by_direction: dict[str, dict[SequenceRegion, RegionProjectionMoments]]
    records: list[SequenceProjectionRecord]


def random_unit_directions(count: int, hidden_size: int, seed: int) -> Float[Tensor, "count hidden"]:
    directions = torch.randn(count, hidden_size, generator=torch.Generator().manual_seed(seed))
    return directions / directions.norm(dim=-1, keepdim=True)


def region_projection_moments[DirectionName: str](
    sequence_projections: Sequence[SequenceProjections[DirectionName]], direction_name: DirectionName
) -> dict[SequenceRegion, RegionProjectionMoments]:
    """Population moments over every labelled token of the sequences, pooled."""
    moments_by_region: dict[SequenceRegion, RegionProjectionMoments] = {}
    for region in SEQUENCE_REGIONS:
        if region == "unlabelled":
            continue
        region_values = np.concatenate(
            [
                projections["projections_by_axis"][direction_name][
                    projections["region_codes"] == SEQUENCE_REGIONS.index(region)
                ]
                for projections in sequence_projections
            ]
        )
        moments_by_region[region] = {
            "token_count": len(region_values),
            "mean": float(region_values.mean()),
            "std": float(region_values.std()),
        }
    return moments_by_region


def norm_ratios(projections: SequenceProjections[str]) -> Float[np.ndarray, " n_tokens"]:
    """Each token's residual norm over its sequence's median."""
    return projections["residual_norms"] / np.median(projections["residual_norms"])


def norm_outlier_tokens(
    sequence_projections: Sequence[SequenceProjections[str]], norm_ratio_threshold: float, example_count: int
) -> list[NormOutlierToken]:
    """Token ids with any occurrence whose norm ratio is above norm_ratio_threshold or below its reciprocal, most
    frequently outlying first."""
    token_ids = np.concatenate([projections["token_ids"] for projections in sequence_projections])
    ratios = np.concatenate([norm_ratios(projections) for projections in sequence_projections])
    positions = np.concatenate([np.arange(len(projections["token_ids"])) for projections in sequence_projections])
    is_outlier = (ratios > norm_ratio_threshold) | (ratios < 1 / norm_ratio_threshold)
    outlier_tokens: list[NormOutlierToken] = []
    for token_id in np.unique(token_ids[is_outlier]):
        is_token_outlier = is_outlier & (token_ids == token_id)
        outlier_tokens.append({
            "token_id": int(token_id),
            "outlier_count": int(is_token_outlier.sum()),
            "occurrence_count": int((token_ids == token_id).sum()),
            "min_outlying_norm_ratio": float(ratios[is_token_outlier].min()),
            "max_outlying_norm_ratio": float(ratios[is_token_outlier].max()),
            "example_positions": sorted({int(position) for position in positions[is_token_outlier]})[:example_count],
        })
    return sorted(outlier_tokens, key=lambda outlier_token: -outlier_token["outlier_count"])
