from collections.abc import Sequence
from typing import TypedDict

import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor

from selfconcept.assistant_axis_cot.records import SEQUENCE_REGIONS, SequenceProjections, SequenceRegion


class RegionProjectionMoments(TypedDict):
    token_count: int
    mean: float
    std: float


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
