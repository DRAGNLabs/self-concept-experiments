from typing import TypedDict

import numpy as np
from jaxtyping import Float, Int

from selfconcept.arm_search.interface import Arm, GroupId


class GroupCoverage(TypedDict):
    group_ids: list[GroupId]
    arm_group_indices: Int[np.ndarray, " arm"]
    uncovered_probability: Float[np.ndarray, "draw group"]


def initial_group_coverage(arms: list[Arm], covered_group_ids: set[GroupId], draw_count: int) -> GroupCoverage:
    """`arms` must be in the posterior's arm-axis order."""
    group_ids = sorted({arm["group_id"] for arm in arms})
    group_index_by_group_id = {group_id: index for index, group_id in enumerate(group_ids)}
    uncovered_by_group = np.array([0.0 if group_id in covered_group_ids else 1.0 for group_id in group_ids])
    return GroupCoverage(
        group_ids=group_ids,
        arm_group_indices=np.array([group_index_by_group_id[arm["group_id"]] for arm in arms]),
        uncovered_probability=np.tile(uncovered_by_group, (draw_count, 1)),
    )


def add_pending_unit(
    coverage: GroupCoverage,
    success_probability: Float[np.ndarray, "draw arm"],
    arm_index: int,
    seeds_per_unit: int,
) -> GroupCoverage:
    group_index = int(coverage["arm_group_indices"][arm_index])
    uncovered_probability = coverage["uncovered_probability"].copy()
    uncovered_probability[:, group_index] *= (1 - success_probability[:, arm_index]) ** seeds_per_unit
    return GroupCoverage(
        group_ids=coverage["group_ids"],
        arm_group_indices=coverage["arm_group_indices"],
        uncovered_probability=uncovered_probability,
    )
