from typing import TypedDict

import numpy as np
from jaxtyping import Float, Int

from selfconcept.assistant_axis_cot.ablation_statistics import RANDOM_MEAN_NAME
from selfconcept.assistant_axis_cot.records import (
    LABELLED_SEQUENCE_REGIONS,
    SEQUENCE_REGIONS,
    Condition,
    SequenceRegion,
    WindowAblationRecord,
)
from selfconcept.assistant_axis_cot.window_ablation import WindowAblationRun


class WindowKLCells(TypedDict):
    """Indexed [window offset][distance], each a window_size-token bin: the window's tokens before its region's end,
    and the measured tokens' distance after the window's start, so the first distance bin is the window itself. None
    where fewer than the minimum conversations reach the cell."""

    mean_kl_by_name: dict[str, list[list[float | None]]]
    conversation_count: list[list[int]]


class WindowAblationAnalysis(TypedDict):
    """Cells by ablated region, then measured region, at or after the ablated one."""

    model: str
    layer: int
    axis_name: str
    condition: Condition
    conversation_count: int
    window_size: int
    min_conversation_count: int
    cells_by_measured_by_ablated: dict[SequenceRegion, dict[SequenceRegion, WindowKLCells]]


class PooledWindowKL(TypedDict):
    """Sums over conversations of each conversation's mean per-token KL in a cell, and how many conversations reach
    it. Axes: ablated and measured regions index LABELLED_SEQUENCE_REGIONS."""

    conversation_mean_sum: Float[np.ndarray, "direction ablated window_offset measured distance"]
    conversation_count: Int[np.ndarray, "ablated window_offset measured distance"]


def pooled_window_kl(
    records: list[WindowAblationRecord], direction_count: int, window_size: int
) -> PooledWindowKL:
    window_offset_count = 1 + max(
        (window["tokens_before_region_end"] // window_size for record in records for window in record["windows"]),
        default=0,
    )
    distance_count = -(-max(len(record["region_codes"]) for record in records) // window_size)
    region_count = len(LABELLED_SEQUENCE_REGIONS)
    conversation_mean_sum = np.zeros((direction_count, region_count, window_offset_count, region_count, distance_count))
    conversation_count = np.zeros((region_count, window_offset_count, region_count, distance_count), dtype=np.int64)
    for record in records:
        for window_index, window in enumerate(record["windows"]):
            positions = np.arange(window["start"], len(record["region_codes"]) - 1)
            measured_codes = record["region_codes"][positions].astype(np.int64)
            is_labelled = measured_codes != SEQUENCE_REGIONS.index("unlabelled")
            cell_indices = (measured_codes * distance_count + (positions - window["start"]) // window_size)[is_labelled]
            token_counts = np.bincount(cell_indices, minlength=region_count * distance_count)
            is_reached = (token_counts > 0).reshape(region_count, distance_count)
            ablated_index = LABELLED_SEQUENCE_REGIONS.index(window["region"])
            window_offset = window["tokens_before_region_end"] // window_size
            conversation_count[ablated_index, window_offset] += is_reached
            for direction_index in range(direction_count):
                kl_sums = np.bincount(
                    cell_indices,
                    weights=record["kl"][direction_index, window_index, positions[is_labelled]],
                    minlength=region_count * distance_count,
                )
                conversation_mean_sum[direction_index, ablated_index, window_offset] += (
                    kl_sums / np.maximum(token_counts, 1)
                ).reshape(region_count, distance_count)
    return {"conversation_mean_sum": conversation_mean_sum, "conversation_count": conversation_count}


def window_kl_cells(
    mean_kl_by_name: dict[str, Float[np.ndarray, "window_offset distance"]],
    conversation_count: Int[np.ndarray, "window_offset distance"],
    min_conversation_count: int,
) -> WindowKLCells:
    """Trimmed to the window offsets and distances with a reached cell."""
    is_reached = conversation_count >= min_conversation_count
    window_offset_count = int(np.flatnonzero(is_reached.any(axis=1)).max(initial=-1)) + 1
    distance_count = int(np.flatnonzero(is_reached.any(axis=0)).max(initial=-1)) + 1
    return {
        "mean_kl_by_name": {
            name: [
                [
                    float(mean_kl[window_offset, distance]) if is_reached[window_offset, distance] else None
                    for distance in range(distance_count)
                ]
                for window_offset in range(window_offset_count)
            ]
            for name, mean_kl in mean_kl_by_name.items()
        },
        "conversation_count": conversation_count[:window_offset_count, :distance_count].tolist(),
    }


def analyze_window_ablation_run(
    window_ablation_run: WindowAblationRun, min_conversation_count: int
) -> WindowAblationAnalysis:
    """Per cell, the mean over conversations of each one's mean per-token KL there; the random mean averages the
    random directions' cells, which equals averaging each conversation's over the directions first since every
    direction reaches the same cells."""
    direction_names = window_ablation_run["direction_names"]
    pooled = pooled_window_kl(window_ablation_run["records"], len(direction_names), window_ablation_run["window_size"])
    mean_kl = pooled["conversation_mean_sum"] / np.maximum(pooled["conversation_count"], 1)[None]
    mean_kl_by_name = {name: mean_kl[index] for index, name in enumerate(direction_names)}
    random_names = [name for name in direction_names if name != window_ablation_run["axis_name"]]
    if random_names:
        mean_kl_by_name[RANDOM_MEAN_NAME] = np.mean([mean_kl_by_name[name] for name in random_names], axis=0)
    return {
        "model": window_ablation_run["model"],
        "layer": window_ablation_run["layer"],
        "axis_name": window_ablation_run["axis_name"],
        "condition": window_ablation_run["condition"],
        "conversation_count": len(window_ablation_run["records"]),
        "window_size": window_ablation_run["window_size"],
        "min_conversation_count": min_conversation_count,
        "cells_by_measured_by_ablated": {
            ablated: {
                measured: window_kl_cells(
                    {
                        name: name_mean_kl[ablated_index, :, measured_index]
                        for name, name_mean_kl in mean_kl_by_name.items()
                    },
                    pooled["conversation_count"][ablated_index, :, measured_index],
                    min_conversation_count,
                )
                for measured_index, measured in enumerate(LABELLED_SEQUENCE_REGIONS)
                if measured_index >= ablated_index
            }
            for ablated_index, ablated in enumerate(LABELLED_SEQUENCE_REGIONS)
        },
    }
