from collections.abc import Sequence
from typing import TypedDict

import numpy as np
from jaxtyping import Bool, Float

from selfconcept.assistant_axis_cot.ablation import AblationRun
from selfconcept.assistant_axis_cot.records import (
    LABELLED_SEQUENCE_REGIONS,
    SEQUENCE_REGIONS,
    AblationRecord,
    Condition,
    SequenceRegion,
)
from selfconcept.assistant_axis_cot.statistics import question_bootstrap_ci

RANDOM_MEAN_NAME = "random_mean"
LABELLED_REGION_INDICES = [SEQUENCE_REGIONS.index(region) for region in LABELLED_SEQUENCE_REGIONS]


class CellEstimate(TypedDict):
    mean: float
    ci_low: float
    ci_high: float
    conversation_count: int


type KLGrid = dict[SequenceRegion, dict[SequenceRegion, CellEstimate]]


class AblationAnalysis(TypedDict):
    """KL grids are ablated region -> measured region; surviving shift fractions are ablated region -> measured
    region -> one value per layer in layers."""

    model: str
    layer: int
    axis_name: str
    condition: Condition
    conversation_count: int
    kl_grid_by_name: dict[str, KLGrid]
    projection_std_by_region_by_direction: dict[str, dict[SequenceRegion, float]]
    layers: list[int]
    surviving_shift_fraction_by_measured_by_ablated: dict[SequenceRegion, dict[SequenceRegion, list[float]]]


def conversation_mean_kl(
    records: Sequence[AblationRecord],
) -> Float[np.ndarray, "conversation direction ablated measured"]:
    """Each conversation's mean per-token KL, measured regions indexing LABELLED_SEQUENCE_REGIONS."""
    return np.stack(
        [record["kl_sum"] / np.maximum(record["kl_token_count"], 1) for record in records]
    )[..., LABELLED_REGION_INDICES]


def has_enough_tokens(
    records: Sequence[AblationRecord], min_region_tokens: int
) -> Bool[np.ndarray, "conversation ablated measured"]:
    region_token_counts = np.stack([record["projection_token_count"][LABELLED_REGION_INDICES] for record in records])
    kl_token_counts = np.stack([record["kl_token_count"][LABELLED_REGION_INDICES] for record in records])
    return (region_token_counts[:, :, None] >= min_region_tokens) & (kl_token_counts[:, None, :] >= min_region_tokens)


def cell_estimate(
    values: Float[np.ndarray, " conversation"],
    question_indices: Sequence[int],
    rng: np.random.Generator,
    resample_count: int,
) -> CellEstimate:
    if len(values) == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "conversation_count": 0}
    ci_low, ci_high = question_bootstrap_ci(values, question_indices, rng, resample_count)
    return {"mean": float(values.mean()), "ci_low": ci_low, "ci_high": ci_high, "conversation_count": len(values)}


def kl_grid(
    mean_kl: Float[np.ndarray, "conversation ablated measured"],
    is_included: Bool[np.ndarray, "conversation ablated measured"],
    question_indices: Sequence[int],
    rng: np.random.Generator,
    resample_count: int,
) -> KLGrid:
    question_index_array = np.asarray(question_indices)
    return {
        ablated: {
            measured: cell_estimate(
                mean_kl[is_included[:, ablated_index, measured_index], ablated_index, measured_index],
                question_index_array[is_included[:, ablated_index, measured_index]].tolist(),
                rng,
                resample_count,
            )
            for measured_index, measured in enumerate(LABELLED_SEQUENCE_REGIONS)
        }
        for ablated_index, ablated in enumerate(LABELLED_SEQUENCE_REGIONS)
    }


def kl_grid_by_name(
    ablation_run: AblationRun, min_region_tokens: int, rng: np.random.Generator, resample_count: int
) -> dict[str, KLGrid]:
    """One grid per direction, and one of each conversation's mean over the random baseline directions."""
    records = ablation_run["records"]
    mean_kl = conversation_mean_kl(records)
    is_included = has_enough_tokens(records, min_region_tokens)
    question_indices = [record["question_index"] for record in records]
    random_direction_indices = [
        index for index, name in enumerate(ablation_run["direction_names"]) if name != ablation_run["axis_name"]
    ]
    mean_kl_by_name = {name: mean_kl[:, index] for index, name in enumerate(ablation_run["direction_names"])}
    if random_direction_indices:
        mean_kl_by_name[RANDOM_MEAN_NAME] = mean_kl[:, random_direction_indices].mean(axis=1)
    return {
        name: kl_grid(name_mean_kl, is_included, question_indices, rng, resample_count)
        for name, name_mean_kl in mean_kl_by_name.items()
    }


def pooled_shift_rms(
    records: Sequence[AblationRecord], direction_index: int
) -> Float[np.ndarray, "ablated measured layer"]:
    """RMS over every token of the conversations of the shift in each layer's axis projection; measured regions
    index SEQUENCE_REGIONS."""
    squared_shift_sum = sum(record["squared_projection_shift_sum"][direction_index] for record in records)
    token_count = sum(record["projection_token_count"] for record in records)
    return np.sqrt(squared_shift_sum / np.maximum(token_count, 1)[None, :, None])


def surviving_shift_fraction(
    shift_rms: Float[np.ndarray, "ablated measured layer"],
) -> dict[SequenceRegion, dict[SequenceRegion, list[float]]]:
    """Each layer's RMS shift over the injected one: the ablated region's at the ablation layer. Only regions at or
    after the ablated one, since earlier ones are causally untouched."""
    return {
        ablated: {
            measured: (
                shift_rms[ablated_index, SEQUENCE_REGIONS.index(measured)]
                / shift_rms[ablated_index, SEQUENCE_REGIONS.index(ablated), 0]
            ).tolist()
            for measured in LABELLED_SEQUENCE_REGIONS[ablated_index:]
        }
        for ablated_index, ablated in enumerate(LABELLED_SEQUENCE_REGIONS)
    }


def analyze_ablation_run(
    ablation_run: AblationRun, min_region_tokens: int, rng: np.random.Generator, resample_count: int
) -> AblationAnalysis:
    axis_index = ablation_run["direction_names"].index(ablation_run["axis_name"])
    return {
        "model": ablation_run["model"],
        "layer": ablation_run["layer"],
        "axis_name": ablation_run["axis_name"],
        "condition": ablation_run["condition"],
        "conversation_count": len(ablation_run["records"]),
        "kl_grid_by_name": kl_grid_by_name(ablation_run, min_region_tokens, rng, resample_count),
        "projection_std_by_region_by_direction": {
            direction_name: {region: moments["std"] for region, moments in moments_by_region.items()}
            for direction_name, moments_by_region in ablation_run["moments_by_region_by_direction"].items()
        },
        "layers": ablation_run["layers"],
        "surviving_shift_fraction_by_measured_by_ablated": surviving_shift_fraction(
            pooled_shift_rms(ablation_run["records"], axis_index)
        ),
    }
