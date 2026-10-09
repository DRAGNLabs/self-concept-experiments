from collections.abc import Mapping, Sequence
from typing import TypedDict

import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Float, Int, Int8
from torch import Tensor, nn
from transformers import PreTrainedModel

from selfconcept.common.ablation import mean_ablate_decoder_layer
from selfconcept.common.activation_extraction.model_specifics import ModelSpecifics
from selfconcept.common.span_targeting.types import PromptCompletion
from selfconcept.common.torch_utils import build_padded_batch

from selfconcept.assistant_axis_cot.projection import capture_layer_projections, sequence_region_codes
from selfconcept.assistant_axis_cot.records import (
    LABELLED_SEQUENCE_REGIONS,
    SEQUENCE_REGIONS,
    AblationResult,
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


class RegionKL(TypedDict):
    """Indexed by SEQUENCE_REGIONS, the region of the token each next-token distribution is read at."""

    kl_sum: Float[np.ndarray, " region"]
    token_count: Int[np.ndarray, " region"]


class RegionProjectionShift(TypedDict):
    """Ablated minus clean projection onto each layer's axis, summed by token region; rows index SEQUENCE_REGIONS."""

    shift_sum: Float[np.ndarray, "region layer"]
    squared_shift_sum: Float[np.ndarray, "region layer"]
    token_count: Int[np.ndarray, " region"]


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
    for region in LABELLED_SEQUENCE_REGIONS:
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


def final_hidden_states(
    model: PreTrainedModel, input_ids: Int[Tensor, "batch seq"], attention_mask: Int[Tensor, "batch seq"]
) -> Float[Tensor, "batch seq hidden"]:
    """The decoder's normed output, which the output embeddings map to logits; far smaller than the logits."""
    with torch.inference_mode():
        return model.get_decoder()(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state


def region_kl_sums(
    clean_hidden: Float[Tensor, "token hidden"],
    ablated_hidden: Float[Tensor, "token hidden"],
    output_embeddings: nn.Module,
    region_codes: Int8[np.ndarray, " token"],
    chunk_size: int,
) -> RegionKL:
    """KL(clean || ablated) of the next-token distribution after each token but the last, summed by that token's
    region; logits are materialized chunk_size positions at a time."""
    kl_chunks: list[Float[Tensor, " chunk"]] = []
    with torch.inference_mode():
        for start in range(0, len(region_codes) - 1, chunk_size):
            end = min(start + chunk_size, len(region_codes) - 1)
            clean_log_probs = F.log_softmax(output_embeddings(clean_hidden[start:end]).float(), dim=-1)
            ablated_log_probs = F.log_softmax(output_embeddings(ablated_hidden[start:end]).float(), dim=-1)
            kl_chunks.append(
                F.kl_div(ablated_log_probs, clean_log_probs, log_target=True, reduction="none").sum(dim=-1)
            )
    kl = torch.cat(kl_chunks).cpu().numpy()
    read_regions = region_codes[:-1]
    return {
        "kl_sum": np.bincount(read_regions, weights=kl, minlength=len(SEQUENCE_REGIONS)),
        "token_count": np.bincount(read_regions, minlength=len(SEQUENCE_REGIONS)),
    }


def region_projection_shift(
    clean_projections: Float[np.ndarray, "layer token"],
    ablated_projections: Float[np.ndarray, "layer token"],
    region_codes: Int8[np.ndarray, " token"],
) -> RegionProjectionShift:
    shift = ablated_projections - clean_projections
    return {
        "shift_sum": np.stack(
            [np.bincount(region_codes, weights=layer_shift, minlength=len(SEQUENCE_REGIONS)) for layer_shift in shift],
            axis=1,
        ),
        "squared_shift_sum": np.stack(
            [
                np.bincount(region_codes, weights=layer_shift**2, minlength=len(SEQUENCE_REGIONS))
                for layer_shift in shift
            ],
            axis=1,
        ),
        "token_count": np.bincount(region_codes, minlength=len(SEQUENCE_REGIONS)),
    }


def padded_region_codes(
    region_codes_by_example: Sequence[Int8[np.ndarray, " token"]], sequence_length: int, device: torch.device
) -> Int8[Tensor, "batch seq"]:
    padded = torch.full((len(region_codes_by_example), sequence_length), SEQUENCE_REGIONS.index("unlabelled"))
    for row, region_codes in enumerate(region_codes_by_example):
        padded[row, : len(region_codes)] = torch.from_numpy(region_codes)
    return padded.to(device)


def example_layer_projections(
    projections_by_layer: Mapping[int, Float[Tensor, "batch seq"]], row: int, token_count: int
) -> Float[np.ndarray, "layer token"]:
    return torch.stack(
        [projections_by_layer[layer][row, :token_count] for layer in sorted(projections_by_layer)]
    ).cpu().numpy()


def ablate_regions(
    model: PreTrainedModel,
    model_specifics: ModelSpecifics,
    examples: Sequence[PromptCompletion],
    region_means: RegionMeans,
    unit_axis_by_layer: Mapping[int, Float[Tensor, " hidden"]],
    pad_token_id: int,
    max_length: int,
    kl_chunk_size: int,
) -> list[AblationResult]:
    """A clean forward pass, then one per direction and labelled region with that region's tokens mean-ablated along the
    direction at the region means' layer. Each ablated pass is compared to the clean one by next-token KL and by the
    shift in every token's projection onto each layer's axis in unit_axis_by_layer."""
    token_ids_by_example = [example.prompt_token_ids + example.completion_token_ids for example in examples]
    input_ids, attention_mask = build_padded_batch(token_ids_by_example, pad_token_id, max_length, model.device)
    region_codes_by_example = [
        sequence_region_codes(example, model_specifics.split_completion(example), max_length) for example in examples
    ]
    padded_codes = padded_region_codes(region_codes_by_example, input_ids.shape[1], input_ids.device)
    decoder_layers = model_specifics.get_decoder_layers(model)
    output_embeddings = model.get_output_embeddings()
    assert output_embeddings is not None

    with capture_layer_projections(decoder_layers, unit_axis_by_layer) as clean_projections_by_layer:
        clean_hidden = final_hidden_states(model, input_ids, attention_mask)
    clean_layer_projections_by_example = [
        example_layer_projections(clean_projections_by_layer, row, len(region_codes))
        for row, region_codes in enumerate(region_codes_by_example)
    ]

    kl_by_run_by_example: list[list[RegionKL]] = [[] for _ in examples]
    shift_by_run_by_example: list[list[RegionProjectionShift]] = [[] for _ in examples]
    for direction_name, unit_direction in region_means["unit_direction_by_name"].items():
        for region in LABELLED_SEQUENCE_REGIONS:
            target_projection = region_means["moments_by_region_by_direction"][direction_name][region]["mean"]
            region_mask = padded_codes == SEQUENCE_REGIONS.index(region)
            with (
                mean_ablate_decoder_layer(
                    decoder_layers[region_means["layer"]], unit_direction, target_projection, region_mask
                ),
                capture_layer_projections(decoder_layers, unit_axis_by_layer) as ablated_projections_by_layer,
            ):
                ablated_hidden = final_hidden_states(model, input_ids, attention_mask)
            for row, region_codes in enumerate(region_codes_by_example):
                token_count = len(region_codes)
                kl_by_run_by_example[row].append(
                    region_kl_sums(
                        clean_hidden[row, :token_count],
                        ablated_hidden[row, :token_count],
                        output_embeddings,
                        region_codes,
                        kl_chunk_size,
                    )
                )
                shift_by_run_by_example[row].append(
                    region_projection_shift(
                        clean_layer_projections_by_example[row],
                        example_layer_projections(ablated_projections_by_layer, row, token_count),
                        region_codes,
                    )
                )

    run_shape = (len(region_means["unit_direction_by_name"]), len(LABELLED_SEQUENCE_REGIONS))
    return [
        {
            "kl_sum": np.stack([kl["kl_sum"] for kl in kl_by_run]).reshape(*run_shape, -1),
            "kl_token_count": kl_by_run[0]["token_count"],
            "projection_shift_sum": np.stack([shift["shift_sum"] for shift in shift_by_run]).reshape(
                *run_shape, *shift_by_run[0]["shift_sum"].shape
            ),
            "squared_projection_shift_sum": np.stack([shift["squared_shift_sum"] for shift in shift_by_run]).reshape(
                *run_shape, *shift_by_run[0]["squared_shift_sum"].shape
            ),
            "projection_token_count": shift_by_run[0]["token_count"],
            "truncated": len(token_ids) > max_length,
        }
        for kl_by_run, shift_by_run, token_ids in zip(
            kl_by_run_by_example, shift_by_run_by_example, token_ids_by_example, strict=True
        )
    ]
