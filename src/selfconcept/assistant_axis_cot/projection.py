from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch
from jaxtyping import Float, Int8
from torch import Tensor
from transformers import PreTrainedModel

from selfconcept.assistant_axis_cot.records import (
    COMPLETION_REGIONS,
    SEQUENCE_REGIONS,
    AxisName,
    CompletionProjections,
    SequenceProjections,
)
from selfconcept.common.activation_extraction.model_specifics import ModelSpecifics
from selfconcept.common.activation_extraction.extraction import extract_token_span_activations
from selfconcept.common.span_targeting.types import CompletionSpans, PromptCompletion, TokenSpan
from selfconcept.measurement.unit_axes import load_unit_axes

ATTENTION_SINK_TOKEN_COUNT = 1


def load_unit_axis_by_name(
    axis_path_by_name: Mapping[AxisName, Path],
    layer: int,
    num_layers: int,
    hidden_size: int,
) -> dict[AxisName, Float[Tensor, " hidden"]]:
    return {
        axis_name: torch.from_numpy(load_unit_axes(axis_path, [layer], num_layers, hidden_size)[layer]["direction"])
        for axis_name, axis_path in axis_path_by_name.items()
    }


def completion_region_codes(
    completion_spans: CompletionSpans,
    prompt_length: int,
    completion_length: int,
) -> Int8[np.ndarray, " n_tokens"]:
    region_codes = np.full(completion_length, COMPLETION_REGIONS.index("delimiter"), dtype=np.int8)
    for region, spans in (("cot", completion_spans.thinking), ("final", completion_spans.response)):
        for span in spans:
            region_codes[span.start - prompt_length : span.end - prompt_length] = COMPLETION_REGIONS.index(region)
    return region_codes


def sequence_region_codes(
    example: PromptCompletion,
    completion_spans: CompletionSpans,
    max_length: int,
) -> Int8[np.ndarray, " n_tokens"]:
    """Per token of the concatenated prompt and completion, up to max_length, an index into SEQUENCE_REGIONS;
    the attention sink and the completion's delimiters are unlabelled."""
    sequence_length = len(example.prompt_token_ids) + len(example.completion_token_ids)
    region_codes = np.full(sequence_length, SEQUENCE_REGIONS.index("unlabelled"), dtype=np.int8)
    region_codes[ATTENTION_SINK_TOKEN_COUNT : len(example.prompt_token_ids)] = SEQUENCE_REGIONS.index("prompt")
    for region, spans in (("cot", completion_spans.thinking), ("final", completion_spans.response)):
        for span in spans:
            region_codes[span.start : span.end] = SEQUENCE_REGIONS.index(region)
    return region_codes[:max_length]


def sequence_projections[DirectionName: str](
    example: PromptCompletion,
    completion_spans: CompletionSpans,
    sequence_hidden: Float[Tensor, "token hidden"],
    unit_direction_by_name: Mapping[DirectionName, Float[Tensor, " hidden"]],
    max_length: int,
) -> SequenceProjections[DirectionName]:
    kept_token_count = sequence_hidden.shape[0]
    token_ids = example.prompt_token_ids + example.completion_token_ids
    return {
        "projections_by_axis": {
            direction_name: (sequence_hidden @ unit_direction.to(sequence_hidden.device)).cpu().numpy()
            for direction_name, unit_direction in unit_direction_by_name.items()
        },
        "residual_norms": sequence_hidden.norm(dim=-1).cpu().numpy(),
        "token_ids": np.asarray(token_ids[:kept_token_count], dtype=np.int32),
        "region_codes": sequence_region_codes(example, completion_spans, max_length),
    }


def project_sequences[DirectionName: str](
    model: PreTrainedModel,
    model_specifics: ModelSpecifics,
    examples: Sequence[PromptCompletion],
    unit_direction_by_name: Mapping[DirectionName, Float[Tensor, " hidden"]],
    layer: int,
    pad_token_id: int,
    max_length: int,
) -> list[SequenceProjections[DirectionName]]:
    """Projects every prompt and completion token at one decoder layer; one batched forward pass."""
    token_ids_by_example = [example.prompt_token_ids + example.completion_token_ids for example in examples]
    hidden_by_example = extract_token_span_activations(
        model,
        token_ids_by_example,
        [[TokenSpan(0, len(token_ids))] for token_ids in token_ids_by_example],
        model_specifics.get_decoder_layers(model),
        pad_token_id,
        max_length,
        layers=[layer],
    )
    return [
        sequence_projections(
            example, model_specifics.split_completion(example), hidden[0].float(), unit_direction_by_name, max_length
        )
        for example, hidden in zip(examples, hidden_by_example, strict=True)
    ]


def completion_projections(
    example: PromptCompletion,
    completion_spans: CompletionSpans,
    projections: SequenceProjections[AxisName],
) -> CompletionProjections:
    prompt_length = len(example.prompt_token_ids)
    completion_residual_norms = projections["residual_norms"][prompt_length:]
    kept_token_count = len(completion_residual_norms)
    region_codes = completion_region_codes(completion_spans, prompt_length, len(example.completion_token_ids))
    return {
        "projections_by_axis": {
            axis_name: axis_projections[prompt_length:]
            for axis_name, axis_projections in projections["projections_by_axis"].items()
        },
        "residual_norms": completion_residual_norms,
        "completion_token_ids": np.asarray(example.completion_token_ids[:kept_token_count], dtype=np.int32),
        "region_codes": region_codes[:kept_token_count],
        "truncated": kept_token_count < len(example.completion_token_ids),
    }


def project_completions(
    model: PreTrainedModel,
    model_specifics: ModelSpecifics,
    examples: Sequence[PromptCompletion],
    unit_axis_by_name: Mapping[AxisName, Float[Tensor, " hidden"]],
    layer: int,
    pad_token_id: int,
    max_length: int,
) -> list[CompletionProjections]:
    """Projects every completion token, delimiters included."""
    projections_by_example = project_sequences(
        model, model_specifics, examples, unit_axis_by_name, layer, pad_token_id, max_length
    )
    return [
        completion_projections(example, model_specifics.split_completion(example), projections)
        for example, projections in zip(examples, projections_by_example, strict=True)
    ]
