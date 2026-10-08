from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch
from jaxtyping import Float, Int8
from torch import Tensor
from transformers import PreTrainedModel

from selfconcept.assistant_axis_cot.records import REGIONS, AxisName, CompletionProjections
from selfconcept.common.activation_extraction.model_specifics import ModelSpecifics
from selfconcept.common.activation_extraction.extraction import extract_token_span_activations
from selfconcept.common.span_targeting.types import CompletionSpans, PromptCompletion, TokenSpan
from selfconcept.measurement.unit_axes import load_unit_axes


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
    region_codes = np.full(completion_length, REGIONS.index("delimiter"), dtype=np.int8)
    for region, spans in (("cot", completion_spans.thinking), ("final", completion_spans.response)):
        for span in spans:
            region_codes[span.start - prompt_length : span.end - prompt_length] = REGIONS.index(region)
    return region_codes


def project_completion(
    example: PromptCompletion,
    completion_spans: CompletionSpans,
    completion_hidden: Float[Tensor, "token hidden"],
    unit_axis_by_name: Mapping[AxisName, Float[Tensor, " hidden"]],
) -> CompletionProjections:
    kept_token_count = completion_hidden.shape[0]
    region_codes = completion_region_codes(
        completion_spans, len(example.prompt_token_ids), len(example.completion_token_ids)
    )
    return {
        "completion_token_ids": np.asarray(example.completion_token_ids[:kept_token_count], dtype=np.int32),
        "region_codes": region_codes[:kept_token_count],
        "projections_by_axis": {
            axis_name: (completion_hidden @ unit_axis.to(completion_hidden.device)).cpu().numpy()
            for axis_name, unit_axis in unit_axis_by_name.items()
        },
        "residual_norms": completion_hidden.norm(dim=-1).cpu().numpy(),
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
    """Projects every completion token, delimiters included, at one decoder layer; one batched forward pass."""
    whole_completion_spans = [
        [TokenSpan(len(example.prompt_token_ids), len(example.prompt_token_ids) + len(example.completion_token_ids))]
        for example in examples
    ]
    hidden_by_example = extract_token_span_activations(
        model,
        [example.prompt_token_ids + example.completion_token_ids for example in examples],
        whole_completion_spans,
        model_specifics.get_decoder_layers(model),
        pad_token_id,
        max_length,
        layers=[layer],
    )
    return [
        project_completion(example, model_specifics.split_completion(example), hidden[0].float(), unit_axis_by_name)
        for example, hidden in zip(examples, hidden_by_example, strict=True)
    ]
