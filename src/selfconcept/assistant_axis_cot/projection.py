from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
from jaxtyping import Float, Int8
from torch import Tensor, nn
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedModel

from selfconcept.assistant_axis_cot.records import (
    COMPLETION_REGIONS,
    SEQUENCE_REGIONS,
    AxisName,
    CompletionProjections,
    SequenceProjections,
)
from selfconcept.assistant_axis_cot.transcripts import padded_token_budget_batches
from selfconcept.common.activation_extraction.model_specifics import ModelSpecifics
from selfconcept.common.activation_extraction.extraction import extract_token_span_activations
from selfconcept.common.hf_strong_types import HFTokenizer
from selfconcept.common.loading import load_causal_lm
from selfconcept.common.span_targeting.types import CompletionSpans, PromptCompletion, TokenSpan
from selfconcept.measurement.unit_axes import load_unit_axes

ATTENTION_SINK_TOKEN_COUNT = 1


def load_tokenizer_and_model(model_name: str, attn_implementation: str | None) -> tuple[HFTokenizer, PreTrainedModel]:
    tokenizer: HFTokenizer = AutoTokenizer.from_pretrained(model_name)
    load_kwargs = {} if attn_implementation is None else {"attn_implementation": attn_implementation}
    model: PreTrainedModel = load_causal_lm(model_name, dtype=torch.bfloat16, device_map="auto", **load_kwargs)
    model.eval()
    return tokenizer, model


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


def load_unit_axis_by_layer(
    axis_path: Path,
    layers: Sequence[int],
    num_layers: int,
    hidden_size: int,
) -> dict[int, Float[Tensor, " hidden"]]:
    return {
        layer: torch.from_numpy(direction_by_name["direction"])
        for layer, direction_by_name in load_unit_axes(axis_path, layers, num_layers, hidden_size).items()
    }


@contextmanager
def capture_layer_projections(
    decoder_layers: nn.ModuleList, unit_axis_by_layer: Mapping[int, Float[Tensor, " hidden"]]
) -> Generator[dict[int, Float[Tensor, "batch seq"]], None, None]:
    """Yields a dict that each forward pass in the context fills with every token's projection onto each layer's axis.
    A hook registered earlier on the same layer, such as an ablation, runs first, so its edit is captured."""
    projections_by_layer: dict[int, Float[Tensor, "batch seq"]] = {}

    def make_hook(layer: int, unit_axis: Float[Tensor, " hidden"]):
        def hook(_module: nn.Module, _inputs: Any, output: Any) -> None:
            hidden_states = output[0] if isinstance(output, tuple) else output
            projections_by_layer[layer] = hidden_states.float() @ unit_axis.to(hidden_states.device).float()

        return hook

    handles = [
        decoder_layers[layer].register_forward_hook(make_hook(layer, unit_axis))
        for layer, unit_axis in unit_axis_by_layer.items()
    ]
    try:
        yield projections_by_layer
    finally:
        for handle in handles:
            handle.remove()


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


def project_sequences_in_batches[DirectionName: str](
    model: PreTrainedModel,
    model_specifics: ModelSpecifics,
    examples: Sequence[PromptCompletion],
    unit_direction_by_name: Mapping[DirectionName, Float[Tensor, " hidden"]],
    layer: int,
    pad_token_id: int,
    max_length: int,
    max_batch_tokens: int,
) -> list[SequenceProjections[DirectionName]]:
    sequence_lengths = [
        min(len(example.prompt_token_ids) + len(example.completion_token_ids), max_length) for example in examples
    ]
    projections_by_index: dict[int, SequenceProjections[DirectionName]] = {}
    for batch_indices in tqdm(padded_token_budget_batches(sequence_lengths, max_batch_tokens), desc="Batches"):
        batch_projections = project_sequences(
            model,
            model_specifics,
            [examples[index] for index in batch_indices],
            unit_direction_by_name,
            layer,
            pad_token_id,
            max_length,
        )
        projections_by_index.update(zip(batch_indices, batch_projections, strict=True))
    return [projections_by_index[index] for index in range(len(examples))]
