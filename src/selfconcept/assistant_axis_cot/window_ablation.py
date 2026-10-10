from typing import TypedDict

import numpy as np
import torch
from jaxtyping import Bool, Float, Float32, Int8
from torch import Tensor, nn
from transformers import PreTrainedModel

from selfconcept.common.ablation import mean_ablate_decoder_layer
from selfconcept.common.activation_extraction.model_specifics import ModelSpecifics
from selfconcept.common.span_targeting.types import PromptCompletion
from selfconcept.common.torch_utils import build_padded_batch

from selfconcept.assistant_axis_cot.ablation import RegionMeans, final_hidden_states, next_token_kl
from selfconcept.assistant_axis_cot.projection import sequence_region_codes
from selfconcept.assistant_axis_cot.records import (
    LABELLED_SEQUENCE_REGIONS,
    SEQUENCE_REGIONS,
    AblationWindow,
    AxisName,
    Condition,
    SequenceRegion,
    WindowAblationRecord,
    WindowAblationResult,
)


class WindowAblationRun(TypedDict):
    """ablate_windows over one condition's conversations; the records' kl direction axes are ordered by
    direction_names."""

    model: str
    layer: int
    axis_name: AxisName
    condition: Condition
    direction_names: list[str]
    window_size: int
    records: list[WindowAblationRecord]


def region_end_anchored_windows(
    region_codes: Int8[np.ndarray, " n_tokens"], region: SequenceRegion, window_size: int
) -> list[AblationWindow]:
    """Back-to-back windows lying wholly within the region, the first ending at its last token, going back until
    a window would leave the region."""
    region_code = SEQUENCE_REGIONS.index(region)
    region_positions = np.flatnonzero(region_codes == region_code)
    if len(region_positions) == 0:
        return []
    region_end = int(region_positions[-1]) + 1
    windows: list[AblationWindow] = []
    for end in range(region_end, window_size - 1, -window_size):
        if not (region_codes[end - window_size : end] == region_code).all():
            break
        windows.append(
            {"region": region, "start": end - window_size, "end": end, "tokens_before_region_end": region_end - end}
        )
    return windows


def window_masks(
    windows: list[AblationWindow], sequence_length: int, device: torch.device
) -> Bool[Tensor, "window seq"]:
    masks = torch.zeros(len(windows), sequence_length, dtype=torch.bool)
    for row, window in enumerate(windows):
        masks[row, window["start"] : window["end"]] = True
    return masks.to(device)


def kl_from_window_start(
    clean_hidden: Float[Tensor, "token hidden"],
    ablated_hidden: Float[Tensor, "token hidden"],
    output_embeddings: nn.Module,
    window: AblationWindow,
    chunk_size: int,
) -> Float32[np.ndarray, " token_minus_one"]:
    """next_token_kl, NaN before the window's start, where the ablation cannot reach."""
    kl = np.full(len(clean_hidden) - 1, np.nan, dtype=np.float32)
    kl[window["start"] :] = next_token_kl(
        clean_hidden[window["start"] :], ablated_hidden[window["start"] :], output_embeddings, chunk_size
    )
    return kl


def ablate_windows(
    model: PreTrainedModel,
    model_specifics: ModelSpecifics,
    example: PromptCompletion,
    region_means: RegionMeans,
    pad_token_id: int,
    max_length: int,
    window_size: int,
    max_batch_tokens: int,
    kl_chunk_size: int,
) -> WindowAblationResult:
    """Per direction, one forward pass per region_end_anchored_windows window of every labelled region with that
    window's tokens mean-ablated along the direction to the region's mean at the region means' layer. Windows share a
    batch of copies of the conversation, up to max_batch_tokens tokens, and each is compared to a clean pass of the
    same batch size, since a different size changes the kernels' rounding."""
    token_ids = example.prompt_token_ids + example.completion_token_ids
    region_codes = sequence_region_codes(example, model_specifics.split_completion(example), max_length)
    input_ids, attention_mask = build_padded_batch([token_ids], pad_token_id, max_length, model.device)
    decoder_layer = model_specifics.get_decoder_layers(model)[region_means["layer"]]
    output_embeddings = model.get_output_embeddings()
    assert output_embeddings is not None

    windows_by_region: dict[SequenceRegion, list[AblationWindow]] = {
        region: region_end_anchored_windows(region_codes, region, window_size) for region in LABELLED_SEQUENCE_REGIONS
    }
    copies_per_pass = max(1, max_batch_tokens // len(region_codes))
    clean_hidden_by_copy_count: dict[int, Float[Tensor, "copy seq hidden"]] = {}
    kl_by_window_by_direction: list[list[Float32[np.ndarray, " token_minus_one"]]] = []
    for direction_name, unit_direction in region_means["unit_direction_by_name"].items():
        kl_by_window: list[Float32[np.ndarray, " token_minus_one"]] = []
        for region, region_windows in windows_by_region.items():
            target_projection = region_means["moments_by_region_by_direction"][direction_name][region]["mean"]
            for batch_start in range(0, len(region_windows), copies_per_pass):
                batch_windows = region_windows[batch_start : batch_start + copies_per_pass]
                batch_input_ids = input_ids.expand(len(batch_windows), -1)
                batch_attention_mask = attention_mask.expand(len(batch_windows), -1)
                if len(batch_windows) not in clean_hidden_by_copy_count:
                    clean_hidden_by_copy_count[len(batch_windows)] = final_hidden_states(
                        model, batch_input_ids, batch_attention_mask
                    )
                with mean_ablate_decoder_layer(
                    decoder_layer,
                    unit_direction,
                    target_projection,
                    window_masks(batch_windows, len(region_codes), input_ids.device),
                ):
                    ablated_hidden = final_hidden_states(model, batch_input_ids, batch_attention_mask)
                clean_hidden = clean_hidden_by_copy_count[len(batch_windows)]
                kl_by_window.extend(
                    kl_from_window_start(
                        clean_hidden[row], ablated_hidden[row], output_embeddings, window, kl_chunk_size
                    )
                    for row, window in enumerate(batch_windows)
                )
        kl_by_window_by_direction.append(kl_by_window)

    windows = [window for region_windows in windows_by_region.values() for window in region_windows]
    return {
        "region_codes": region_codes,
        "windows": windows,
        "kl": np.array(kl_by_window_by_direction, dtype=np.float32).reshape(
            len(kl_by_window_by_direction), len(windows), len(region_codes) - 1
        ),
        "truncated": len(token_ids) > max_length,
    }
