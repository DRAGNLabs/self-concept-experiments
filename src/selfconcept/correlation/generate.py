"""Generate while scoring every processed token with one residual-stream measurement."""
import hashlib
from itertools import accumulate
from pathlib import Path
from typing import Any, cast, NamedTuple

from jaxtyping import Float
import numpy as np
import torch
from transformers import PreTrainedModel

from selfconcept.codebench import harness
from selfconcept.common.hf_strong_types import configure_call, Conversation, HFTokenizer, OffsetMappingPresent
from selfconcept.common.jsonl import append_jsonl
from selfconcept.measurement.capture import capture
from selfconcept.measurement.interface import Measurement, MeasurementName
from selfconcept.measurement.templates import (
    content_token_indices, final_answer_text, ModelFamily, render_generation_prompt, response_spans, ResponseSpans)
from .records import GenerationRecord, GenerationStatus, Region, RegionSummary, Sampling

type TokenScores = Float[np.ndarray, "token"]


class MeasuredModel(NamedTuple):
    model: PreTrainedModel
    tokenizer: HFTokenizer
    family: ModelFamily
    measurement: Measurement


class GenerationSettings(NamedTuple):
    output_directory: Path
    sampling: Sampling
    seed: int
    max_new_tokens: int


class ScoredGeneration(NamedTuple):
    token_ids: list[int]
    token_scores_by_layer: dict[int, TokenScores]


def sampling_kwargs(temperature: float, top_p: float, top_k: int) -> Sampling:
    if temperature > 0:
        return {"do_sample": True, "temperature": temperature, "top_p": top_p, "top_k": top_k}
    return {"do_sample": False}


def summarize_region_scores(token_scores: TokenScores, indices: list[int]) -> RegionSummary:
    selected = token_scores[[index for index in indices if index < len(token_scores)]]
    if not len(selected):
        return {"n": 0, "mean": None, "p90": None}
    return {"n": len(selected), "mean": float(selected.mean()), "p90": float(np.quantile(selected, .9))}


def generated_token_offsets(tokenizer: HFTokenizer, token_ids: list[int], text: str) -> OffsetMappingPresent | None:
    # Single-token decoding is exact for ordinary text, but not split UTF-8.
    pieces = [tokenizer.decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)
              for token_id in token_ids]
    if "".join(pieces) == text:
        ends = list(accumulate(len(piece) for piece in pieces))
        return list(zip([0] + ends[:-1], ends))
    encoding = configure_call(tokenizer).return_offsets_mapping(True)(text, add_special_tokens=False)
    if encoding["input_ids"] == token_ids:
        return encoding["offset_mapping"]
    return None  # Never assign an activation to the wrong token.


def generation_seed(seed: int, scenario: str, example_id: str, turn: int) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{scenario}:{example_id}:{turn}".encode()).digest()[:4], "big")


def check_context_budget(model: PreTrainedModel, prompt_length: int, max_new_tokens: int) -> None:
    context_limit = getattr(getattr(model.config, "text_config", model.config), "max_position_embeddings", None)
    if context_limit and prompt_length + max_new_tokens > context_limit:
        raise ValueError("Prompt plus generation budget exceeds model context; refusing silent truncation")


def generate_scored(measured_model: MeasuredModel, prompt_ids: list[int], sampling: Sampling,
                    max_new_tokens: int, seed: int) -> ScoredGeneration:
    model, tokenizer, _, measurement = measured_model
    input_ids = torch.tensor([prompt_ids], device=next(model.get_input_embeddings().parameters()).device)
    try:
        torch.manual_seed(seed)
        with torch.inference_mode(), capture(model, measurement.layers, measurement.score) as scores_by_layer:
            output = cast(Any, model).generate(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                                               max_new_tokens=max_new_tokens, **sampling,
                                               pad_token_id=tokenizer.eos_token_id, use_cache=True)
    except torch.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        raise harness.GenerationOOM(str(exc)[:200]) from None
    token_ids: list[int] = output[0].tolist()
    token_scores_by_layer = {layer: torch.cat(chunks).numpy() for layer, chunks in scores_by_layer.items()}
    for token_scores in token_scores_by_layer.values():
        if len(token_scores) != len(token_ids) - 1:
            raise ValueError(f"Unexpected cached generation alignment: {len(token_scores)} vs {len(token_ids) - 1}")
    return ScoredGeneration(token_ids, token_scores_by_layer)


def last_message_token_indices(tokenizer: HFTokenizer, messages: Conversation, prompt: str,
                               prompt_offsets: OffsetMappingPresent, prompt_ids: list[int]) -> list[int]:
    # Measure only content from the last externally supplied message, excluding
    # assistant history, system instructions and generation-role delimiters.
    # Native templates may trim user content (notably coding feedback's
    # leading newline). Match the shared non-whitespace content exactly.
    last_message = messages[-1]["content"].strip()
    start = prompt.rfind(last_message)
    if start < 0:
        raise ValueError("Cannot exactly locate the last message in the rendered prompt")
    return content_token_indices(prompt_offsets, [(start, start + len(last_message))], tokenizer.all_special_ids, prompt_ids)


def response_token_indices_by_region(tokenizer: HFTokenizer, new_token_ids: list[int], raw_response: str,
                                     spans: ResponseSpans, prompt_length: int) -> dict[Region, list[int]] | None:
    offsets = generated_token_offsets(tokenizer, new_token_ids, raw_response)
    if offsets is None:
        return None
    return {region: [prompt_length + index
                     for index in content_token_indices(offsets, spans[region], tokenizer.all_special_ids, new_token_ids)]
            for region in ("cot", "final")}


def generation_status(new_token_ids: list[int], max_new_tokens: int, eos_token_ids: list[int],
                      final_answer: str) -> tuple[GenerationStatus, bool]:
    truncated = len(new_token_ids) >= max_new_tokens and new_token_ids[-1] not in eos_token_ids
    return ("truncated" if truncated else "complete" if final_answer else "no_final"), truncated


def write_trace(output_directory: Path, scenario: str, example_id: str, turn: int, measurement_name: MeasurementName,
                scored_generation: ScoredGeneration, prompt_length: int,
                token_indices_by_region: dict[Region, list[int]]) -> Path:
    token_ids, token_scores_by_layer = scored_generation
    trace_key = hashlib.sha256(f"{scenario}:{example_id}:{turn}".encode()).hexdigest()[:24]
    trace_path = output_directory / "traces" / f"{trace_key}.npz"
    trace_path.parent.mkdir(exist_ok=True)
    np.savez_compressed(
        trace_path, allow_pickle=True,
        input_ids=np.asarray(token_ids, dtype=np.int32), prompt_length=np.array(prompt_length),
        **{f"{measurement_name.replace('-', '_')}_layer_{layer}": token_scores
           for layer, token_scores in token_scores_by_layer.items()},
        **{f"indices_{region}": np.asarray([index for index in indices if index < len(token_ids) - 1], dtype=np.int32)
           for region, indices in token_indices_by_region.items()})
    return trace_path


def measured_generate(measured_model: MeasuredModel, settings: GenerationSettings, scenario: str,
                      example_id: str, messages: Conversation, turn: int) -> GenerationRecord:
    """Generate one turn, write its token trace, and append its record to generations.jsonl."""
    model, tokenizer, family, measurement = measured_model
    prompt = render_generation_prompt(tokenizer, family, messages)
    prompt_encoding = configure_call(tokenizer).return_offsets_mapping(True)(prompt, add_special_tokens=False)
    prompt_ids = prompt_encoding["input_ids"]
    prompt_length = len(prompt_ids)
    check_context_budget(model, prompt_length, settings.max_new_tokens)
    seed = generation_seed(settings.seed, scenario, example_id, turn)
    scored_generation = generate_scored(measured_model, prompt_ids, settings.sampling, settings.max_new_tokens, seed)
    new_token_ids = scored_generation.token_ids[prompt_length:]
    raw_response = tokenizer.decode(new_token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    spans = response_spans(prompt, raw_response, family)
    final_answer = final_answer_text(raw_response, spans, tokenizer)
    eos_token_id = model.generation_config.eos_token_id
    eos_token_ids = eos_token_id if isinstance(eos_token_id, list) else [eos_token_id]
    status, truncated = generation_status(new_token_ids, settings.max_new_tokens, eos_token_ids, final_answer)
    response_indices_by_region = response_token_indices_by_region(tokenizer, new_token_ids, raw_response, spans, prompt_length)
    token_indices_by_region: dict[Region, list[int]] = {
        "prompt": last_message_token_indices(tokenizer, messages, prompt, prompt_encoding["offset_mapping"], prompt_ids),
        **(response_indices_by_region or {})}
    trace_path = write_trace(settings.output_directory, scenario, example_id, turn, measurement.name,
                             scored_generation, prompt_length, token_indices_by_region)
    record: GenerationRecord = {
        "example_id": example_id, "scenario": scenario, "turn": turn,
        "response": final_answer, "raw_response": raw_response, "rendered_prompt": prompt,
        "status": status, "truncated": truncated, "generated_tokens": len(new_token_ids),
        "sampling": settings.sampling, "seed": seed,
        "prompt_tokens": prompt_length, "spans": spans,
        "scores": {str(layer): {region: summarize_region_scores(token_scores, indices)
                                for region, indices in token_indices_by_region.items()}
                   for layer, token_scores in scored_generation.token_scores_by_layer.items()},
        "generated_alignment": "exact" if response_indices_by_region is not None else "unavailable",
        "trace": str(trace_path.relative_to(settings.output_directory)), "measurement": measurement.name}
    append_jsonl(settings.output_directory / "generations.jsonl", record)
    return record
