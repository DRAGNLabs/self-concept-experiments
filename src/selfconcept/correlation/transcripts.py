"""Score existing transcripts with one teacher-forced forward pass per transcript."""
import logging
from pathlib import Path
from typing import Literal, NamedTuple

import torch

from selfconcept.codebench import harness
from selfconcept.common.hf_strong_types import (
    configure_apply_chat_template, configure_call, Conversation, HFTokenizer, OffsetMappingPresent)
from selfconcept.common.jsonl import append_jsonl, read_jsonl
from selfconcept.common.resumable_jsonl import process_unrecorded_items
from selfconcept.measurement.capture import capture
from selfconcept.measurement.templates import final_answer_text, ResponseSpans, response_spans
from .generate import (
    last_message_token_indices, MeasuredModel, response_token_indices_by_region, ScoredGeneration,
    summarize_region_scores, write_trace)
from .config import RunConfig
from .records import GenerationRecord, GenerationStatus, OutcomeRecord, Region, RegionSummary, TranscriptRecord
from .scenarios import oom_outcome, write_judge_inputs

logger = logging.getLogger(__name__)

type Alignment = Literal["exact", "unavailable"]


class TokenizedTranscript(NamedTuple):
    prompt: str
    prompt_ids: list[int]
    prompt_offsets: OffsetMappingPresent
    response_ids: list[int]


def tokenize_transcript(tokenizer: HFTokenizer, transcript: TranscriptRecord) -> TokenizedTranscript:
    prompt = configure_apply_chat_template(tokenizer).tokenize(False)(
        transcript["messages"], add_generation_prompt=True, **transcript["chat_kwargs"])
    prompt_encoding = configure_call(tokenizer).return_offsets_mapping(True)(prompt, add_special_tokens=False)
    response_ids: list[int] = configure_call(tokenizer)(transcript["raw_response"], add_special_tokens=False)["input_ids"]
    return TokenizedTranscript(prompt, prompt_encoding["input_ids"], prompt_encoding["offset_mapping"], response_ids)


def forward_scored(measured_model: MeasuredModel, token_ids: list[int]) -> ScoredGeneration:
    model, _, _, measurement = measured_model
    input_ids = torch.tensor([token_ids], device=next(model.get_input_embeddings().parameters()).device)
    try:
        with torch.inference_mode(), capture(model, measurement.layers, measurement.score) as scores_by_layer:
            model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), use_cache=False, logits_to_keep=1)
    except torch.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        raise harness.GenerationOOM(str(exc)[:200]) from None
    token_scores_by_layer = {layer: torch.cat(chunks)[:-1].numpy() for layer, chunks in scores_by_layer.items()}
    for token_scores in token_scores_by_layer.values():
        if len(token_scores) != len(token_ids) - 1:
            raise ValueError(f"Unexpected teacher-forced alignment: {len(token_scores)} vs {len(token_ids) - 1}")
    return ScoredGeneration(token_ids, token_scores_by_layer)


def transcript_token_indices_by_region(tokenizer: HFTokenizer, messages: Conversation, raw_response: str,
                                       tokenized: TokenizedTranscript,
                                       spans: ResponseSpans) -> tuple[dict[Region, list[int]], Alignment]:
    prompt_indices = last_message_token_indices(
        tokenizer, messages, tokenized.prompt, tokenized.prompt_offsets, tokenized.prompt_ids)
    response_indices_by_region = response_token_indices_by_region(
        tokenizer, tokenized.response_ids, raw_response, spans, len(tokenized.prompt_ids))
    if response_indices_by_region is None:
        return {"prompt": prompt_indices}, "unavailable"
    return {"prompt": prompt_indices, **response_indices_by_region}, "exact"


def region_summaries_by_layer(scored_transcript: ScoredGeneration, token_indices_by_region: dict[Region, list[int]]
                              ) -> dict[str, dict[Region, RegionSummary]]:
    return {str(layer): {region: summarize_region_scores(token_scores, indices)
                         for region, indices in token_indices_by_region.items()}
            for layer, token_scores in scored_transcript.token_scores_by_layer.items()}


def transcript_status(truncated: bool, final_answer: str) -> GenerationStatus:
    if truncated:
        return "truncated"
    if not final_answer:
        return "no_final"
    return "complete"


def measured_score(measured_model: MeasuredModel, output_directory: Path,
                   transcript: TranscriptRecord) -> GenerationRecord:
    """Score one transcript turn, write its token trace, and append its record to generations.jsonl."""
    _, tokenizer, family, measurement = measured_model
    raw_response = transcript["raw_response"]
    tokenized = tokenize_transcript(tokenizer, transcript)
    scored_transcript = forward_scored(measured_model, tokenized.prompt_ids + tokenized.response_ids)
    spans = response_spans(tokenized.prompt, raw_response, family)
    final_answer = final_answer_text(raw_response, spans, tokenizer)
    token_indices_by_region, alignment = transcript_token_indices_by_region(
        tokenizer, transcript["messages"], raw_response, tokenized, spans)
    trace_path = write_trace(output_directory, transcript["scenario"], transcript["example_id"], transcript["turn"],
                             measurement.name, scored_transcript, len(tokenized.prompt_ids), token_indices_by_region)
    record: GenerationRecord = {
        "example_id": transcript["example_id"], "scenario": transcript["scenario"], "turn": transcript["turn"],
        "response": final_answer, "raw_response": raw_response, "rendered_prompt": tokenized.prompt,
        "status": transcript_status(transcript["truncated"], final_answer), "truncated": transcript["truncated"],
        "generated_tokens": len(tokenized.response_ids), "prompt_tokens": len(tokenized.prompt_ids), "spans": spans,
        "scores": region_summaries_by_layer(scored_transcript, token_indices_by_region),
        "generated_alignment": alignment,
        "trace": str(trace_path.relative_to(output_directory)), "measurement": measurement.name}
    append_jsonl(output_directory / "generations.jsonl", record)
    return record


def score_transcript(measured_model: MeasuredModel, output_directory: Path,
                     transcript: TranscriptRecord) -> OutcomeRecord:
    try:
        measured_score(measured_model, output_directory, transcript)
        outcome = transcript["outcome"]
    except harness.GenerationOOM as error:
        outcome = oom_outcome(transcript["scenario"], transcript["example_id"], transcript["outcome"]["model_key"], error)
    logger.info(f"{transcript['scenario']}/{transcript['example_id']}: {outcome['status']} {outcome['label']}")
    return outcome


def outcome_key(outcome: OutcomeRecord | TranscriptRecord) -> tuple[str, str]:
    return outcome["scenario"], outcome["example_id"]


def score_transcripts(config: RunConfig, measured_model: MeasuredModel) -> None:
    assert config.transcripts is not None, "main only scores transcripts when --transcripts is set"
    transcripts: list[TranscriptRecord] = read_jsonl(config.transcripts)
    if len({outcome_key(transcript) for transcript in transcripts}) != len(transcripts):
        raise ValueError("Transcript (scenario, example_id) pairs must be unique")
    outcomes_path = config.out / "outcomes.jsonl"
    process_unrecorded_items(
        transcripts, outcomes_path, outcome_key, outcome_key,
        lambda chunk: [score_transcript(measured_model, config.out, transcript) for transcript in chunk], chunk_size=1)
    write_judge_inputs(config.out, outcomes_path, config.scenarios)
