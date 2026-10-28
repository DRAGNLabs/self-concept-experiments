"""Extract one model's self-recognition vector: mean own-text minus mean other-text activations over the
candidate spans of its stage-2 judge prompts, on train pairs it recognised in both orders.

Usage (from experiments/self-recognition):
    python scripts/3_vectors.py --run.model Qwen/Qwen3-32B
    python scripts/3_vectors.py --run.model zai-org/GLM-4.5-Air-FP8 --run.batch_size 2 --run.smoke_load true
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict, cast

import torch
from jaxtyping import Float
from torch import Tensor, nn
from transformers import AutoTokenizer, PreTrainedModel

from selfconcept.common.activation_extraction.extraction import extract_token_span_activations
from selfconcept.common.activation_extraction.model_specifics import get_model_specifics
from selfconcept.common.hf_strong_types import HFTokenizer
from selfconcept.common.span_targeting.types import TokenSpan
from selfconcept.self_recognition.hf_loading import load_model
from selfconcept.self_recognition.judge_prompt_spans import tokenize_generation_prompt
from selfconcept.self_recognition.judging import judge_chat_template_kwargs
from selfconcept.self_recognition.schemas import (
    DEFAULT_RUN_DIR,
    ActivationMeans,
    ActivationMeansInputs,
    JudgementRecord,
    PooledSpanMeans,
    SelfRecognitionVectors,
    TranscriptRecord,
    activation_means_path,
    judgements_path,
    length_matched_vectors_path,
    prompts_path,
    read_judgements,
    read_prompts,
    read_transcripts,
    transcripts_path,
    vectors_path,
)
from selfconcept.self_recognition.success_filter import LENGTH_MATCHED_MAX_LOG_RATIO, pair_key_tuple, recognised_pairs
from selfconcept.self_recognition.vectors import (
    JudgementsInBothOrders,
    JudgePromptSpans,
    TranscriptKey,
    authorship_balanced_length_axis,
    build_self_recognition_vectors,
    candidate_char_counts,
    difference_in_means,
    is_length_matched,
    judge_prompt_token_spans,
    length_projection_spearman,
    log_length_ratios,
    longer_minus_shorter_axis,
    non_recognised_judgement_pairs,
    pair_differences,
    per_layer_cosine,
    pooled_own_and_other,
    sample_without_replacement,
    span_length,
    split_half_cosine,
    transcript_key,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MIN_LENGTH_MATCHED_PAIRS = 30


@dataclass(frozen=True)
class RunConfig:
    model: str
    run_dir: Path = DEFAULT_RUN_DIR
    data_dir: Path = Path("data")
    batch_size: int = 4
    max_length: int = 16384
    limit_pairs: int | None = None
    smoke_load: bool = False
    seed: int = 0
    recompute: bool = False


class PairExample(TypedDict):
    pair_index: int
    judgement: JudgementRecord
    spans: JudgePromptSpans
    own_char_count: int
    other_char_count: int


def log_peak_gpu_memory(stage: str) -> None:
    peak_gib_by_device = {
        device: round(torch.cuda.max_memory_allocated(device) / 2**30, 1) for device in range(torch.cuda.device_count())
    }
    logger.info("peak GPU memory (GiB) %s: %s", stage, peak_gib_by_device)


def load_tokenizer(model_name: str) -> tuple[HFTokenizer, int]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    return cast(HFTokenizer, tokenizer), pad_token_id


def load_model_and_decoder_layers(model_name: str, tokenizer: HFTokenizer) -> tuple[PreTrainedModel, nn.ModuleList]:
    model = load_model(model_name)
    decoder_layers = get_model_specifics(model_name, tokenizer).get_decoder_layers(model)
    log_peak_gpu_memory("after load")
    return model, decoder_layers


def check_load_and_one_forward(model_name: str) -> None:
    tokenizer, pad_token_id = load_tokenizer(model_name)
    model, decoder_layers = load_model_and_decoder_layers(model_name, tokenizer)
    logger.info(
        "%d decoder layers of %s; parameter dtypes %s",
        len(decoder_layers),
        type(decoder_layers[0]).__name__,
        sorted({str(parameter.dtype) for parameter in model.parameters()}),
    )
    token_ids = tokenize_generation_prompt(
        [{"role": "user", "content": "Say hello in one sentence."}], tokenizer, **judge_chat_template_kwargs(model_name)
    )
    (activations,) = extract_token_span_activations(
        model, [token_ids], [[TokenSpan(0, len(token_ids))]], decoder_layers, pad_token_id, len(token_ids)
    )
    if not torch.isfinite(activations).all():
        raise ValueError("non-finite activations in the smoke forward pass")
    logger.info("smoke forward: finite activations of shape %s", tuple(activations.shape))
    log_peak_gpu_memory("after forward")


def read_transcripts_by_key(run_dir: Path, models: set[str]) -> dict[TranscriptKey, TranscriptRecord]:
    return {
        transcript_key(transcript): transcript
        for model in sorted(models)
        for transcript in read_transcripts(transcripts_path(run_dir, model))
    }


def fitting_pair_examples(
    pairs: Sequence[JudgementsInBothOrders],
    transcripts_by_key: dict[TranscriptKey, TranscriptRecord],
    tokenizer: HFTokenizer,
    chat_template_kwargs: dict[str, Any],
    max_length: int,
) -> list[PairExample]:
    """Both orders of every pair whose two prompts fit in max_length, so the own and other texts stay balanced
    across slots."""
    pair_examples: list[PairExample] = []
    for pair_index, pair in enumerate(pairs):
        examples: list[PairExample] = [
            {
                "pair_index": pair_index,
                "judgement": judgement,
                "spans": judge_prompt_token_spans(judgement, transcripts_by_key, tokenizer, chat_template_kwargs),
                "own_char_count": own_char_count,
                "other_char_count": other_char_count,
            }
            for judgement in pair
            for own_char_count, other_char_count in [candidate_char_counts(judgement, transcripts_by_key)]
        ]
        if all(len(example["spans"][0]) <= max_length for example in examples):
            pair_examples.extend(examples)
    skipped_pairs = len(pairs) - len({example["pair_index"] for example in pair_examples})
    logger.info("%d of %d pairs skipped: a prompt is longer than %d tokens", skipped_pairs, len(pairs), max_length)
    return pair_examples


def pool_in_batches(
    model: PreTrainedModel,
    pair_examples: Sequence[PairExample],
    decoder_layers: nn.ModuleList,
    pad_token_id: int,
    max_length: int,
    batch_size: int,
) -> PooledSpanMeans:
    """Longest prompts first, so an out-of-memory batch fails early."""
    by_length = sorted(pair_examples, key=lambda example: len(example["spans"][0]), reverse=True)
    pooled_batches = [
        pooled_own_and_other(
            model,
            [example["spans"] for example in by_length[batch_start : batch_start + batch_size]],
            decoder_layers,
            pad_token_id,
            max_length,
        )
        for batch_start in range(0, len(by_length), batch_size)
    ]
    return {
        "own": torch.cat([own for own, _ in pooled_batches]),
        "other": torch.cat([other for _, other in pooled_batches]),
        "pair_indices": torch.tensor([example["pair_index"] for example in by_length]),
        "prompt_ids": [example["judgement"]["prompt_id"] for example in by_length],
        "other_models": [example["judgement"]["other_model"] for example in by_length],
        "target_sample_indices": [example["judgement"]["target_sample_index"] for example in by_length],
        "other_sample_indices": [example["judgement"]["other_sample_index"] for example in by_length],
        "target_positions": [example["judgement"]["target_position"] for example in by_length],
        "own_token_counts": [span_length(example["spans"][1]) for example in by_length],
        "other_token_counts": [span_length(example["spans"][2]) for example in by_length],
        "own_char_counts": [example["own_char_count"] for example in by_length],
        "other_char_counts": [example["other_char_count"] for example in by_length],
    }


def compute_activation_means(
    run: RunConfig,
    inputs: ActivationMeansInputs,
    recognised: Sequence[JudgementsInBothOrders],
    non_recognised: Sequence[JudgementsInBothOrders],
    judgements: Sequence[JudgementRecord],
) -> ActivationMeans:
    tokenizer, pad_token_id = load_tokenizer(run.model)
    chat_template_kwargs = judge_chat_template_kwargs(run.model)
    transcripts_by_key = read_transcripts_by_key(
        run.run_dir, {run.model, *(judgement["other_model"] for judgement in judgements)}
    )
    recognised_examples = fitting_pair_examples(
        recognised, transcripts_by_key, tokenizer, chat_template_kwargs, run.max_length
    )
    non_recognised_examples = fitting_pair_examples(
        non_recognised, transcripts_by_key, tokenizer, chat_template_kwargs, run.max_length
    )
    model, decoder_layers = load_model_and_decoder_layers(run.model, tokenizer)
    return {
        **inputs,
        "recognised": pool_in_batches(
            model, recognised_examples, decoder_layers, pad_token_id, run.max_length, run.batch_size
        ),
        "non_recognised": (
            pool_in_batches(
                model, non_recognised_examples, decoder_layers, pad_token_id, run.max_length, run.batch_size
            )
            if non_recognised_examples
            else None
        ),
    }


def activation_means_inputs(means: ActivationMeansInputs) -> ActivationMeansInputs:
    return {
        "model": means["model"],
        "limit_pairs": means["limit_pairs"],
        "seed": means["seed"],
        "max_length": means["max_length"],
        "recognised_pair_keys": means["recognised_pair_keys"],
        "non_recognised_pair_keys": means["non_recognised_pair_keys"],
    }


def read_matching_activation_means(path: Path, inputs: ActivationMeansInputs) -> ActivationMeans | None:
    if not path.exists():
        logger.info("no cached activation means at %s", path)
        return None
    means = cast(ActivationMeans, torch.load(path, map_location="cpu", weights_only=True))
    if activation_means_inputs(means) != inputs:
        logger.info("cached activation means at %s are for other inputs; recomputing", path)
        return None
    logger.info("loaded cached activation means from %s", path)
    return means


def log_span_lengths(label: str, pooled: PooledSpanMeans) -> None:
    logger.info(
        "%s: mean span length own %.1f, other %.1f tokens; own %.1f, other %.1f characters; own longer in %.1f%%",
        label,
        sum(pooled["own_token_counts"]) / len(pooled["own_token_counts"]),
        sum(pooled["other_token_counts"]) / len(pooled["other_token_counts"]),
        sum(pooled["own_char_counts"]) / len(pooled["own_char_counts"]),
        sum(pooled["other_char_counts"]) / len(pooled["other_char_counts"]),
        100 * (log_length_ratios(pooled) > 0).double().mean().item(),
    )


def format_per_layer(values: Float[Tensor, " layer"]) -> str:
    return " ".join(f"{value:.3f}" for value in values.tolist())


def log_length_checks(run: RunConfig, recognised: PooledSpanMeans, axis: Float[Tensor, "layer hidden"]) -> None:
    pairs = pair_differences(recognised)
    logger.info(
        "per-layer cosine with the longer-minus-shorter contrast (%d untied pairs): %s",
        int((pairs["log_length_ratio"] != 0).sum()),
        format_per_layer(per_layer_cosine(axis, longer_minus_shorter_axis(pairs))),
    )
    balanced_length_axis = authorship_balanced_length_axis(pairs)
    if balanced_length_axis is None:
        logger.info("no authorship-balanced length contrast: own text is longer in all or none of the pairs")
    else:
        logger.info(
            "per-layer cosine with the authorship-balanced length contrast: %s",
            format_per_layer(per_layer_cosine(axis, balanced_length_axis)),
        )
    layer_count = axis.shape[0]
    representative_layers = [layer_count // 4, layer_count // 2, 3 * layer_count // 4, layer_count - 1]
    spearman_by_layer = dict(
        zip(representative_layers, length_projection_spearman(pairs, axis, representative_layers), strict=True)
    )
    logger.info(
        "Spearman over %d pairs of log(own/other chars) vs pair projection on the unit axis, by layer: %s",
        len(pairs["log_length_ratio"]),
        {layer: round(spearman, 3) for layer, spearman in spearman_by_layer.items()},
    )

    length_matched = is_length_matched(recognised)
    length_matched_pair_count = len(recognised["pair_indices"][length_matched].unique())
    length_matched_prompt_ids = [
        prompt_id for prompt_id, matched in zip(recognised["prompt_ids"], length_matched.tolist(), strict=True) if matched
    ]
    logger.info(
        "length-matched (|log length ratio| <= %.3f): %d of %d pairs over %d prompts",
        LENGTH_MATCHED_MAX_LOG_RATIO,
        length_matched_pair_count,
        len(pairs["log_length_ratio"]),
        len(set(length_matched_prompt_ids)),
    )
    if length_matched_pair_count < MIN_LENGTH_MATCHED_PAIRS:
        logger.info("fewer than %d length-matched pairs: no length-matched vector", MIN_LENGTH_MATCHED_PAIRS)
        return
    length_matched_vectors = build_self_recognition_vectors(
        run.model, recognised["own"][length_matched], recognised["other"][length_matched], length_matched_prompt_ids
    )
    logger.info(
        "per-layer cosine of the length-matched vector with the full vector: %s",
        format_per_layer(per_layer_cosine(axis, length_matched_vectors["axis"])),
    )
    save(length_matched_vectors, length_matched_vectors_path(run.run_dir, run.model))


def save(contents: SelfRecognitionVectors | ActivationMeans, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(contents, path)
    logger.info("saved %s", path)


def main(run: RunConfig) -> None:
    logger.info("%s", run)
    if run.smoke_load:
        check_load_and_one_forward(run.model)
        return

    train_prompt_ids = {prompt["id"] for prompt in read_prompts(prompts_path(run.data_dir)) if prompt["split"] == "train"}
    judgements: list[JudgementRecord] = [
        judgement
        for judgement in read_judgements(judgements_path(run.run_dir, run.model))
        if judgement["prompt_id"] in train_prompt_ids
    ]
    recognised = [(pair["target_first"], pair["target_second"]) for pair in recognised_pairs(judgements)]
    recognised = recognised[: run.limit_pairs]
    non_recognised = sample_without_replacement(non_recognised_judgement_pairs(judgements), len(recognised), run.seed)
    logger.info(
        "%d train judgements; %d recognised pairs used; %d non-recognised pairs sampled",
        len(judgements),
        len(recognised),
        len(non_recognised),
    )

    inputs: ActivationMeansInputs = {
        "model": run.model,
        "limit_pairs": run.limit_pairs,
        "seed": run.seed,
        "max_length": run.max_length,
        "recognised_pair_keys": [pair_key_tuple(target_first) for target_first, _ in recognised],
        "non_recognised_pair_keys": [pair_key_tuple(target_first) for target_first, _ in non_recognised],
    }
    means_path = activation_means_path(run.run_dir, run.model)
    cached_means = None if run.recompute else read_matching_activation_means(means_path, inputs)
    means = (
        cached_means
        if cached_means is not None
        else compute_activation_means(run, inputs, recognised, non_recognised, judgements)
    )
    if cached_means is None:
        save(means, means_path)

    pooled = means["recognised"]
    log_span_lengths("recognised", pooled)
    vectors = build_self_recognition_vectors(run.model, pooled["own"], pooled["other"], pooled["prompt_ids"])
    logger.info(
        "axis %s from %d own / %d other spans over %d prompts",
        tuple(vectors["axis"].shape),
        vectors["positive_count"],
        vectors["negative_count"],
        len(vectors["train_prompt_ids"]),
    )
    logger.info("per-layer axis norm: %s", format_per_layer(vectors["axis"].norm(dim=-1)))
    logger.info(
        "per-layer split-half cosine: %s",
        format_per_layer(split_half_cosine(pooled["own"], pooled["other"], pooled["pair_indices"], run.seed)),
    )

    non_recognised_pooled = means["non_recognised"]
    if non_recognised_pooled is not None:
        log_span_lengths("non-recognised", non_recognised_pooled)
        logger.info(
            "per-layer cosine with the non-recognised contrast: %s",
            format_per_layer(
                per_layer_cosine(
                    vectors["axis"], difference_in_means(non_recognised_pooled["own"], non_recognised_pooled["other"])
                )
            ),
        )

    log_length_checks(run, pooled, vectors["axis"])
    save(vectors, vectors_path(run.run_dir, run.model))


if __name__ == "__main__":
    from jsonargparse import auto_cli

    auto_cli(main)
