"""Project one model's self-recognition vector onto (A) the candidate-text spans of its train and holdout judge
prompts and (B) plain transcripts with its own or another model's response in its assistant turn.

Train records only choose each reading's layer. Holdout records also get projections onto seeded random
directions at that layer. Reading A also writes the mean per-token residual norm per layer, which scales the
stage-4b steering coefficients.

Usage (from experiments/self-recognition):
    python scripts/4a_read_vectors.py --config configs/4_validation.yaml --run.model Qwen/Qwen3-32B
"""

import json
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from itertools import batched
from pathlib import Path
from typing import Any, Literal, cast

import jsonlines
import torch
from jaxtyping import Float
from torch import Tensor, nn
from transformers import AutoTokenizer, PreTrainedModel

from selfconcept.common.activation_extraction.extraction import (
    extract_span_activations,
    extract_token_span_activations,
)
from selfconcept.common.activation_extraction.model_specifics import ModelSpecifics, get_model_specifics
from selfconcept.common.hf_strong_types import HFTokenizer
from selfconcept.common.span_targeting.span_targeters import ResponseTargeter
from selfconcept.common.span_targeting.types import PromptCompletion
from selfconcept.measurement.interface import TokenScorer
from selfconcept.self_recognition.hf_loading import load_model, peak_gpu_memory_gib_by_device
from selfconcept.self_recognition.judging import judge_chat_template_kwargs, read_transcripts_by_model
from selfconcept.self_recognition.reading import (
    RANDOM_DIRECTION_COUNT,
    generation_prompt_completion,
    TranscriptReadingItem,
    axis_scorer,
    mean_token_norm_by_layer,
    own_vs_other_auroc_by_layer,
    project_by_layer,
    random_unit_directions,
    select_best_layer,
    separation_auroc_by_layer,
    span_projection_record,
    token_norms,
    transcript_projection_record,
    transcript_reading_items,
)
from selfconcept.self_recognition.schemas import (
    DEFAULT_RUN_DIR,
    JudgementRecord,
    SpanProjectionRecord,
    Split,
    TranscriptProjectionRecord,
    TranscriptRecord,
    judgements_path,
    layer_norms_path,
    prompts_path,
    read_judgements,
    read_prompts,
    reading_path,
    transcript_reading_path,
    vectors_path,
)
from selfconcept.self_recognition.vectors import (
    JudgePromptSpans,
    TranscriptKey,
    judge_prompt_token_spans,
    pooled_own_and_other,
    span_length,
    transcript_key,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

type Reading = Literal["judge", "transcript"]
SPLITS: tuple[Split, Split] = ("train", "holdout")


@dataclass(frozen=True)
class RunConfig:
    model: str
    run_dir: Path = DEFAULT_RUN_DIR
    data_dir: Path = Path("data")
    readings: list[Reading] = field(default_factory=lambda: ["judge", "transcript"])
    batch_size: int = 4
    max_length: int = 16384
    layer_norm_record_count: int = 100
    limit_per_split: int | None = None


@dataclass(frozen=True)
class ReadingModel:
    name: str
    model: PreTrainedModel
    tokenizer: HFTokenizer
    model_specifics: ModelSpecifics
    decoder_layers: nn.ModuleList
    pad_token_id: int
    scorer: TokenScorer
    random_directions: Float[Tensor, "seed hidden"]


def load_reading_model(run: RunConfig) -> ReadingModel:
    hf_tokenizer = AutoTokenizer.from_pretrained(run.model)
    tokenizer = cast(HFTokenizer, hf_tokenizer)
    model = load_model(run.model)
    model_specifics = get_model_specifics(run.model, tokenizer)
    decoder_layers = model_specifics.get_decoder_layers(model)
    hidden_size = cast(nn.Embedding, model.get_input_embeddings()).embedding_dim
    return ReadingModel(
        name=run.model,
        model=model,
        tokenizer=tokenizer,
        model_specifics=model_specifics,
        decoder_layers=decoder_layers,
        pad_token_id=hf_tokenizer.pad_token_id if hf_tokenizer.pad_token_id is not None else hf_tokenizer.eos_token_id,
        scorer=axis_scorer(vectors_path(run.run_dir, run.model), run.model, len(decoder_layers), hidden_size),
        random_directions=random_unit_directions(RANDOM_DIRECTION_COUNT, hidden_size),
    )


def random_projections(
    pooled_at_layer: Float[Tensor, "example hidden"],
    random_directions: Float[Tensor, "seed hidden"],
) -> list[list[float]]:
    return (pooled_at_layer @ random_directions.T).tolist()


def write_jsonl(path: Path, records: Sequence[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(path, "w") as writer:
        writer.write_all(records)
    logger.info("wrote %d records to %s", len(records), path)


def longest_first[T](items: Sequence[T], length: Callable[[T], int]) -> list[T]:
    """Longest first, so an out-of-memory batch fails early."""
    return sorted(items, key=length, reverse=True)


def fitting_judge_examples(
    judgements: Sequence[JudgementRecord],
    transcripts_by_key: dict[TranscriptKey, TranscriptRecord],
    reading_model: ReadingModel,
    max_length: int,
) -> list[tuple[JudgementRecord, JudgePromptSpans]]:
    chat_template_kwargs = judge_chat_template_kwargs(reading_model.name)
    examples = [
        (judgement, judge_prompt_token_spans(judgement, transcripts_by_key, reading_model.tokenizer, chat_template_kwargs))
        for judgement in judgements
    ]
    fitting = [(judgement, example) for judgement, example in examples if len(example[0]) <= max_length]
    logger.info("%d of %d judge prompts skipped: longer than %d tokens", len(examples) - len(fitting), len(examples), max_length)
    return longest_first(fitting, lambda judgement_example: len(judgement_example[1][0]))


def judge_split_records(
    judge_examples: Sequence[tuple[JudgementRecord, JudgePromptSpans]],
    split: Split,
    random_layer: int | None,
    reading_model: ReadingModel,
    run: RunConfig,
) -> list[SpanProjectionRecord]:
    records: list[SpanProjectionRecord] = []
    for batch in batched(judge_examples, run.batch_size):
        own, other = pooled_own_and_other(
            reading_model.model, [example for _, example in batch], reading_model.decoder_layers, reading_model.pad_token_id, run.max_length
        )
        own_projections = project_by_layer(own, reading_model.scorer).tolist()
        other_projections = project_by_layer(other, reading_model.scorer).tolist()
        random_own = None if random_layer is None else random_projections(own[:, random_layer], reading_model.random_directions)
        random_other = None if random_layer is None else random_projections(other[:, random_layer], reading_model.random_directions)
        records.extend(
            span_projection_record(
                judgement,
                split,
                span_length(own_span),
                span_length(other_span),
                own_projections[index],
                other_projections[index],
                None if random_own is None else random_own[index],
                None if random_other is None else random_other[index],
            )
            for index, (judgement, (_, own_span, other_span)) in enumerate(batch)
        )
        logger.info("reading A %s: %d/%d", split, len(records), len(judge_examples))
    return records


def span_layer_norms(
    judge_examples: Sequence[tuple[JudgementRecord, JudgePromptSpans]],
    reading_model: ReadingModel,
    run: RunConfig,
) -> list[float]:
    """Mean per-token L2 norm of each decoder layer's output over the own and other spans."""
    norms: list[Float[Tensor, "layer token"]] = []
    for batch in batched(judge_examples, run.batch_size):
        norms.extend(
            token_norms(
                extract_token_span_activations(
                    reading_model.model,
                    [token_ids for _, (token_ids, _, _) in batch],
                    [sorted([own_span, other_span]) for _, (_, own_span, other_span) in batch],
                    reading_model.decoder_layers,
                    reading_model.pad_token_id,
                    run.max_length,
                )
            )
        )
    return mean_token_norm_by_layer(norms)


def run_judge_reading(
    reading_model: ReadingModel,
    split_by_prompt_id: dict[int, Split],
    run: RunConfig,
) -> None:
    judgements = read_judgements(judgements_path(run.run_dir, run.model))
    transcripts_by_key = {
        transcript_key(transcript): transcript
        for transcripts in read_transcripts_by_model(run.run_dir).values()
        for transcript in transcripts
    }
    examples_by_split = {
        split: fitting_judge_examples(
            [judgement for judgement in judgements if split_by_prompt_id.get(judgement["prompt_id"]) == split][: run.limit_per_split],
            transcripts_by_key,
            reading_model,
            run.max_length,
        )
        for split in SPLITS
    }
    train_records = judge_split_records(examples_by_split["train"], "train", None, reading_model, run)
    best_layer = select_best_layer(separation_auroc_by_layer(train_records))
    logger.info("reading A train-chosen best layer: %d", best_layer)
    holdout_records = judge_split_records(examples_by_split["holdout"], "holdout", best_layer, reading_model, run)
    write_jsonl(reading_path(run.run_dir, run.model), [*train_records, *holdout_records])

    layer_norm_by_layer = span_layer_norms(examples_by_split["train"][: run.layer_norm_record_count], reading_model, run)
    logger.info("per-layer mean residual norm: %s", " ".join(f"{norm:.1f}" for norm in layer_norm_by_layer))
    layer_norms_path(run.run_dir, run.model).write_text(json.dumps(layer_norm_by_layer))


def response_text_mismatch(
    item: TranscriptReadingItem,
    example: PromptCompletion,
    reading_model: ReadingModel,
) -> str | None:
    """The decoded response span when it is not exactly the response text."""
    token_ids = example.prompt_token_ids + example.completion_token_ids
    decoded = "".join(
        reading_model.tokenizer.decode(token_ids[span.start : span.end])
        for span in ResponseTargeter().target_spans(reading_model.model_specifics.split_completion(example))
    )
    return None if decoded.strip() == item.conversation[-1]["content"].strip() else decoded


def fitting_transcript_examples(
    items: Sequence[TranscriptReadingItem],
    reading_model: ReadingModel,
    max_length: int,
) -> list[tuple[TranscriptReadingItem, PromptCompletion]]:
    chat_template_kwargs = judge_chat_template_kwargs(reading_model.name)
    examples = [
        (item, generation_prompt_completion(item.conversation, reading_model.tokenizer, **chat_template_kwargs))
        for item in items
    ]
    mismatches = [
        (item, decoded) for item, example in examples if (decoded := response_text_mismatch(item, example, reading_model)) is not None
    ]
    logger.info("reading B: %d of %d response spans do not decode to the response text", len(mismatches), len(examples))
    for item, decoded in mismatches[:3]:
        logger.warning("span %r vs response %r", decoded[:200], item.conversation[-1]["content"][:200])
    fitting = [
        (item, example)
        for item, example in examples
        if len(example.prompt_token_ids) + len(example.completion_token_ids) <= max_length
    ]
    logger.info("%d of %d transcripts skipped: longer than %d tokens", len(examples) - len(fitting), len(examples), max_length)
    return longest_first(fitting, lambda item_example: len(item_example[1].prompt_token_ids) + len(item_example[1].completion_token_ids))


def transcript_split_records(
    transcript_examples: Sequence[tuple[TranscriptReadingItem, PromptCompletion]],
    random_layer: int | None,
    reading_model: ReadingModel,
    run: RunConfig,
) -> list[TranscriptProjectionRecord]:
    records: list[TranscriptProjectionRecord] = []
    for batch in batched(transcript_examples, run.batch_size):
        response_activations = extract_span_activations(
            reading_model.model,
            [example for _, example in batch],
            ResponseTargeter(),
            reading_model.model_specifics,
            reading_model.pad_token_id,
            run.max_length,
        )
        pooled = torch.stack([activations.float().mean(dim=1).cpu() for activations in response_activations])
        projections = project_by_layer(pooled, reading_model.scorer).tolist()
        random = None if random_layer is None else random_projections(pooled[:, random_layer], reading_model.random_directions)
        records.extend(
            transcript_projection_record(
                item, activations.shape[1], projections[index], None if random is None else random[index]
            )
            for index, ((item, _), activations) in enumerate(zip(batch, response_activations, strict=True))
        )
        logger.info("reading B: %d/%d", len(records), len(transcript_examples))
    return records


def run_transcript_reading(
    reading_model: ReadingModel,
    split_by_prompt_id: dict[int, Split],
    run: RunConfig,
) -> None:
    items = transcript_reading_items(run.model, read_transcripts_by_model(run.run_dir), split_by_prompt_id)
    examples_by_split = {
        split: fitting_transcript_examples(
            [item for item in items if item.split == split][: run.limit_per_split], reading_model, run.max_length
        )
        for split in SPLITS
    }
    train_records = transcript_split_records(examples_by_split["train"], None, reading_model, run)
    best_layer = select_best_layer(own_vs_other_auroc_by_layer(train_records))
    logger.info("reading B train-chosen best layer: %d", best_layer)
    holdout_records = transcript_split_records(examples_by_split["holdout"], best_layer, reading_model, run)
    write_jsonl(transcript_reading_path(run.run_dir, run.model), [*train_records, *holdout_records])


def main(run: RunConfig) -> None:
    logger.info("%s", run)
    split_by_prompt_id: dict[int, Split] = {
        prompt["id"]: prompt["split"] for prompt in read_prompts(prompts_path(run.data_dir))
    }
    reading_model = load_reading_model(run)
    if "judge" in run.readings:
        run_judge_reading(reading_model, split_by_prompt_id, run)
    if "transcript" in run.readings:
        run_transcript_reading(reading_model, split_by_prompt_id, run)
    logger.info("peak GPU memory (GiB): %s", peak_gpu_memory_gib_by_device())


if __name__ == "__main__":
    from jsonargparse import auto_cli

    auto_cli(main)
