"""Steer one model's self-recognition vector into its stage-2 judge prompts and record the "1"/"2" answer
probabilities, plus greedy continuations for the coherence check.

Three modes:
- default: holdout judge records, alpha in {0, +-alphas, +-alpha_max} and random directions at the same nonzero alphas.
- --run.pilot true: train records of the first pilot_prompt_count train prompts, the pilot alpha grid at the
  reading best layer and a coarse layer sweep, written under run_dir/pilot. Read its table (4c --run.pilot)
  to choose alpha_max.
- --run.individual true: the qualitative "Did you write this?" run on single holdout texts.

The steering layer defaults to Reading A's train-chosen best layer (needs 4a's outputs). --run.shard_layers and
--run.shard_alphas (|alpha| values) run one shard of the default or pilot settings as an independent job: each shard
writes its own tagged files and resumes per judge record, and 4c reads all of a model's shards together.

Usage (from experiments/self-recognition):
    python scripts/4b_steer_judgements.py --config configs/4_validation.yaml --run.model Qwen/Qwen3-32B --run.pilot true
    python scripts/4b_steer_judgements.py --config configs/4_validation.yaml --run.model Qwen/Qwen3-32B --run.alpha_max 0.4
    python scripts/4b_steer_judgements.py --config configs/4_validation.yaml --run.model Qwen/Qwen3-32B --run.alpha_max 0.4 \
        --run.shard_layers [24] --run.shard_alphas [0.1]
"""

import json
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from itertools import batched
from pathlib import Path
from typing import cast

from jaxtyping import Float
from torch import Tensor, nn
from transformers import AutoTokenizer, PreTrainedModel

from selfconcept.common.activation_extraction.model_specifics import get_model_specifics
from selfconcept.common.generation.model_specifics import get_generation_specifics
from selfconcept.common.hf_strong_types import HFTokenizer, configure_call
from selfconcept.common.resumable_jsonl import process_unrecorded_items
from selfconcept.self_recognition.hf_loading import load_model, peak_gpu_memory_gib_by_device
from selfconcept.self_recognition.judging import (
    JudgementKey,
    answer_label_token_ids,
    is_judgeable,
    judge_candidate_text,
    judge_prompt_text,
    judgement_record_key,
    read_transcripts_by_model,
)
from selfconcept.self_recognition.reading import (
    random_unit_directions,
    select_best_layer,
    separation_auroc_by_layer,
    unit_direction,
)
from selfconcept.self_recognition.schemas import (
    DEFAULT_RUN_DIR,
    ContinuationRecord,
    IndividualRecord,
    JudgementRecord,
    Split,
    SteeringRecord,
    TranscriptRecord,
    continuation_path,
    individual_path,
    judgements_path,
    layer_norms_path,
    prompts_path,
    read_judgements,
    read_prompts,
    read_span_projections,
    reading_path,
    steering_path,
    vectors_path,
)
from selfconcept.self_recognition.steering_validation import (
    ScoringExample,
    SteeringSetting,
    capture_decoder_layer_outputs,
    condition_spans,
    continuation_record,
    depth_layers,
    final_settings,
    individual_conversation,
    last_token_label_probabilities,
    pilot_settings,
    replay_decoder_layers,
    scoring_example,
    shard_continuation_settings,
    shard_settings,
    shard_tag,
    steer_positions,
    steered_generation,
    steered_label_probabilities,
    steering_judgement_key,
    steering_record,
    token_ids_decoding_to,
)
from selfconcept.self_recognition.vectors import sample_without_replacement, transcript_key

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

type IndividualKey = tuple[int, str, int]
"""(prompt_id, response model, sample_index)"""


@dataclass(frozen=True)
class RunConfig:
    model: str
    run_dir: Path = DEFAULT_RUN_DIR
    data_dir: Path = Path("data")
    batch_size: int = 4
    max_length: int = 16384
    chunk_size: int = 16
    alphas: list[float] = field(default_factory=lambda: [0.1, 0.2])
    alpha_max: float | None = None
    steering_layers: list[int] | None = None
    shard_layers: list[int] | None = None
    shard_alphas: list[float] | None = None
    random_direction_count: int = 5
    coherence_record_count: int = 20
    continuation_tokens: int = 64
    record_limit: int | None = None
    pilot: bool = False
    pilot_name: str = "pilot"
    pilot_prompt_count: int = 20
    pilot_alphas: list[float] = field(default_factory=lambda: [0.05, 0.1, 0.2, 0.4, 0.8])
    pilot_layer_depths: list[float] = field(default_factory=lambda: [0.3, 0.4, 0.5, 0.6, 0.7])
    pilot_layer_alpha: float = 0.4
    individual: bool = False
    individual_text_count: int = 50
    individual_sample_count: int = 3
    seed: int = 0


@dataclass(frozen=True)
class SteeringModel:
    name: str
    model: PreTrainedModel
    tokenizer: HFTokenizer
    decoder_layers: nn.ModuleList
    pad_token_id: int
    eos_token_ids: frozenset[int]
    layer_norm_by_layer: list[float]
    direction_by_layer: dict[int, Float[Tensor, "hidden"]]
    random_directions: Float[Tensor, "seed hidden"]


def reading_best_layer(run_dir: Path, model: str) -> int:
    return select_best_layer(
        separation_auroc_by_layer([record for record in read_span_projections(reading_path(run_dir, model)) if record["split"] == "train"])
    )


def load_steering_model(run: RunConfig, layers: Sequence[int]) -> SteeringModel:
    hf_tokenizer = AutoTokenizer.from_pretrained(run.model)
    tokenizer = cast(HFTokenizer, hf_tokenizer)
    model = load_model(run.model)
    decoder_layers = get_model_specifics(run.model, tokenizer).get_decoder_layers(model)
    hidden_size = cast(nn.Embedding, model.get_input_embeddings()).embedding_dim
    eos_token_id = model.generation_config.eos_token_id
    return SteeringModel(
        name=run.model,
        model=model,
        tokenizer=tokenizer,
        decoder_layers=decoder_layers,
        pad_token_id=hf_tokenizer.pad_token_id if hf_tokenizer.pad_token_id is not None else hf_tokenizer.eos_token_id,
        eos_token_ids=frozenset(eos_token_id if isinstance(eos_token_id, list) else [eos_token_id]),
        layer_norm_by_layer=json.loads(layer_norms_path(run.run_dir, run.model).read_text()),
        direction_by_layer={
            layer: unit_direction(vectors_path(run.run_dir, run.model), run.model, layer, len(decoder_layers), hidden_size)
            for layer in layers
        },
        random_directions=random_unit_directions(run.random_direction_count, hidden_size),
    )


def setting_direction(steering_model: SteeringModel, setting: SteeringSetting) -> Float[Tensor, "hidden"]:
    if setting.random_seed is None:
        return steering_model.direction_by_layer[setting.layer]
    return steering_model.random_directions[setting.random_seed]


def setting_coefficient(steering_model: SteeringModel, setting: SteeringSetting) -> float:
    return setting.alpha * steering_model.layer_norm_by_layer[setting.layer]


def fitting_scoring_examples(
    judgements: Sequence[JudgementRecord],
    transcripts_by_key: dict[tuple[str, int, int], TranscriptRecord],
    steering_model: SteeringModel,
    max_length: int,
) -> list[ScoringExample]:
    examples = [scoring_example(judgement, transcripts_by_key, steering_model.tokenizer, steering_model.name) for judgement in judgements]
    fitting = [example for example in examples if len(example.token_ids) <= max_length]
    logger.info("%d of %d judge prompts skipped: longer than %d tokens", len(examples) - len(fitting), len(examples), max_length)
    return fitting


def steering_chunk_processor(
    steering_model: SteeringModel,
    settings: Sequence[SteeringSetting],
    batch_size: int,
) -> Callable[[Sequence[ScoringExample]], list[SteeringRecord]]:
    """Each batch gets one unsteered pass, which gives every alpha-0 setting and the steering layers' outputs; a
    steered pass replays those outputs and computes only the layers after its steering layer."""
    label_token_ids_by_position = answer_label_token_ids(steering_model.tokenizer)
    label_token_ids = [label_token_ids_by_position[1], label_token_ids_by_position[2]]
    steered_layers = {setting.layer for setting in settings if setting.alpha != 0.0}

    def steered_probabilities(
        batch: Sequence[ScoringExample],
        setting: SteeringSetting,
        unsteered_output_by_layer: dict[int, Float[Tensor, "batch seq hidden"]],
    ) -> Float[Tensor, "batch label"]:
        with replay_decoder_layers(steering_model.decoder_layers, setting.layer, unsteered_output_by_layer[setting.layer]):
            return steered_label_probabilities(
                steering_model.model,
                steering_model.decoder_layers[setting.layer],
                [example.token_ids for example in batch],
                condition_spans(setting.condition, batch),
                setting_direction(steering_model, setting),
                setting_coefficient(steering_model, setting),
                label_token_ids,
                steering_model.pad_token_id,
            )

    def process_chunk(examples: Sequence[ScoringExample]) -> list[SteeringRecord]:
        records: list[SteeringRecord] = []
        for batch in batched(sorted(examples, key=lambda example: len(example.token_ids), reverse=True), batch_size):
            with capture_decoder_layer_outputs(steering_model.decoder_layers, steered_layers) as unsteered_output_by_layer:
                unsteered_probabilities = last_token_label_probabilities(
                    steering_model.model, [example.token_ids for example in batch], label_token_ids, steering_model.pad_token_id
                )
            for setting in settings:
                probabilities = (
                    unsteered_probabilities
                    if setting.alpha == 0.0
                    else steered_probabilities(batch, setting, unsteered_output_by_layer)
                )
                records.extend(
                    steering_record(example.judgement, setting, setting_coefficient(steering_model, setting), example_probabilities)
                    for example, example_probabilities in zip(batch, probabilities.tolist(), strict=True)
                )
        return records

    return process_chunk


def generated_text_and_stop(steering_model: SteeringModel, generated_token_ids: list[int], max_new_tokens: int) -> tuple[str, bool]:
    stopped = len(generated_token_ids) < max_new_tokens or (
        bool(generated_token_ids) and generated_token_ids[-1] in steering_model.eos_token_ids
    )
    text = steering_model.tokenizer.decode(
        generated_token_ids, skip_special_tokens=get_generation_specifics(steering_model.name).skip_special_tokens
    )
    return text, stopped


def continuation_chunk_processor(
    steering_model: SteeringModel,
    settings: Sequence[SteeringSetting],
    max_new_tokens: int,
) -> Callable[[Sequence[ScoringExample]], list[ContinuationRecord]]:
    def process_chunk(examples: Sequence[ScoringExample]) -> list[ContinuationRecord]:
        records: list[ContinuationRecord] = []
        for example in examples:
            for setting in settings:
                spans = condition_spans(setting.condition, [example])
                generated_token_ids = steered_generation(
                    steering_model.model,
                    steering_model.decoder_layers[setting.layer],
                    example.token_ids,
                    steer_positions(spans, len(example.token_ids), steering_model.model.device),
                    setting_direction(steering_model, setting),
                    setting_coefficient(steering_model, setting),
                    max_new_tokens,
                    steering_model.pad_token_id,
                    sampling_seed=None,
                )
                text, stopped = generated_text_and_stop(steering_model, generated_token_ids, max_new_tokens)
                records.append(continuation_record(example.judgement, setting, generated_token_ids, text, stopped))
        return records

    return process_chunk


def example_key(example: ScoringExample) -> JudgementKey:
    return judgement_record_key(example.judgement)


def run_judgement_steering(
    examples: Sequence[ScoringExample],
    settings: Sequence[SteeringSetting],
    continuation_settings: Sequence[SteeringSetting],
    steering_model: SteeringModel,
    output_dir: Path,
    run: RunConfig,
) -> None:
    tag = shard_tag(settings, continuation_settings)
    process_unrecorded_items(
        examples,
        steering_path(output_dir, run.model, tag),
        item_key=example_key,
        record_key=steering_judgement_key,
        process_chunk=steering_chunk_processor(steering_model, settings, run.batch_size),
        chunk_size=run.chunk_size,
    )
    process_unrecorded_items(
        sorted(examples, key=example_key)[: run.coherence_record_count],
        continuation_path(output_dir, run.model, tag),
        item_key=example_key,
        record_key=steering_judgement_key,
        process_chunk=continuation_chunk_processor(steering_model, continuation_settings, run.continuation_tokens),
        chunk_size=1,
    )


def check_shard_selectors(run_settings: Sequence[SteeringSetting], run: RunConfig) -> None:
    unknown_layers = set(run.shard_layers or []) - {setting.layer for setting in run_settings}
    unknown_alphas = set(run.shard_alphas or []) - {abs(setting.alpha) for setting in run_settings if setting.alpha != 0.0}
    if unknown_layers or unknown_alphas:
        raise ValueError(f"shard selectors outside the run's settings: layers {sorted(unknown_layers)}, alphas {sorted(unknown_alphas)}")


def individual_texts(
    target_model: str,
    transcripts_by_model: dict[str, list[TranscriptRecord]],
    split_by_prompt_id: dict[int, Split],
    text_count: int,
    seed: int,
) -> list[TranscriptRecord]:
    """text_count of the target's own judgeable holdout texts, then text_count of the other models'."""
    holdout_texts = [
        transcript
        for transcripts in transcripts_by_model.values()
        for transcript in transcripts
        if split_by_prompt_id.get(transcript["prompt_id"]) == "holdout" and is_judgeable(transcript)
    ]
    own = [transcript for transcript in holdout_texts if transcript["model"] == target_model]
    other = [transcript for transcript in holdout_texts if transcript["model"] != target_model]
    return [*sample_without_replacement(own, text_count, seed), *sample_without_replacement(other, text_count, seed)]


def individual_key(record: IndividualRecord) -> IndividualKey:
    return record["prompt_id"], record["response_model"], record["sample_index"]


def transcript_individual_key(transcript: TranscriptRecord) -> IndividualKey:
    return transcript["prompt_id"], transcript["model"], transcript["sample_index"]


def individual_chunk_processor(
    steering_model: SteeringModel,
    layer: int,
    alphas: Sequence[float],
    run: RunConfig,
) -> Callable[[Sequence[TranscriptRecord]], list[IndividualRecord]]:
    yes_no_token_ids = [token_ids_decoding_to(steering_model.tokenizer, "Yes"), token_ids_decoding_to(steering_model.tokenizer, "No")]
    direction = steering_model.direction_by_layer[layer]

    def process_chunk(transcripts: Sequence[TranscriptRecord]) -> list[IndividualRecord]:
        records: list[IndividualRecord] = []
        for transcript in transcripts:
            token_ids = configure_call(steering_model.tokenizer)(
                judge_prompt_text(
                    steering_model.tokenizer, steering_model.name, individual_conversation(judge_candidate_text(transcript))
                ),
                add_special_tokens=False,
            )["input_ids"]
            for alpha in alphas:
                coefficient = alpha * steering_model.layer_norm_by_layer[layer]
                (yes_probability, no_probability), = steered_label_probabilities(
                    steering_model.model,
                    steering_model.decoder_layers[layer],
                    [token_ids],
                    None,
                    direction,
                    coefficient,
                    yes_no_token_ids,
                    steering_model.pad_token_id,
                ).tolist()
                sampled_generations = [
                    generated_text_and_stop(
                        steering_model,
                        steered_generation(
                            steering_model.model,
                            steering_model.decoder_layers[layer],
                            token_ids,
                            "all",
                            direction,
                            coefficient,
                            run.continuation_tokens,
                            steering_model.pad_token_id,
                            sampling_seed=sample_seed,
                        ),
                        run.continuation_tokens,
                    )[0]
                    for sample_seed in range(run.individual_sample_count)
                ]
                records.append(
                    {
                        "prompt_id": transcript["prompt_id"],
                        "sample_index": transcript["sample_index"],
                        "response_model": transcript["model"],
                        "is_own": transcript["model"] == steering_model.name,
                        "layer": layer,
                        "alpha": alpha,
                        "coefficient": coefficient,
                        "yes_probability": yes_probability,
                        "no_probability": no_probability,
                        "sampled_generations": sampled_generations,
                    }
                )
        return records

    return process_chunk


def main(run: RunConfig) -> None:
    logger.info("%s", run)
    if not run.pilot and run.alpha_max is None:
        raise ValueError("--run.alpha_max is required outside the pilot; choose it from the pilot table")
    split_by_prompt_id: dict[int, Split] = {prompt["id"]: prompt["split"] for prompt in read_prompts(prompts_path(run.data_dir))}
    best_layer = reading_best_layer(run.run_dir, run.model)
    primary_layers = run.steering_layers or [best_layer]
    logger.info("reading best layer %d; steering layers %s", best_layer, primary_layers)
    transcripts_by_model = read_transcripts_by_model(run.run_dir)

    if run.individual:
        assert run.alpha_max is not None
        steering_model = load_steering_model(run, primary_layers[:1])
        process_unrecorded_items(
            individual_texts(run.model, transcripts_by_model, split_by_prompt_id, run.individual_text_count, run.seed),
            individual_path(run.run_dir, run.model),
            item_key=transcript_individual_key,
            record_key=individual_key,
            process_chunk=individual_chunk_processor(
                steering_model, primary_layers[0], [0.0, -run.alpha_max, run.alpha_max], run
            ),
            chunk_size=run.chunk_size,
        )
        return

    if run.pilot:
        layer_count = len(json.loads(layer_norms_path(run.run_dir, run.model).read_text()))
        run_settings = pilot_settings(
            primary_layers, run.pilot_alphas, depth_layers(layer_count, run.pilot_layer_depths), run.pilot_layer_alpha
        )
        output_dir = run.run_dir / run.pilot_name
    else:
        assert run.alpha_max is not None
        run_settings = final_settings(primary_layers, run.alphas, run.alpha_max, run.random_direction_count)
        output_dir = run.run_dir
    check_shard_selectors(run_settings, run)
    settings = shard_settings(run_settings, run.shard_layers, run.shard_alphas)
    steering_model = load_steering_model(run, sorted({setting.layer for setting in settings}))

    split: Split = "train" if run.pilot else "holdout"
    split_prompt_ids = sorted(prompt_id for prompt_id, prompt_split in split_by_prompt_id.items() if prompt_split == split)
    selected_prompt_ids = set(split_prompt_ids[: run.pilot_prompt_count] if run.pilot else split_prompt_ids)
    judgements = [
        judgement
        for judgement in read_judgements(judgements_path(run.run_dir, run.model))
        if judgement["prompt_id"] in selected_prompt_ids
    ][: run.record_limit]
    transcripts_by_key = {
        transcript_key(transcript): transcript for transcripts in transcripts_by_model.values() for transcript in transcripts
    }
    examples = fitting_scoring_examples(judgements, transcripts_by_key, steering_model, run.max_length)
    logger.info("%d judge records x %d of the run's %d settings", len(examples), len(settings), len(run_settings))
    run_judgement_steering(
        examples,
        settings,
        shard_continuation_settings(run_settings, run.shard_layers, run.shard_alphas),
        steering_model,
        output_dir,
        run,
    )
    logger.info("peak GPU memory (GiB): %s", peak_gpu_memory_gib_by_device())


if __name__ == "__main__":
    from jsonargparse import auto_cli

    auto_cli(main)
