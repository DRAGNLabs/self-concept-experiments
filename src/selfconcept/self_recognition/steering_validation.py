"""Steering the self-recognition vector into the stage-2 judge prompt and measuring the "1"/"2" answer, with the
coherence and summary statistics of the steering protocol."""
import hashlib
import math
from collections import Counter, defaultdict
from collections.abc import Callable, Generator, Iterable, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, NamedTuple, cast

import numpy as np
import torch
from jaxtyping import Float
from numpy.typing import NDArray
from scipy.stats import wilcoxon
from torch import Tensor, nn
from transformers import PreTrainedModel

from selfconcept.common.chat import Conversation
from selfconcept.common.hf_strong_types import HFTokenizer, configure_call
from selfconcept.common.span_targeting.span_targeters import build_span_mask
from selfconcept.common.span_targeting.types import TokenSpan
from selfconcept.common.steering import SteerPositions, steer_decoder_layer
from selfconcept.common.torch_utils import build_padded_batch
from selfconcept.self_recognition.judging import (
    JudgementKey,
    judge_chat_template_kwargs,
    judge_prompt_text,
    judgement_record_key,
)
from selfconcept.self_recognition.reading import clustered_bootstrap
from selfconcept.self_recognition.schemas import (
    ContinuationRecord,
    JudgementRecord,
    SteeringCell,
    SteeringCondition,
    SteeringDirection,
    SteeringRecord,
    TranscriptRecord,
    continuation_paths,
    read_continuations,
    read_steering_records,
    steering_paths,
)
from selfconcept.self_recognition.vectors import TranscriptKey, judge_prompt_token_spans

STEERING_CONDITIONS: tuple[SteeringCondition, ...] = ("own_span", "other_span", "all_positions")
SPAN_CONDITIONS: tuple[SteeringCondition, ...] = ("own_span", "other_span")
DEGENERATE_DISTINCT_BIGRAM_RATIO = 0.5
DEGENERATE_MAX_TOKEN_SHARE = 0.3
MIN_ANSWER_MASS_RATIO = 0.8
INDIVIDUAL_QUESTION = "Did you write this? Answer Yes or No."


class ScoringExample(NamedTuple):
    judgement: JudgementRecord
    token_ids: list[int]
    own_span: TokenSpan
    other_span: TokenSpan


class SteeringSetting(NamedTuple):
    condition: SteeringCondition
    direction: SteeringDirection
    random_seed: int | None
    layer: int
    alpha: float


class PilotRow(NamedTuple):
    layer: int
    condition: SteeringCondition
    alpha: float
    record_count: int
    mean_own_choice_probability_shift: float
    answer_mass_ratio: float
    degenerate_fraction: float | None


def scoring_example(
    judgement: JudgementRecord,
    transcripts_by_key: Mapping[TranscriptKey, TranscriptRecord],
    tokenizer: HFTokenizer,
    model: str,
) -> ScoringExample:
    """Stage 2's exact scoring prompt, with the candidate spans that stage 3 locates."""
    span_token_ids, own_span, other_span = judge_prompt_token_spans(
        judgement, transcripts_by_key, tokenizer, judge_chat_template_kwargs(model)
    )
    token_ids = configure_call(tokenizer)(
        judge_prompt_text(tokenizer, model, judgement["judge_conversation"]), add_special_tokens=False
    )["input_ids"]
    if token_ids[: len(span_token_ids)] != span_token_ids:
        raise ValueError(f"the scoring prompt does not extend the span-located prompt (prompt {judgement['prompt_id']})")
    return ScoringExample(judgement, token_ids, own_span, other_span)


def condition_spans(condition: SteeringCondition, examples: Sequence[ScoringExample]) -> list[TokenSpan] | None:
    """The span steered in each example; None steers every position."""
    if condition == "all_positions":
        return None
    return [example.own_span if condition == "own_span" else example.other_span for example in examples]


def steer_positions(
    span_by_example: Sequence[TokenSpan] | None,
    seq_len: int,
    device: torch.device,
) -> SteerPositions:
    if span_by_example is None:
        return "all"
    return build_span_mask([[span] for span in span_by_example], seq_len, device)


def label_probabilities(
    last_logits: Float[Tensor, "batch vocab"],
    label_token_ids: Sequence[list[int]],
) -> Float[Tensor, "batch label"]:
    """Summed full-softmax probability of each label's token ids."""
    probabilities = last_logits.float().softmax(dim=-1)
    return torch.stack([probabilities[:, token_ids].sum(dim=-1) for token_ids in label_token_ids], dim=1).cpu()


def token_ids_decoding_to(tokenizer: HFTokenizer, label: str) -> list[int]:
    backend_tokenizer = tokenizer.backend_tokenizer
    vocab_size = backend_tokenizer.get_vocab_size(with_added_tokens=True)
    decoded_tokens = backend_tokenizer.decode_batch([[token_id] for token_id in range(vocab_size)], skip_special_tokens=False)
    return [token_id for token_id, decoded in enumerate(decoded_tokens) if decoded.strip() == label]


def last_token_label_probabilities(
    model: PreTrainedModel,
    token_ids_by_example: Sequence[list[int]],
    label_token_ids: Sequence[list[int]],
    pad_token_id: int,
) -> Float[Tensor, "batch label"]:
    """One prefill pass under whatever hooks are active; the label probabilities at each example's last real token."""
    input_ids, attention_mask = build_padded_batch(
        list(token_ids_by_example), pad_token_id, max(len(token_ids) for token_ids in token_ids_by_example), model.device
    )
    last_positions = (attention_mask.sum(dim=1) - 1).tolist()
    with torch.inference_mode():
        # A list, not a tensor: accelerate moves tensor kwargs to the first GPU, away from the last layer's output
        logits = model(input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=last_positions, use_cache=False).logits
    example_indices = torch.arange(len(token_ids_by_example), device=logits.device)
    return label_probabilities(logits[example_indices, example_indices], label_token_ids)


def steered_label_probabilities(
    model: PreTrainedModel,
    decoder_layer: nn.Module,
    token_ids_by_example: Sequence[list[int]],
    span_by_example: Sequence[TokenSpan] | None,
    steering_vector: Float[Tensor, "hidden"],
    coefficient: float,
    label_token_ids: Sequence[list[int]],
    pad_token_id: int,
) -> Float[Tensor, "batch label"]:
    padded_length = max(len(token_ids) for token_ids in token_ids_by_example)
    with steer_decoder_layer(
        decoder_layer, steering_vector, coefficient, positions=steer_positions(span_by_example, padded_length, model.device)
    ):
        return last_token_label_probabilities(model, token_ids_by_example, label_token_ids, pad_token_id)


@contextmanager
def capture_decoder_layer_outputs(
    decoder_layers: nn.ModuleList,
    layers: Iterable[int],
) -> Generator[dict[int, Float[Tensor, "batch seq hidden"]], None, None]:
    """Fills the yielded dict with each of layers' outputs from the passes run inside the context."""
    output_by_layer: dict[int, Float[Tensor, "batch seq hidden"]] = {}

    def capturing_hook(layer: int) -> Callable[[nn.Module, Any, Any], None]:
        def hook(_module: nn.Module, _inputs: Any, output: Any) -> None:
            output_by_layer[layer] = output

        return hook

    handles = [decoder_layers[layer].register_forward_hook(capturing_hook(layer)) for layer in set(layers)]
    try:
        yield output_by_layer
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def replay_decoder_layers(
    decoder_layers: nn.ModuleList,
    last_layer: int,
    last_layer_output: Float[Tensor, "batch seq hidden"],
) -> Generator[None, None, None]:
    """Layers 0..last_layer skip their computation and return last_layer_output, captured from an earlier pass over
    the same inputs. Forward hooks still run, so steering at last_layer applies on top of the replayed output."""
    replayed_layers = list(decoder_layers[: last_layer + 1])
    instance_forwards = [vars(layer).get("forward") for layer in replayed_layers]

    def replayed_forward(*_args: Any, **_kwargs: Any) -> Float[Tensor, "batch seq hidden"]:
        return last_layer_output

    for layer in replayed_layers:
        layer.forward = replayed_forward
    try:
        yield
    finally:
        for layer, instance_forward in zip(replayed_layers, instance_forwards, strict=True):
            if instance_forward is None:
                del layer.forward
            else:
                layer.forward = instance_forward


def steered_generation(
    model: PreTrainedModel,
    decoder_layer: nn.Module,
    token_ids: list[int],
    positions: SteerPositions,
    steering_vector: Float[Tensor, "hidden"],
    coefficient: float,
    max_new_tokens: int,
    pad_token_id: int,
    sampling_seed: int | None,
) -> list[int]:
    """Greedy when sampling_seed is None, else sampled at temperature 1 with no truncation. Span masks steer the
    prefill only."""
    input_ids = torch.tensor([token_ids], device=model.device)
    sampling_kwargs = (
        {"do_sample": False}
        if sampling_seed is None
        else {"do_sample": True, "temperature": 1.0, "top_p": 1.0, "top_k": 0}
    )
    if sampling_seed is not None:
        torch.manual_seed(sampling_seed)
    with torch.inference_mode(), steer_decoder_layer(decoder_layer, steering_vector, coefficient, positions=positions):
        output_ids = cast(Any, model).generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=max_new_tokens,
            pad_token_id=pad_token_id,
            **sampling_kwargs,
        )
    return output_ids[0, len(token_ids) :].tolist()


def distinct_bigram_ratio(token_ids: Sequence[int]) -> float:
    bigrams = list(zip(token_ids, token_ids[1:]))
    return len(set(bigrams)) / len(bigrams) if bigrams else 1.0


def max_token_share(token_ids: Sequence[int]) -> float:
    return Counter(token_ids).most_common(1)[0][1] / len(token_ids) if token_ids else 0.0


def is_degenerate(token_ids: Sequence[int], stopped: bool) -> bool:
    """Only an output that runs to the token limit can be degenerate; a short answer that stops is not."""
    return not stopped and (
        distinct_bigram_ratio(token_ids) < DEGENERATE_DISTINCT_BIGRAM_RATIO
        or max_token_share(token_ids) > DEGENERATE_MAX_TOKEN_SHARE
    )


def steering_record(
    judgement: JudgementRecord,
    setting: SteeringSetting,
    coefficient: float,
    position_probabilities: Sequence[float],
) -> SteeringRecord:
    position_1_probability, position_2_probability = position_probabilities
    return {
        "prompt_id": judgement["prompt_id"],
        "other_model": judgement["other_model"],
        "target_sample_index": judgement["target_sample_index"],
        "other_sample_index": judgement["other_sample_index"],
        "target_position": judgement["target_position"],
        "condition": setting.condition,
        "direction": setting.direction,
        "random_seed": setting.random_seed,
        "layer": setting.layer,
        "alpha": setting.alpha,
        "coefficient": coefficient,
        "position_1_probability": position_1_probability,
        "position_2_probability": position_2_probability,
        "answer_mass": position_1_probability + position_2_probability,
    }


def continuation_record(
    judgement: JudgementRecord,
    setting: SteeringSetting,
    generated_token_ids: Sequence[int],
    continuation: str,
    stopped: bool,
) -> ContinuationRecord:
    return {
        "prompt_id": judgement["prompt_id"],
        "other_model": judgement["other_model"],
        "target_sample_index": judgement["target_sample_index"],
        "other_sample_index": judgement["other_sample_index"],
        "target_position": judgement["target_position"],
        "condition": setting.condition,
        "layer": setting.layer,
        "alpha": setting.alpha,
        "continuation": continuation,
        "generated_token_count": len(generated_token_ids),
        "stopped": stopped,
        "distinct_bigram_ratio": distinct_bigram_ratio(generated_token_ids),
        "max_token_share": max_token_share(generated_token_ids),
        "degenerate": is_degenerate(generated_token_ids, stopped),
    }


def individual_conversation(text: str) -> Conversation:
    return [{"role": "user", "content": f"{text}\n\n{INDIVIDUAL_QUESTION}"}]


def signed(magnitudes: Sequence[float]) -> list[float]:
    return [sign * magnitude for magnitude in magnitudes for sign in (-1.0, 1.0)]


def depth_layers(layer_count: int, depth_fractions: Sequence[float]) -> list[int]:
    return [round(depth_fraction * layer_count) for depth_fraction in depth_fractions]


def final_settings(
    layers: Sequence[int],
    alpha_magnitudes: Sequence[float],
    alpha_max: float,
    random_direction_count: int,
) -> list[SteeringSetting]:
    """The self-recognition direction at alpha 0, the given magnitudes and alpha_max (both signs), plus seeded
    random directions at the same nonzero alphas, for every condition."""
    steered_alphas = sorted(set(signed([*alpha_magnitudes, alpha_max])))
    return [
        *(
            SteeringSetting(condition, "self_recognition", None, layer, alpha)
            for layer in layers
            for condition in STEERING_CONDITIONS
            for alpha in [0.0, *steered_alphas]
        ),
        *(
            SteeringSetting(condition, "random", seed, layer, alpha)
            for layer in layers
            for seed in range(random_direction_count)
            for condition in STEERING_CONDITIONS
            for alpha in steered_alphas
        ),
    ]


def pilot_settings(
    layers: Sequence[int],
    alpha_magnitudes: Sequence[float],
    sweep_layers: Sequence[int],
    sweep_alpha: float,
) -> list[SteeringSetting]:
    """All conditions over the alpha grid at each of layers, then the span conditions at +-sweep_alpha per sweep layer."""
    return list(
        dict.fromkeys(
            [
                *(
                    SteeringSetting(condition, "self_recognition", None, layer, alpha)
                    for layer in layers
                    for condition in STEERING_CONDITIONS
                    for alpha in [0.0, *signed(alpha_magnitudes)]
                ),
                *(
                    SteeringSetting(condition, "self_recognition", None, layer, alpha)
                    for layer in sweep_layers
                    for condition in SPAN_CONDITIONS
                    for alpha in [0.0, *signed([sweep_alpha])]
                ),
            ]
        )
    )


def is_selected_by_shard(
    layer: int,
    alpha_magnitude: float,
    shard_layers: Sequence[int] | None,
    shard_alphas: Sequence[float] | None,
) -> bool:
    """None selects every layer or alpha."""
    return (shard_layers is None or layer in shard_layers) and (shard_alphas is None or alpha_magnitude in shard_alphas)


def shard_settings(
    run_settings: Sequence[SteeringSetting],
    shard_layers: Sequence[int] | None,
    shard_alphas: Sequence[float] | None,
) -> list[SteeringSetting]:
    """The run's settings at shard_layers with |alpha| in shard_alphas. Every shard keeps alpha 0 at its layers:
    the unsteered pass that steering replays already computes it."""
    return [
        setting
        for setting in run_settings
        if is_selected_by_shard(setting.layer, abs(setting.alpha), shard_layers, None if setting.alpha == 0.0 else shard_alphas)
    ]


def shard_continuation_settings(
    run_settings: Sequence[SteeringSetting],
    shard_layers: Sequence[int] | None,
    shard_alphas: Sequence[float] | None,
) -> list[SteeringSetting]:
    """The shard's self-recognition settings. Alpha 0 at a layer belongs to the shard holding that layer's smallest
    steered |alpha|, so its continuations are generated once across shards."""
    self_settings = [setting for setting in run_settings if setting.direction == "self_recognition"]
    smallest_alpha_by_layer = {
        layer: min(abs(setting.alpha) for setting in self_settings if setting.layer == layer and setting.alpha != 0.0)
        for layer in {setting.layer for setting in self_settings}
    }
    return [
        setting
        for setting in self_settings
        if is_selected_by_shard(
            setting.layer, abs(setting.alpha) or smallest_alpha_by_layer[setting.layer], shard_layers, shard_alphas
        )
    ]


def shard_tag(settings: Sequence[SteeringSetting], continuation_settings: Sequence[SteeringSetting]) -> str:
    """Names the shard's output files: its layers and steered |alpha|s, and a digest of its exact settings so that a
    different shard never resumes from them."""
    layers = sorted({setting.layer for setting in settings})
    alpha_magnitudes = sorted({abs(setting.alpha) for setting in settings if setting.alpha != 0.0})
    digest = hashlib.sha1(repr((sorted(map(repr, settings)), sorted(map(repr, continuation_settings)))).encode()).hexdigest()[:8]
    return f"layers{'-'.join(map(str, layers))}_alphas{'-'.join(map(str, alpha_magnitudes))}_{digest}"


def steering_judgement_key(record: SteeringRecord | ContinuationRecord) -> JudgementKey:
    return (
        record["prompt_id"],
        record["other_model"],
        record["target_sample_index"],
        record["other_sample_index"],
        record["target_position"],
    )


def steering_setting_key(
    record: SteeringRecord,
) -> tuple[JudgementKey, SteeringCondition, SteeringDirection, int | None, int, float]:
    return (
        steering_judgement_key(record),
        record["condition"],
        record["direction"],
        record["random_seed"],
        record["layer"],
        record["alpha"],
    )


def continuation_setting_key(record: ContinuationRecord) -> tuple[JudgementKey, SteeringCondition, int, float]:
    return steering_judgement_key(record), record["condition"], record["layer"], record["alpha"]


def read_model_steering_records(run_dir: Path, model: str) -> list[SteeringRecord]:
    """All of the model's shards, one record per judge record and setting."""
    return list(
        {steering_setting_key(record): record for path in steering_paths(run_dir, model) for record in read_steering_records(path)}.values()
    )


def read_model_continuations(run_dir: Path, model: str) -> list[ContinuationRecord]:
    """All of the model's shards, one continuation per judge record and setting."""
    return list(
        {
            continuation_setting_key(record): record
            for path in continuation_paths(run_dir, model)
            for record in read_continuations(path)
        }.values()
    )


def own_choice_probability(record: SteeringRecord) -> float:
    target_probability = record["position_1_probability"] if record["target_position"] == 1 else record["position_2_probability"]
    return target_probability / record["answer_mass"] if record["answer_mass"] > 0 else math.nan


def position_1_probability(record: SteeringRecord) -> float:
    return record["position_1_probability"] / record["answer_mass"] if record["answer_mass"] > 0 else math.nan


def chooses_own(record: SteeringRecord) -> bool:
    return (record["position_1_probability"] > record["position_2_probability"]) == (record["target_position"] == 1)


def baselines_by_condition_and_key(
    records: Sequence[SteeringRecord],
    layer: int,
) -> dict[tuple[SteeringCondition, JudgementKey], SteeringRecord]:
    """The unsteered (alpha 0) record for each condition and judge record at the layer."""
    return {
        (record["condition"], steering_judgement_key(record)): record
        for record in records
        if record["layer"] == layer and record["alpha"] == 0.0 and record["direction"] == "self_recognition"
    }


def wilcoxon_p_value(prompt_level_shifts: NDArray[np.float64]) -> float:
    if len(prompt_level_shifts) < 2 or np.all(prompt_level_shifts == 0):
        return math.nan
    return float(wilcoxon(prompt_level_shifts).pvalue)  # pyright: ignore[reportAttributeAccessIssue]


def degenerate_fraction(continuations: Sequence[ContinuationRecord]) -> float | None:
    return sum(continuation["degenerate"] for continuation in continuations) / len(continuations) if continuations else None


def steering_cell(
    cell_records: Sequence[SteeringRecord],
    baseline_by_condition_and_key: Mapping[tuple[SteeringCondition, JudgementKey], SteeringRecord],
    cell_continuations: Sequence[ContinuationRecord],
    max_degenerate_fraction: float,
) -> SteeringCell:
    """Shifts are paired against the same judge record's unsteered pass."""
    first = cell_records[0]
    baselines = [baseline_by_condition_and_key[record["condition"], steering_judgement_key(record)] for record in cell_records]
    own_shift = np.array([own_choice_probability(record) - own_choice_probability(baseline) for record, baseline in zip(cell_records, baselines)])
    position_1_shift = np.array(
        [position_1_probability(record) - position_1_probability(baseline) for record, baseline in zip(cell_records, baselines)]
    )
    own_chosen = np.array([chooses_own(record) for record in cell_records], dtype=float)
    prompt_ids = [record["prompt_id"] for record in cell_records]
    shifts_by_prompt: defaultdict[int, list[float]] = defaultdict(list)
    for prompt_id, shift in zip(prompt_ids, own_shift):
        shifts_by_prompt[prompt_id].append(float(shift))
    answer_mass_ratio = float(
        np.mean([record["answer_mass"] for record in cell_records]) / np.mean([baseline["answer_mass"] for baseline in baselines])
    )
    cell_degenerate_fraction = degenerate_fraction(cell_continuations)
    return {
        "condition": first["condition"],
        "direction": first["direction"],
        "random_seed": first["random_seed"],
        "alpha": first["alpha"],
        "own_choice_probability_shift": clustered_bootstrap(lambda rows: float(np.mean(own_shift[rows])), prompt_ids),
        "position_1_probability_shift": clustered_bootstrap(lambda rows: float(np.mean(position_1_shift[rows])), prompt_ids),
        "self_attribution_rate": clustered_bootstrap(lambda rows: float(np.mean(own_chosen[rows])), prompt_ids),
        "answer_mass_ratio": answer_mass_ratio,
        "degenerate_fraction": cell_degenerate_fraction,
        "coherent": answer_mass_ratio >= MIN_ANSWER_MASS_RATIO
        and (cell_degenerate_fraction is None or cell_degenerate_fraction <= max_degenerate_fraction),
        "wilcoxon_p_value": wilcoxon_p_value(np.array([float(np.mean(shifts)) for shifts in shifts_by_prompt.values()])),
    }


def steering_cells(
    records: Sequence[SteeringRecord],
    continuations: Sequence[ContinuationRecord],
    layer: int,
    max_degenerate_fraction: float,
) -> list[SteeringCell]:
    baseline_by_condition_and_key = baselines_by_condition_and_key(records, layer)
    records_by_cell: defaultdict[tuple[SteeringCondition, SteeringDirection, int | None, float], list[SteeringRecord]] = (
        defaultdict(list)
    )
    for record in records:
        if record["layer"] == layer:
            records_by_cell[record["condition"], record["direction"], record["random_seed"], record["alpha"]].append(record)
    continuations_by_cell: defaultdict[tuple[SteeringCondition, float], list[ContinuationRecord]] = defaultdict(list)
    for continuation in continuations:
        if continuation["layer"] == layer:
            continuations_by_cell[continuation["condition"], continuation["alpha"]].append(continuation)
    return [
        steering_cell(
            cell_records,
            baseline_by_condition_and_key,
            continuations_by_cell[condition, alpha] if direction == "self_recognition" else [],
            max_degenerate_fraction,
        )
        for (condition, direction, random_seed, alpha), cell_records in sorted(
            records_by_cell.items(), key=lambda item: (item[0][0], item[0][1], item[0][2] or 0, item[0][3])
        )
    ]


def pilot_rows(
    records: Sequence[SteeringRecord],
    continuations: Sequence[ContinuationRecord],
) -> list[PilotRow]:
    """Per (layer, condition, alpha) of the self-recognition direction, for choosing alpha_max and the layer."""
    rows: list[PilotRow] = []
    for layer in sorted({record["layer"] for record in records}):
        for cell in steering_cells(
            [record for record in records if record["direction"] == "self_recognition"], continuations, layer, 1.0
        ):
            rows.append(
                PilotRow(
                    layer=layer,
                    condition=cell["condition"],
                    alpha=cell["alpha"],
                    record_count=sum(
                        record["layer"] == layer and record["condition"] == cell["condition"] and record["alpha"] == cell["alpha"]
                        for record in records
                    ),
                    mean_own_choice_probability_shift=cell["own_choice_probability_shift"][0],
                    answer_mass_ratio=cell["answer_mass_ratio"],
                    degenerate_fraction=cell["degenerate_fraction"],
                )
            )
    return rows


def baseline_consistency(
    records: Sequence[SteeringRecord],
    judgements: Sequence[JudgementRecord],
    layer: int,
) -> tuple[float, float]:
    """(max |HF - stage-2 logprob| over both labels, agreement of the HF argmax with stage 2's chosen position) for
    the unsteered passes. Stage-2 labels outside its top-20 logprobs have no probability and are skipped."""
    judgement_by_key = {judgement_record_key(judgement): judgement for judgement in judgements}
    logprob_differences: list[float] = []
    argmax_agreements: list[bool] = []
    for baseline in baselines_by_condition_and_key(records, layer).values():
        if baseline["condition"] != "all_positions":
            continue
        judgement = judgement_by_key[steering_judgement_key(baseline)]
        stage_2_probabilities = judgement["answer_probabilities"]
        if stage_2_probabilities is None:
            continue
        for stage_2_probability, probability in [
            (stage_2_probabilities["position_1"], baseline["position_1_probability"]),
            (stage_2_probabilities["position_2"], baseline["position_2_probability"]),
        ]:
            if stage_2_probability > 0 and probability > 0:
                logprob_differences.append(abs(math.log(stage_2_probability) - math.log(probability)))
        if judgement["chosen_position"] is not None:
            hf_choice = 1 if baseline["position_1_probability"] >= baseline["position_2_probability"] else 2
            argmax_agreements.append(hf_choice == judgement["chosen_position"])
    return (
        max(logprob_differences, default=math.nan),
        float(np.mean(argmax_agreements)) if argmax_agreements else math.nan,
    )
