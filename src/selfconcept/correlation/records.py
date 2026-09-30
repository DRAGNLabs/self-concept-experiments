"""On-disk record shapes. Field names are the keys existing runs were written with."""
from typing import Any, Literal, NotRequired, TypedDict

from selfconcept.measurement.interface import MeasurementName
from selfconcept.measurement.templates import ResponseSpans

type Region = Literal["prompt", "cot", "final"]
type GenerationStatus = Literal["complete", "truncated", "no_final"]
type OutcomeStatus = GenerationStatus | Literal["error_oom"]
type RunStage = Literal["started", "axis_ready", "probe_ready", "probe_failed_validation", "complete"]


class GreedySampling(TypedDict):
    do_sample: Literal[False]


class NucleusSampling(TypedDict):
    do_sample: Literal[True]
    temperature: float
    top_p: float
    top_k: int


type Sampling = GreedySampling | NucleusSampling


class RegionSummary(TypedDict):
    n: int
    mean: float | None
    p90: float | None


class GenerationRecord(TypedDict):
    example_id: str
    scenario: str
    turn: int
    response: str
    raw_response: str
    rendered_prompt: str
    status: GenerationStatus
    truncated: bool
    generated_tokens: int
    sampling: Sampling
    seed: int
    prompt_tokens: int
    spans: ResponseSpans
    scores: dict[str, dict[Region, RegionSummary]]
    generated_alignment: Literal["exact", "unavailable"]
    trace: str
    measurement: MeasurementName


class OutcomeRecord(TypedDict):
    example_id: str
    scenario: str
    status: OutcomeStatus
    label: str
    model_key: str
    response: NotRequired[str]
    generation_turns: NotRequired[int]
    behavior_label: NotRequired[str | None]
    final_code: NotRequired[str]
    evidence: NotRequired[dict[str, Any]]
    expected: NotRequired[str]
    topic: NotRequired[str]
    correct: NotRequired[bool | None]
    error: NotRequired[str]


class JudgeGrade(TypedDict):
    example_id: str
    label: str


class MeasurementManifest(TypedDict):
    name: MeasurementName
    layers: list[int]
    primary_layer: int


class AssistantAxisManifest(TypedDict):
    sha256: str
    layers: list[int]
    primary_layer: int
    artifact: str
    site: Literal["post_decoder_layer_residual"]
    normalization: Literal["unit_direction"]


class Manifest(TypedDict):
    args: dict[str, object]
    model: dict[str, object]
    chat_kwargs: dict[str, bool | str]
    corpus_sha256: str | None
    torch: str
    transformers: str
    sklearn: str
    numpy: str
    python: str
    stage: RunStage
    axis_sha256: NotRequired[str]
    probe_sha256: NotRequired[dict[str, str]]
    assistant_axis: NotRequired[AssistantAxisManifest]
    probe_usable: NotRequired[bool]
    measurement: NotRequired[MeasurementManifest]
