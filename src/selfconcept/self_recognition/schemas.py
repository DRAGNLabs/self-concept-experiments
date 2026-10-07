"""Records passed between the self-recognition stages: prompts -> transcripts -> judgements -> vectors."""
from pathlib import Path
from typing import Literal, TypedDict, cast

from jaxtyping import Float
from torch import Tensor

from selfconcept.common.chat import Conversation
from selfconcept.common.jsonl import read_jsonl
from selfconcept.common.paths import scratch_dir

DEFAULT_RUN_DIR = scratch_dir("self-recognition")

type Split = Literal["train", "holdout"]
type CandidatePosition = Literal[1, 2]


class PromptRecord(TypedDict):
    id: int
    question: str
    split: Split


class TranscriptRecord(TypedDict):
    prompt_id: int
    model: str
    sample_index: int
    conversation: Conversation
    truncated: bool


class JudgementRecord(TypedDict):
    prompt_id: int
    target_model: str
    other_model: str
    target_sample_index: int
    other_sample_index: int
    target_position: CandidatePosition
    judge_conversation: Conversation
    raw_response: str
    chosen_position: CandidatePosition | None
    correct: bool


class SelfRecognitionVectors(TypedDict):
    """Saved with torch.save; the model/axis keys match assistant_axis.projection.load_unit_axes."""

    model: str
    axis: Float[Tensor, "layer hidden"]
    train_prompt_ids: list[int]
    positive_count: int
    negative_count: int


def model_slug(model: str) -> str:
    return model.replace("/", "__")


def prompts_path(data_dir: Path) -> Path:
    return data_dir / "prompts.jsonl"


def transcripts_path(run_dir: Path, model: str) -> Path:
    return run_dir / "transcripts" / f"{model_slug(model)}.jsonl"


def judgements_path(run_dir: Path, target_model: str) -> Path:
    return run_dir / "judgements" / f"{model_slug(target_model)}.jsonl"


def vectors_path(run_dir: Path, model: str) -> Path:
    return run_dir / "vectors" / f"{model_slug(model)}.pt"


def read_prompts(path: Path) -> list[PromptRecord]:
    return cast(list[PromptRecord], read_jsonl(path))


def read_transcripts(path: Path) -> list[TranscriptRecord]:
    return cast(list[TranscriptRecord], read_jsonl(path))


def read_judgements(path: Path) -> list[JudgementRecord]:
    return cast(list[JudgementRecord], read_jsonl(path))
