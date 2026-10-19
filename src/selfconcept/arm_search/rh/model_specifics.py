"""What the reward-hack arm search needs to know about the model it searches: how to serve it, how its reasoning is
controlled and how to sample it."""

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final, Literal, Protocol

from selfconcept.arm_search.rh.conditions import Effort

# codebench.run backends that talk to the `vllm serve` of vllm_server_workers.sbatch
type VllmServerBackend = Literal["vllm-server-harmony", "vllm-server-chat"]


class RhModelSpecifics(Protocol):
    @property
    def model_id(self) -> str: ...

    @property
    def backend(self) -> VllmServerBackend: ...

    @property
    def generation_vram_gb(self) -> float: ...

    @property
    def temperature(self) -> float: ...

    @property
    def efforts(self) -> tuple[Effort, ...]:
        """The first is the ladder model's reference level."""
        ...

    def vllm_serve_args(self) -> list[str]:
        """No argument may contain spaces: the sbatch word-splits them."""
        ...

    def chat_kwargs(self, effort: Effort) -> dict[str, object]: ...


@dataclass(frozen=True)
class GptOssSpecifics:
    """Harmony prompts; the analysis channel is the reasoning and its length is set by reasoning_effort."""

    model_id: str
    backend: VllmServerBackend = "vllm-server-harmony"
    generation_vram_gb: float = 130
    temperature: float = 1.0
    efforts: tuple[Effort, ...] = ("medium", "low")

    def vllm_serve_args(self) -> list[str]:
        return []

    def chat_kwargs(self, effort: Effort) -> dict[str, object]:
        return {"reasoning_effort": effort}


@dataclass(frozen=True)
class Qwen3Specifics:
    """Thinks by default; YaRN x4 stretches its native 40,960-token window to 128k for the long workspace episodes
    (vLLM wants max_position_embeddings already scaled)."""

    model_id: str
    backend: VllmServerBackend = "vllm-server-chat"
    generation_vram_gb: float = 100
    temperature: float = 0.6
    efforts: tuple[Effort, ...] = ("thinking",)

    def vllm_serve_args(self) -> list[str]:
        yarn_overrides = {
            "max_position_embeddings": 131072,
            "rope_parameters": {
                "rope_type": "yarn",
                "rope_theta": 1000000,
                "factor": 4.0,
                "original_max_position_embeddings": 32768,
            },
        }
        return [
            *["--reasoning-parser", "qwen3", "--max-model-len", "131072"],
            *["--hf-overrides", json.dumps(yarn_overrides, separators=(",", ":"))],
        ]

    def chat_kwargs(self, effort: Effort) -> dict[str, object]:
        return {"enable_thinking": True}


@dataclass(frozen=True)
class Gemma4Specifics:
    """Its template leaves thinking off (an empty thought channel) unless enable_thinking is set."""

    model_id: str
    backend: VllmServerBackend = "vllm-server-chat"
    generation_vram_gb: float = 100
    temperature: float = 1.0
    efforts: tuple[Effort, ...] = ("thinking",)

    def vllm_serve_args(self) -> list[str]:
        return ["--reasoning-parser", "gemma4", "--max-model-len", "131072"]

    def chat_kwargs(self, effort: Effort) -> dict[str, object]:
        return {"enable_thinking": True}


RH_MODEL_SPECIFICS_FACTORY_BY_NAME_SUBSTRING: Final[dict[str, Callable[[str], RhModelSpecifics]]] = {
    "gpt-oss": GptOssSpecifics,
    "qwen3": Qwen3Specifics,
    "gemma-4": Gemma4Specifics,
}


def rh_model_specifics(model_id: str) -> RhModelSpecifics:
    for name_substring, specifics_factory in RH_MODEL_SPECIFICS_FACTORY_BY_NAME_SUBSTRING.items():
        if name_substring in model_id.lower():
            return specifics_factory(model_id)
    raise ValueError(f"no reward-hack model specifics registered for {model_id}")
