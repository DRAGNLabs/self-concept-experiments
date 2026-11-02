"""What the residual-stream steering harness needs to know about a model family, in the model-specifics pattern of
``selfconcept.arm_search.rh.model_specifics``: a Protocol, one frozen dataclass per family, a name-substring registry
and a factory. Used by ``experiments/schemer-axis/scripts/steered_rh.py`` (HF hooks), ``steered_rh_vllm.py`` (class
patch before the vLLM engine is built) and ``capability_eval_vllm.py``; it replaces the ``FAMILIES`` dicts those
scripts carried.

Per family: the record label (``backend`` is ``hf-{family}`` or ``vllm-{family}-patched``), the HF attention kernel
to load with (None = the model default), the vLLM decoder-block class to patch (module path, class name), the patcher
matching that class's forward contract (``patch_vllm_blocks`` when forward returns ``(mlp_output, residual)``,
``patch_vllm_stream_layers`` when it returns the residual stream itself), how to read the residual stream off a
forward-hook output of that class, and the completion splitter (raw text -> (reasoning, final), reasoning "" when
the model opened no reasoning channel). ``completion_splitter(chat_kwargs)`` lets a family whose template can turn
thinking off return the answer-only splitter in that case (Qwen's ``enable_thinking: false``)."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Final, Protocol

import torch

from selfconcept.common.generation.model_specifics.gemma4 import split_gemma4_completion
from selfconcept.common.generation.model_specifics.muse_glimmer import split_muse_glimmer_completion
from selfconcept.common.generation.model_specifics.think_flag import split_qwen_think_completion
from selfconcept.common.harmony import split_harmony_completion
from selfconcept.measurement.intervene import patch_vllm_blocks, patch_vllm_stream_layers

type CompletionSplitter = Callable[[str], tuple[str, str]]
type Restore = Callable[[], None]


class SteeringModelSpecifics(Protocol):
    @property
    def family(self) -> str:
        """Record label; the same names the FAMILIES tables used (gpt-oss, gemma4, muse-glimmer, qwen3.5)."""
        ...

    @property
    def hf_attn_implementation(self) -> str | None: ...

    @property
    def vllm_block(self) -> tuple[str, str]:
        """(module path, class name) of the vLLM decoder-block class whose forward is patched."""
        ...

    def patch_vllm(self, block_cls: type, transforms_by_layer: dict[int, object]) -> Restore:
        """Patch ``block_cls.forward`` so the listed layers apply their transform to the residual stream."""
        ...

    def vllm_stream(self, output: object) -> torch.Tensor:
        """The residual stream after the block, read off a forward hook's ``output`` of ``vllm_block``."""
        ...

    def split_completion(self, raw_completion: str) -> tuple[str, str]: ...

    def completion_splitter(self, chat_kwargs: Mapping[str, object]) -> CompletionSplitter:
        """The splitter for a run generated with these chat-template kwargs."""
        ...


def split_answer_only(raw_completion: str) -> tuple[str, str]:
    """Thinking off: the template closes the thought channel inside the prompt, so every generated token is the
    answer."""
    return "", raw_completion.strip()


def _sum_stream(output) -> torch.Tensor:
    """``(mlp_output, residual)`` blocks: the residual stream after the block is the sum."""
    return output[0] + output[1]


def _first_stream(output) -> torch.Tensor:
    """Blocks whose forward returns the residual stream as the first element."""
    return output[0]


class _FixedSplitter:
    def completion_splitter(self, chat_kwargs: Mapping[str, object]) -> CompletionSplitter:
        return self.split_completion  # type: ignore[attr-defined]


@dataclass(frozen=True)
class GptOssSteeringSpecifics(_FixedSplitter):
    """Harmony; vLLM's ``TransformerBlock`` returns ``(mlp_output, residual)``; HF loads with the FA3 kernel."""

    family: str = "gpt-oss"
    hf_attn_implementation: str | None = "kernels-community/vllm-flash-attn3"
    vllm_block: tuple[str, str] = ("vllm.model_executor.models.gpt_oss", "TransformerBlock")

    def patch_vllm(self, block_cls: type, transforms_by_layer: dict[int, object]) -> Restore:
        return patch_vllm_blocks(block_cls, transforms_by_layer)

    def vllm_stream(self, output: object) -> torch.Tensor:
        return _sum_stream(output)

    def split_completion(self, raw_completion: str) -> tuple[str, str]:
        return split_harmony_completion(raw_completion)


@dataclass(frozen=True)
class Gemma4SteeringSpecifics(_FixedSplitter):
    """``Gemma4DecoderLayer`` returns ``(hidden_states, None)`` with the residual stream first (after the layer
    scalar, the same tensor the HF ``Gemma4TextDecoderLayer`` returns)."""

    family: str = "gemma4"
    hf_attn_implementation: str | None = None
    vllm_block: tuple[str, str] = ("vllm.model_executor.models.gemma4", "Gemma4DecoderLayer")

    def patch_vllm(self, block_cls: type, transforms_by_layer: dict[int, object]) -> Restore:
        return patch_vllm_stream_layers(block_cls, transforms_by_layer)

    def vllm_stream(self, output: object) -> torch.Tensor:
        return _first_stream(output)

    def split_completion(self, raw_completion: str) -> tuple[str, str]:
        reasoning, content = split_gemma4_completion(raw_completion)
        return reasoning or "", content


@dataclass(frozen=True)
class MuseGlimmerSteeringSpecifics(_FixedSplitter):
    """``MuseGlimmerDecoderLayer.forward(positions, hidden_states, residual) -> (hidden_states, residual)``: the first
    element is the full residual stream after the layer (the HF ``MuseGlimmerTextDecoderLayer`` output); the second
    is stale. The ATEM ``to=self`` channel is the reasoning."""

    family: str = "muse-glimmer"
    hf_attn_implementation: str | None = None
    vllm_block: tuple[str, str] = ("vllm.model_executor.models.muse_glimmer", "MuseGlimmerDecoderLayer")

    def patch_vllm(self, block_cls: type, transforms_by_layer: dict[int, object]) -> Restore:
        return patch_vllm_stream_layers(block_cls, transforms_by_layer)

    def vllm_stream(self, output: object) -> torch.Tensor:
        return _first_stream(output)

    def split_completion(self, raw_completion: str) -> tuple[str, str]:
        reasoning, content = split_muse_glimmer_completion(raw_completion)
        return reasoning or "", content


@dataclass(frozen=True)
class Qwen35SteeringSpecifics:
    """Qwen3.5/3.8: ``Qwen3_5DecoderLayer`` (``Qwen3NextDecoderLayer.forward``, called as ``layer(positions=,
    hidden_states=, residual=)``) returns ``(mlp_output, residual)`` like gpt-oss's block. Thinking is on by default
    and the completion is ``{reasoning}</think>{answer}``; with ``enable_thinking: false`` the template closes the
    thought channel in the prompt and the whole completion is the answer."""

    family: str = "qwen3.5"
    hf_attn_implementation: str | None = None
    vllm_block: tuple[str, str] = ("vllm.model_executor.models.qwen3_5", "Qwen3_5DecoderLayer")

    def patch_vllm(self, block_cls: type, transforms_by_layer: dict[int, object]) -> Restore:
        return patch_vllm_blocks(block_cls, transforms_by_layer)

    def vllm_stream(self, output: object) -> torch.Tensor:
        return _sum_stream(output)

    def split_completion(self, raw_completion: str) -> tuple[str, str]:
        return split_qwen_think_completion(raw_completion)

    def completion_splitter(self, chat_kwargs: Mapping[str, object]) -> CompletionSplitter:
        if chat_kwargs.get("enable_thinking", True) is False:
            return split_answer_only
        return self.split_completion


# First substring match (on the lower-cased model name) wins.
STEERING_SPECIFICS_FACTORY_BY_NAME_SUBSTRING: Final[dict[str, Callable[[], SteeringModelSpecifics]]] = {
    "gpt-oss": GptOssSteeringSpecifics,
    "gpt_oss": GptOssSteeringSpecifics,
    "gemma-4": Gemma4SteeringSpecifics,
    "gemma4": Gemma4SteeringSpecifics,
    "glimmer": MuseGlimmerSteeringSpecifics,
    "qwen3.5": Qwen35SteeringSpecifics,
    "qwen3_5": Qwen35SteeringSpecifics,
    "qwen3.8": Qwen35SteeringSpecifics,
    "qwen3_8": Qwen35SteeringSpecifics,
}


def steering_model_specifics(model_name: str) -> SteeringModelSpecifics:
    name = model_name.lower()
    for name_substring, specifics_factory in STEERING_SPECIFICS_FACTORY_BY_NAME_SUBSTRING.items():
        if name_substring in name:
            return specifics_factory()
    raise ValueError(f"no steering model specifics registered for {model_name!r} "
                     f"(known: {sorted(STEERING_SPECIFICS_FACTORY_BY_NAME_SUBSTRING)})")
