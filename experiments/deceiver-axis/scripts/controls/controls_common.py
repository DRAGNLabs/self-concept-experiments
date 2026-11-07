"""Shared pieces of the control-direction scripts: vector helpers, diversity statistics, and the per-model downstream
read used to match a control's steering coefficient to gi_v2's.

Downstream read of a unit direction d added after layer L (first order): root-sum-square over layers L+1 .. last of
||W (g ⊙ d)|| for the matrices that read the residual stream after each norm with gain g. The coefficient matching gi_v2
at dose α is α · read(gi_v2) / read(d). Expert weights of MoE layers are left out (gpt-oss's are MXFP4).
"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Protocol, TypedDict

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from jaxtyping import Float
from safetensors import safe_open


class Spread(TypedDict):
    median: float
    p95: float
    max: float


class EffectiveDimension(TypedDict):
    participation_ratio: float
    entropy_effective_rank: float
    hidden: int


def unit(vector: Float[torch.Tensor, "... hidden"]) -> Float[torch.Tensor, "... hidden"]:
    return vector / vector.norm(dim=-1, keepdim=True)


def spread(values: Float[torch.Tensor, "n"]) -> Spread:
    """Median, 95th percentile and maximum of |values|."""
    magnitudes = values.abs().numpy()
    return {"median": float(np.median(magnitudes)), "p95": float(np.percentile(magnitudes, 95)), "max": float(magnitudes.max())}


def off_diagonal(directions: Float[torch.Tensor, "n hidden"]) -> Float[torch.Tensor, "pairs"]:
    cosines = directions @ directions.T
    return cosines[~torch.eye(len(directions), dtype=torch.bool)]


def effective_dimension(centered: Float[torch.Tensor, "n hidden"]) -> EffectiveDimension:
    """Participation ratio (Σλ)² / Σλ² and entropy effective rank exp(H(λ / Σλ)) of the rows' covariance spectrum."""
    eigenvalues = torch.linalg.svdvals(centered.double()).pow(2)
    shares = eigenvalues / eigenvalues.sum()
    positive = shares[shares > 0]
    return {"participation_ratio": (eigenvalues.sum().pow(2) / eigenvalues.pow(2).sum()).item(),
            "entropy_effective_rank": torch.exp(-(positive * positive.log()).sum()).item(),
            "hidden": centered.shape[1]}


class DownstreamReadSpecifics(Protocol):
    def downstream_read(self, directions: Float[torch.Tensor, "n hidden"], layer: int) -> Float[torch.Tensor, "n"]: ...


def load_weight(repo: str, name: str) -> Float[torch.Tensor, "..."]:
    weight_map = json.loads(Path(hf_hub_download(repo, "model.safetensors.index.json")).read_text())["weight_map"]
    with safe_open(hf_hub_download(repo, weight_map[name]), "pt") as shard:
        return shard.get_tensor(name).float()


@dataclass(frozen=True)
class NormReaders:
    """Weights read through one norm: matrices whose input is the norm's output."""

    norm: str
    matrices: tuple[str, ...]


def read_through(repo: str, prefix: str, num_layers: int, readers: tuple[NormReaders, ...], gain_offset: float,
                 directions: Float[torch.Tensor, "n hidden"], layer: int) -> Float[torch.Tensor, "n"]:
    squared = torch.zeros(len(directions))
    for downstream in range(layer + 1, num_layers):
        for reader in readers:
            scaled = directions * (gain_offset + load_weight(repo, f"{prefix}.layers.{downstream}.{reader.norm}.weight"))
            for matrix in reader.matrices:
                squared += (scaled @ load_weight(repo, f"{prefix}.layers.{downstream}.{matrix}.weight").T).norm(dim=1).pow(2)
    return squared.sqrt()


@dataclass(frozen=True)
class GptOssDownstreamRead(DownstreamReadSpecifics):
    """q/k/v after input_layernorm and the MoE router after post_attention_layernorm; RMSNorm gain used as is."""

    repo: str = "openai/gpt-oss-120b"
    num_layers: int = 36

    def downstream_read(self, directions: Float[torch.Tensor, "n hidden"], layer: int) -> Float[torch.Tensor, "n"]:
        readers = (NormReaders("input_layernorm", ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj")),
                   NormReaders("post_attention_layernorm", ("mlp.router",)))
        return read_through(self.repo, "model", self.num_layers, readers, 0.0, directions, layer)


@dataclass(frozen=True)
class MuseGlimmerDownstreamRead(DownstreamReadSpecifics):
    """q/k/v/gate after input_layernorm and the MLP gate/up after pre_feedforward_layernorm; the norm's gain is 1 + w."""

    repo: str = "meta-models/Muse-Glimmer-30B"
    num_layers: int = 52

    def downstream_read(self, directions: Float[torch.Tensor, "n hidden"], layer: int) -> Float[torch.Tensor, "n"]:
        readers = (NormReaders("input_layernorm", ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.gate_proj")),
                   NormReaders("pre_feedforward_layernorm", ("mlp.gate_proj", "mlp.up_proj")))
        return read_through(self.repo, "model.language_model", self.num_layers, readers, 1.0, directions, layer)


DOWNSTREAM_READ_SPECIFICS_BY_MODEL_SUBSTRING: Final[dict[str, DownstreamReadSpecifics]] = {
    "gpt-oss-120b": GptOssDownstreamRead(),
    "muse-glimmer-30b": MuseGlimmerDownstreamRead(),
}


def downstream_read_specifics_for(model: str) -> DownstreamReadSpecifics:
    for substring, specifics in DOWNSTREAM_READ_SPECIFICS_BY_MODEL_SUBSTRING.items():
        if substring in model.lower():
            return specifics
    raise ValueError(f"no downstream read for {model!r}: implement DownstreamReadSpecifics.downstream_read and add the model "
                     f"to DOWNSTREAM_READ_SPECIFICS_BY_MODEL_SUBSTRING (known: {sorted(DOWNSTREAM_READ_SPECIFICS_BY_MODEL_SUBSTRING)})")
