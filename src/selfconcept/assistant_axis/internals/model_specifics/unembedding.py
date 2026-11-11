"""Narrow model-specifics protocol for decoding residual-stream directions through a checkpoint's output head
(logit lens), read straight from the safetensors without building the model."""
from typing import Protocol

from jaxtyping import Float
import torch


class CheckpointTensorLoader(Protocol):
    def __call__(self, tensor_name: str) -> Float[torch.Tensor, "..."]:
        """The named tensor from the checkpoint's safetensors, as float32, without building the model."""
        ...


class UnembeddingSpecifics(Protocol):
    def gain_scaled_unembedding(self, load_tensor: CheckpointTensorLoader) -> Float[torch.Tensor, "vocab hidden"]:
        """The output head with the final RMSNorm gain folded in: logits = this @ (residual / rms(residual))."""
        ...
