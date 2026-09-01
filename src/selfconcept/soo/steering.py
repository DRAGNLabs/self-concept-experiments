"""Steering-vector variant of self-other overlap.

Instead of training LoRA adapters until self/other o_proj activations match,
measure the mean activation difference v = E[a_self - a_other] once on the
base model (scripts/extract_steering.py), then intervene at inference:

    mode 'add':     h <- h - alpha * v
        alpha is in natural units: alpha=1 subtracts the full mean
        self-minus-other difference, pushing self-referencing activations
        toward their other-referencing counterparts.
    mode 'project': h <- h - alpha * (h . v_hat) v_hat
        removes the component along the self/other axis; alpha=1 deletes it
        entirely (directional ablation), fractional alpha attenuates it.

Applied at the same site the SOO loss trained on (one layer's
self_attn.o_proj output, every token position), so results are directly
comparable to the LoRA rounds.
"""

from contextlib import contextmanager
from pathlib import Path

import torch

from .activations import get_decoder_layers

TOKEN_MODES = ("last", "mean")


def load_vectors(path: str | Path) -> dict:
    """Load an extract_steering.py output: metadata plus per-layer vectors."""
    return torch.load(path, map_location="cpu", weights_only=True)


def get_vector(data: dict, layer: int, token_mode: str) -> torch.Tensor:
    """One layer's self-minus-other vector (float32, [hidden_size])."""
    if token_mode not in TOKEN_MODES:
        raise ValueError(f"token_mode must be one of {TOKEN_MODES}, got {token_mode!r}")
    return data[f"vectors_{token_mode}"][layer]


def random_matched_vector(vector: torch.Tensor, seed: int) -> torch.Tensor:
    """Random Gaussian direction scaled to the real vector's norm.

    Damage control: separates "this direction matters" from "any perturbation
    of this magnitude at this layer changes behavior".
    """
    gen = torch.Generator().manual_seed(seed)
    rand = torch.randn(vector.shape, generator=gen, dtype=torch.float32)
    return rand / rand.norm() * vector.norm()


def steer_o_proj(model, layer: int, vector: torch.Tensor, alpha: float, mode: str = "add"):
    """Register a steering hook on one layer's o_proj output; returns the handle.

    The hook stays active for the model's lifetime unless the handle is
    removed — use apply_steering() when a scoped intervention is needed.
    """
    module = get_decoder_layers(model)[layer].get_submodule("self_attn.o_proj")
    vector = vector.float()
    unit = vector / vector.norm()

    def hook(_module, _inputs, output):
        v = vector.to(output.device, output.dtype)
        if mode == "add":
            return output - alpha * v
        if mode == "project":
            u = unit.to(output.device, output.dtype)
            return output - alpha * (output * u).sum(-1, keepdim=True) * u
        raise ValueError(f"unknown steering mode {mode!r}")

    return module.register_forward_hook(hook)


@contextmanager
def apply_steering(model, layer: int, vector: torch.Tensor, alpha: float, mode: str = "add"):
    """Context-managed steer_o_proj (removes the hook on exit)."""
    handle = steer_o_proj(model, layer, vector, alpha, mode)
    try:
        yield
    finally:
        handle.remove()
