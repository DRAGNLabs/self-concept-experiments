"""Forward-hook utilities for capturing activations at self_attn.o_proj outputs."""

from contextlib import contextmanager

import torch


def get_decoder_layers(model):
    """Return the decoder layer list of a (possibly PEFT-wrapped) causal LM."""
    base = model
    while not hasattr(base, "model") or not hasattr(base.model, "layers"):
        if hasattr(base, "base_model"):  # PeftModel -> LoraModel
            base = base.base_model
        elif hasattr(base, "model"):
            base = base.model
        else:
            raise AttributeError(f"cannot find decoder layers on {type(model)}")
    return base.model.layers


@contextmanager
def capture_o_proj(model, layer_idx: int, store: list):
    """Capture the o_proj output tensor of one decoder layer during forward passes.

    Appends one tensor per forward pass to `store` (gradients flow through it,
    so it is usable directly in a loss).
    """
    module = get_decoder_layers(model)[layer_idx].self_attn.o_proj

    def hook(_module, _inputs, output):
        store.append(output)

    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()
