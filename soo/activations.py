"""Forward-hook utilities for capturing activations at self_attn.o_proj outputs."""

from contextlib import contextmanager

import torch


def get_decoder_layers(model):
    """Return the decoder layer list of a (possibly PEFT-wrapped) causal LM."""
    # Bounded unwrap (PeftModel -> LoraModel -> *ForCausalLM -> decoder): on the
    # bare decoder, transformers' `base_model` property returns `self`, so an
    # unguarded walk through `base_model` never terminates.
    base = model
    for _ in range(8):
        layers = getattr(base, "layers", None)
        if isinstance(layers, torch.nn.ModuleList):
            return layers
        # "language_model" hops into the text tower of multimodal wrappers
        # (e.g. MuseGlimmerForConditionalGeneration.model.language_model).
        for attr in ("base_model", "model", "language_model"):
            nxt = getattr(base, attr, None)
            if isinstance(nxt, torch.nn.Module) and nxt is not base:
                base = nxt
                break
        else:
            break
    raise AttributeError(f"cannot find decoder layers on {type(model)}")


@contextmanager
def capture_o_proj(model, layer_idx: int, store: list):
    """Capture the o_proj output tensor of one decoder layer during forward passes.

    Appends one tensor per forward pass to `store` (gradients flow through it,
    so it is usable directly in a loss).
    """
    module = get_decoder_layers(model)[layer_idx].get_submodule("self_attn.o_proj")

    def hook(_module, _inputs, output):
        store.append(output)

    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()
