"""Restrict a loaded PEFT LoRA adapter to a subset of decoder layers or target modules.

The first study's adapters (and every LoRA in this repository) put rank-r
updates on q_proj and v_proj in *every* decoder layer, while the loss read one
layer's attention output. To ask where the behavioral effect lives, keep the
trained deltas in a chosen set of layers and remove the rest: each unselected
LoRA module is swapped back for its frozen base layer, so the kept layers run
the unmerged adapter path bit-for-bit as in the full model and the removed
layers run the base model exactly (LAYER_ROUND.md). No merging, no bf16
rounding of small deltas into the base weights.

Layer specs (N = number of decoder layers):
  all            every layer
  only:L         layer L
  except:L       every layer but L
  below:L        0 .. L-1
  above:L        L+1 .. N-1
  range:A-B      A .. B inclusive
  layers:3,5,7   an explicit list
"""

import re

import torch

from .activations import get_decoder_layers

LAYER_INDEX = re.compile(r"\.layers\.(\d+)\.")
DEFAULT_MODULES = ("q_proj", "v_proj")


def parse_layer_spec(spec: str, n_layers: int) -> set[int]:
    spec = spec.strip()
    if spec == "all":
        return set(range(n_layers))
    kind, _, arg = spec.partition(":")
    if not arg:
        raise ValueError(f"bad layer spec {spec!r}")
    if kind in ("only", "except", "below", "above"):
        layer = int(arg)
        if not 0 <= layer < n_layers:
            raise ValueError(f"layer {layer} out of range for {n_layers} layers")
        if kind == "only":
            return {layer}
        if kind == "except":
            return set(range(n_layers)) - {layer}
        if kind == "below":
            return set(range(layer))
        return set(range(layer + 1, n_layers))
    if kind == "range":
        a, b = (int(x) for x in arg.split("-"))
        if not 0 <= a <= b < n_layers:
            raise ValueError(f"range {a}-{b} out of order or out of range for {n_layers} layers")
        return set(range(a, b + 1))
    if kind == "layers":
        layers = {int(x) for x in arg.split(",") if x.strip()}
        if any(not 0 <= l < n_layers for l in layers):
            raise ValueError(f"layers {sorted(layers)} out of range for {n_layers} layers")
        return layers
    raise ValueError(f"bad layer spec {spec!r}")


def parse_module_spec(spec: str | None) -> tuple[str, ...]:
    if spec is None or spec in ("both", "all"):
        return DEFAULT_MODULES
    modules = tuple(m.strip() for m in spec.split(",") if m.strip())
    if not modules:
        raise ValueError(f"bad module spec {spec!r}")
    return modules


def restrict_lora(model, keep_layers: set[int], keep_modules: tuple[str, ...] = DEFAULT_MODULES) -> dict:
    """Swap every LoRA module outside (keep_layers x keep_modules) for its base layer, in place.

    Returns counts: LoRA modules found, kept, removed, and the kept (layer, module) pairs.
    """
    from peft.tuners.lora.layer import LoraLayer

    found = [(name, module) for name, module in model.named_modules() if isinstance(module, LoraLayer)]
    if not found:
        raise ValueError("no LoRA modules found; is this a PeftModel with a loaded adapter?")
    kept, removed = [], 0
    for name, module in found:
        m = LAYER_INDEX.search(name)
        if m is None:
            raise ValueError(f"cannot read a decoder-layer index from LoRA module name {name!r}")
        layer = int(m.group(1))
        leaf = name.rsplit(".", 1)[-1]
        if layer in keep_layers and leaf in keep_modules:
            kept.append((layer, leaf))
            continue
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name)
        setattr(parent, child, module.get_base_layer())
        removed += 1
    return {"found": len(found), "kept": len(kept), "removed": removed,
            "kept_layers": sorted({l for l, _ in kept}), "kept_modules": sorted({m for _, m in kept})}


def apply_adapter_subset(model, layer_spec: str | None, module_spec: str | None) -> dict | None:
    """Apply --adapter-layers / --adapter-modules to a PeftModel; None when both are unset (full adapter)."""
    if layer_spec is None and module_spec is None:
        return None
    n_layers = len(get_decoder_layers(model))
    keep_layers = parse_layer_spec(layer_spec or "all", n_layers)
    keep_modules = parse_module_spec(module_spec)
    stats = restrict_lora(model, keep_layers, keep_modules)
    stats.update({"layer_spec": layer_spec or "all", "module_spec": module_spec or "both", "n_layers": n_layers})
    return stats
