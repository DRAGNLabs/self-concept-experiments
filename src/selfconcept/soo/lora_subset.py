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

Module specs: comma-separated LoRA leaf names (q_proj, v_proj, in_proj_qkv, ...);
"all" or unset keeps every LoRA module and "both" means q_proj,v_proj. A fused
projection can be cut to part of its output with a slice suffix, e.g.
in_proj_qkv.v (value channels only) or in_proj_qkv.qk: the kept module's lora_B
rows outside the slice are zeroed, so the delta s*B*A*x reaches only those
output channels. Slices are defined for the Gated DeltaNet in_proj_qkv of
Qwen3.5/3.8, whose output is [query | key | value] with key_dim, key_dim and
value_dim channels, followed by a per-channel (depthwise) convolution.
"""

import re

import torch

from .activations import get_decoder_layers

LAYER_INDEX = re.compile(r"\.layers\.(\d+)\.")
DEFAULT_MODULES = ("q_proj", "v_proj")
SLICE_PARTS = {"q", "k", "v"}


def fused_qkv_slices(parent) -> dict[str, tuple[int, int]]:
    """Output-channel ranges of a Gated DeltaNet in_proj_qkv, read from its parent block."""
    key_dim, value_dim = int(parent.key_dim), int(parent.value_dim)
    return {"q": (0, key_dim), "k": (key_dim, 2 * key_dim), "v": (2 * key_dim, 2 * key_dim + value_dim)}


def split_module(entry: str) -> tuple[str, str | None]:
    """'in_proj_qkv.v' -> ('in_proj_qkv', 'v'); 'v_proj' -> ('v_proj', None)."""
    leaf, _, part = entry.partition(".")
    if part and (not set(part) <= SLICE_PARTS or len(set(part)) != len(part)):
        raise ValueError(f"bad slice {part!r} in module spec entry {entry!r}; use letters from q, k, v")
    return leaf, part or None


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


def parse_module_spec(spec: str | None) -> tuple[str, ...] | None:
    """None keeps every LoRA module; otherwise a tuple of leaf names, each optionally with a slice suffix."""
    if spec is None or spec == "all":
        return None
    if spec == "both":
        return DEFAULT_MODULES
    modules = tuple(m.strip() for m in spec.split(",") if m.strip())
    if not modules:
        raise ValueError(f"bad module spec {spec!r}")
    leaves = [split_module(m)[0] for m in modules]
    if len(set(leaves)) != len(leaves):
        raise ValueError(f"module spec {spec!r} names a module twice")
    return modules


def restrict_lora(model, keep_layers: set[int], keep_modules: tuple[str, ...] | None = DEFAULT_MODULES) -> dict:
    """Swap every LoRA module outside (keep_layers x keep_modules) for its base layer, in place.

    keep_modules None keeps every module type. A kept entry with a slice suffix
    (in_proj_qkv.v) also zeroes the module's lora_B rows outside that slice.
    Returns counts: LoRA modules found, kept, removed, sliced, and the kept (layer, module) pairs.
    """
    from peft.tuners.lora.layer import LoraLayer

    wanted = None if keep_modules is None else dict(split_module(m) for m in keep_modules)
    found = [(name, module) for name, module in model.named_modules() if isinstance(module, LoraLayer)]
    if not found:
        raise ValueError("no LoRA modules found; is this a PeftModel with a loaded adapter?")
    kept, removed, sliced = [], 0, 0
    for name, module in found:
        m = LAYER_INDEX.search(name)
        if m is None:
            raise ValueError(f"cannot read a decoder-layer index from LoRA module name {name!r}")
        layer = int(m.group(1))
        leaf = name.rsplit(".", 1)[-1]
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name)
        if layer in keep_layers and (wanted is None or leaf in wanted):
            part = None if wanted is None else wanted[leaf]
            if part is not None:
                if leaf != "in_proj_qkv":
                    raise ValueError(f"slices are defined for in_proj_qkv only, not {leaf!r}")
                ranges = fused_qkv_slices(parent)
                keep = torch.zeros(module.out_features, dtype=torch.bool)
                for p in part:
                    a, b = ranges[p]
                    keep[a:b] = True
                if ranges["v"][1] != module.out_features:
                    raise ValueError(f"{name}: slices cover {ranges['v'][1]} channels, module has {module.out_features}")
                with torch.no_grad():
                    for lora_b in module.lora_B.values():
                        lora_b.weight[~keep.to(lora_b.weight.device)] = 0
                sliced += 1
            kept.append((layer, leaf if part is None else f"{leaf}.{part}"))
            continue
        setattr(parent, child, module.get_base_layer())
        removed += 1
    return {"found": len(found), "kept": len(kept), "removed": removed, "sliced": sliced,
            "kept_layers": sorted({l for l, _ in kept}), "kept_modules": sorted({m for _, m in kept})}


def apply_adapter_subset(model, layer_spec: str | None, module_spec: str | None) -> dict | None:
    """Apply --adapter-layers / --adapter-modules to a PeftModel; None when both are unset (full adapter)."""
    if layer_spec is None and module_spec is None:
        return None
    n_layers = len(get_decoder_layers(model))
    keep_layers = parse_layer_spec(layer_spec or "all", n_layers)
    keep_modules = parse_module_spec(module_spec)
    stats = restrict_lora(model, keep_layers, keep_modules)
    stats.update({"layer_spec": layer_spec or "all", "module_spec": module_spec or "all", "n_layers": n_layers})
    return stats
