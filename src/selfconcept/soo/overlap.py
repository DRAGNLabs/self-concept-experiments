"""Endpoint measurements for matched self/other forward passes.

All distances use float32 activations at each prompt's last non-padding token.
The old full-tensor latent measurement is deliberately kept separate.
"""

from contextlib import ExitStack, nullcontext

import numpy as np
import torch

from .activations import attn_out_proj, get_decoder_layers


def last_valid_indices(mask: torch.Tensor) -> torch.Tensor:
    """Support either padding side; do not assume length - 1 is the endpoint."""
    if mask.ndim != 2 or not ((mask == 0) | (mask == 1)).all():
        raise ValueError("attention_mask must be a binary [batch, sequence] tensor")
    positions = torch.arange(mask.shape[1], device=mask.device)
    indices = positions.expand_as(mask).masked_fill(~mask.bool(), -1).max(1).values
    if (indices < 0).any():
        raise ValueError("cannot measure an empty prompt")
    return indices


def endpoint(output, mask: torch.Tensor, offset: int = 0) -> torch.Tensor:
    """Return a detached CPU copy, including for tuple-valued block outputs.

    `offset` counts valid tokens back from the last one: 0 is the last valid
    token, 1 the one before it. Every prompt must be long enough.
    """
    value = output[0] if isinstance(output, tuple) else output
    if not isinstance(value, torch.Tensor) or value.ndim != 3:
        raise ValueError("capture site must return [batch, sequence, hidden]")
    if value.shape[:2] != mask.shape:
        raise ValueError("capture and attention mask shapes differ")
    if offset < 0:
        raise ValueError("endpoint offset must be nonnegative")
    indices = last_valid_indices(mask).to(value.device) - offset
    if (indices < 0).any() or not mask.bool()[torch.arange(len(indices), device=mask.device), indices.to(mask.device)].all():
        raise ValueError("endpoint offset reaches before the first valid token")
    rows = torch.arange(len(indices), device=value.device)
    result = value[rows, indices].detach().float().cpu().clone()
    if not torch.isfinite(result).all():
        raise ValueError("non-finite captured activation")
    return result


def decoder_final_norm(model):
    """Find the final norm beside the decoder layers, including PEFT wrappers.

    Fail explicitly for an unsupported architecture instead of silently calling
    the last block's pre-normalization output the final decoder representation.
    """
    layers = get_decoder_layers(model)
    for _, module in model.named_modules():
        if getattr(module, "layers", None) is layers:
            for name in ("norm", "final_layernorm"):
                norm = getattr(module, name, None)
                if isinstance(norm, torch.nn.Module):
                    return norm
    raise ValueError("cannot locate final decoder norm on this model")


def measure_condition(model, batches, layer: int, residual_layers=None,
                      intervention=None, endpoint_offset: int = 0) -> dict[str, torch.Tensor]:
    """Capture fixed, interleaved self/other inputs under one intervention.

    `batches` contain ordinary model keyword tensors (no cache or generation).
    `intervention` is a factory returning a context manager, e.g. apply_steering.
    Install the pre-capture, intervention, and post-capture in that order.
    Saved tensors are [pairs, self_or_other, hidden].
    """
    layers = get_decoder_layers(model)
    if layer < 0 or layer >= len(layers):
        raise ValueError("intervention layer is outside the decoder")
    if residual_layers is None:
        residual_layers = list(range(layer, len(layers)))
    if not residual_layers or any(i < layer or i >= len(layers) for i in residual_layers):
        raise ValueError("residual captures must be at or after the intervention layer")
    residual_layers = sorted(set([layer, *residual_layers]))
    sites = ["hook_before", "hook_after", *[f"residual_L{i}" for i in residual_layers], "final_norm"]
    saved = {site: [] for site in sites}
    current = {}
    mask = None

    def capture(site):
        def hook(_module, _inputs, output):
            if site in current:
                raise ValueError(f"site {site} executed twice in one forward pass")
            current[site] = endpoint(output, mask, endpoint_offset)
        return hook

    # eval() disables adapter dropout as well as ordinary model dropout.
    model.eval()
    module = attn_out_proj(layers[layer])
    with ExitStack() as stack:
        stack.callback(module.register_forward_hook(capture("hook_before")).remove)
        stack.enter_context(intervention() if intervention else nullcontext())
        stack.callback(module.register_forward_hook(capture("hook_after")).remove)
        for i in residual_layers:
            stack.callback(layers[i].register_forward_hook(capture(f"residual_L{i}")).remove)
        stack.callback(decoder_final_norm(model).register_forward_hook(capture("final_norm")).remove)
        with torch.inference_mode():
            for batch in batches:
                mask = batch["attention_mask"]
                if len(mask) % 2:
                    raise ValueError("each batch must contain complete self/other pairs")
                current.clear()
                model(**batch, use_cache=False)
                if set(current) != set(sites):
                    raise ValueError(f"missing capture sites: {set(sites) - set(current)}")
                for site in sites:
                    saved[site].append(current[site].reshape(-1, 2, current[site].shape[-1]))
    if not saved["hook_before"]:
        raise ValueError("no pairs to measure")
    return {site: torch.cat(values) for site, values in saved.items()}


def site_metrics(values: torch.Tensor) -> tuple[dict, dict]:
    """Raw squared distances, norms, and population variation; no normalization."""
    values = values.float()
    if values.ndim != 3 or values.shape[1] != 2 or len(values) == 0:
        raise ValueError("expected nonempty [pairs, 2, hidden] activations")
    s, o = values[:, 0], values[:, 1]
    gaps = (s - o).square().mean(-1)
    per_pair = {
        "gap": gaps.tolist(),
        "self_norm": s.norm(dim=-1).tolist(),
        "other_norm": o.norm(dim=-1).tolist(),
    }
    flat = values.flatten(0, 1)
    summary = {
        "n_pairs": len(values),
        "gap": gaps.mean().item(),
        "centroid_gap": (s.mean(0) - o.mean(0)).square().mean().item(),
        "mean_self_norm": s.norm(dim=-1).mean().item(),
        "mean_other_norm": o.norm(dim=-1).mean().item(),
        "prompt_variance": flat.var(dim=0, correction=0).mean().item(),
        "self_prompt_variance": s.var(dim=0, correction=0).mean().item(),
        "other_prompt_variance": o.var(dim=0, correction=0).mean().item(),
    }
    return summary, per_pair


def paired_interval(changes, families, n_bootstrap=2000, seed=0):
    """Percentile interval for a pair-weighted mean, resampling whole families.

    A single family cannot estimate between-family uncertainty: return no CI.
    """
    changes = np.asarray(changes, dtype=np.float64)
    if len(changes) != len(families) or not len(changes):
        raise ValueError("changes and families must be nonempty and aligned")
    labels = sorted(set(families))
    groups = [changes[np.array(families) == label] for label in labels]
    if len(groups) < 2 or n_bootstrap < 1:
        return None
    sums = np.array([group.sum() for group in groups])
    sizes = np.array([len(group) for group in groups])
    rng = np.random.default_rng(seed)
    samples = rng.integers(len(groups), size=(n_bootstrap, len(groups)))
    means = sums[samples].sum(1) / sizes[samples].sum(1)
    return np.quantile(means, [0.025, 0.975]).tolist()


def summarize(conditions, pairs, n_bootstrap=2000, seed=0):
    """Summaries by pair kind and site, plus paired changes against base."""
    base = conditions["base"]
    report, records = {}, []
    # Keep self/other and nonsocial controls separate, never average them together.
    for name, sites in conditions.items():
        report[name] = {}
        for kind in sorted({p["kind"] for p in pairs}):
            idx = [i for i, p in enumerate(pairs) if p["kind"] == kind]
            families = [pairs[i]["family"] for i in idx]
            report[name][kind] = {}
            for site, values in sites.items():
                summary, per_pair = site_metrics(values[idx])
                _, base_pair = site_metrics(base[site][idx])
                delta = np.array(per_pair["gap"]) - np.array(base_pair["gap"])
                displacement = (values[idx].float() - base[site][idx].float()).square().mean((1, 2))
                summary.update({
                    "n_families": len(set(families)),
                    "gap_change_vs_base": float(delta.mean()),
                    "gap_change_ci95": paired_interval(delta, families, n_bootstrap, seed),
                    "mean_squared_displacement_vs_base": displacement.mean().item(),
                })
                report[name][kind][site] = summary
                for j, i in enumerate(idx):
                    records.append({
                        "id": pairs[i]["id"], "family": pairs[i]["family"],
                        "kind": kind, "split": pairs[i]["split"],
                        "condition": name, "site": site,
                        **{key: vals[j] for key, vals in per_pair.items()},
                        "gap_change_vs_base": float(delta[j]),
                        "mean_squared_displacement_vs_base": displacement[j].item(),
                    })
    return report, records
