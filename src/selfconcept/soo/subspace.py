"""Fit uncentered difference subspaces and remove them at attention outputs."""

from contextlib import contextmanager
import math

import torch

from .activations import attn_out_proj, get_decoder_layers


def fit_subspace(differences, max_rank):
    """Rows are matched self-minus-other differences; do not subtract the mean."""
    x = differences.detach().float().cpu()
    if x.ndim != 2 or min(x.shape) < 1 or not torch.isfinite(x).all() or max_rank < 1:
        raise ValueError("expected finite [pairs, hidden] differences and positive rank")
    _, singular, vh = torch.linalg.svd(x, full_matrices=False)
    tolerance = max(x.shape) * torch.finfo(x.dtype).eps * singular[0]
    available = int((singular > tolerance).sum())
    rank = min(max_rank, available)
    if rank == 0:
        raise ValueError("fit differences have zero numerical rank")
    return {"basis": vh[:rank].T.contiguous(), "singular_values": singular,
            "numerical_rank": available, "rank_tolerance": float(tolerance),
            "mean_difference": x.mean(0), "n_pairs": len(x),
            "centered": False,
            "captured_energy_fraction": singular[:rank].square().cumsum(0) / singular.square().sum()}


def random_subspace(hidden, rank, seed):
    if not 1 <= rank <= hidden:
        raise ValueError("random rank must be between 1 and hidden size")
    generator = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(hidden, rank, generator=generator), mode="reduced")
    return q


@contextmanager
def apply_subspace(model, layer, basis, strength, positions=None):
    """h <- h - strength * h U U^T, with optional existing positional mask.

    The basis has orthonormal columns. Projection arithmetic is float32;
    return the original model dtype. Cache only the small basis on each device.
    """
    basis = basis.detach().float()
    if basis.ndim != 2 or basis.shape[1] < 1 or not torch.isfinite(basis).all():
        raise ValueError("basis must be finite [hidden, rank]")
    if not torch.allclose(basis.T @ basis, torch.eye(basis.shape[1], device=basis.device), atol=1e-5, rtol=1e-5):
        raise ValueError("basis columns must be orthonormal")
    if not math.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError("subspace strength must lie between 0 and 1")
    if layer < 0 or layer >= len(get_decoder_layers(model)):
        raise ValueError("layer is outside the decoder")
    module = attn_out_proj(get_decoder_layers(model)[layer])
    cache = {}
    if positions is not None and positions.mode == "all":
        positions = None
    if positions is not None:
        positions.attach(model)

    def hook(_module, _inputs, output):
        if output.shape[-1] != basis.shape[0]:
            raise ValueError("basis width does not match activation width")
        if strength == 0:
            return output
        if output.device not in cache:
            cache[output.device] = basis.to(output.device)
        u = cache[output.device]
        value = output.float()
        delta = strength * ((value @ u) @ u.T)
        if positions is not None:
            delta *= positions.mask(output).float()
        return (value - delta).to(output.dtype)

    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()
        if positions is not None and positions._handle is not None:
            positions._handle.remove()
