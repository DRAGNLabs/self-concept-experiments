"""Small torch helpers shared across experiments."""

from __future__ import annotations

import torch
from jaxtyping import Int


def build_padded_batch(
    token_id_lists: list[list[int]],
    pad_token_id: int,
    max_length: int,
    device: torch.device,
) -> tuple[Int[torch.Tensor, "batch seq"], Int[torch.Tensor, "batch seq"]]:
    """Right-pad a batch of token-id lists to a common length, returning ids and attention mask.

    Sequences longer than ``max_length`` are truncated; the attention mask is 1 for real tokens
    and 0 for padding.
    """
    max_len = min(max_length, max(len(ids) for ids in token_id_lists))
    input_id_rows: list[list[int]] = []
    attention_mask_rows: list[list[int]] = []
    for ids in token_id_lists:
        truncated = ids[:max_len]
        pad_count = max_len - len(truncated)
        input_id_rows.append(truncated + [pad_token_id] * pad_count)
        attention_mask_rows.append([1] * len(truncated) + [0] * pad_count)
    input_ids = torch.tensor(input_id_rows, dtype=torch.long, device=device)
    attention_mask = torch.tensor(attention_mask_rows, dtype=torch.long, device=device)
    return input_ids, attention_mask
