"""SpanMapper - Map token spans to activations and compute per-turn aggregates."""

from __future__ import annotations

from typing import Any
import torch


class SpanMapper:
    """Map token spans to per-turn mean activations for a batch of conversations."""

    def map_spans(
        self,
        batch_activations: torch.Tensor,
        batch_spans: list[dict[str, Any]],
        batch_metadata: dict[str, Any],
    ) -> list[torch.Tensor]:
        """
        Map span indices to activations and compute per-turn mean activations.
        Optimized for GPU computation with bf16 consistency.

        Args:
            batch_activations: torch.Tensor shape (num_layers, batch_size, max_seq_len, hidden_size)
            batch_spans: List of span dicts with conversation_id and local indices
            batch_metadata: Dict with batching information

        Returns:
            List of per-conversation activations, each with shape (num_turns, num_layers, hidden_size)
        """
        num_layers, batch_size, max_seq_len, hidden_size = batch_activations.shape
        device = batch_activations.device
        dtype = batch_activations.dtype  # Preserve bf16

        conversation_activations: list[torch.Tensor] = [
            torch.empty(0, num_layers, hidden_size, dtype=dtype, device=device)
            for _ in range(batch_metadata['total_conversations'])
        ]

        # Group spans by conversation
        spans_by_conversation = {}
        for span in batch_spans:
            conv_id = span['conversation_id']
            if conv_id not in spans_by_conversation:
                spans_by_conversation[conv_id] = []
            spans_by_conversation[conv_id].append(span)

        # Sort spans by turn within each conversation
        for conv_id in spans_by_conversation:
            spans_by_conversation[conv_id].sort(key=lambda x: x['turn'])

        # Extract per-turn activations for each conversation
        for conv_id in range(batch_metadata['total_conversations']):
            if conv_id not in spans_by_conversation:
                # Empty conversation - maintain dtype and device consistency
                conversation_activations[conv_id] = torch.empty(0, num_layers, hidden_size, dtype=dtype, device=device)
                continue

            spans = spans_by_conversation[conv_id]
            turn_activations = []

            for span in spans:
                # Use local indices since batch_activations[conv_id] corresponds to this conversation
                start_idx = span['start']  # Local start within the conversation
                end_idx = span['end']      # Local end within the conversation

                # Check bounds to handle truncation
                actual_length = batch_metadata['truncated_lengths'][conv_id]
                if start_idx >= actual_length:
                    # Span is beyond truncated length, skip
                    continue

                # Adjust end index if it exceeds actual length
                end_idx = min(end_idx, actual_length)

                if start_idx >= end_idx:
                    # Invalid span, skip
                    continue

                # Extract activations for this span from the conversation
                # batch_activations[:, conv_id, start_idx:end_idx, :] has shape (num_layers, span_length, hidden_size)
                span_activations = batch_activations[:, conv_id, start_idx:end_idx, :]

                # Compute mean across tokens in this span (optimized for GPU)
                span_length = span_activations.size(1)
                if span_length > 0:
                    if span_length == 1:
                        # Single token - avoid mean computation
                        mean_activation = span_activations.squeeze(1)  # (num_layers, hidden_size)
                    else:
                        # Multi-token span - compute mean on GPU
                        mean_activation = span_activations.mean(dim=1)  # (num_layers, hidden_size)
                    turn_activations.append(mean_activation)

            if turn_activations:
                # Stack to get (num_turns, num_layers, hidden_size)
                conversation_activations[conv_id] = torch.stack(turn_activations)
            else:
                # No valid activations for this conversation - maintain dtype and device consistency
                conversation_activations[conv_id] = torch.empty(0, num_layers, hidden_size, dtype=dtype, device=device)

        return conversation_activations
