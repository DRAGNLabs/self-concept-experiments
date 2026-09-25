"""ActivationExtractor - Extract hidden state activations from model layers."""

from __future__ import annotations

from typing import TYPE_CHECKING
import torch

from selfconcept.assistant_axis.internals.model_specifics import get_model_specifics_by_name

if TYPE_CHECKING:
    from selfconcept.common.hf_strong_types import Conversation
    from .model import ProbingModel
    from .conversation import ConversationEncoder


class ActivationExtractor:
    """Extract per-layer activations for a batch of conversations using forward hooks."""

    def __init__(self, probing_model: ProbingModel, encoder: ConversationEncoder):
        """
        Initialize the activation extractor.

        Args:
            probing_model: ProbingModel instance with loaded model and tokenizer
            encoder: ConversationEncoder for formatting conversations
        """
        assert probing_model.model is not None and probing_model.tokenizer is not None
        self.model = probing_model.model
        self.tokenizer = probing_model.tokenizer
        self.probing_model = probing_model
        self.encoder = encoder

    def batch_conversations(
        self,
        conversations: list[Conversation],
        layer: int | list[int] | None = None,
        max_length: int = 4096,
        **chat_kwargs,
    ) -> tuple[torch.Tensor, dict]:
        """
        Extract activations for a batch of conversations.

        Args:
            conversations: List of conversations, each being a list of {"role", "content"} dicts
            layer: int for single layer, list of ints for multiple layers, or None for all layers
            max_length: Maximum sequence length for padding
            **chat_kwargs: Additional arguments for apply_chat_template

        Returns:
            Tuple of (batch_activations, batch_metadata):
            - batch_activations: torch.Tensor shape (num_layers, batch_size, max_seq_len, hidden_size)
            - batch_metadata: Dict with batching information (lengths, attention_mask, etc.)
        """
        # Get tokenized conversations and spans
        batch_full_ids, batch_spans, span_metadata = self.encoder.build_batch_turn_spans(
            conversations, **chat_kwargs
        )

        # Handle layer specification
        if isinstance(layer, int):
            layer_list = [layer]
        elif isinstance(layer, list):
            layer_list = layer
        else:
            model_specifics = get_model_specifics_by_name(self.probing_model.model_name)
            layer_list = list(range(len(model_specifics.get_layers(self.probing_model.model))))

        # Prepare batch tensors
        device = self.model.device

        # Find max length and pad sequences - ALWAYS respect max_length limit
        actual_max_len = max(len(ids) for ids in batch_full_ids)
        max_seq_len = min(max_length, actual_max_len)

        # Log warning if truncation will occur
        if actual_max_len > max_length:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Truncating sequences: max conversation length {actual_max_len} > max_length {max_length}")

        input_ids_batch = []
        attention_mask_batch = []

        for ids in batch_full_ids:
            # Truncate if too long
            if len(ids) > max_seq_len:
                ids = ids[:max_seq_len]

            # Pad to max length
            padded_ids = ids + [self.tokenizer.pad_token_id] * (max_seq_len - len(ids))
            attention_mask = [1] * len(ids) + [0] * (max_seq_len - len(ids))

            input_ids_batch.append(padded_ids)
            attention_mask_batch.append(attention_mask)

        # Convert to tensors
        input_ids_tensor = torch.tensor(input_ids_batch, dtype=torch.long, device=device)
        attention_mask_tensor = torch.tensor(attention_mask_batch, dtype=torch.long, device=device)

        # Extract activations using hooks (more reliable than output_hidden_states)
        layer_outputs = {}  # Will store {layer_idx: tensor} after forward pass
        handles = []

        def create_hook_fn(layer_idx):
            def hook_fn(module, input, output):
                # Extract the activation tensor (handle tuple output)
                act_tensor = output[0] if isinstance(output, tuple) else output
                layer_outputs[layer_idx] = act_tensor
            return hook_fn

        # Register hooks for target layers
        model_specifics = get_model_specifics_by_name(self.probing_model.model_name)
        model_layers = model_specifics.get_layers(self.probing_model.model)
        for layer_idx in layer_list:
            target_layer = model_layers[layer_idx]
            handle = target_layer.register_forward_hook(create_hook_fn(layer_idx))
            handles.append(handle)

        try:
            with torch.inference_mode():
                _ = self.model(
                    input_ids=input_ids_tensor,
                    attention_mask=attention_mask_tensor,
                )
        finally:
            # Clean up hooks
            for handle in handles:
                handle.remove()

        # Stack activations in layer order, moving to consistent device (first layer's device)
        target_device = layer_outputs[layer_list[0]].device
        selected_activations = torch.stack([
            layer_outputs[i].to(target_device) for i in layer_list
        ])  # (num_layers, batch_size, seq_len, hidden_size)

        # Ensure consistent bf16 dtype
        if selected_activations.dtype != torch.bfloat16:
            selected_activations = selected_activations.to(torch.bfloat16)

        batch_metadata = {
            'conversation_lengths': span_metadata['conversation_lengths'],
            'total_conversations': span_metadata['total_conversations'],
            'conversation_offsets': span_metadata['conversation_offsets'],
            'max_seq_len': max_seq_len,
            'attention_mask': attention_mask_tensor,
            'actual_lengths': [len(ids) for ids in batch_full_ids],
            'truncated_lengths': [min(len(ids), max_seq_len) for ids in batch_full_ids]
        }

        return selected_activations, batch_metadata
