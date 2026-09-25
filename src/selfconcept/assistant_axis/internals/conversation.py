"""ConversationEncoder - Handles chat formatting and token indexing."""

from __future__ import annotations

import itertools
from typing import Any
import torch
import re


from selfconcept.assistant_axis.internals.model_specifics import get_model_specifics_by_name
from selfconcept.common.hf_strong_types import (
    Conversation,
    HFTokenizer,
    configure_apply_chat_template,
)

def flatten[T](lst: list[list[T]]) -> list[T]:
    return list(itertools.chain(*lst))


class ConversationEncoder:
    """
    Handles conversation formatting, tokenization, and response index extraction.

    This class knows about model-specific quirks (Qwen vs LLaMA vs Gemma) and
    provides a unified interface for working with chat templates.
    """

    def __init__(self, tokenizer: HFTokenizer, model_name: str | None = None):
        """
        Initialize the conversation encoder.

        Args:
            tokenizer: HuggingFace tokenizer with chat template support
            model_name: Optional model name for detecting model-specific behavior
        """
        self.tokenizer = tokenizer
        self.model_name = (model_name or getattr(tokenizer, "name_or_path", "")).lower()

    def format_chat(
        self,
        conversation: str | Conversation,
        swap: bool = False,
        **chat_kwargs,
    ) -> str:
        """
        Format a conversation using the chat template.

        Args:
            conversation: Either a string prompt or list of {"role", "content"} dicts
            swap: If True, use swapped role formatting. Note, this is dead and only works with gemma anyway. Use with caution.
            **chat_kwargs: Additional arguments for apply_chat_template

        Returns:
            Formatted string ready for tokenization
        """
        if isinstance(conversation, str):
            # Single prompt - convert to conversation format
            conversation = [{"role": "user", "content": conversation}]

        if swap:
            # Swapped format for special use cases
            messages: Conversation = [{"role": "user", "content": "Hello."}, {"role": "assistant", "content": conversation[0]["content"]}]
            formatted_prompt = configure_apply_chat_template(self.tokenizer).tokenize(False)(
                messages, add_generation_prompt=True, **chat_kwargs
            )
            parts = formatted_prompt.rsplit('model', 1) # gemma internally uses "model" as the label for the assistant role
            if len(parts) == 2:
                formatted_prompt = 'user'.join(parts)
            return formatted_prompt
        else:
            return configure_apply_chat_template(self.tokenizer).tokenize(False)(
                conversation, add_generation_prompt=True, **chat_kwargs
            )

    def token_ids(
        self,
        conversation: Conversation,
        add_generation_prompt: bool = False,
        **chat_kwargs,
    ) -> list[int]:
        """
        Tokenize a conversation into token IDs.

        Args:
            conversation: List of {"role", "content"} dicts
            add_generation_prompt: Whether to add generation prompt at end
            **chat_kwargs: Additional arguments for apply_chat_template

        Returns:
            List of token IDs
        """
        return configure_apply_chat_template(self.tokenizer).tokenize(True)(
            conversation,
            add_generation_prompt=add_generation_prompt,
            **chat_kwargs,
        )["input_ids"]

    def response_indices(
        self,
        conversation: Conversation,
        per_turn: bool = False,
        **chat_kwargs,
    ) -> list[int] | list[list[int]]:
        """
        Get token indices for assistant responses in a conversation.

        Args:
            conversation: List of {"role", "content"} dicts
            per_turn: If True, return list of lists (one per assistant turn)
                     If False, return single flat list
            **chat_kwargs: Additional arguments for apply_chat_template

        Returns:
            Token indices for assistant responses
        """
        model_specifics = get_model_specifics_by_name(self.model_name)
        response_indices = model_specifics.get_response_indices(conversation, self.tokenizer, **chat_kwargs)
        return self._flatten_conditional(
            not per_turn,
            response_indices,
        )
        
    def _flatten_conditional[T](self, should_flatten: bool, list_of_lists: list[list[T]]) -> list[T] | list[list[T]]:
        return flatten(list_of_lists) if should_flatten else list_of_lists

    def build_turn_spans(
        self,
        conversation: Conversation,
        **chat_kwargs,
    ) -> tuple[list[int], list[dict[str, Any]]]:
        """
        Build token spans for each turn in a conversation.

        Args:
            conversation: List of {"role", "content"} dicts
            **chat_kwargs: Additional arguments for apply_chat_template

        Returns:
            Tuple of (full_ids, spans) where:
            - full_ids: tokenized ids of the whole conversation
            - spans: list of dicts with absolute [start, end) token spans for content per turn
        """
        # Tokenize the full conversation first
        full_ids = configure_apply_chat_template(self.tokenizer).tokenize(True)(
            conversation, add_generation_prompt=False, **chat_kwargs
        )["input_ids"]

        model_specifics = get_model_specifics_by_name(self.model_name)
        return model_specifics.build_turn_spans(conversation, self.tokenizer, full_ids, **chat_kwargs)

    def build_batch_turn_spans(
        self,
        conversations: list[Conversation],
        **chat_kwargs,
    ) -> tuple[list[list[int]], list[dict[str, Any]], dict[str, Any]]:
        """
        Process multiple conversations and build spans for batched processing.

        Args:
            conversations: List of conversations, each being a list of {"role", "content"} dicts
            **chat_kwargs: Additional arguments for apply_chat_template

        Returns:
            Tuple of (batch_full_ids, batch_spans, batch_metadata):
            - batch_full_ids: List of tokenized ids for each conversation
            - batch_spans: List of span dicts with conversation_id, local and global indices
            - batch_metadata: Dict with batching information (lengths, padding info, etc.)
        """
        batch_full_ids = []
        batch_spans = []
        batch_metadata = {
            'conversation_lengths': [],
            'total_conversations': len(conversations),
            'conversation_offsets': []  # Global token offsets for each conversation in batch
        }

        global_offset = 0

        for conv_id, conversation in enumerate(conversations):
            # Get spans for this conversation using existing function
            full_ids, spans = self.build_turn_spans(conversation, **chat_kwargs)

            batch_full_ids.append(full_ids)
            batch_metadata['conversation_lengths'].append(len(full_ids))
            batch_metadata['conversation_offsets'].append(global_offset)

            # Add conversation ID and global indices to each span
            for span in spans:
                enhanced_span = span.copy()
                enhanced_span['conversation_id'] = conv_id
                enhanced_span['local_start'] = span['start']
                enhanced_span['local_end'] = span['end']
                enhanced_span['global_start'] = global_offset + span['start']
                enhanced_span['global_end'] = global_offset + span['end']
                batch_spans.append(enhanced_span)

            global_offset += len(full_ids)

        return batch_full_ids, batch_spans, batch_metadata

    def code_block_token_mask(self, text: str) -> torch.Tensor:
        """
        Identify which tokens in a text span are within code blocks (single or triple backticks).

        Args:
            text: The text string to analyze

        Returns:
            Boolean mask tensor of shape (n_tokens,) where True indicates tokens to exclude
        """
        # Tokenize the text to get tokens and their character offsets
        tokenized = self.tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        token_ids = tokenized['input_ids']
        offset_mapping = tokenized['offset_mapping']

        n_tokens = len(token_ids)
        exclude_mask = torch.zeros(n_tokens, dtype=torch.bool)

        if n_tokens == 0:
            return exclude_mask

        # Find all code block regions (both single and triple backticks)
        code_regions = []

        # First, find triple backtick regions (these take precedence)
        triple_pattern = r'```[\s\S]*?```'
        for match in re.finditer(triple_pattern, text):
            code_regions.append((match.start(), match.end()))

        # Then find single backtick regions, but only if they're not within triple backtick regions
        single_pattern = r'`[^`\n]*?`'
        for match in re.finditer(single_pattern, text):
            start, end = match.start(), match.end()
            # Check if this single backtick region overlaps with any triple backtick region
            overlaps = any(triple_start <= start < triple_end or triple_start < end <= triple_end
                          for triple_start, triple_end in code_regions)
            if not overlaps:
                code_regions.append((start, end))

        # Map character regions to token indices
        for char_start, char_end in code_regions:
            for i, (token_start, token_end) in enumerate(offset_mapping):
                # Check if token overlaps with code region
                if (token_start < char_end and token_end > char_start):
                    exclude_mask[i] = True

        return exclude_mask
