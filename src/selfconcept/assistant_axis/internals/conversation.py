"""ConversationEncoder - Handles chat formatting and token indexing."""

from __future__ import annotations

import itertools
from typing import Any, Protocol
import torch
import re


from selfconcept.common.hf_strong_types import (
    AllRoles,
    Conversation,
    _Conversation,
    HFTokenizer,
    configure_apply_chat_template,
    configure_call,
)

class ModelSpecifics[RoleT: AllRoles = AllRoles](Protocol): # TODO: include tokenizer?
    def get_response_indices(self, conversation: _Conversation[RoleT], tokenizer: HFTokenizer[RoleT], **apply_chat_template_kwargs: Any) -> list[list[int]]: ... # TODO: figure out what to do with apply_chat_template_kwargs

    def build_turn_spans(
        self,
        conversation: _Conversation[RoleT],
        tokenizer: HFTokenizer[RoleT],
        full_ids: list[int],
        **apply_chat_template_kwargs,
    ) -> tuple[list[int], list[dict[str, Any]]]: ...

    def content_only_ids_and_offset(
        self,
        messages_before: _Conversation[RoleT],
        tokenizer: HFTokenizer[RoleT],
        role: RoleT,
        content: str,
        **apply_chat_template_kwargs,
    ) -> tuple[list[int], int]: ...
    
# PORT_ASSUMPTION[model-specific]: Qwen-specific response-index / turn-span extraction.
# spans from assistant turns; a thinking model would need enable_thinking=True and its
# reasoning spans handled deliberately rather than dropped.
class QwenModelSpecifics(ModelSpecifics):
    def get_response_indices(self, conversation: Conversation, tokenizer: HFTokenizer, **apply_chat_template_kwargs: Any) -> list[list[int]]:
        """Qwen-specific implementation for extracting response token indices."""
        all_turn_indices = []

        # Check if thinking is enabled
        enable_thinking = apply_chat_template_kwargs.get('enable_thinking', False) # TODO: extract enable_thinking

        # Get the full formatted conversation
        full_formatted = configure_apply_chat_template(tokenizer).tokenize(False)(
            conversation, add_generation_prompt=False, **apply_chat_template_kwargs
        )
        full_tokens = tokenizer(full_formatted, add_special_tokens=False)
        all_token_ids = full_tokens['input_ids']

        # Get special token IDs for Qwen
        try:
            im_start_id = tokenizer.convert_tokens_to_ids('<|im_start|>')
            im_end_id = tokenizer.convert_tokens_to_ids('<|im_end|>')
            assistant_token_id = tokenizer.convert_tokens_to_ids('assistant')

            # Thinking tokens (may not exist in all Qwen variants)
            try:
                think_start_id = tokenizer.convert_tokens_to_ids('<think>')
                think_end_id = tokenizer.convert_tokens_to_ids('</think>')
            except (KeyError, ValueError):
                think_start_id = None
                think_end_id = None

        except (KeyError, ValueError):
            # Fallback if special tokens not found
            return _get_response_indices_simple(conversation, tokenizer, **apply_chat_template_kwargs)

        # Find assistant response sections
        i = 0
        while i < len(all_token_ids):
            # Look for <|im_start|>assistant pattern
            if (i + 1 < len(all_token_ids) and
                all_token_ids[i] == im_start_id and
                all_token_ids[i + 1] == assistant_token_id):

                # Found start of assistant response, skip the <|im_start|>assistant tokens
                response_start = i + 2

                # Find the corresponding <|im_end|>
                response_end = None
                for j in range(response_start, len(all_token_ids)):
                    if all_token_ids[j] == im_end_id:
                        response_end = j  # Don't include the <|im_end|> token
                        break

                if response_end is not None:
                    # Extract tokens in this range
                    raw_turn_indices = list(range(response_start, response_end))

                    # Filter out thinking tokens if thinking disabled
                    if not enable_thinking and think_start_id is not None and think_end_id is not None:
                        filtered_indices: list[int] = []
                        skip_until_think_end = False

                        for idx in raw_turn_indices:
                            token_id = all_token_ids[idx]

                            # Check if we hit a <think> token
                            if token_id == think_start_id:
                                skip_until_think_end = True
                                continue

                            # Check if we hit a </think> token
                            if token_id == think_end_id:
                                skip_until_think_end = False
                                continue

                            # Skip tokens that are inside thinking blocks
                            if skip_until_think_end:
                                continue

                            # Include all tokens that are not inside thinking blocks
                            filtered_indices.append(idx)

                        # Clean up extracted text by removing extra whitespace/newlines at boundaries
                        if filtered_indices:
                            # Get the text to check for leading/trailing cleanup
                            extracted_token_ids = [all_token_ids[i] for i in filtered_indices]
                            extracted_text = tokenizer.decode(extracted_token_ids)

                            # If text starts/ends with excessive whitespace, find better boundaries
                            if extracted_text.strip() != extracted_text:
                                # Remove leading whitespace-only tokens
                                while (filtered_indices and
                                       tokenizer.decode([all_token_ids[filtered_indices[0]]]).strip() == ''):
                                    filtered_indices.pop(0)

                                # Remove trailing whitespace-only tokens
                                while (filtered_indices and
                                       tokenizer.decode([all_token_ids[filtered_indices[-1]]]).strip() == ''):
                                    filtered_indices.pop()

                        turn_indices = filtered_indices
                    else:
                        turn_indices = raw_turn_indices

                    all_turn_indices.append(turn_indices)

                    i = response_end + 1
                else:
                    # No matching <|im_end|> found, skip this token
                    i += 1
            else:
                i += 1

        return all_turn_indices
    
    def build_turn_spans(
        self,
        conversation: Conversation,
        tokenizer: HFTokenizer,
        full_ids: list[int],
        **apply_chat_template_kwargs,
    ) -> tuple[list[int], list[dict[str, Any]]]:
        """
        Build turn spans for Qwen models using pattern-matching approach.

        This matches the behavior of persona-subspace's response_indices() method,
        which includes all tokens between <|im_start|>role and <|im_end|> markers
        (excluding the markers themselves but including boundary tokens like newlines).

        When enable_thinking=False, thinking tokens (<think>...</think>) are filtered out.
        """
        spans = []

        # Check if thinking is enabled
        enable_thinking = apply_chat_template_kwargs.get('enable_thinking', False)

        # Get special token IDs for Qwen
        try:
            im_start_id = tokenizer.convert_tokens_to_ids('<|im_start|>')
            im_end_id = tokenizer.convert_tokens_to_ids('<|im_end|>')
            user_token_id = tokenizer.convert_tokens_to_ids('user')
            assistant_token_id = tokenizer.convert_tokens_to_ids('assistant')

            # Thinking tokens (may not exist in all Qwen variants)
            try:
                think_start_id = tokenizer.convert_tokens_to_ids('<think>')
                think_end_id = tokenizer.convert_tokens_to_ids('</think>')
            except (KeyError, ValueError):
                think_start_id = None
                think_end_id = None

        except (KeyError, ValueError):
            # Fallback to standard approach if special tokens not found
            return _build_turn_spans_fallback(conversation, self, tokenizer, full_ids, **apply_chat_template_kwargs)

        # Build a list of (role, text) for non-system messages to match with found spans
        expected_turns = []
        for msg in conversation:
            if msg["role"] != "system":
                expected_turns.append((msg["role"], msg.get("content", "")))

        turn_idx = 0
        i = 0

        while i < len(full_ids):
            # Look for <|im_start|>user or <|im_start|>assistant pattern
            if i + 1 < len(full_ids) and full_ids[i] == im_start_id:
                role_token = full_ids[i + 1]

                if role_token == user_token_id:
                    role = "user"
                elif role_token == assistant_token_id:
                    role = "assistant"
                else:
                    i += 1
                    continue

                # Found start of a turn, skip the <|im_start|>role tokens
                content_start = i + 2

                # Find the corresponding <|im_end|>
                content_end = None
                for j in range(content_start, len(full_ids)):
                    if full_ids[j] == im_end_id:
                        content_end = j  # Don't include the <|im_end|> token
                        break

                if content_end is not None and turn_idx < len(expected_turns):
                    expected_role, expected_text = expected_turns[turn_idx]

                    # Verify role matches
                    if role == expected_role:
                        raw_indices = list(range(content_start, content_end))

                        # Filter out thinking tokens for assistant turns if thinking disabled
                        if role == "assistant" and not enable_thinking and think_start_id is not None and think_end_id is not None:
                            filtered_indices = []
                            skip_until_think_end = False

                            for idx in raw_indices:
                                token_id = full_ids[idx]

                                # Check if we hit a <think> token
                                if token_id == think_start_id:
                                    skip_until_think_end = True
                                    continue

                                # Check if we hit a </think> token
                                if token_id == think_end_id:
                                    skip_until_think_end = False
                                    continue

                                # Skip tokens that are inside thinking blocks
                                if skip_until_think_end:
                                    continue

                                # Include all tokens that are not inside thinking blocks
                                filtered_indices.append(idx)

                            # Clean up extracted text by removing extra whitespace/newlines at boundaries
                            if filtered_indices:
                                # Get the text to check for leading/trailing cleanup
                                extracted_token_ids = [full_ids[idx] for idx in filtered_indices]
                                extracted_text = tokenizer.decode(extracted_token_ids)

                                # If text starts/ends with excessive whitespace, find better boundaries
                                if extracted_text.strip() != extracted_text:
                                    # Remove leading whitespace-only tokens
                                    while (filtered_indices and
                                           tokenizer.decode([full_ids[filtered_indices[0]]]).strip() == ''):
                                        filtered_indices.pop(0)

                                    # Remove trailing whitespace-only tokens
                                    while (filtered_indices and
                                           tokenizer.decode([full_ids[filtered_indices[-1]]]).strip() == ''):
                                        filtered_indices.pop()

                            final_indices = filtered_indices
                        else:
                            final_indices = raw_indices

                        if final_indices:
                            spans.append({
                                "turn": turn_idx,
                                "role": role,
                                "start": min(final_indices),
                                "end": max(final_indices) + 1,  # exclusive
                                "n_tokens": len(final_indices),
                                "text": expected_text,
                            })
                        turn_idx += 1

                    i = content_end + 1
                else:
                    i += 1
            else:
                i += 1

        return full_ids, spans

    def content_only_ids_and_offset(
        self,
        messages_before: Conversation,
        tokenizer: HFTokenizer,
        role: AllRoles,
        content: str,
        **chat_kwargs,
    ) -> tuple[list[int], int]:
        """Qwen-specific version that handles thinking tokens properly."""
        # For Qwen assistant turns, thinking tokens interfere even when disabled
        if role == "assistant":
            # Find where content appears in the full tokenized conversation
            msgs_full = messages_before + [{"role": role, "content": content}]
            ids_full = configure_apply_chat_template(tokenizer).tokenize(True)(
                msgs_full, add_generation_prompt=False, **chat_kwargs
            )["input_ids"]

            # Find the content tokens in the full sequence
            plain = tokenizer(content, add_special_tokens=False).input_ids
            content_start = _find_subsequence(ids_full, plain)

            if content_start != -1:
                # Calculate offset from the beginning of the conversation
                if messages_before:
                    ids_before = configure_apply_chat_template(tokenizer).tokenize(True)(
                        messages_before, add_generation_prompt=False, **chat_kwargs
                    )["input_ids"]
                    prefix_len = len(ids_before)
                else:
                    prefix_len = 0

                start_in_delta = content_start - prefix_len
                return plain, max(0, start_in_delta)

        # Fall back to standard approach for user turns or if assistant approach fails
        return _content_only_ids_and_offset_standard(messages_before, tokenizer, role, content, **chat_kwargs)



# PORT_ASSUMPTION[model-specific]: Gemma/Llama offset-mapping response-index / turn-span extraction.
class GemmaModelSpecifics(ModelSpecifics):
    def get_response_indices(self, conversation: Conversation, tokenizer: HFTokenizer, **apply_chat_template_kwargs: Any) -> list[list[int]]:
        """Gemma/Llama-specific implementation using offset mapping approach."""
        all_turn_indices = []

        # Process conversation incrementally to find assistant response boundaries
        for i, turn in enumerate(conversation):
            if turn['role'] != 'assistant':
                continue

            # Get conversation up to but not including this assistant turn
            conversation_before = conversation[:i]

            # Get conversation up to and including this assistant turn
            conversation_including = conversation[:i+1]

            # Format and tokenize both versions
            if conversation_before:
                before_formatted = configure_apply_chat_template(tokenizer).tokenize(False)(
                    conversation_before, add_generation_prompt=True, **apply_chat_template_kwargs
                )
                before_tokens = tokenizer(before_formatted, add_special_tokens=False)
                before_length = len(before_tokens['input_ids'])
            else:
                before_length = 0

            including_formatted = configure_apply_chat_template(tokenizer).tokenize(False)(
                conversation_including, add_generation_prompt=False, **apply_chat_template_kwargs
            )
            including_tokens = tokenizer(including_formatted, add_special_tokens=False)
            including_length = len(including_tokens['input_ids'])

            # Find the actual content of this assistant response (excluding formatting tokens)
            assistant_content = turn['content'].strip()

            # Collect indices for this turn
            turn_indices = []

            # Find where the assistant content appears in the formatted text
            content_start_in_formatted = including_formatted.find(assistant_content)
            if content_start_in_formatted != -1:
                content_end_in_formatted = content_start_in_formatted + len(assistant_content)

                # Convert character positions to token indices using offset mapping
                tokens_with_offsets = configure_call(tokenizer).return_offsets_mapping(True)(including_formatted, add_special_tokens=False)
                offset_mapping = tokens_with_offsets['offset_mapping']

                # Find tokens that overlap with the assistant content
                for token_idx, (start_char, end_char) in enumerate(offset_mapping):
                    if (start_char >= content_start_in_formatted and start_char < content_end_in_formatted) or \
                       (end_char > content_start_in_formatted and end_char <= content_end_in_formatted) or \
                       (start_char < content_start_in_formatted and end_char > content_end_in_formatted):
                        turn_indices.append(token_idx)
            else:
                # Fallback to original method if content not found
                assistant_start = before_length
                assistant_end = including_length
                turn_indices.extend(range(assistant_start, assistant_end))

            # Store indices based on per_turn flag
            all_turn_indices.append(turn_indices)

        return all_turn_indices

    def build_turn_spans(
        self,
        conversation: _Conversation,
        tokenizer: HFTokenizer,
        full_ids: list[int],
        **apply_chat_template_kwargs,
    ) -> tuple[list[int], list[dict[str, Any]]]:
        spans = []
        msgs_before = []
        turn_idx = 0

        for msg in conversation:
            role = msg["role"]
            text = msg.get("content", "")

            if role == "system":
                msgs_before.append(msg)
                continue

            content_ids, start_in_delta = self.content_only_ids_and_offset(
                msgs_before, tokenizer, role, text, **apply_chat_template_kwargs
            )

            # Standard approach for non-Qwen models
            # Calculate absolute start based on the empty message template
            msgs_empty_for_this = msgs_before + [{"role": role, "content": ""}]
            ids_empty_full = configure_apply_chat_template(tokenizer).tokenize(True)(
                msgs_empty_for_this, add_generation_prompt=False, **apply_chat_template_kwargs
            )["input_ids"]

            # Find where the content appears in the full sequence
            ids_full_for_this = configure_apply_chat_template(tokenizer).tokenize(True)(
                msgs_before + [{"role": role, "content": text}], add_generation_prompt=False, **apply_chat_template_kwargs
            )["input_ids"]

            pref_len = _longest_common_prefix_len(ids_full_for_this, ids_empty_full)
            abs_start = pref_len + start_in_delta
            abs_end = abs_start + len(content_ids)

            spans.append({
                "turn": turn_idx,
                "role": role,
                "start": abs_start,
                "end": abs_end,   # exclusive
                "n_tokens": len(content_ids),
                "text": text,
            })
            msgs_before.append(msg)
            turn_idx += 1

        return full_ids, spans

    def content_only_ids_and_offset(
        self,
        messages_before: _Conversation,
        tokenizer: HFTokenizer,
        role: AllRoles,
        content: str,
        **apply_chat_template_kwargs,
    ) -> tuple[list[int], int]:
        return _content_only_ids_and_offset_standard(messages_before, tokenizer, role, content, **apply_chat_template_kwargs)

def _get_response_indices_simple(
    conversation: Conversation,
    tokenizer: HFTokenizer,
    **apply_chat_template_kwargs, # TODO
) -> list[list[int]]:
    """Simple fallback implementation using range-based approach."""
    all_turn_indices = []

    # Process conversation incrementally to find assistant response boundaries
    for i, turn in enumerate(conversation):
        if turn['role'] != 'assistant':
            continue

        # Get conversation up to but not including this assistant turn
        conversation_before = conversation[:i]

        # Get conversation up to and including this assistant turn
        conversation_including = conversation[:i+1]

        # Format and tokenize both versions
        if conversation_before:
            before_formatted = configure_apply_chat_template(tokenizer).tokenize(False)(
                conversation_before, add_generation_prompt=True, **apply_chat_template_kwargs
            )
            before_tokens = tokenizer(before_formatted, add_special_tokens=False)
            before_length = len(before_tokens['input_ids'])
        else:
            before_length = 0

        including_formatted = configure_apply_chat_template(tokenizer).tokenize(False)(
            conversation_including, add_generation_prompt=False, **apply_chat_template_kwargs
        )
        including_tokens = tokenizer(including_formatted, add_special_tokens=False)
        including_length = len(including_tokens['input_ids'])

        # The assistant response tokens are between before_length and including_length
        assistant_start = before_length
        assistant_end = including_length

        turn_indices = list(range(assistant_start, assistant_end))

        # Store indices based on per_turn flag
        all_turn_indices.append(turn_indices)

    return all_turn_indices

def _build_turn_spans_fallback(
    conversation: Conversation,
    model_specifics: ModelSpecifics,
    tokenizer: HFTokenizer,
    full_ids: list[int],
    **chat_kwargs,
) -> tuple[list[int], list[dict[str, Any]]]:
    """Fallback method for building turn spans when pattern matching fails."""
    spans = []
    msgs_before = []
    turn_idx = 0

    for msg in conversation:
        role = msg["role"]
        text = msg.get("content", "")

        if role == "system":
            msgs_before.append(msg)
            continue

        content_ids, start_in_delta = model_specifics.content_only_ids_and_offset(
            msgs_before, tokenizer, role, text, **chat_kwargs
        )

        # Find where the content appears in the full conversation
        abs_start = _find_subsequence(full_ids, content_ids)
        if abs_start == -1:
            msgs_before.append(msg)
            continue
        abs_end = abs_start + len(content_ids)

        spans.append({
            "turn": turn_idx,
            "role": role,
            "start": abs_start,
            "end": abs_end,
            "n_tokens": len(content_ids),
            "text": text,
        })
        msgs_before.append(msg)
        turn_idx += 1

    return full_ids, spans

def _content_only_ids_and_offset_standard(
    messages_before: Conversation,
    tokenizer: HFTokenizer,
    role: AllRoles,
    content: str,
    **chat_kwargs,
) -> tuple[list[int], int]:
    """Standard implementation for most models."""
    msgs_empty = messages_before + [{"role": role, "content": ""}]
    msgs_full  = messages_before + [{"role": role, "content": content}]

    ids_empty = configure_apply_chat_template(tokenizer).tokenize(True)(
        msgs_empty, add_generation_prompt=False, **chat_kwargs
    )["input_ids"]
    ids_full  = configure_apply_chat_template(tokenizer).tokenize(True)(
        msgs_full, add_generation_prompt=False, **chat_kwargs
    )["input_ids"]

    # Suffix introduced by adding this message (template + content)
    pref = _longest_common_prefix_len(ids_full, ids_empty)
    delta = ids_full[pref:]
    delta = _strip_trailing_special(delta, set(tokenizer.all_special_ids))

    # Try to locate the raw content (with/without a leading space) inside delta
    plain = tokenizer(content, add_special_tokens=False).input_ids
    sp    = tokenizer(" " + content, add_special_tokens=False).input_ids

    start = _find_subsequence(delta, plain)
    use = plain
    if start == -1:
        start = _find_subsequence(delta, sp)
        use = sp if start != -1 else plain

    if start == -1:
        # Fallback: keep the whole delta (may include a fused leading-space token)
        return delta, 0
    else:
            return delta[start:start+len(use)], start

def _longest_common_prefix_len(a: list[int], b: list[int]) -> int:
    """Find the length of the longest common prefix between two sequences."""
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i

def _strip_trailing_special(ids: list[int], special_ids: set) -> list[int]:
    """Strip trailing special tokens from a sequence."""
    i = len(ids)
    while i > 0 and ids[i-1] in special_ids:
        i -= 1
    return ids[:i]


def _find_subsequence(hay: list[int], needle: list[int]) -> int:
    """Find the starting index of needle in hay, or -1 if not found."""
    if not needle or len(needle) > len(hay):
        return -1
    for i in range(len(hay) - len(needle) + 1):
        if hay[i:i+len(needle)] == needle:
            return i
    return -1

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

    def _is_qwen(self) -> bool:
        """Check if this is a Qwen model."""
        return 'qwen' in self.model_name

    def _is_llama(self) -> bool:
        """Check if this is a Llama model."""
        return 'llama' in self.model_name or 'meta-llama' in self.model_name

    def _is_gemma(self) -> bool:
        """Check if this is a Gemma model."""
        return 'gemma' in self.model_name

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
        # PORT_ASSUMPTION[model-specific]: dispatch by family — Qwen vs Llama/Gemma; unknown
        # families fall back to the simple offset method.
        # Dispatch to model-specific implementation
        if self._is_qwen():
            return self._flatten_conditional(
                not per_turn,
                QwenModelSpecifics().get_response_indices(conversation, self.tokenizer, **chat_kwargs)
            )
        elif self._is_llama() or self._is_gemma():
            return self._flatten_conditional(
                not per_turn,
                GemmaModelSpecifics().get_response_indices(conversation, self.tokenizer, **chat_kwargs)
            )
        else:
            # Fallback to simple method
            return self._flatten_conditional(
                not per_turn,
                _get_response_indices_simple(conversation, self.tokenizer, **chat_kwargs)
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

        # PORT_ASSUMPTION[model-specific]: Qwen uses a pattern-matching turn-span path; every
        # other family (Llama/Gemma/unknown) uses the Gemma offset-mapping path.
        # For Qwen models, use pattern-matching approach (matches persona-subspace behavior)
        if self._is_qwen():
            return QwenModelSpecifics().build_turn_spans(conversation, self.tokenizer, full_ids, **chat_kwargs)
        
        return GemmaModelSpecifics().build_turn_spans(conversation, self.tokenizer, full_ids, **chat_kwargs)

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
