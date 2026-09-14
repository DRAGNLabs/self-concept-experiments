from typing import Any, Iterator

from selfconcept.assistant_axis.internals.conversation_utils import content_only_ids_and_offset_standard, find_subsequence
from selfconcept.common.hf_strong_types import AllRoles, Conversation, HFTokenizer, configure_apply_chat_template


def get_response_indices_chatml(
    conversation: Conversation,
    tokenizer: HFTokenizer,
    **apply_chat_template_kwargs: Any
) -> list[list[int]]:
    all_turn_indices = []

    # Check if thinking is enabled
    enable_thinking = apply_chat_template_kwargs.get('enable_thinking', False) # TODO: extract enable_thinking

    # Get the full formatted conversation
    full_formatted = configure_apply_chat_template(tokenizer).tokenize(False)(
        conversation, add_generation_prompt=False, **apply_chat_template_kwargs
    )
    full_tokens = tokenizer(full_formatted, add_special_tokens=False)
    all_token_ids = full_tokens['input_ids']

    for _, role, response_start, response_end in _iter_over_turns(tokenizer, all_token_ids):
        if role != 'assistant':
            continue

        raw_turn_indices = list(range(response_start, response_end))
        turn_indices = _get_turn_indices(raw_turn_indices, all_token_ids, role, tokenizer, enable_thinking)
        all_turn_indices.append(turn_indices)


    return all_turn_indices

def _get_chatml_think_tokens(tokenizer: HFTokenizer, required: bool) -> tuple[int, int] | None:
    try:
        think_start_id = tokenizer.convert_tokens_to_ids('<think>')
        think_end_id = tokenizer.convert_tokens_to_ids('</think>')
    except (KeyError, ValueError):
        if required:
            raise ValueError("Could not find <think> token ids for tokenizer")

        return None

    return think_start_id, think_end_id

def build_turn_spans_chatml(
    conversation: Conversation,
    tokenizer: HFTokenizer,
    full_ids: list[int],
    **apply_chat_template_kwargs,
) -> tuple[list[int], list[dict[str, Any]]]:
    """
    Build turn spans for ChatML models using pattern-matching approach.

    This matches the behavior of persona-subspace's response_indices() method,
    which includes all tokens between <|im_start|>role and <|im_end|> markers
    (excluding the markers themselves but including boundary tokens like newlines).

    When enable_thinking=False, thinking tokens (<think>...</think>) are filtered out.
    """
    spans = []

    enable_thinking = apply_chat_template_kwargs.get('enable_thinking', False)

    # Build a list of (role, text) for non-system messages to match with found spans
    expected_turns = []
    for msg in conversation:
        if msg["role"] != "system":
            expected_turns.append((msg["role"], msg.get("content", "")))

    for turn_idx, role, content_start, content_end in _iter_over_turns(tokenizer, full_ids):
        if content_end is not None and turn_idx < len(expected_turns):
            expected_role, expected_text = expected_turns[turn_idx]

            # Verify role matches
            if role == expected_role:
                raw_indices = list(range(content_start, content_end))
                final_indices = _get_turn_indices(raw_indices, full_ids, role, tokenizer, enable_thinking)
                
                if final_indices:
                    spans.append({
                        "turn": turn_idx,
                        "role": role,
                        "start": min(final_indices),
                        "end": max(final_indices) + 1,  # exclusive
                        "n_tokens": len(final_indices),
                        "text": expected_text,
                    })

    return full_ids, spans

    
def _iter_over_turns(
    tokenizer: HFTokenizer,
    full_ids: list[int],
) -> Iterator[tuple[int, AllRoles, int, int]]:
    im_start_id = tokenizer.convert_tokens_to_ids('<|im_start|>')
    im_end_id = tokenizer.convert_tokens_to_ids('<|im_end|>')
    user_token_id = tokenizer.convert_tokens_to_ids('user')
    assistant_token_id = tokenizer.convert_tokens_to_ids('assistant')

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

            if content_end is None:
                raise ValueError(f"Could not find content end for turn {turn_idx}")

            yield turn_idx, role, content_start, content_end

            turn_idx += 1
            i = content_end + 1
        else:
            i += 1


def _get_turn_indices(raw_indices: list[int], full_ids: list[int], role: AllRoles, tokenizer: HFTokenizer, enable_thinking: bool) -> list[int]:
    thinking_token_ids = _get_chatml_think_tokens(tokenizer, required=not enable_thinking)

    if (
        role != "assistant"
        or enable_thinking
        or thinking_token_ids is None
    ):
        return raw_indices

    think_start_id, think_end_id = thinking_token_ids
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

    return filtered_indices

def content_only_ids_and_offset_string_search(
    messages_before: Conversation,
    tokenizer: HFTokenizer,
    role: AllRoles,
    content: str,
    **chat_kwargs,
) -> tuple[list[int], int]:
    """Handles thinking tokens by searching for the plain message"""
    if role == "assistant":
        # Find where content appears in the full tokenized conversation
        msgs_full = messages_before + [{"role": role, "content": content}]
        ids_full = configure_apply_chat_template(tokenizer).tokenize(True)(
            msgs_full, add_generation_prompt=False, **chat_kwargs
        )["input_ids"]

        # Find the content tokens in the full sequence
        plain = tokenizer(content, add_special_tokens=False).input_ids
        content_start = find_subsequence(ids_full, plain)

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
    return content_only_ids_and_offset_standard(messages_before, tokenizer, role, content, **chat_kwargs)

