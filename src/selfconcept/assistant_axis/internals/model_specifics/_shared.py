from typing import Any, Iterator

from selfconcept.assistant_axis.internals.conversation_utils import content_only_ids_and_offset_standard, find_subsequence
from selfconcept.common.hf_strong_types import AllRoles, Conversation, HFTokenizer, configure_apply_chat_template, configure_call


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

def _think_char_spans(text: str) -> list[tuple[int, int]]:
    """Char ranges of <think>...</think> blocks to drop from `text`.

    Works at the character level because models like OLMo 3 encode the tags as
    multi-token text whose closing '>' fuses with the following character under
    BPE, so the tags have no stable token-id sequence. Handles a missing opening
    <think> (block runs from the start) and a missing closing </think> (block
    runs to the end), which is how reasoning turns are emitted and stored.
    """
    open_tag, close_tag = "<think>", "</think>"
    spans = []
    pos = 0
    while True:
        start = text.find(open_tag, pos)
        end = text.find(close_tag, pos)
        if start == -1 and end == -1:
            break
        # Block starts at <think> when it precedes the close, else at the turn/scan start
        block_start = start if start != -1 and (end == -1 or start < end) else pos
        if end == -1:
            spans.append((block_start, len(text)))
            break
        block_end = end + len(close_tag)
        spans.append((block_start, block_end))
        pos = block_end
    return spans

def _overlaps_any(start: int, end: int, spans: list[tuple[int, int]]) -> bool:
    return any(start < span_end and end > span_start for span_start, span_end in spans)

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
    # Some ChatML variants (e.g. OLMo 3) close the final assistant turn with eos
    # instead of <|im_end|>, so treat both as turn terminators.
    turn_end_ids: set[int | None] = {im_end_id, tokenizer.eos_token_id} # type: ignore
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

            # Find the corresponding turn terminator (<|im_end|> or eos)
            content_end = None
            for j in range(content_start, len(full_ids)):
                if full_ids[j] in turn_end_ids:
                    content_end = j  # Don't include the terminator token
                    break

            if content_end is None:
                raise ValueError(f"Could not find content end for turn {turn_idx}")

            yield turn_idx, role, content_start, content_end

            turn_idx += 1
            i = content_end + 1
        else:
            i += 1


def _get_turn_indices(raw_indices: list[int], full_ids: list[int], role: AllRoles, tokenizer: HFTokenizer, enable_thinking: bool) -> list[int]:
    if role != "assistant" or enable_thinking:
        return raw_indices

    turn_ids = [full_ids[i] for i in raw_indices]
    turn_text = tokenizer.decode(turn_ids)
    drop_spans = _think_char_spans(turn_text)

    if not drop_spans:
        filtered_indices = list(raw_indices)
    else:
        # Re-encoding recovers per-token char offsets, which only align if it
        # reproduces the original tokens.
        enc = configure_call(tokenizer).return_offsets_mapping(True)(turn_text, add_special_tokens=False)
        if enc['input_ids'] != turn_ids:
            raise ValueError(
                "Re-encoding the decoded assistant turn did not reproduce its tokens, "
                "so think-block offsets cannot be aligned"
            )
        filtered_indices = [
            raw_indices[k]
            for k, (char_start, char_end) in enumerate(enc['offset_mapping'])
            if not _overlaps_any(char_start, char_end, drop_spans)
        ]

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

