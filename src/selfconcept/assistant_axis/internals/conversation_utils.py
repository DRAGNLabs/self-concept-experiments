
import itertools
from typing import Any, Protocol

from selfconcept.common.hf_strong_types import AllRoles, Conversation, _Conversation, HFTokenizer, configure_apply_chat_template

class ContentOnlyIdsAndOffsetFn[RoleT: AllRoles = AllRoles](Protocol):
    def __call__(
        self,
        messages_before: _Conversation[RoleT],
        tokenizer: HFTokenizer[RoleT],
        role: RoleT,
        content: str,
        **apply_chat_template_kwargs,
    ) -> tuple[list[int], int]: ...
    

def get_response_indices_simple(
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



def build_turn_spans_fallback(
    conversation: Conversation,
    content_only_ids_and_offset: ContentOnlyIdsAndOffsetFn,
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

        content_ids, start_in_delta = content_only_ids_and_offset(
            msgs_before, tokenizer, role, text, **chat_kwargs
        )

        # Find where the content appears in the full conversation
        abs_start = find_subsequence(full_ids, content_ids)
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


def find_subsequence(hay: list[int], needle: list[int]) -> int:
    """Find the starting index of needle in hay, or -1 if not found."""
    if not needle or len(needle) > len(hay):
        return -1
    for i in range(len(hay) - len(needle) + 1):
        if hay[i:i+len(needle)] == needle:
            return i
    return -1

def flatten[T](lst: list[list[T]]) -> list[T]:
    return list(itertools.chain(*lst))

def content_only_ids_and_offset_standard(
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
    pref = longest_common_prefix_len(ids_full, ids_empty)
    delta = ids_full[pref:]
    delta = _strip_trailing_special(delta, set(tokenizer.all_special_ids))

    # Try to locate the raw content (with/without a leading space) inside delta
    plain = tokenizer(content, add_special_tokens=False).input_ids
    sp    = tokenizer(" " + content, add_special_tokens=False).input_ids

    start = find_subsequence(delta, plain)
    use = plain
    if start == -1:
        start = find_subsequence(delta, sp)
        use = sp if start != -1 else plain

    if start == -1:
        # Fallback: keep the whole delta (may include a fused leading-space token)
        return delta, 0
    else:
            return delta[start:start+len(use)], start


def longest_common_prefix_len(a: list[int], b: list[int]) -> int:
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

