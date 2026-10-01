from typing import Iterator, NamedTuple

from selfconcept.common.hf_strong_types import HFTokenizer


class HarmonyMessage(NamedTuple):
    role: str
    channel: str | None
    content_indices: list[int]


def _iter_harmony_messages(tokenizer: HFTokenizer, full_ids: list[int]) -> Iterator[HarmonyMessage]:
    """Messages are "<|start|>ROLE[<|channel|>CHANNEL ...]<|message|>CONTENT" ended by <|end|>, <|return|> or <|call|>."""
    start_id = tokenizer.convert_tokens_to_ids("<|start|>")
    message_id = tokenizer.convert_tokens_to_ids("<|message|>")
    message_end_ids = {tokenizer.convert_tokens_to_ids(token) for token in ("<|end|>", "<|return|>", "<|call|>")}

    position = 0
    while start_id in full_ids[position:]:
        header_start = full_ids.index(start_id, position) + 1
        content_start = full_ids.index(message_id, header_start) + 1
        content_end = next(
            (index for index in range(content_start, len(full_ids)) if full_ids[index] in message_end_ids),
            len(full_ids),
        )
        header = tokenizer.decode(full_ids[header_start:content_start - 1])
        role, _, channel_and_format = header.partition("<|channel|>")
        channel = channel_and_format.split()[0] if channel_and_format.strip() else None
        yield HarmonyMessage(role.strip(), channel, list(range(content_start, content_end)))
        position = content_end
