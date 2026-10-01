"""Text-level parsing of Harmony-format (gpt-oss) completions."""

import re

# One message of an assistant completion: "<|channel|>NAME[ to=RECIPIENT][ <|constrain|>FORMAT]<|message|>BODY",
# ended by <|end|> (the next message then opens with "<|start|>assistant"), <|return|>, <|call|>, truncation, or
# the next message opening without an <|end|>.
HARMONY_MESSAGE = re.compile(
    r"<\|channel\|>(\w+)[^<]*(?:<\|constrain\|>[^<]*)?<\|message\|>(.*?)"
    r"(?:<\|end\|>|<\|return\|>|<\|call\|>|$|(?=<\|start\|>|<\|channel\|>))",
    re.DOTALL,
)
HARMONY_SPECIAL_TOKENS = (
    "<|start|>", "<|end|>", "<|return|>", "<|call|>", "<|channel|>", "<|message|>", "<|constrain|>"
)


def strip_harmony_tokens(text: str) -> str:
    for token in HARMONY_SPECIAL_TOKENS:
        text = text.replace(token, "")
    return text


def split_harmony_completion(raw_completion: str) -> tuple[str, str]:
    """-> (reasoning, final): the bodies of the messages before the last final-channel message, and
    that message's body. final is "" when the model never opened the final channel."""
    messages: list[tuple[str, str]] = HARMONY_MESSAGE.findall(raw_completion)
    if not messages:
        return strip_harmony_tokens(raw_completion), ""
    final_indices = [index for index, (channel, _) in enumerate(messages) if channel == "final"]
    if not final_indices:
        return "\n\n".join(body for _, body in messages), ""
    last_final_index = final_indices[-1]
    return "\n\n".join(body for _, body in messages[:last_final_index]), messages[last_final_index][1].strip()


def join_harmony_completion(reasoning: str, final: str) -> str:
    """Inverse of split_harmony_completion for a completion with one analysis and one final message."""
    return (f"<|channel|>analysis<|message|>{reasoning}<|end|>"
            f"<|start|>assistant<|channel|>final<|message|>{final}<|return|>")
