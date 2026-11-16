THINK_OPEN = "[THINK]"
THINK_CLOSE = "[/THINK]"
# skip_special_tokens=False leaves the end-of-sequence piece in the completion text.
END_OF_TURN_MARKERS = ("</s>",)


def split_mistral_think_completion(raw_completion: str) -> tuple[str, str]:
    """-> (reasoning, answer) of a Mistral-Small-4 completion with reasoning_effort high.

    The generation prompt ends at ``[/INST]`` and does not open ``[THINK]``; the model emits
    ``[THINK]...[/THINK]`` and then the answer. No think tags means the model skipped the channel
    and the whole completion is the answer. An opened channel that never closes was cut off
    mid-thought, so the answer is empty."""
    text = raw_completion
    for marker in END_OF_TURN_MARKERS:
        text = text.replace(marker, "")
    close = text.find(THINK_CLOSE)
    if close != -1:
        reasoning = text[:close]
        opened = reasoning.find(THINK_OPEN)
        if opened != -1:
            reasoning = reasoning[opened + len(THINK_OPEN):]
        return reasoning.strip(), text[close + len(THINK_CLOSE):].strip()
    opened = text.find(THINK_OPEN)
    if opened != -1:
        return text[opened + len(THINK_OPEN):].strip(), ""
    return "", text.strip()
