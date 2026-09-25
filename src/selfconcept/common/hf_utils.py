"""HuggingFace helpers shared across experiments."""

from __future__ import annotations

from selfconcept.common.hf_strong_types import (
    Conversation,
    HFTokenizer,
    configure_apply_chat_template,
)

_SYSTEM_SUPPORT_SENTINEL = "__SYSTEM_TEST__"


def supports_system_role(tokenizer: HFTokenizer) -> bool:
    """Whether the tokenizer's chat template keeps a ``system`` turn's content."""
    probe: Conversation = [
        {"role": "system", "content": _SYSTEM_SUPPORT_SENTINEL},
        {"role": "user", "content": "hello"},
    ]
    try:
        rendered = configure_apply_chat_template(tokenizer).tokenize(False)(
            probe, add_generation_prompt=False
        )
    except Exception:
        return False
    return _SYSTEM_SUPPORT_SENTINEL in rendered


def build_conversation(
    question: str,
    system_prompt: str | None,
    tokenizer: HFTokenizer,
) -> Conversation:
    """A user turn asking ``question``, with ``system_prompt`` as a system turn if supported."""
    if system_prompt is None:
        return [{"role": "user", "content": question}]
    if supports_system_role(tokenizer):
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]
    return [{"role": "user", "content": f"{system_prompt}\n\n{question}"}]
