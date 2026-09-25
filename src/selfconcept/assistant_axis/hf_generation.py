# Derived from safety-research/assistant-axis (https://github.com/safety-research/assistant-axis),
# MIT licensed. See the NOTICE file at the repository root for the full license text.
"""Single-response HuggingFace generation for steered traces.

The steering hook in ``steering.py`` needs a forward pass it can intercept, so steered
generation runs through HuggingFace ``model.generate`` here rather than the vLLM batch
path in ``generation.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import torch

from selfconcept.assistant_axis.internals.conversation import ConversationEncoder
from selfconcept.assistant_axis.internals.model import ProbingModel
from selfconcept.assistant_axis.internals.model_specifics import get_model_specifics_by_name
from selfconcept.common.hf_strong_types import Conversation, HFTokenizer, configure_call


@dataclass(frozen=True)
class Answer:
    """A generated assistant answer."""

    text: str


@dataclass(frozen=True)
class UnclosedReasoning:
    """Generation ran to the token budget still reasoning, so there is no answer to extract."""


type GenerationResult = Answer | UnclosedReasoning

# On-disk marker for the degenerate case in trace files, consumed only by the serialize/parse
# pair below. Distinct from any plausible answer so parse_response can round-trip it.
_UNCLOSED_MARKER = "[DEGENERATE_UNCLOSED_COT]"


def serialize_response(result: GenerationResult) -> str:
    """Encode a result into the ``response`` field of a trace record."""
    match result:
        case Answer(text=text):
            return text
        case UnclosedReasoning():
            return _UNCLOSED_MARKER


def parse_response(serialized: str) -> GenerationResult:
    """Decode a trace record's ``response`` field back into a result."""
    if serialized == _UNCLOSED_MARKER:
        return UnclosedReasoning()
    return Answer(text=serialized)


def _answer_after_thinking(tokenizer: HFTokenizer, gen_ids: list[int], close_str: str) -> GenerationResult:
    """The answer following ``close_str``; ``UnclosedReasoning`` when thinking never closed."""
    raw = tokenizer.decode(gen_ids, skip_special_tokens=False)
    idx = raw.find(close_str)
    if idx == -1:
        return UnclosedReasoning()
    n_prefix = len(configure_call(tokenizer)(raw[: idx + len(close_str)], add_special_tokens=False)["input_ids"])
    return Answer(text=tokenizer.decode(gen_ids[n_prefix:], skip_special_tokens=True).strip())


def generate_response(
    probing_model: ProbingModel,
    conversation: Conversation,
    *,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
    enable_thinking: bool = False,
) -> GenerationResult:
    """Generate and decode a single assistant response for ``conversation``.

    With ``enable_thinking`` the model reasons first and the returned answer is only the text
    after ``</think>`` -- or ``UnclosedReasoning`` if it never closed its reasoning; otherwise
    thinking is force-closed and the full decode returned as an ``Answer``.
    """
    assert probing_model.model is not None and probing_model.tokenizer is not None
    tokenizer = probing_model.tokenizer
    encoder = ConversationEncoder(tokenizer, probing_model.model_name)
    model_specifics = get_model_specifics_by_name(probing_model.model_name)
    chat_kwargs = model_specifics.set_enable_thinking({}, enable_thinking=enable_thinking)

    token_ids = encoder.token_ids(conversation, add_generation_prompt=True, **chat_kwargs)
    if not enable_thinking:
        token_ids = token_ids + model_specifics.thinking_close_ids(tokenizer)
    token_ids_tensor = torch.tensor([token_ids], device=probing_model.device)
    attention_mask = torch.ones_like(token_ids_tensor)

    with torch.inference_mode():
        output_ids = cast(Any, probing_model.model).generate(
            input_ids=token_ids_tensor,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
        )

    gen_ids = output_ids[0][token_ids_tensor.shape[1] :].tolist()
    if enable_thinking:
        close_str = tokenizer.decode(model_specifics.thinking_close_ids(tokenizer)).strip()
        return _answer_after_thinking(tokenizer, gen_ids, close_str)
    return Answer(text=tokenizer.decode(gen_ids, skip_special_tokens=True))
