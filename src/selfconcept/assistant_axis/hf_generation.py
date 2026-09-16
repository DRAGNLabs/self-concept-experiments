# Derived from safety-research/assistant-axis (https://github.com/safety-research/assistant-axis),
# MIT licensed. See the NOTICE file at the repository root for the full license text.
"""Single-response HuggingFace generation for steered traces.

The steering hook in ``steering.py`` needs a forward pass it can intercept, so steered
generation runs through HuggingFace ``model.generate`` here rather than the vLLM batch
path in ``generation.py``.
"""

from __future__ import annotations

from typing import Any, cast

import torch

from selfconcept.assistant_axis.internals.conversation import ConversationEncoder
from selfconcept.assistant_axis.internals.model import ProbingModel
from selfconcept.assistant_axis.internals.model_specifics import get_model_specifics_by_name
from selfconcept.common.hf_strong_types import Conversation


def generate_response(
    probing_model: ProbingModel,
    conversation: Conversation,
    *,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
) -> str:
    """Generate and decode a single assistant response for ``conversation``."""
    assert probing_model.model is not None and probing_model.tokenizer is not None
    encoder = ConversationEncoder(probing_model.tokenizer, probing_model.model_name)
    model_specifics = get_model_specifics_by_name(probing_model.model_name)
    chat_kwargs = model_specifics.set_enable_thinking({}, enable_thinking=False)

    token_ids = encoder.token_ids(conversation, add_generation_prompt=True, **chat_kwargs)
    token_ids = token_ids + model_specifics.thinking_close_ids(probing_model.tokenizer)
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
            pad_token_id=probing_model.tokenizer.pad_token_id,
        )

    return probing_model.tokenizer.decode(
        output_ids[0][token_ids_tensor.shape[1] :], skip_special_tokens=True
    )
