from typing import Any

from selfconcept.common.hf_strong_types import Conversation, HFTokenizer, configure_apply_chat_template
from selfconcept.common.span_targeting.types import PromptCompletion


def tokenize_as_prompt_completion(
    conversation: Conversation,
    tokenizer: HFTokenizer,
    **chat_template_kwargs: Any,
) -> PromptCompletion:
    """Splits the conversation's token ids at the generation prompt of its final turn."""
    apply_chat_template = configure_apply_chat_template(tokenizer).tokenize(True)
    prompt_token_ids = apply_chat_template(
        conversation[:-1], add_generation_prompt=True, **chat_template_kwargs
    )["input_ids"]
    full_token_ids = apply_chat_template(
        conversation, add_generation_prompt=False, **chat_template_kwargs
    )["input_ids"]
    if full_token_ids[:len(prompt_token_ids)] != prompt_token_ids:
        raise ValueError("the chat template does not render the prompt as a prefix of the full conversation")
    return PromptCompletion(prompt_token_ids, full_token_ids[len(prompt_token_ids):])
