from typing import Any

from selfconcept.common.hf_strong_types import Conversation, HFTokenizer, configure_apply_chat_template, configure_call
from selfconcept.common.span_targeting.types import PromptCompletion


def tokenize_as_prompt_completion(
    conversation: Conversation,
    tokenizer: HFTokenizer,
    **chat_template_kwargs: Any,
) -> PromptCompletion:
    """Splits the conversation at the generation prompt of its final turn. The completion is tokenized
    on its own, as generation would produce it, since tokenizing the full text can merge tokens across the split."""
    apply_chat_template = configure_apply_chat_template(tokenizer).tokenize(False)
    prompt_text = apply_chat_template(conversation[:-1], add_generation_prompt=True, **chat_template_kwargs)
    full_text = apply_chat_template(conversation, add_generation_prompt=False, **chat_template_kwargs)
    if not full_text.startswith(prompt_text):
        raise ValueError("the chat template does not render the prompt as a prefix of the full conversation")
    tokenize = configure_call(tokenizer)
    return PromptCompletion(
        tokenize(prompt_text, add_special_tokens=False)["input_ids"],
        tokenize(full_text[len(prompt_text):], add_special_tokens=False)["input_ids"],
    )
