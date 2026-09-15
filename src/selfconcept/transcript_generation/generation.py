# Derived from safety-research/assistant-axis (https://github.com/safety-research/assistant-axis),
# MIT licensed. See the NOTICE file at the repository root for the full license text.
from typing import Optional, Protocol, runtime_checkable

from selfconcept.common.conversation_types import Conversation


@runtime_checkable
class BatchEngine(Protocol):
    """Minimal interface the persona-drift loop needs from a generation engine.
    """

    model_name: str

    def generate_batch(self, conversations: list[Conversation]) -> list[str]: ...


def format_conversation(
    instruction: Optional[str],
    question: str,
    tokenizer,
) -> Conversation:
    """
    Format a conversation for model input.

    Args:
        instruction: Optional system instruction
        question: User question
        tokenizer: HuggingFace tokenizer (to check system prompt support)

    Returns:
        List of message dicts for the conversation
    """
    # Check system prompt support by testing the chat template
    test_message = "__SYSTEM_TEST__"
    test_conversation = [
        {"role": "system", "content": test_message},
        {"role": "user", "content": "hello"},
    ]

    try:
        output = tokenizer.apply_chat_template(
            test_conversation,
            tokenize=False,
            add_generation_prompt=False,
        )
        supports_system = test_message in output
    except Exception:
        supports_system = False

    if supports_system:
        messages = []
        if instruction:
            messages.append({"role": "system", "content": instruction})
        messages.append({"role": "user", "content": question})
        return messages
    else:
        # Concatenate instruction with question for models without system support
        if instruction:
            formatted = f"{instruction}\n\n{question}"
        else:
            formatted = question
        return [{"role": "user", "content": formatted}]


