from collections.abc import Sequence
from pathlib import Path

import jsonlines

from selfconcept.assistant_axis_cot.records import ConversationMetadata, condition_of
from selfconcept.common.hf_strong_types import Conversation, HFTokenizer
from selfconcept.common.span_targeting.types import PromptCompletion
from selfconcept.transcript_generation.vllm_generation import generated_prompt_completion


def load_labelled_conversations(
    responses_dir: Path, conversations_per_role: int | None
) -> list[tuple[ConversationMetadata, Conversation]]:
    labelled_conversations: list[tuple[ConversationMetadata, Conversation]] = []
    for responses_file in sorted(responses_dir.glob("*.jsonl")):
        with jsonlines.open(responses_file) as reader:
            responses = list(reader)[:conversations_per_role]
        for response in responses:
            condition = condition_of(responses_file.stem, response["prompt_index"])
            if condition is None:
                continue
            metadata: ConversationMetadata = {
                "role": responses_file.stem,
                "prompt_index": response["prompt_index"],
                "question_index": response["question_index"],
                "condition": condition,
            }
            labelled_conversations.append((metadata, response["conversation"]))
    return labelled_conversations


def thinking_prompt_completions(
    tokenizer: HFTokenizer, model_name: str, conversations: Sequence[Conversation]
) -> list[PromptCompletion]:
    return [
        generated_prompt_completion(tokenizer, model_name, conversation, enable_thinking=True, chat_template_kwargs={})
        for conversation in conversations
    ]


def padded_token_budget_batches(sequence_lengths: Sequence[int], max_batch_tokens: int) -> list[list[int]]:
    """Index batches, longest sequences first, whose right-padded size stays within max_batch_tokens
    (a sequence longer than the budget gets a batch of its own)."""
    indices_longest_first = sorted(range(len(sequence_lengths)), key=lambda index: -sequence_lengths[index])
    batches: list[list[int]] = []
    for index in indices_longest_first:
        if batches and sequence_lengths[batches[-1][0]] * (len(batches[-1]) + 1) <= max_batch_tokens:
            batches[-1].append(index)
        else:
            batches.append([index])
    return batches
