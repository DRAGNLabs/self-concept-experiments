from collections.abc import Callable, Sequence
from typing import NamedTuple, TypedDict

from selfconcept.common.chat import Conversation, ConversationTurn
from selfconcept.common.generation.model_specifics import get_generation_specifics
from selfconcept.common.llm_judge import GeneratedResponse
from selfconcept.self_recognition.schemas import PromptRecord, TranscriptRecord
from selfconcept.transcript_generation.vllm_generation import VLLMGenerator


class GenerationItem(NamedTuple):
    prompt: PromptRecord
    sample_index: int


class TranscriptQualityCounts(TypedDict):
    records: int
    truncated: int
    empty_content: int


def generation_items(prompts: Sequence[PromptRecord], samples_per_prompt: int) -> list[GenerationItem]:
    return [GenerationItem(prompt, sample_index) for prompt in prompts for sample_index in range(samples_per_prompt)]


def user_turn_conversation(prompt: PromptRecord) -> Conversation:
    return [{"role": "user", "content": prompt["question"]}]


def with_stripped_content(assistant_turn: ConversationTurn) -> ConversationTurn:
    return {**assistant_turn, "content": assistant_turn["content"].strip()}


def transcript_record(
    model: str,
    prompt: PromptRecord,
    sample_index: int,
    response: GeneratedResponse,
    assistant_turn: ConversationTurn,
) -> TranscriptRecord:
    return {
        "prompt_id": prompt["id"],
        "model": model,
        "sample_index": sample_index,
        "conversation": user_turn_conversation(prompt) + [assistant_turn],
        "truncated": response["truncated"],
    }


def transcript_chunk_processor(
    generator: VLLMGenerator,
    model: str,
) -> Callable[[Sequence[GenerationItem]], list[TranscriptRecord]]:
    generation_specifics = get_generation_specifics(model)

    def process_chunk(items: Sequence[GenerationItem]) -> list[TranscriptRecord]:
        responses = generator.generate_responses([user_turn_conversation(item.prompt) for item in items])
        return [
            transcript_record(
                model,
                item.prompt,
                item.sample_index,
                response,
                with_stripped_content(generation_specifics.to_assistant_turn(response["text"])),
            )
            for item, response in zip(items, responses, strict=True)
        ]

    return process_chunk


def transcript_quality_counts(records: Sequence[TranscriptRecord]) -> TranscriptQualityCounts:
    return {
        "records": len(records),
        "truncated": sum(record["truncated"] for record in records),
        "empty_content": sum(record["conversation"][-1]["content"] == "" for record in records),
    }
