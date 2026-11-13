"""Finished multi-turn episodes rendered turn by turn exactly as the model saw them, for teacher forcing: the chat
template's prompt over the history (earlier answers only, as the templates drop earlier reasoning), then this turn's
reasoning and answer in the model's raw completion markup, every token tagged with its segment."""
from collections.abc import Sequence
from enum import IntEnum
from typing import Any, Literal, NamedTuple

import numpy as np
from jaxtyping import Int8

from selfconcept.common.chat import Conversation
from selfconcept.common.hf_strong_types import HFTokenizer, configure_apply_chat_template, configure_call
from selfconcept.common.teacher_forcing.completion_markup import CompletionMarkup, CompletionMarkupSpecifics


class Segment(IntEnum):
    OTHER = 0
    REASONING = 1
    ANSWER = 2


type EpisodeStep = tuple[Literal["user", "model_turn"], str]  # a user message's text, or a model turn's index


class RenderedTurn(NamedTuple):
    turn: int
    ids: list[int]  # capped at max_length
    segments: Int8[np.ndarray, "tokens"]
    uncapped_length: int
    reasoning_missing: bool


def completion_pieces(markup: CompletionMarkup, reasoning: str, answer: str) -> list[tuple[str, Segment]]:
    if not reasoning:
        return [(markup.direct_answer_open, Segment.OTHER), (answer, Segment.ANSWER), (markup.end_of_turn, Segment.OTHER)]
    if not answer:  # the reasoning never closed
        return [(markup.thought_open, Segment.OTHER), (reasoning, Segment.REASONING)]
    return [(markup.thought_open, Segment.OTHER), (reasoning, Segment.REASONING), (markup.thought_close, Segment.OTHER),
            (answer, Segment.ANSWER), (markup.end_of_turn, Segment.OTHER)]


def render_turn(specifics: CompletionMarkupSpecifics, tokenizer: HFTokenizer, history: Conversation, reasoning: str, answer: str,
                chat_kwargs: dict[str, Any]) -> tuple[list[int], Int8[np.ndarray, "tokens"]]:
    prompt = configure_apply_chat_template(tokenizer)(history, add_generation_prompt=True, **chat_kwargs)["input_ids"]
    tokenize = configure_call(tokenizer)
    pieces = [(prompt, Segment.OTHER)] + [(tokenize(text, add_special_tokens=False)["input_ids"] if text else [], segment)
                                           for text, segment in completion_pieces(specifics.completion_markup(), reasoning, answer)]
    ids = [token for piece_ids, _ in pieces for token in piece_ids]
    segments = np.concatenate([np.full(len(piece_ids), segment, dtype=np.int8) for piece_ids, segment in pieces])
    return ids, segments


def episode_turns(specifics: CompletionMarkupSpecifics, tokenizer: HFTokenizer, steps: Sequence[EpisodeStep], answer_by_turn: Sequence[str],
                  reasoning_by_turn: dict[int, str], max_length: int, chat_kwargs: dict[str, Any]) -> list[RenderedTurn]:
    history: Conversation = []
    turns: list[RenderedTurn] = []
    for role, text in steps:
        if role == "user":
            history.append({"role": "user", "content": text})
            continue
        turn = int(text)
        ids, segments = render_turn(specifics, tokenizer, history, reasoning_by_turn.get(turn, ""), answer_by_turn[turn], chat_kwargs)
        turns.append(RenderedTurn(turn, ids[:max_length], segments[:max_length], len(ids), turn not in reasoning_by_turn))
        history.append({"role": "assistant", "content": answer_by_turn[turn]})
    return turns
