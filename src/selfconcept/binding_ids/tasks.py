"""Binding task templates: contexts that bind entities to attributes, plus queries.

CAPITALS task (adapted from Feng & Steinhardt 2023): each context binds names to
countries, and the query asks for the capital of the queried name's country.

ROOMS tasks: "<entity> is in the <room>." with the same frame for every entity type
(see ROOM_TASKS), queried with "Where is <entity>?" and answered with the room itself.

ROLE tasks: "You are the doctor. I am the teacher." inside a chat user turn, so "you" is the
model and "I" the user; queried "Which role do you have?" and answered in the assistant
turn ("I am the ___"). The model's true role, "assistant", can be placed on one entity
(see ROLE_PAIRS) to test whether self-knowledge changes binding.
"""

import json
import random
from pathlib import Path
from typing import NotRequired, TypedDict

from .vocab import ANIMALS, COUNTRY_CAPITALS, NAMES, OBJECTS, PRONOUNS, ROLES, ROOMS, SELF_ROLE

PREAMBLE = "Answer the question based on the context below. Keep the answer short.\n\nContext:"
QUERY = "\n\nQuestion: Which city does {entity} live in?\n\nAnswer: {entity} lives in the city of"


class BindingRow(TypedDict):
    context: str
    entities: list[str]
    attributes: list[str]
    answers: list[str]
    # [start, end) character spans of each mention inside `context`
    entity_spans: list[list[int]]
    attribute_spans: list[list[int]]
    # Query text appended to `context` for each entity; absent for CAPITALS rows (uses QUERY)
    queries: NotRequired[list[str]]
    task: NotRequired[str]


def build_row(entities: list[str], attributes: list[str]) -> BindingRow:
    context = PREAMBLE
    entity_spans, attribute_spans = [], []
    for entity, attribute in zip(entities, attributes):
        context += " "
        entity_spans.append([len(context), len(context) + len(entity)])
        context += entity + " lives in the capital city of "
        attribute_spans.append([len(context), len(context) + len(attribute)])
        context += attribute + "."
    return BindingRow(
        context=context,
        entities=entities,
        attributes=attributes,
        answers=[COUNTRY_CAPITALS[a] for a in attributes],
        entity_spans=entity_spans,
        attribute_spans=attribute_spans,
    )


def query_prompt(row: BindingRow, query_index: int) -> str:
    if "queries" in row:
        return row["context"] + row["queries"][query_index]
    return row["context"] + QUERY.format(entity=row["entities"][query_index])


# Entity pools for the two pairs in each context. One shared pool: two distinct members.
# Two pools: one member of each, with their order alternating across contexts so neither
# pool is always first (otherwise the binding-vector estimate would absorb the
# difference between the pools).
ROOM_TASKS: dict[str, tuple[list[str], list[str]]] = {
    "people": (NAMES, NAMES),
    "animals": (ANIMALS, ANIMALS),
    "objects": (OBJECTS, OBJECTS),
    "i_you": (["I"], ["You"]),
    "animal_object": (ANIMALS, OBJECTS),
}

ROOM_QUERY = "\n\nQuestion: Where {verb} {entity}?\n\nAnswer: {Entity} {verb} in the"


def _room_phrase(entity: str, sentence_start: bool) -> tuple[str, int]:
    """Surface form of `entity` and the character offset of the entity word inside it."""
    if entity in PRONOUNS:
        return (entity if sentence_start or entity == "I" else entity.lower()), 0
    if entity in NAMES:
        return entity, 0
    article = "The " if sentence_start else "the "
    return article + entity, len(article)


def _copula(entity: str) -> str:
    return {"I": "am", "You": "are"}.get(entity, "is")


def build_room_row(task: str, entities: list[str], rooms: list[str]) -> BindingRow:
    context = PREAMBLE
    entity_spans, attribute_spans, queries = [], [], []
    for entity, room in zip(entities, rooms):
        phrase, offset = _room_phrase(entity, sentence_start=True)
        context += " "
        entity_spans.append([len(context) + offset, len(context) + offset + len(entity)])
        context += f"{phrase} {_copula(entity)} in the "
        attribute_spans.append([len(context), len(context) + len(room)])
        context += room + "."
        queries.append(
            ROOM_QUERY.format(
                verb=_copula(entity),
                entity=_room_phrase(entity, sentence_start=False)[0],
                Entity=phrase,
            )
        )
    return BindingRow(
        context=context,
        entities=entities,
        attributes=rooms,
        answers=rooms,
        entity_spans=entity_spans,
        attribute_spans=attribute_spans,
        queries=queries,
        task=task,
    )


def sample_room_row(task: str, rng: random.Random, index: int) -> BindingRow:
    first_pool, second_pool = ROOM_TASKS[task]
    if first_pool is second_pool:
        entities = rng.sample(first_pool, 2)
    else:
        entities = [rng.choice(first_pool), rng.choice(second_pool)]
        if index % 2:
            entities.reverse()
    return build_room_row(task, entities, rng.sample(ROOMS, 2))


def load_rows(path: Path) -> list[BindingRow]:
    with open(path) as f:
        return [json.loads(line) for line in f]


# OLMo-2-Instruct chat template (checked against tokenizer.apply_chat_template): one user turn,
# then the assistant turn the model continues. Base models get the same text, for comparison.
CHAT_USER = "<|endoftext|><|user|>\n"
CHAT_ASSISTANT = "\n<|assistant|>\n"

# Entity kinds for each pair; "name" is a third person drawn from NAMES.
ROLE_PAIRS: dict[str, tuple[str, str]] = {
    "you_i": ("you", "I"),
    "you_name": ("you", "name"),
    "i_name": ("I", "name"),
    "names": ("name", "name"),
}

# kind -> (context subject, verb, question, answer prefix); {n} is the name
_ROLE_FORMS = {
    "you": ("You", "are", "Which role do you have?", "I am the"),
    "I": ("I", "am", "Which role do I have?", "You are the"),
    "name": ("{n}", "is", "Which role does {n} have?", "{n} is the"),
}


def build_role_row(task: str, kinds: list[str], entities: list[str], roles: list[str]) -> BindingRow:
    """`entities` are the surface subjects ("You", "I" or a name), one per kind."""
    context = CHAT_USER + PREAMBLE
    entity_spans, attribute_spans, queries = [], [], []
    for kind, entity, role in zip(kinds, entities, roles):
        _, verb, question, answer = _ROLE_FORMS[kind]
        context += " "
        entity_spans.append([len(context), len(context) + len(entity)])
        context += f"{entity} {verb} the "
        attribute_spans.append([len(context), len(context) + len(role)])
        context += role + "."
        queries.append(f"\n\nQuestion: {question.format(n=entity)}{CHAT_ASSISTANT}{answer.format(n=entity)}")
    return BindingRow(
        context=context,
        entities=entities,
        attributes=roles,
        answers=roles,
        entity_spans=entity_spans,
        attribute_spans=attribute_spans,
        queries=queries,
        task=task,
    )


def sample_role_row(pair: str, assistant: str | None, rng: random.Random, index: int) -> BindingRow:
    """One ROLE context. `assistant` is the entity kind that gets the role "assistant" (None:
    neither does). Entity order alternates with `index`; for `names`, which name holds
    "assistant" alternates every two rows, so all four order/holder combinations are balanced.
    """
    kinds = list(ROLE_PAIRS[pair])
    names = iter(rng.sample(NAMES, 2))
    entities = [next(names) if k == "name" else {"you": "You", "I": "I"}[k] for k in kinds]
    roles = rng.sample(ROLES, 2)
    if assistant is not None:
        if assistant not in kinds:
            raise ValueError(f"pair {pair} has no {assistant!r} entity")
        holder = (index // 2) % 2 if kinds[0] == kinds[1] else kinds.index(assistant)
        roles[holder] = SELF_ROLE
    if kinds[0] != kinds[1] and index % 2:
        kinds.reverse(), entities.reverse(), roles.reverse()
    suffix = "neutral" if assistant is None else f"asst_{assistant.lower()}"
    return build_role_row(f"role_{pair}_{suffix}", kinds, entities, roles)
