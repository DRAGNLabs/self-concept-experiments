"""Binding task templates: contexts that bind entities to attributes, plus queries.

CAPITALS task (adapted from Feng & Steinhardt 2023): each context binds names to
countries, and the query asks for the capital of the queried name's country.
"""

import json
from pathlib import Path
from typing import TypedDict

from .vocab import COUNTRY_CAPITALS

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
    return row["context"] + QUERY.format(entity=row["entities"][query_index])


def load_rows(path: Path) -> list[BindingRow]:
    with open(path) as f:
        return [json.loads(line) for line in f]
