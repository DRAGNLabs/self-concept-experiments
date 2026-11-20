"""Tally the Haiku episode-audit labels written for episode_audit_sample.py's set (CPU).

Checks every quote field (label_sets.AuditQuoteField) is verbatim in its item (whitespace normalised), joins the key
and writes per (model, arm) counts of every label present in the rows, and theme counts, to <dir>/tally.json; one row
per item with its key to <dir>/labels_joined.jsonl. Works for label sets with a subset of the labels (e.g. a re-judge
of explanation_mismatch only, whose quotes are prose_quote and code_quote).

    python experiments/deceiver-axis/scripts/episode_audit_tally.py --dir <audit dir>
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import TypedDict

from selfconcept.deceiver_axis.behaviour_measures.blind_labels import (labelled_pairs, normalise_whitespace,
                                                                       read_label_rows, write_joined_and_tally)
from selfconcept.deceiver_axis.behaviour_measures.label_sets import (AuditKey, AuditLabel, AuditLabelField,
                                                                     AuditQuoteField)

QUOTE_FIELD_BY_LABEL: dict[AuditLabelField, AuditQuoteField] = {
    "impossible_action": "impossible_action_quote", "explanation_mismatch": "explanation_mismatch_quote",
    "claims_correct_when_rejected": "claims_correct_when_rejected_quote", "damaging_code": "damaging_code_quote"}
QUOTE_FIELDS: tuple[AuditQuoteField, ...] = (*QUOTE_FIELD_BY_LABEL.values(), "prose_quote", "code_quote")
NO_QUOTE_VALUES = {"no", "none", "not_applicable", "unclear"}


class QuoteProblem(TypedDict):
    item: str
    label: str
    problem: str


def quote_problems(label_row: AuditLabel, item_text: str) -> list[QuoteProblem]:
    """Quotes not verbatim in the item, and labels other than no/none/... without a quote of their own (the
    single-label task's quotes all count as its own)."""
    labels_present = [label for label in QUOTE_FIELD_BY_LABEL if label in label_row]
    quote_by_field = {field: label_row.get(field, "") for field in QUOTE_FIELDS if field in label_row}
    problems: list[QuoteProblem] = [{"item": label_row["item"], "label": field, "problem": "quote not verbatim"}
                                    for field, quote in quote_by_field.items()
                                    if quote and normalise_whitespace(quote) not in normalise_whitespace(item_text)]
    for label in labels_present:
        own_quotes = [quote for field, quote in quote_by_field.items()
                      if field.startswith(label) or len(labels_present) == 1]
        if label_row.get(label) not in NO_QUOTE_VALUES and not any(own_quotes):
            problems.append({"item": label_row["item"], "label": label, "problem": "missing quote"})
    return problems


def counts_by_model_arm(pairs: list[tuple[AuditKey, AuditLabel]]) -> dict[str, dict[str, Counter]]:
    counts_by_group: dict[str, dict[str, Counter]] = {}
    for key, label_row in pairs:
        counters = counts_by_group.setdefault(f"{key['model']}/{key['arm']}",
                                              {field: Counter() for field in [*QUOTE_FIELD_BY_LABEL, "themes"]})
        for label in QUOTE_FIELD_BY_LABEL:
            if label in label_row:
                counters[label][label_row.get(label)] += 1
        counters["themes"].update(label_row.get("themes") or [])
    return counts_by_group


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", type=Path, required=True)
    args = parser.parse_args()
    key_by_item: dict[str, AuditKey] = json.loads((args.dir / "key.json").read_text())
    item_text_by_item: dict[str, str] = json.loads((args.dir / "items.json").read_text())
    label_rows = read_label_rows(args.dir, AuditLabel)
    label_row_by_item = {row["item"]: row for row in label_rows}
    problems = [problem for row in label_rows if row["item"] in item_text_by_item
                for problem in quote_problems(row, item_text_by_item[row["item"]])]
    write_joined_and_tally(args.dir, key_by_item, label_row_by_item, problems,
                           counts_by_model_arm(labelled_pairs(key_by_item, label_row_by_item)))


if __name__ == "__main__":
    main()
