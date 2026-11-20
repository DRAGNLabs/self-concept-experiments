"""Tally the Haiku abstention / admission labels written for abstention_admission_sample.py's set (CPU).

Checks every quote is verbatim in its item (whitespace normalised), joins the key and writes per (arm, pool, role)
label counts and theme counts to <dir>/tally.json; one row per item with its key to <dir>/labels_joined.jsonl.

    python experiments/deceiver-axis/scripts/abstention_admission_tally.py --dir <abstention_admission_pilot>
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import TypedDict

from selfconcept.deceiver_axis.behaviour_measures.blind_labels import (labelled_pairs, normalise_whitespace,
                                                                       read_label_rows, write_joined_and_tally)
from selfconcept.deceiver_axis.behaviour_measures.label_sets import (AbstentionAdmissionItem, AbstentionAdmissionKey,
                                                                     AbstentionAdmissionLabel,
                                                                     AbstentionAdmissionLabelField,
                                                                     AbstentionAdmissionQuoteField)

QUOTE_FIELD_BY_LABEL: dict[AbstentionAdmissionLabelField, AbstentionAdmissionQuoteField] = {
    "abstained": "abstained_quote", "stuck_to_previous": "stuck_quote", "admitted_hack": "admitted_quote",
    "tests_incorrect": "tests_incorrect_quote"}
NO_QUOTE_VALUES = {"no", "not_applicable", "unclear"}


class QuoteProblem(TypedDict):
    item: str
    field: str
    problem: str


def quote_problems(label: AbstentionAdmissionLabel, item_text: str) -> list[QuoteProblem]:
    problems: list[QuoteProblem] = []
    for field, quote_field in QUOTE_FIELD_BY_LABEL.items():
        quote = label.get(quote_field) or ""
        if label.get(field) not in NO_QUOTE_VALUES and not quote:
            problems.append({"item": label["item"], "field": field, "problem": "missing quote"})
        elif quote and normalise_whitespace(quote) not in normalise_whitespace(item_text):
            problems.append({"item": label["item"], "field": field, "problem": "quote not verbatim"})
    return problems


def counts_by_arm_pool_role(pairs: list[tuple[AbstentionAdmissionKey, AbstentionAdmissionLabel]]
                            ) -> dict[str, dict[str, Counter]]:
    counts_by_group: dict[str, dict[str, Counter]] = {}
    for key, label in pairs:
        group = f"{key['arm']}/{key['pool']}/{key['role']}"
        counters = counts_by_group.setdefault(group, {field: Counter() for field in [*QUOTE_FIELD_BY_LABEL, "themes"]})
        for field in QUOTE_FIELD_BY_LABEL:
            counters[field][label[field]] += 1
        counters["themes"].update(label.get("themes") or [])
    return counts_by_group


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", type=Path, required=True)
    args = parser.parse_args()
    key_by_item: dict[str, AbstentionAdmissionKey] = json.loads((args.dir / "key.json").read_text())
    items: list[AbstentionAdmissionItem] = json.loads((args.dir / "items.json").read_text())
    item_text_by_item = {item["item_id"]: "\n".join(turn["text"] for turn in item["shown_turns"]) for item in items}
    labels = read_label_rows(args.dir, AbstentionAdmissionLabel)
    label_by_item = {label["item"]: label for label in labels}
    problems = [problem for label in labels for problem in quote_problems(label, item_text_by_item[label["item"]])]
    write_joined_and_tally(args.dir, key_by_item, label_by_item, problems,
                           counts_by_arm_pool_role(labelled_pairs(key_by_item, label_by_item)))


if __name__ == "__main__":
    main()
