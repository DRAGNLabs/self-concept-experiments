"""Summarize ROLE results: binding-swap accuracy per test set, split by who was asked about.

For each test set and queried entity (you = the model, I = the user, name = a third person),
prints "says" accuracy (top-1 token is the expected role) under each condition, plus how often
the model says "assistant" when the expected answer is something else ("asst intrusions"),
the sign of self-knowledge overriding the context.

Usage (from the repo root):
    python experiments/binding_ids/scripts/summarize_roles.py experiments/binding_ids/results/olmo2-13b-instruct-roles
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

CONDITIONS = ["none", "swap_entities", "swap_attributes", "swap_both", "random_both"]
SWAPPED = {"swap_entities", "swap_attributes"}  # expected answer moves to the other pair


def kind(entity: str) -> str:
    return {"You": "you", "I": "I"}.get(entity, "name")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=Path)
    args = parser.parse_args()
    data_dir = Path("experiments/binding_ids/data")

    print("| test set | asked | n | " + " | ".join(CONDITIONS) + " | asst intrusions (none / swaps) |")
    print("|---|---|---|" + "---|" * (len(CONDITIONS) + 1))
    for path in sorted(args.results_dir.glob("role_*_queries.jsonl")):
        stem = path.name.removesuffix("_queries.jsonl")
        rows = [json.loads(line) for line in open(data_dir / f"{stem}.jsonl")]
        correct: dict[tuple[str, str], list[bool]] = defaultdict(list)
        intrusions: dict[tuple[str, str], list[bool]] = defaultdict(list)
        for rec in map(json.loads, open(path)):
            row, q, cond = rows[rec["row"]], rec["query_index"], rec["condition"]
            expected = 1 - q if cond in SWAPPED else q
            asked = kind(row["entities"][q])
            correct[(asked, cond)].append(rec["top_slot"] == expected)
            if row["answers"][expected] != "assistant":
                group = "none" if cond == "none" else "swaps" if cond in SWAPPED | {"swap_both"} else None
                if group:
                    intrusions[(asked, group)].append(rec["top_token"].strip() == "assistant")
        for asked in sorted({k for k, _ in correct}):
            n = len(correct[(asked, "none")])
            accs = " | ".join(f"{sum(v) / len(v):.2f}" for v in (correct[(asked, c)] for c in CONDITIONS))
            intr = " / ".join(
                f"{sum(v) / len(v):.2f}" if (v := intrusions.get((asked, g))) else "–" for g in ("none", "swaps")
            )
            print(f"| {stem.removeprefix('role_')} | {asked} | {n} | {accs} | {intr} |")


if __name__ == "__main__":
    main()
