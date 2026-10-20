"""Summarize ASSISTANT results: does the model bind itself with ordinary binding IDs?

One row per (frame, pair, binding-vector source, queried entity). Accuracy is "raw": the
expected in-context room beats the other one (Instruct often rewords its answer, so the
top-1 "says" metric undercounts; it is printed last for reference, for `none` only).
"from others" rows use binding vectors estimated on two non-self person nouns.

Usage (from the repo root):
    python experiments/binding_ids/scripts/summarize_assistant.py experiments/binding_ids/results/olmo2-13b-instruct-asst
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

CONDITIONS = ["none", "swap_entities", "swap_attributes", "swap_both", "random_both"]
SWAPPED = {"swap_entities", "swap_attributes"}
FRAMES = ["model_voice", "user_voice", "plain"]
PAIRS = ["others", "assistant_user", "assistant_other", "user_other"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=Path)
    args = parser.parse_args()
    data_dir = Path("experiments/binding_ids/data")

    print("| frame | pair | vectors from | asked | n | " + " | ".join(CONDITIONS) + " | says (none) |")
    print("|---|---|---|---|---|" + "---|" * (len(CONDITIONS) + 1))
    for frame in FRAMES:
        for pair in PAIRS:
            for source in ("own", "others"):
                stem = f"asst_{frame}_{pair}" + ("_from_others" if source == "others" and pair != "others" else "")
                if source == "others" and pair == "others":
                    continue
                path = args.results_dir / f"{stem}_queries.jsonl"
                if not path.exists():
                    continue
                rows = [json.loads(line) for line in open(data_dir / f"asst_{frame}_{pair}.jsonl")]
                raw: dict[tuple[str, str], list[bool]] = defaultdict(list)
                says: dict[str, list[bool]] = defaultdict(list)
                for rec in map(json.loads, open(path)):
                    q, cond = rec["query_index"], rec["condition"]
                    entity = rows[rec["row"]]["entities"][q]
                    asked = entity if entity in ("assistant", "user") else "other"
                    expected = 1 - q if cond in SWAPPED else q
                    lp = rec["log_probs"]
                    raw[(asked, cond)].append(lp[expected] > lp[1 - expected])
                    if cond == "none":
                        says[asked].append(rec["top_slot"] == expected)
                for asked in sorted(says):
                    accs = " | ".join(f"{sum(v) / len(v):.2f}" for v in (raw[(asked, c)] for c in CONDITIONS))
                    s = says[asked]
                    print(f"| {frame} | {pair} | {source} | {asked} | {len(s)} | {accs} | {sum(s) / len(s):.2f} |")


if __name__ == "__main__":
    main()
