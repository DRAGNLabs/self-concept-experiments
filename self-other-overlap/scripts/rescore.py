"""Re-label archived eval responses with the current classifier.

The original classifier used a bare substring search, so the room 'den'
matched inside words like 'hidden'. This joins every archived response file
back to its eval dataset via example_id, recomputes labels with
soo.evaluate.classify, rewrites the .jsonl labels and _summary.json counts in
place, and prints every file whose rates changed.

Usage: python scripts/rescore.py [--dry-run]
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from soo.evaluate import classify  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    answers = {}
    for data_file in (ROOT / "data" / "eval").glob("*.jsonl"):
        for line in data_file.open():
            ex = json.loads(line)
            answers[ex["example_id"]] = (ex["honest_answer"], ex["deceptive_answer"])

    total = changed = 0
    changed_files = []
    for resp_file in sorted((ROOT / "results" / "eval").glob("*.jsonl")):
        records = [json.loads(l) for l in resp_file.open()]
        flips = []
        for rec in records:
            honest, deceptive = answers[rec["example_id"]]
            new_label = classify(rec["response"], honest, deceptive)
            total += 1
            if new_label != rec["label"]:
                flips.append((rec["label"], new_label))
                rec["label"] = new_label
        if not flips:
            continue
        changed += len(flips)
        counts = {label: 0 for label in ("honest", "deceptive", "refusal", "other")}
        for rec in records:
            counts[rec["label"]] += 1
        changed_files.append((resp_file.name, flips, counts))
        if args.dry_run:
            continue
        with resp_file.open("w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
        summary_path = resp_file.with_name(resp_file.stem + "_summary.json")
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
            summary["counts"] = counts
            summary["rates"] = {k: round(v / len(records), 4) for k, v in counts.items()}
            summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"{total} labels checked, {changed} changed across {len(changed_files)} files\n")
    for name, flips, counts in changed_files:
        moves = {}
        for old, new in flips:
            moves[f"{old}->{new}"] = moves.get(f"{old}->{new}", 0) + 1
        print(f"{name}: {len(flips)} flips {moves} -> new counts {counts}")


if __name__ == "__main__":
    main()
