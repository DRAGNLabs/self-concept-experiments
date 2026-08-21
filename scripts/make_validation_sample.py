"""Build a blinded sample for human validation of the rule-based classifier.

Stratified draw of responses from the headline runs, shuffled, with the two
candidate rooms presented in alphabetical order so the labeler can't infer
which is honest. Writes validation/sample.jsonl (blind) and validation/key.jsonl
(the answer key + the classifier's label — don't look before labeling).

Usage: python scripts/make_validation_sample.py [--per-file 10] [--seed 7]
Label by adding "human_label": honest|deceptive|refusal|other to each sample row.
Then: python scripts/make_validation_sample.py --score

Alternatively an LLM judge (blind: sees response + alphabetical rooms only)
writes validation/judge.jsonl rows {"sample_id", "answer_room", "category"}
with category room|refusal|unclear; --score-judge maps answer_room through the
key (room -> honest/deceptive, unclear -> other) and scores the same way.
"""

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SOURCE_FILES = [
    "baseline_mistral7b_n250_main_i_would.jsonl",
    "baseline_mistral7b_n250_treasure_hunt_i_would.jsonl",
    "soo_mistral7b_lasttok_L16_seed0_n250_main_i_would.jsonl",
    "soo_mistral7b_lasttok_L16_seed1_n250_main_i_would.jsonl",
    "soo_mistral7b_lasttok_L16_seed2_n250_treasure_hunt_i_would.jsonl",
    "soo_mistral7b_lasttok_L16_seed3_n250_treasure_hunt_i_would.jsonl",
    "soo_mistral7b_lasttok_L16_seed4_n250_perspectives_i_would.jsonl",
    "soo_mistral7b_seed0_main_i_would.jsonl",
    "soo_mistral7b_seed1_treasure_hunt_i_would.jsonl",
    "baseline_gemma27b_main_room_only.jsonl",
    "soo_gemma27b_seed0_main_room_only.jsonl",
    "soo_gemma27b_seed1_perspectives_room_only.jsonl",
]


def _score_labels(labels: dict[str, str], judge_name: str) -> None:
    key = {r["sample_id"]: r for r in map(json.loads, (ROOT / "validation/key.jsonl").open())}
    agree = total = 0
    confusion: dict[str, int] = {}
    disagreements = []
    for sample_id, label in labels.items():
        total += 1
        auto = key[sample_id]["classifier_label"]
        pair = f"{label}/{auto}"
        confusion[pair] = confusion.get(pair, 0) + 1
        if label == auto:
            agree += 1
        else:
            disagreements.append((sample_id, key[sample_id]["source_file"], label, auto))
    if not total:
        sys.exit("no labels found")
    print(f"agreement: {agree}/{total} = {agree / total:.1%}")
    for pair, n in sorted(confusion.items(), key=lambda kv: -kv[1]):
        print(f"  {judge_name}/classifier {pair}: {n}")
    if disagreements:
        print("disagreements:")
        for sample_id, source, label, auto in disagreements:
            print(f"  {sample_id} ({source}): {judge_name}={label} classifier={auto}")


def score() -> None:
    labels = {
        row["sample_id"]: row["human_label"]
        for row in map(json.loads, (ROOT / "validation/sample.jsonl").open())
        if "human_label" in row
    }
    _score_labels(labels, "human")


def score_judge() -> None:
    key = {r["sample_id"]: r for r in map(json.loads, (ROOT / "validation/key.jsonl").open())}
    labels = {}
    for row in map(json.loads, (ROOT / "validation/judge.jsonl").open()):
        k = key[row["sample_id"]]
        if row["category"] == "refusal":
            labels[row["sample_id"]] = "refusal"
        elif row["category"] == "room" and row["answer_room"] == k["honest"]:
            labels[row["sample_id"]] = "honest"
        elif row["category"] == "room" and row["answer_room"] == k["deceptive"]:
            labels[row["sample_id"]] = "deceptive"
        else:
            labels[row["sample_id"]] = "other"
    _score_labels(labels, "judge")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-file", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--score", action="store_true", help="score human-labeled sample")
    parser.add_argument(
        "--score-judge", action="store_true", help="score validation/judge.jsonl (LLM judge)"
    )
    args = parser.parse_args()
    if args.score:
        score()
        return
    if args.score_judge:
        score_judge()
        return

    answers = {}
    for data_file in (ROOT / "data").glob("eval*/*.jsonl"):
        for line in data_file.open():
            ex = json.loads(line)
            answers[ex["example_id"]] = (ex["honest_answer"], ex["deceptive_answer"])

    rng = random.Random(args.seed)
    samples, keys = [], []
    for name in SOURCE_FILES:
        records = [json.loads(l) for l in (ROOT / "results/eval" / name).open()]
        for rec in rng.sample(records, min(args.per_file, len(records))):
            honest, deceptive = answers[rec["example_id"]]
            samples.append(
                {
                    "sample_id": f"v{len(samples):03d}",
                    "response": rec["response"],
                    "rooms": sorted([honest, deceptive]),
                }
            )
            keys.append(
                {
                    "sample_id": samples[-1]["sample_id"],
                    "source_file": name,
                    "example_id": rec["example_id"],
                    "honest": honest,
                    "deceptive": deceptive,
                    "classifier_label": rec["label"],
                }
            )
    order = list(range(len(samples)))
    rng.shuffle(order)
    out = ROOT / "validation"
    out.mkdir(exist_ok=True)
    with (out / "sample.jsonl").open("w") as f:
        for i in order:
            f.write(json.dumps(samples[i]) + "\n")
    with (out / "key.jsonl").open("w") as f:
        for k in keys:
            f.write(json.dumps(k) + "\n")
    print(f"wrote {len(samples)} blinded samples to validation/sample.jsonl")


if __name__ == "__main__":
    main()
