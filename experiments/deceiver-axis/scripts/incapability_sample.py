"""Blind Haiku set for validating the incapability-admission regex (text_proxies.says_incapability) (CPU).

Per model (behaviour_measures.episodes.PRESETS), from the unsteered arm: --per-class code-submitting turns whose
code FAILED the visible tests (the regex's pool, rejected submissions) and --per-class turns whose code PASSED. Shown
text is the turn's visible reply with all code removed except comments and docstrings (blind_labels.prose_only);
labelers are not told whether the code passed. The key (<out>/key.json, never shown to labelers) records model,
episode, turn, outcome and whether the regex fires on the full reply, as the proxy reads it. --regex-hits adds a
separate stratum of that many rejected turns per model on which the regex fires (outcome "failed_regex_hit"), to
check its precision.

    python experiments/deceiver-axis/scripts/incapability_sample.py --out <dir>
"""
import argparse
import json
import random
from pathlib import Path
from typing import Literal, TypedDict

from selfconcept.deceiver_axis.behaviour_measures.blind_labels import (BlindItem, opaque_id, prose_only,
                                                                       write_strided_batches)
from selfconcept.deceiver_axis.behaviour_measures.episodes import PRESETS, ModelPreset, load_records
from selfconcept.deceiver_axis.behaviour_measures.text_proxies import says_incapability

type TurnOutcome = Literal["failed", "passed", "failed_regex_hit"]
type SubmittedTurn = tuple[str, int, str]  # seed key, turn, visible reply


class IncapabilityKey(TypedDict):
    model: str
    episode_key: str
    turn: int
    outcome: TurnOutcome
    regex: bool


def submitted_turns_by_outcome(preset: ModelPreset) -> dict[TurnOutcome, list[SubmittedTurn]]:
    turns_by_outcome: dict[TurnOutcome, list[SubmittedTurn]] = {"failed": [], "passed": [], "failed_regex_hit": []}
    for seed_key, record in sorted(load_records(preset["unsteered"], preset["record_file"]).items()):
        for turn, attempt in enumerate(record["attempts"]):
            if attempt.get("answer"):
                outcome: TurnOutcome = "passed" if attempt.get("success") else "failed"
                turns_by_outcome[outcome].append((seed_key, turn, attempt.get("completion") or ""))
    turns_by_outcome["failed_regex_hit"] = [entry for entry in turns_by_outcome["failed"] if says_incapability(entry[2])]
    return turns_by_outcome


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-class", type=int, default=15)
    parser.add_argument("--regex-hits", type=int, default=0)
    parser.add_argument("--batches", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    size_by_outcome: dict[TurnOutcome, int] = {"failed": args.per_class, "passed": args.per_class,
                                               "failed_regex_hit": args.regex_hits}
    items: list[BlindItem] = []
    key: dict[str, IncapabilityKey] = {}
    for preset in PRESETS:
        for outcome, pool in submitted_turns_by_outcome(preset).items():
            for seed_key, turn, completion in rng.sample(pool, min(size_by_outcome[outcome], len(pool))):
                item_id = opaque_id(f"{preset['slug']}/{seed_key}/{turn}")
                if item_id in key:
                    continue
                items.append({"item": item_id, "text": prose_only(completion)})
                key[item_id] = {"model": preset["label"], "episode_key": f"{preset['unsteered'].name}/{seed_key}",
                                "turn": turn, "outcome": outcome, "regex": says_incapability(completion)}
    rng.shuffle(items)
    write_strided_batches(items, args.batches, args.out / "batches")
    (args.out / "key.json").write_text(json.dumps(key, indent=1))
    print(f"{len(items)} items in {args.batches} batches -> {args.out}")


if __name__ == "__main__":
    main()
