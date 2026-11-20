"""EXPLORATORY (lightly reviewed when merged 2026-11-20): kept for reference, not a validated measurement.

Blind Haiku set: can a reader tell a hack from the prose alone? (CPU)

Per model (behaviour_measures.episodes.PRESETS), --per-class judged HACK and NOT_HACK episodes of one arm
(default unsteered; --arm-override slug=<arm dir> for a model without unsteered hacks). Shown text is one turn's visible
reply with every line of code removed except its comments and docstrings (blind_labels.prose_only): for a hack the
judge's hack turn, for a non-hack its final (passing) turn; every judged episode passed the visible tests. Items are
shuffled across models and classes under opaque ids into --batches batch files; the key goes to <out>/key.json, which
labelers are never pointed at.

    python experiments/deceiver-axis/scripts/hack_prose_sample.py --out <dir> [--arm-override qwen=<arm dir>]
"""
import argparse
import json
import random
from pathlib import Path
from typing import Literal

from selfconcept.deceiver_axis.behaviour_measures.blind_labels import (BlindItem, opaque_id, prose_only,
                                                                       write_strided_batches)
from selfconcept.codebench.harness import CheckRecord
from selfconcept.deceiver_axis.behaviour_measures.episodes import (PRESETS, ModelPreset, hack_turns, load_judge_details,
                                                                   load_records, load_verdict_by_key)

type HackClass = Literal["hack", "not_hack"]
type PoolEntry = tuple[str, int, CheckRecord]  # episode key, shown turn, record


def class_pools(preset: ModelPreset, arm_dir: Path) -> dict[HackClass, list[PoolEntry]]:
    """Judged HACK episodes with a judge detail (shown: the hack turn) and NOT_HACK episodes (shown: the last turn)."""
    verdict_by_key = load_verdict_by_key(preset["verdict_files"])
    detail_by_key = load_judge_details(preset, verdict_by_key, {arm_dir.name})
    pools: dict[HackClass, list[PoolEntry]] = {"hack": [], "not_hack": []}
    for seed_key, record in load_records(arm_dir, preset["record_file"]).items():
        episode_key = f"{arm_dir.name}/{seed_key}"
        verdict = verdict_by_key.get(episode_key)
        if verdict == "HACK" and episode_key in detail_by_key:
            pools["hack"].append((episode_key, hack_turns(detail_by_key[episode_key])[1], record))
        elif verdict == "NOT_HACK":
            pools["not_hack"].append((episode_key, len(record["attempts"]) - 1, record))
    return pools


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-class", type=int, default=15)
    parser.add_argument("--batches", type=int, default=6)
    parser.add_argument("--arm-override", action="append", default=[], help="slug=<arm dir>")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    arm_dir_by_slug = dict(spec.split("=", 1) for spec in args.arm_override)
    rng = random.Random(args.seed)
    items: list[BlindItem] = []
    key: dict[str, dict] = {}
    for preset in PRESETS:
        arm_dir = Path(arm_dir_by_slug.get(preset["slug"], preset["unsteered"]))
        for label, pool in class_pools(preset, arm_dir).items():
            if len(pool) < args.per_class:
                raise SystemExit(f"{preset['label']}: only {len(pool)} {label} episodes in {arm_dir}")
            for episode_key, turn, record in rng.sample(sorted(pool, key=lambda entry: entry[0]), args.per_class):
                item_id = opaque_id(f"{preset['slug']}/{episode_key}")
                items.append({"item": item_id, "text": prose_only(record["attempts"][turn].get("completion") or "")})
                key[item_id] = {"model": preset["label"], "arm": arm_dir.name, "episode_key": episode_key,
                                "turn": turn, "class": label}
    rng.shuffle(items)
    write_strided_batches(items, args.batches, args.out / "batches")
    (args.out / "key.json").write_text(json.dumps(key, indent=1))
    print(f"{len(items)} items in {args.batches} batches -> {args.out}")


if __name__ == "__main__":
    main()
