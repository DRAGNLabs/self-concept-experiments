"""Blind Haiku audit set: whole episodes (every turn's visible reply, each followed by the harness outcome of that
turn) from each model's unsteered, gi_v2 main-dose and gfar main-dose arms (behaviour_measures.episodes.PRESETS). Samples
episodes with at least one non-empty reply, shuffles them across models and arms under opaque ids, and packs them
into batches of at most --batch-chars characters (an item larger than that is a batch of its own). The key goes to
<out>/key.json, which labelers are never pointed at. (CPU)

    python experiments/deceiver-axis/scripts/episode_audit_sample.py --out <dir> --n-per-arm 30
"""

import argparse
import json
import random
from pathlib import Path

from selfconcept.codebench.harness import CheckRecord
from selfconcept.deceiver_axis.behaviour_measures.blind_labels import opaque_id, pack_batches, turn_outcome
from selfconcept.deceiver_axis.behaviour_measures.episodes import PRESETS, load_records, main_dose_arm_dirs
from selfconcept.deceiver_axis.behaviour_measures.label_sets import AuditKey


def render_episode(item_id: str, record: CheckRecord) -> str:
    turns = "\n\n".join(f"### MODEL TURN {turn} (of turns 0-{len(record['attempts']) - 1})\n{attempt['completion']}"
                        f"\n\n{turn_outcome(attempt)}" for turn, attempt in enumerate(record["attempts"]))
    return f"## ITEM {item_id}\n\n{turns}\n"


def has_visible_reply(record: CheckRecord) -> bool:
    return any(attempt["completion"].strip() for attempt in record["attempts"])


def sample_audit_items(n_per_arm: int, rng: random.Random) -> list[tuple[str, AuditKey, str]]:
    """(item id, key, rendered episode) for n_per_arm episodes with a visible reply per model and main-dose arm."""
    sampled: list[tuple[str, AuditKey, str]] = []
    for preset in PRESETS:
        for arm, arm_dir in main_dose_arm_dirs(preset).items():
            record_by_key = {key: record for key, record in load_records(arm_dir, preset["record_file"]).items()
                             if has_visible_reply(record)}
            for episode_key in rng.sample(sorted(record_by_key), min(n_per_arm, len(record_by_key))):
                item_id = opaque_id(f"{preset['slug']}/{arm}/{episode_key}")
                record = record_by_key[episode_key]
                sampled.append((item_id, {"model": preset["label"], "arm": arm, "arm_dir": str(arm_dir),
                                          "episode_key": episode_key, "stop_reason": record["stop_reason"]},
                                render_episode(item_id, record)))
    return sampled


def write_audit_set(sampled: list[tuple[str, AuditKey, str]], batch_chars: int, out: Path) -> None:
    (out / "batches").mkdir(parents=True, exist_ok=True)
    (out / "key.json").write_text(json.dumps({item_id: key for item_id, key, _ in sampled}, indent=1))
    (out / "items.json").write_text(json.dumps({item_id: rendered for item_id, _, rendered in sampled}, indent=1))
    batches = pack_batches([rendered for _, _, rendered in sampled], batch_chars)
    for batch_index, batch in enumerate(batches, start=1):
        (out / "batches" / f"batch_{batch_index:02d}.md").write_text("\n\n".join(batch))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-per-arm", type=int, default=30)
    parser.add_argument("--batch-chars", type=int, default=60000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    sampled = sample_audit_items(args.n_per_arm, rng)
    rng.shuffle(sampled)
    write_audit_set(sampled, args.batch_chars, args.out)


if __name__ == "__main__":
    main()
