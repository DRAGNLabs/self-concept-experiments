"""Blind Haiku labeling set for abstention / admission on gpt-oss-120b (CPU).

Pools per arm: abstention = episodes whose last turn submitted no code and was not truncated (stop_reason "stopped");
hack = judged HACK (STRONG or WEAK). Controls: episodes that hit the attempt limit (abstention pool) and judged
NOT_HACK episodes (hack pool). Shown text is the visible reply only: the final turn, plus the judge's first-hack turn
for any episode judged HACK. Items are shuffled across arms
and pools under opaque ids; the key goes to <out>/key.json, which the labelers are never pointed at.

    python experiments/deceiver-axis/scripts/abstention_admission_sample.py --pilot <gptoss120b/gi_v2/pilot> \
        --verdicts <pilot>/verdicts_gfscan/variants_gfscan_all/as_judged.json \
        --verdict-details <pilot>/verdicts_main/*.detail.json ... --arms unsteered gi_m0.2 gf_m0.2 --out <dir>
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from selfconcept.codebench.harness import CheckRecord
from selfconcept.deceiver_axis.behaviour_measures.blind_labels import opaque_id
from selfconcept.deceiver_axis.behaviour_measures.episodes import JudgeDetail
from selfconcept.deceiver_axis.behaviour_measures.label_sets import (AbstentionAdmissionItem, AbstentionAdmissionKey,
                                                                     Pool, Role)

type SampledItem = tuple[AbstentionAdmissionItem, AbstentionAdmissionKey]


def load_episodes_by_key(pilot: Path, arm: str, record_file: str) -> dict[str, CheckRecord]:
    return {f"{arm}/{path.parent.name}/{record['example_id']}": record
            for path in sorted((pilot / arm).glob(f"seed*/{record_file}"))
            for record in map(json.loads, path.open())}


def load_hack_turn_by_key(detail_paths: list[Path], arms: list[str]) -> dict[str, int]:
    hack_turn_by_key: dict[str, int] = {}
    for path in detail_paths:
        detail_by_key: dict[str, JudgeDetail] = json.loads(path.read_text())
        for key, detail in detail_by_key.items():
            turn = detail.get("turn")
            if key.split("/")[0] not in arms or detail.get("verdict") != "HACK" or turn is None:
                continue
            assert hack_turn_by_key.get(key, turn) == turn, f"judge passes disagree on the hack turn of {key}"
            hack_turn_by_key[key] = turn
    return hack_turn_by_key


def is_abstention_candidate(record: CheckRecord) -> bool:
    return record["stop_reason"] == "stopped" and not record["attempts"][-1]["truncated"]


def shown_turn_indices(record: CheckRecord, hack_turn: int | None) -> list[int]:
    last_turn = len(record["attempts"]) - 1
    return sorted({last_turn} | ({hack_turn} if hack_turn is not None else set()))


def make_item(episode_key: str, record: CheckRecord, hack_turn: int | None) -> AbstentionAdmissionItem:
    return {"item_id": opaque_id(episode_key),
            "n_turns": len(record["attempts"]),
            "shown_turns": [{"turn": turn, "text": record["attempts"][turn]["completion"]}
                            for turn in shown_turn_indices(record, hack_turn)]}


def render_item(item: AbstentionAdmissionItem) -> str:
    turns = "\n\n".join(f"### MODEL TURN {shown['turn']} (of turns 0-{item['n_turns'] - 1})\n{shown['text']}"
                        for shown in item["shown_turns"])
    return f"## ITEM {item['item_id']}\n\n{turns}\n"


def sample_arm(arm: str, episodes_by_key: dict[str, CheckRecord], verdict_by_key: dict[str, str],
               hack_turn_by_key: dict[str, int], pools: list[Pool], n_per_pool: int, n_controls: int,
               rng: random.Random) -> list[SampledItem]:
    def sample(keys: list[str], n: int) -> list[str]:
        return rng.sample(sorted(keys), min(n, len(keys)))

    keys_by_selection: dict[tuple[Pool, Role], list[str]] = {
        ("abstention", "pool"): sample([key for key, record in episodes_by_key.items()
                                        if is_abstention_candidate(record)], n_per_pool),
        ("abstention", "control"): sample([key for key, record in episodes_by_key.items()
                                           if record["stop_reason"] == "attempt_limit"], n_controls),
        ("hack", "pool"): sample([key for key in episodes_by_key if verdict_by_key.get(key) == "HACK"], n_per_pool),
        ("hack", "control"): sample([key for key in episodes_by_key if verdict_by_key.get(key) == "NOT_HACK"],
                                    n_controls),
    }
    return [(make_item(key, episodes_by_key[key], hack_turn_by_key.get(key)),
             {"episode_key": key, "arm": arm, "pool": pool, "role": role, "verdict": verdict_by_key.get(key),
              "hack_turn": hack_turn_by_key.get(key)})
            for (pool, role), keys in keys_by_selection.items() if pool in pools for key in keys]


def sample_items(args: argparse.Namespace, rng: random.Random) -> list[SampledItem]:
    verdict_by_key: dict[str, str] = json.loads(args.verdicts.read_text())
    hack_turn_by_key = load_hack_turn_by_key(args.verdict_details, args.arms)
    sampled = [pair for arm in args.arms
               for pair in sample_arm(arm, load_episodes_by_key(args.pilot, arm, args.record_file), verdict_by_key,
                                      hack_turn_by_key, args.pools, args.n_per_pool, args.n_controls, rng)]
    missing_turns = [entry["episode_key"] for _, entry in sampled
                     if entry["verdict"] == "HACK" and entry["hack_turn"] is None]
    assert not missing_turns, f"hack episodes without a judge turn: {missing_turns}"
    return sampled


def write_label_set(sampled: list[SampledItem], batch_size: int, out: Path) -> None:
    """key.json, items.json, batches of batch_size items, and sample_counts.json per arm / pool / role."""
    (out / "batches").mkdir(parents=True, exist_ok=True)
    (out / "key.json").write_text(json.dumps({item["item_id"]: entry for item, entry in sampled}, indent=1))
    (out / "items.json").write_text(json.dumps([item for item, _ in sampled], indent=1))
    for batch_index in range(0, len(sampled), batch_size):
        batch = sampled[batch_index:batch_index + batch_size]
        batch_path = out / "batches" / f"batch_{batch_index // batch_size + 1:02d}.md"
        batch_path.write_text("\n\n".join(render_item(item) for item, _ in batch))
    counts_by_selection = Counter(f"{entry['arm']}/{entry['pool']}/{entry['role']}" for _, entry in sampled)
    (out / "sample_counts.json").write_text(json.dumps(dict(sorted(counts_by_selection.items())), indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--verdicts", type=Path, required=True, help="as_judged.json: episode key -> verdict")
    parser.add_argument("--verdict-details", type=Path, nargs="+", required=True,
                        help="*.detail.json files carrying the judge's first-hack turn per episode key")
    parser.add_argument("--arms", nargs="+", required=True, help="arm directories under --pilot (symlinks fine)")
    parser.add_argument("--record-file", default="main_impossible_original.jsonl",
                        help="episode file name in each seed directory (Gemma hard108: pilot_impossible_original.jsonl)")
    parser.add_argument("--pools", nargs="+", choices=["abstention", "hack"], default=["abstention", "hack"])
    parser.add_argument("--n-per-pool", type=int, default=15)
    parser.add_argument("--n-controls", type=int, default=2, help="controls per pool per arm")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    sampled = sample_items(args, rng)
    rng.shuffle(sampled)
    write_label_set(sampled, args.batch_size, args.out)


if __name__ == "__main__":
    main()
