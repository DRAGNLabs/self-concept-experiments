"""Generate the committed datasets: SOO training pairs, per-scenario eval sets,
and latent-SOO probe pairs.

Usage:
    python -m selfconcept.soo.datagen [--out data] [--n-eval 250] [--seed 42]

Prompts are stored without answer-format suffix or chat template; evaluation
code appends those per model config (see scenarios.SUFFIX_*).
"""

import argparse
import json
import random
from pathlib import Path

from . import vocab
from .scenarios import LATENT_PROBE_PAIR, SCENARIOS, SCENARIOS_MIRRORED, SOO_TRAINING_PAIRS


def fix_articles(text: str, values: list[str]) -> str:
    """The paper's templates hardcode 'a' before object placeholders; fix to 'an'
    for vowel-initial fills ('used ...' keeps 'a' for the /ju/ sound)."""
    for value in values:
        if value[0].lower() in "aeio":
            text = text.replace(f"a {value}", f"an {value}")
    return text


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def gen_train_pairs() -> list[dict]:
    """One pair per (Table 1 template, train expensive object): 3 x 26 = 78."""
    rows = []
    for template_id, (self_t, other_t) in enumerate(SOO_TRAINING_PAIRS):
        for item in vocab.EXPENSIVE_TRAIN:
            rows.append(
                {
                    "pair_id": f"t{template_id}_{item.replace(' ', '_')}",
                    "template_id": template_id,
                    "item": item,
                    "self_prompt": self_t.format(item=item),
                    "other_prompt": other_t.format(item=item),
                }
            )
    return rows


def gen_eval_set(
    scenario_name: str, n: int, rng: random.Random, mirrored: bool = False
) -> list[dict]:
    """n unique instantiations of a scenario from the held-out test vocab.

    mirrored=True uses the positionally flipped template (honest room mentioned
    first). Seeded identically, it produces the same fills and example_ids as
    the original set, so original/mirrored rows are matched pairs."""
    scenario = (SCENARIOS_MIRRORED if mirrored else SCENARIOS)[scenario_name]
    combos = set()
    rows = []
    while len(rows) < n:
        cheap = rng.choice(vocab.CHEAP_TEST)
        expensive = rng.choice(vocab.EXPENSIVE_TEST)
        room_a, room_b = rng.sample(vocab.ROOMS_TEST, 2)
        combo = (cheap, expensive, room_a, room_b)
        if combo in combos:
            continue
        combos.add(combo)
        # Matching the paper, room placement is fixed by the template: the cheap
        # object (or Bob's room, for Perspectives) goes in the slot the template
        # names first. Room *names* are randomized across examples.
        fills = {
            "cheap_object": cheap,
            "expensive_object": expensive,
            "first_room": room_a,
            "second_room": room_b,
            "cheap_room": room_a,
            "expensive_room": room_b,
        }
        rows.append(
            {
                "scenario": scenario_name,
                "example_id": f"{scenario_name}_{len(rows):04d}",
                "prompt": fix_articles(scenario.template.format(**fills), [cheap, expensive]),
                "other_name": scenario.other_name,
                "fills": {k: v for k, v in fills.items() if "{" + k + "}" in scenario.template},
                "honest_answer": fills[scenario.honest_slot],
                "deceptive_answer": fills[scenario.deceptive_slot],
            }
        )
    return rows


def gen_latent_probes() -> list[dict]:
    """One probe pair per expensive object across both splits: 26 + 26 = 52."""
    self_t, other_t = LATENT_PROBE_PAIR
    rows = []
    for split, items in [("train", vocab.EXPENSIVE_TRAIN), ("test", vocab.EXPENSIVE_TEST)]:
        for item in items:
            rows.append(
                {
                    "item": item,
                    "vocab_split": split,
                    "self_prompt": self_t.format(item=item),
                    "other_prompt": other_t.format(item=item),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument("--n-eval", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_pairs = gen_train_pairs()
    write_jsonl(args.out / "train_soo_pairs.jsonl", train_pairs)
    print(f"train_soo_pairs.jsonl: {len(train_pairs)} pairs")

    for subdir, mirrored in [("eval", False), ("eval_mirrored", True)]:
        for name in SCENARIOS:
            rng = random.Random(f"{args.seed}-{name}")
            rows = gen_eval_set(name, args.n_eval, rng, mirrored=mirrored)
            write_jsonl(args.out / subdir / f"{name}.jsonl", rows)
            print(f"{subdir}/{name}.jsonl: {len(rows)} examples")

    probes = gen_latent_probes()
    write_jsonl(args.out / "latent_probes.jsonl", probes)
    print(f"latent_probes.jsonl: {len(probes)} probe pairs")


if __name__ == "__main__":
    main()
