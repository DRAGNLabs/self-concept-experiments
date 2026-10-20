"""Instantiate binding task templates into JSONL datasets under experiments/binding_ids/data.

Usage:
    python -m selfconcept.binding_ids.datagen                       # capitals.jsonl (eval)
    python -m selfconcept.binding_ids.datagen --n 500 --seed 1 \
        --name capitals_estimate --exclude capitals                  # binding-vector estimation set
    python -m selfconcept.binding_ids.datagen --task animals         # a ROOMS task (see tasks.ROOM_TASKS)
    python -m selfconcept.binding_ids.datagen --task animals --n 500 --seed 1 \
        --name animals_estimate --exclude animals
    python -m selfconcept.binding_ids.datagen --task role_you_i --assistant you --n 100
        # a ROLE task (see tasks.ROLE_PAIRS); writes role_you_i_asst_you.jsonl
"""

import argparse
import json
import random

from selfconcept.common.paths import experiment_dir

from .tasks import ROLE_PAIRS, ROOM_TASKS, build_row, load_rows, sample_role_row, sample_room_row
from .vocab import COUNTRY_CAPITALS, NAMES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200, help="number of two-pair contexts")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--task", default="capitals", choices=["capitals", *ROOM_TASKS, *(f"role_{p}" for p in ROLE_PAIRS)]
    )
    parser.add_argument(
        "--assistant", default=None, choices=["you", "I", "name"],
        help="ROLE tasks: entity kind that gets the role 'assistant' (default: neither)",
    )
    parser.add_argument("--name", default=None, help="output file stem under data/ (default: the task name)")
    parser.add_argument("--exclude", default=None, help="dataset stem whose contexts must not be reused")
    args = parser.parse_args()

    data_dir = experiment_dir("binding_ids") / "data"
    seen = set()
    if args.exclude:
        seen = {row["context"] for row in load_rows(data_dir / f"{args.exclude}.jsonl")}

    rng = random.Random(args.seed)
    countries = sorted(COUNTRY_CAPITALS)
    rows = []
    while len(rows) < args.n:
        if args.task == "capitals":
            row = build_row(rng.sample(NAMES, 2), rng.sample(countries, 2))
        elif args.task.startswith("role_"):
            row = sample_role_row(args.task.removeprefix("role_"), args.assistant, rng, len(rows))
        else:
            row = sample_room_row(args.task, rng, len(rows))
        if row["context"] not in seen:
            seen.add(row["context"])
            rows.append(row)

    default_name = rows[0].get("task", args.task)  # ROLE rows carry the assistant placement
    out = data_dir / f"{args.name or default_name}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
