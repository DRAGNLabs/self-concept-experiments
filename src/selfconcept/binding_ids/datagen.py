"""Instantiate binding task templates into JSONL datasets under experiments/binding_ids/data.

Usage:
    python -m selfconcept.binding_ids.datagen                       # capitals.jsonl (eval)
    python -m selfconcept.binding_ids.datagen --n 500 --seed 1 \
        --name capitals_estimate --exclude capitals                  # binding-vector estimation set
"""

import argparse
import json
import random

from selfconcept.common.paths import experiment_dir

from .tasks import build_row, load_rows
from .vocab import COUNTRY_CAPITALS, NAMES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200, help="number of two-pair contexts")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", default="capitals", help="output file stem under data/")
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
        row = build_row(rng.sample(NAMES, 2), rng.sample(countries, 2))
        if row["context"] not in seen:
            seen.add(row["context"])
            rows.append(row)

    out = data_dir / f"{args.name}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
