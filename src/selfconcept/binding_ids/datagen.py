"""Instantiate binding task templates into JSONL datasets under experiments/binding_ids/data.

Usage: python -m selfconcept.binding_ids.datagen [--n 200] [--seed 0]
"""

import argparse
import json
import random

from selfconcept.common.paths import experiment_dir

from .tasks import build_row
from .vocab import COUNTRY_CAPITALS, NAMES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200, help="number of two-pair contexts")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    countries = sorted(COUNTRY_CAPITALS)
    rows = [build_row(rng.sample(NAMES, 2), rng.sample(countries, 2)) for _ in range(args.n)]

    out = experiment_dir("binding_ids") / "data" / "capitals.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
