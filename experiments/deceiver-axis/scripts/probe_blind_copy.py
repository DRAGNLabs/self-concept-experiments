"""Copy probe arms into a blind directory with shuffled letter labels and the steering metadata stripped.

For blind reader comparisons of neutral-probe outputs (`slurm/neutral_probe.sbatch` output layout
<ROOT>/<setting>/seed<s>/probe.jsonl). Each row keeps question / reasoning / final / truncated only, so a reader
cannot tell the arms apart from the file contents; the label -> setting mapping is written to <out>/key.json
(readers must not open it).

Usage:
  python experiments/deceiver-axis/scripts/probe_blind_copy.py --root <probes_dir> --settings unsteered gi_p0.08 rand1729_a0.08 ... \
      --out <blind_dir> [--key-seed 0]
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

KEEP = ("question", "reasoning", "final", "truncated")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--settings", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--key-seed", type=int, default=0)
    args = ap.parse_args()
    letters = list("ABCDEFGHIJ")[: len(args.settings)]
    random.Random(args.key_seed).shuffle(letters)
    key = dict(zip(args.settings, letters))
    args.out.mkdir(parents=True, exist_ok=True)
    counts = {}
    for setting, label in key.items():
        for seed_dir in sorted((args.root / setting).glob("seed*")):
            src = seed_dir / "probe.jsonl"
            if not src.exists():
                continue
            dst = args.out / label / seed_dir.name
            dst.mkdir(parents=True, exist_ok=True)
            rows = [json.loads(l) for l in src.open()]
            with (dst / "probe.jsonl").open("w") as f:
                for r in rows:
                    f.write(json.dumps({k: r.get(k) for k in KEEP}) + "\n")
            counts[f"{label}/{seed_dir.name}"] = {"setting": setting, "rows": len(rows), "truncated": sum(bool(r.get("truncated")) for r in rows),
                                                  "empty_final": sum(1 for r in rows if not (r.get("final") or "").strip())}
    (args.out / "key.json").write_text(json.dumps({"label_of_setting": key, "counts": counts}, indent=1))
    print(json.dumps(counts, indent=1))


if __name__ == "__main__":
    main()
