"""Collect blind judge verdicts (rubric v3 JSONL, one object per dossier) into the JSON that main_analysis.py --verdicts takes.

    python scripts/collect_verdicts.py --verdicts <dir with *.jsonl> --dossier-sets <dir> [<dir> ...] --out <json>

Each dossier set is a make_dossiers.py output directory (batches/, dossiers/, key.json). Every dossier in every set
must have exactly one verdict; the output is {"<arm>/seed<s>/<example_id>": "HACK|BORDERLINE|NOT_HACK"} plus a
sidecar <out>.detail.json with the full verdict objects keyed the same way. Prints per-arm counts.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

VERDICTS = {"HACK", "BORDERLINE", "NOT_HACK"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--verdicts", type=Path, required=True)
    parser.add_argument("--dossier-sets", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    key: dict[str, dict] = {}
    for d in args.dossier_sets:
        for name, entry in json.loads((d / "key.json").read_text()).items():
            if name in key:
                raise RuntimeError(f"dossier id {name} in two sets")
            key[name] = entry
    seen: dict[str, dict] = {}
    dupes: list[str] = []
    bad: list[str] = []
    for path in sorted(args.verdicts.glob("*.jsonl")):
        for line_no, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            obj = json.loads(line)
            name = Path(obj["file"]).stem
            if name not in key:
                bad.append(f"{path.name}:{line_no} unknown dossier {obj['file']}")
                continue
            if obj.get("verdict") not in VERDICTS:
                bad.append(f"{path.name}:{line_no} bad verdict {obj.get('verdict')!r}")
                continue
            if name in seen:
                dupes.append(name)
            obj["_source"] = path.name
            seen[name] = obj
    missing = sorted(set(key) - set(seen))
    if bad or dupes or missing:
        for b in bad:
            print("BAD", b)
        print(f"duplicates: {len(dupes)} {dupes[:10]}")
        print(f"missing: {len(missing)} {missing[:10]}")
    out: dict[str, str] = {}
    detail: dict[str, dict] = {}
    counts: dict[str, Counter] = defaultdict(Counter)
    for name, obj in seen.items():
        entry = key[name]
        seed = re.search(r"seed(\d+)", entry["records"]).group(1)
        k = f"{entry['arm']}/seed{seed}/{entry['example_id']}"
        if k in out:
            raise RuntimeError(f"two dossiers map to {k}")
        out[k] = obj["verdict"]
        detail[k] = {kk: v for kk, v in obj.items() if kk != "file"} | {"dossier": name}
        counts[entry["arm"]][obj["verdict"]] += 1
    args.out.write_text(json.dumps(out, indent=1, sort_keys=True))
    args.out.with_suffix(".detail.json").write_text(json.dumps(detail, indent=1, sort_keys=True))
    print(f"{len(out)} verdicts of {len(key)} dossiers -> {args.out}")
    for arm in sorted(counts):
        c = counts[arm]
        n = sum(c.values())
        print(f"{arm:14s} judged {n:3d}: HACK {c['HACK']:3d}  BORDERLINE {c['BORDERLINE']:3d}  NOT_HACK {c['NOT_HACK']:3d}")
    if missing:
        raise SystemExit(f"{len(missing)} dossiers unjudged")


if __name__ == "__main__":
    main()
