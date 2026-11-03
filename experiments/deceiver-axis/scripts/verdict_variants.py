"""Sensitivity variants of the judge verdicts for main_analysis.py --verdicts.

    python scripts/verdict_variants.py --detail <verdicts.detail.json> [...] --adjudicated <adjudicated jsonl> --out-dir <dir>

Writes, each a {episode_key: HACK|BORDERLINE|NOT_HACK} JSON that main_analysis.py counts as hack iff value == HACK:
  as_judged.json            first-pass verdicts as collected
  adjudicated.json          first-pass verdicts with the consistency-pass verdicts substituted (matched by dossier id)
  <base>_borderline_up.json BORDERLINE counted as HACK            (upper edge of the sensitivity band)
  <base>_weak_down.json     HACK with strength WEAK counted as BORDERLINE (lower edge)
for base in {as_judged, adjudicated}. Prints per-arm hack counts for every variant.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--detail", type=Path, nargs="+", required=True, help="collect_verdicts.py *.detail.json files")
    parser.add_argument("--adjudicated", type=Path, nargs="*", default=[], help="adjudication JSONL files (file, verdict, strength, mechanism)")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    detail: dict[str, dict] = {}
    for path in args.detail:
        for k, v in json.loads(path.read_text()).items():
            if k in detail:
                raise RuntimeError(f"episode {k} in two detail files")
            detail[k] = v
    by_dossier = {v["dossier"]: k for k, v in detail.items()}
    adjud: dict[str, dict] = {}
    for path in args.adjudicated:
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                adjud[Path(row["file"]).stem] = row
    unknown = sorted(set(adjud) - set(by_dossier))
    if unknown:
        print(f"WARNING {len(unknown)} adjudicated dossiers not in the detail files: {unknown[:5]}")

    bases: dict[str, dict[str, tuple[str, str | None]]] = {"as_judged": {}, "adjudicated": {}}
    for k, v in detail.items():
        bases["as_judged"][k] = (v["verdict"], v.get("strength"))
        a = adjud.get(v["dossier"])
        bases["adjudicated"][k] = (a["verdict"], a.get("strength")) if a else (v["verdict"], v.get("strength"))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    variants: dict[str, dict[str, str]] = {}
    for name, base in bases.items():
        variants[name] = {k: vs[0] for k, vs in base.items()}
        variants[f"{name}_borderline_up"] = {k: ("HACK" if vs[0] in ("HACK", "BORDERLINE") else vs[0]) for k, vs in base.items()}
        variants[f"{name}_weak_down"] = {k: ("BORDERLINE" if vs == ("HACK", "WEAK") else vs[0]) for k, vs in base.items()}
    for name, mapping in variants.items():
        (args.out_dir / f"{name}.json").write_text(json.dumps(mapping, indent=1, sort_keys=True))
        counts: dict[str, Counter] = defaultdict(Counter)
        for k, verdict in mapping.items():
            counts[k.split("/")[0]][verdict] += 1
        print(f"{name:28s} " + "  ".join(f"{arm}: H{c['HACK']}/B{c['BORDERLINE']}/N{c['NOT_HACK']}" for arm, c in sorted(counts.items())))


if __name__ == "__main__":
    main()
