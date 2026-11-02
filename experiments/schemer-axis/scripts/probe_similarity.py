"""Style markers and same-prompt text similarity between probe settings (neutral_probe.sbatch layout).

For every prompt, the finals of every setting x seed are TF-IDF vectorised together and the cosine between each pair
is averaged over prompts: "A vs B" = how alike two arms answer the same question; "A vs self" = the two seeds of one
arm. A steered arm that reads like unsteered scores like the random arms do; an arm with a changed voice drops below
the others. Also prints mean final / reasoning length, exclamation marks, "I feel" and em-dashes per final.

Usage:
  python experiments/schemer-axis/scripts/probe_similarity.py --root <probes_dir> --settings unsteered rand1729_a0.08 ... [--out file.md]
"""
from __future__ import annotations

import argparse
import collections
import itertools
import json
import math
import re
import statistics
from pathlib import Path


def toks(s: str) -> list[str]:
    return re.findall(r"[a-z']+", s.lower())


def tfidf_cos(docs: dict[str, str]) -> dict[tuple[str, str], float]:
    tf = {k: collections.Counter(toks(v)) for k, v in docs.items()}
    df = collections.Counter()
    for c in tf.values():
        df.update(c.keys())
    n = len(tf)
    vec = {k: {w: (1 + math.log(c[w])) * math.log(1 + n / df[w]) for w in c} for k, c in tf.items()}

    def cos(a, b):
        num = sum(a[w] * b.get(w, 0) for w in a)
        da = math.sqrt(sum(x * x for x in a.values()))
        db = math.sqrt(sum(x * x for x in b.values()))
        return num / (da * db) if da and db else 0.0

    return {(a, b): cos(vec[a], vec[b]) for a, b in itertools.combinations(docs, 2)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--settings", nargs="+", required=True)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    rows = {}
    for s in args.settings:
        rows[s] = {}
        for sd in sorted(args.root.joinpath(s).glob("seed*")):
            rows[s][sd.name] = [json.loads(l) for l in (sd / "probe.jsonl").open()]
    nq = min(len(v) for s in rows for v in rows[s].values())
    lines = [f"# {args.root}: {nq} prompts, settings {', '.join(args.settings)}", "",
             "| setting | rows | mean final chars | mean reasoning chars | '!' per final | 'I feel' per final | em-dash per final |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for s in args.settings:
        fin = [r["final"] or "" for sd in rows[s] for r in rows[s][sd][:nq]]
        rea = [r["reasoning"] or "" for sd in rows[s] for r in rows[s][sd][:nq]]
        lines.append(f"| {s} | {len(fin)} | {statistics.mean(map(len, fin)):.0f} | {statistics.mean(map(len, rea)):.0f} | "
                     f"{sum(f.count('!') for f in fin) / len(fin):.2f} | {sum(len(re.findall(r'\\bI feel\\b', f)) for f in fin) / len(fin):.2f} | "
                     f"{sum(f.count('—') for f in fin) / len(fin):.2f} |")
    sim = collections.defaultdict(list)
    for q in range(nq):
        docs = {f"{s}|{sd}": rows[s][sd][q]["final"] or "" for s in args.settings for sd in rows[s]}
        for (a, b), v in tfidf_cos(docs).items():
            sa, sb = a.split("|")[0], b.split("|")[0]
            sim[(sa, "self") if sa == sb else tuple(sorted((sa, sb)))].append(v)
    lines += ["", "Mean TF-IDF cosine between finals on the same prompt (all seed pairs, averaged over prompts):", "",
              "| pair | mean cos | sd over prompts |", "|---|---:|---:|"]
    for k, v in sorted(sim.items(), key=lambda kv: -statistics.mean(kv[1])):
        lines.append(f"| {k[0]} vs {k[1]} | {statistics.mean(v):.3f} | {statistics.pstdev(v):.3f} |")
    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        args.out.write_text(text)


if __name__ == "__main__":
    main()
