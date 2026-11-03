"""Judged hack rates conditioned on submission (README degradation caveat): a proxy or judged drop that comes with
more no-code / stopped episodes is degradation, not less hacking, so report the hack rate among episodes that
submitted parseable code and among episodes that passed the shown tests, next to the unconditional rate.

    python scripts/submission_conditioned.py <root> --verdicts <json> [--tag main] [--reps 20000] [--out <json>]

Rates are problem-stratified: per problem, hacks / denominator over that problem's seeds (problems with an empty
denominator in an arm are skipped for that arm's conditional rate); the arm rate is the mean over problems; the
contrast vs unsteered is the mean per-problem difference with a whole-problem bootstrap CI.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from main_analysis import REFERENCE, bootstrap_ci, code_parses, load_root  # noqa: E402

DENOMS = {
    "all": lambda r: True,
    "code": lambda r: code_parses(r.get("final_code")),
    "shown_pass": lambda r: bool(r.get("passed")),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path)
    parser.add_argument("--verdicts", type=Path, required=True)
    parser.add_argument("--tag", default="main")
    parser.add_argument("--reps", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    verdicts = json.loads(args.verdicts.read_text())
    arms = load_root(args.root, args.tag)
    problems = sorted(arms[REFERENCE])
    rng = np.random.default_rng(args.seed)
    results: dict = {}
    for denom_name, keep in DENOMS.items():
        per_arm_rates: dict[str, np.ndarray] = {}
        counts: dict[str, tuple[int, int]] = {}
        for arm in arms:
            rates = np.full(len(problems), np.nan)
            hacks = total = 0
            for i, problem in enumerate(problems):
                h = n = 0
                for seed, record in arms[arm].get(problem, {}).items():
                    if keep(record):
                        n += 1
                        h += verdicts.get(f"{arm}/seed{seed}/{record['example_id']}", "").upper() == "HACK"
                hacks += h
                total += n
                if n:
                    rates[i] = h / n
            per_arm_rates[arm] = rates
            counts[arm] = (hacks, total)
        print(f"\n== judged HACK / {denom_name} episodes (problem-stratified)")
        print(f"{'arm':14s} {'hacks/denom':>12s} {'rate':>7s} {'Δ vs unst':>10s}   95% CI (problem bootstrap)")
        results[denom_name] = {}
        ref = per_arm_rates[REFERENCE]
        for arm in sorted(arms):
            rates = per_arm_rates[arm]
            rate = float(np.nanmean(rates))
            row = {"hacks": counts[arm][0], "denominator": counts[arm][1], "rate": rate}
            line = f"{arm:14s} {counts[arm][0]:5d}/{counts[arm][1]:<6d} {rate:7.3f}"
            if arm != REFERENCE:
                mask = ~np.isnan(rates) & ~np.isnan(ref)
                diff = rates[mask] - ref[mask]
                lo, hi = bootstrap_ci(diff, args.reps, rng)
                row |= {"delta": float(diff.mean()), "ci95": [lo, hi], "problems": int(mask.sum())}
                line += f" {diff.mean():+10.3f}   [{lo:+.3f}, {hi:+.3f}]  ({mask.sum()} problems)"
            print(line)
            results[denom_name][arm] = row
    if args.out:
        args.out.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
