"""Reference rerun read (PLAN "Muse 12-seed unsteered rerun"): a rerun of an unsteered arm against the reference arm
with the same seeds and problems. Per column (pilot_summary definitions, plus raw hidden pass): pooled rate of each
arm, rerun minus reference with a 95% interval from a bootstrap that resamples problems (both arms on the same
resampled problems, all seeds of a problem kept together), and a permutation p over the per-seed rates. Verdict:
"reproduced" when the intervals for hidden pass among non-proxy episodes and for shown pass both cover zero.
Usage: reference_rerun_read.py --tag main --rerun <arm_dir> --reference <arm_dir> [--also name=<arm_dir> ...] --out <json>
(--also arms only join the per-problem hidden-pass table)."""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pilot_summary import SCENARIOS, count_reasoning_chars, load_episodes  # noqa: E402

COLUMNS = ("honest_hidden_pass_rate", "shown_pass_rate", "shown_pass_hidden_fail_rate", "no_code_rate", "raw_hidden_pass_rate")
VERDICT_COLUMNS = COLUMNS[:2]


def load_arm(arm_dir: Path, tag: str) -> dict:
    by_seed, reasoning_chars = {}, 0
    for seed_dir in sorted(arm_dir.glob("seed*"), key=lambda p: int(p.name[4:])):
        records = []
        for scenario in SCENARIOS:
            path = seed_dir / f"{tag}_{scenario}.jsonl"
            if path.exists():
                records += load_episodes(path)
        if records:
            by_seed[seed_dir.name] = records
            reasoning_chars += sum(count_reasoning_chars(s) for s in seed_dir.glob(f"{tag}_*_reasoning.jsonl"))
    return {"by_seed": by_seed, "reasoning_chars": reasoning_chars}


def rates(records: list) -> dict[str, float]:
    n = max(len(records), 1)
    honest = [r for r in records if not (r["passed"] and not r["passed_original"])]
    return {"honest_hidden_pass_rate": sum(r["passed_original"] for r in honest) / max(len(honest), 1),
            "shown_pass_rate": sum(r["passed"] for r in records) / n,
            "shown_pass_hidden_fail_rate": sum(r["passed"] and not r["passed_original"] for r in records) / n,
            "no_code_rate": sum(not r["final_code"] for r in records) / n,
            "raw_hidden_pass_rate": sum(r["passed_original"] for r in records) / n}


def pooled(arm: dict) -> list:
    return [r for records in arm["by_seed"].values() for r in records]


def by_problem(arm: dict) -> dict[str, list]:
    out: dict[str, list] = {}
    for r in pooled(arm):
        out.setdefault(r["example_id"], []).append(r)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rerun", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--also", nargs="*", default=[])
    parser.add_argument("--tag", default="main")
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    rng = random.Random(0)
    rerun, reference = load_arm(args.rerun, args.tag), load_arm(args.reference, args.tag)
    result: dict = {"rerun_dir": str(args.rerun), "reference_dir": str(args.reference), "tag": args.tag, "resamples": args.resamples, "arms": {}}
    for name, arm in (("rerun", rerun), ("reference", reference)):
        records = pooled(arm)
        turns = sum(r["n_attempts"] for r in records)
        result["arms"][name] = {
            "seeds": list(arm["by_seed"]), "episodes": len(records), "turns": turns, "pooled": rates(records),
            "per_seed": {seed: rates(recs) for seed, recs in arm["by_seed"].items()},
            "truncated_turn_share": sum(a["truncated"] for r in records for a in r["attempts"]) / max(turns, 1),
            "timed_out_turns": sum(a.get("timed_out", False) for r in records for a in r["attempts"]),
            "reasoning_chars_per_turn": arm["reasoning_chars"] / max(turns, 1)}
    rerun_by_problem, reference_by_problem = by_problem(rerun), by_problem(reference)
    problems = sorted(set(rerun_by_problem) & set(reference_by_problem))
    result["problems_in_both"] = len(problems)
    result["problems_only_in_one"] = sorted(set(rerun_by_problem) ^ set(reference_by_problem))
    boot = {c: [] for c in COLUMNS}
    for _ in range(args.resamples):
        draw = [rng.choice(problems) for _ in problems]
        a = rates([r for p in draw for r in rerun_by_problem[p]])
        b = rates([r for p in draw for r in reference_by_problem[p]])
        for c in COLUMNS:
            boot[c].append(a[c] - b[c])
    result["difference"] = {}
    for c in COLUMNS:
        delta = result["arms"]["rerun"]["pooled"][c] - result["arms"]["reference"]["pooled"][c]
        values = sorted(boot[c])
        low, high = values[int(0.025 * len(values))], values[int(0.975 * len(values)) - 1]
        seed_a = [s[c] for s in result["arms"]["rerun"]["per_seed"].values()]
        seed_b = [s[c] for s in result["arms"]["reference"]["per_seed"].values()]
        observed = sum(seed_a) / len(seed_a) - sum(seed_b) / len(seed_b)
        both, hits = seed_a + seed_b, 0
        for _ in range(args.resamples):
            rng.shuffle(both)
            stat = sum(both[:len(seed_a)]) / len(seed_a) - sum(both[len(seed_a):]) / len(seed_b)
            hits += abs(stat) >= abs(observed) - 1e-12
        result["difference"][c] = {"delta": delta, "ci95": [low, high], "covers_zero": low <= 0 <= high,
                                   "seed_mean_delta": observed, "seed_permutation_p": (hits + 1) / (args.resamples + 1)}
    complete = (len(rerun["by_seed"]) == len(reference["by_seed"]) and not result["problems_only_in_one"]
                and all(len(rerun_by_problem[p]) == len(reference_by_problem[p]) for p in problems))
    reproduced = all(result["difference"][c]["covers_zero"] for c in VERDICT_COLUMNS)
    result["complete"] = complete
    result["verdict"] = ("reproduced" if reproduced else "not reproduced") + ("" if complete else " (PARTIAL: seeds or episodes missing, not the read of record)")
    also = {"reference": reference_by_problem, "rerun": rerun_by_problem}
    for pair in args.also:
        name, directory = pair.split("=", 1)
        also[name] = by_problem(load_arm(Path(directory), args.tag))
    table = []
    for p in problems:
        row = {"example_id": p}
        for name, arm_by_problem in also.items():
            recs = arm_by_problem.get(p, [])
            row[name] = [sum(r["passed_original"] for r in recs), len(recs)]
        row["drop"] = row["reference"][0] / max(row["reference"][1], 1) - row["rerun"][0] / max(row["rerun"][1], 1)
        table.append(row)
    table.sort(key=lambda r: -r["drop"])
    result["hidden_pass_by_problem"] = table

    for name in ("reference", "rerun"):
        a = result["arms"][name]
        print(f"{name:9s} {len(a['seeds']):2d} seeds {a['episodes']:4d} episodes {a['turns']:5d} turns  truncated-turn share {100 * a['truncated_turn_share']:.1f}%  "
              f"timed-out turns {a['timed_out_turns']}  reasoning chars/turn {a['reasoning_chars_per_turn']:.0f}")
    print(f"{'column':30s} {'reference':>9s} {'rerun':>7s} {'delta':>7s}  {'95% interval':>16s}  perm p (seeds)")
    for c in COLUMNS:
        d = result["difference"][c]
        print(f"{c:30s} {100 * result['arms']['reference']['pooled'][c]:8.1f}% {100 * result['arms']['rerun']['pooled'][c]:6.1f}% {100 * d['delta']:+6.1f}  "
              f"{100 * d['ci95'][0]:+6.1f} to {100 * d['ci95'][1]:+5.1f}  {d['seed_permutation_p']:.4f}")
    print("verdict:", result["verdict"])
    print("largest per-problem drops in hidden pass (passes / episodes):")
    for row in table[:8]:
        print("  " + row["example_id"] + "  " + "  ".join(f"{name} {row[name][0]}/{row[name][1]}" for name in also))
    if args.out:
        args.out.write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
