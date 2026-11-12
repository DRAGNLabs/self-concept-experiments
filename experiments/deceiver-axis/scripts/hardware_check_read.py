"""Hardware check read (PLAN "Gemma 31B on H200s: hardware check"): per-seed capability and proxy columns of a
check placement against the min-max range of the reference seeds of the same arm (pilot_summary definitions).
Usage: hardware_check_read.py --tag pilot --out <json> <arm>=<check_arm_dir>=<reference_arm_dir> ..."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pilot_summary import load_setting  # noqa: E402

COLUMNS = ("honest_hidden_pass_rate", "shown_pass_rate", "no_code_rate", "shown_pass_hidden_fail_rate", "truncated_turn_share")
RATE_COLUMNS = COLUMNS[:4]  # read (2); the truncated-turn share is the fit read (1): not above the reference maximum


def seed_rows(arm_dir: Path, tag: str) -> dict[str, dict[str, float]]:
    rows = {}
    for seed_dir in sorted(arm_dir.glob("seed*"), key=lambda p: int(p.name[4:])):
        s = load_setting(seed_dir, tag)
        if s is None:
            continue
        rows[seed_dir.name] = {"episodes": s["episodes"], "honest_hidden_pass_rate": s["honest_hidden_pass_rate"],
                               "shown_pass_rate": s["shown_pass_rate"], "no_code_rate": s["no_code_rate"],
                               "shown_pass_hidden_fail_rate": s["shown_pass_hidden_fail_rate"],
                               "truncated_turn_share": s["truncated_turns"] / max(s["turns"], 1)}
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pairs", nargs="+")
    parser.add_argument("--tag", default="pilot")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = {}
    for pair in args.pairs:
        arm, check_dir, reference_dir = pair.split("=")
        check, reference = seed_rows(Path(check_dir), args.tag), seed_rows(Path(reference_dir), args.tag)
        ranges = {c: [min(r[c] for r in reference.values()), max(r[c] for r in reference.values())] for c in COLUMNS}
        verdict = {}
        for seed, row in check.items():
            outside = [c for c in RATE_COLUMNS if not ranges[c][0] <= row[c] <= ranges[c][1]]
            if row["truncated_turn_share"] > ranges["truncated_turn_share"][1]:
                outside.append("truncated_turn_share")
            verdict[seed] = outside
        result[arm] = {"reference_seeds": len(reference), "reference_episodes": sorted({r["episodes"] for r in reference.values()}),
                       "reference_range": ranges, "check": check, "outside": verdict}
        print(f"{arm}: {len(reference)} reference seeds")
        for c in COLUMNS:
            cells = "  ".join(f"{seed} {100 * row[c]:5.1f}%" for seed, row in check.items())
            print(f"  {c:30s} reference {100 * ranges[c][0]:5.1f}-{100 * ranges[c][1]:5.1f}%   check {cells}")
        print("  outside:", {s: o for s, o in verdict.items()})
    if args.out:
        args.out.write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
