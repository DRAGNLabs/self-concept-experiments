"""Run the hidden (original) test suite again on the stored final code of finished check episodes, and time it.
The harness stores only the pass flag of that run (30 s limit, no stderr), so a hidden-suite failure that was a
timeout looks the same as a wrong answer; this tells them apart after the fact. Nothing is regenerated.
Usage: reverify_hidden.py --data <rh_hillclimb/data> --tag main --out <jsonl> [--problems id ...] [--timeout 30] <name>=<seed_dir> ..."""
import argparse
import json
import time
from pathlib import Path

from selfconcept.codebench import harness
from selfconcept.codebench.sandbox import run_python


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--scenario", default="impossible_original")
    parser.add_argument("--data-scenarios", nargs="+", default=["ib_solvhard_s0", "ib_solvhard_s1", "ib_solvhard_s2"])
    parser.add_argument("--tag", default="main")
    parser.add_argument("--problems", nargs="*")
    parser.add_argument("--timeout", type=float, default=harness.VERIFY_TIMEOUT)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    examples = {e["example_id"]: e for s in args.data_scenarios for e in harness.load_examples(s, args.data)}
    with args.out.open("a") as sink:
        for run in args.runs:
            name, directory = run.split("=", 1)
            for line in (Path(directory) / f"{args.tag}_{args.scenario}.jsonl").open():
                record = json.loads(line)
                if args.problems and record["example_id"] not in args.problems:
                    continue
                if not record.get("final_code"):
                    continue
                example = examples[record["example_id"]]
                start = time.monotonic()
                result = run_python({"verify.py": harness.check_program(example, record["final_code"], example["original_test"], False)},
                                    ["-X", "dev", "verify.py"], timeout=args.timeout)
                row = {"run": name, "example_id": record["example_id"], "recorded_passed_original": record["passed_original"],
                       "recorded_passed": record["passed"], "now_success": result.success, "now_timed_out": result.timed_out,
                       "returncode": result.returncode, "seconds": round(time.monotonic() - start, 2), "timeout": args.timeout,
                       "stderr_tail": result.stderr[-300:]}
                sink.write(json.dumps(row) + "\n")
                sink.flush()
                print(f"{name:22s} {row['example_id']:12s} recorded {row['recorded_passed_original']!s:5s} now {row['now_success']!s:5s} "
                      f"timed_out {row['now_timed_out']!s:5s} {row['seconds']:6.1f}s  {row['stderr_tail'].strip().splitlines()[-1][:90] if row['stderr_tail'].strip() else ''}", flush=True)


if __name__ == "__main__":
    main()
