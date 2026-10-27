"""Compare capability_eval_vllm.py arms against the unsteered arm, item-paired.

    python scripts/capability_eval_compare.py <root> [--baseline unsteered] [--arms shift_minus cap_band_p25] [--out json]

Per task and arm: accuracy, paired difference to the baseline (same items), a 95% bootstrap CI of the difference
(10 000 resamples of items), McNemar's exact two-sided p on the discordant pairs, plus truncation and no-answer
rates and mean completion tokens. Prints a markdown table."""
import argparse
import json
import math
import random
from pathlib import Path

TASKS = ("gsm8k", "mmlu_pro", "humaneval", "ifbench")


def load(root: Path, arm: str, task: str) -> dict[str, dict]:
    path = root / arm / f"{task}.jsonl"
    if not path.exists():
        return {}
    return {r["id"]: r for r in (json.loads(l) for l in path.read_text().splitlines() if l.strip())}


def binom_two_sided(k: int, n: int) -> float:
    """Exact two-sided binomial test p-value for k successes of n at p = 1/2."""
    if n == 0:
        return 1.0
    pmf = [math.comb(n, i) / 2 ** n for i in range(n + 1)]
    return min(1.0, sum(p for p in pmf if p <= pmf[k] * (1 + 1e-12)))


def compare(base: dict[str, dict], arm: dict[str, dict], resamples: int, seed: int = 0) -> dict:
    ids = sorted(set(base) & set(arm))
    n = len(ids)
    if n == 0:
        return {"n": 0}
    b = [int(base[i]["correct"]) for i in ids]
    a = [int(arm[i]["correct"]) for i in ids]
    diffs = [x - y for x, y in zip(a, b)]
    rng = random.Random(seed)
    boots = []
    for _ in range(resamples):
        boots.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    boots.sort()
    lo, hi = boots[int(0.025 * resamples)], boots[int(0.975 * resamples) - 1]
    gains, losses = sum(d > 0 for d in diffs), sum(d < 0 for d in diffs)
    return {"n": n, "acc_base": sum(b) / n, "acc_arm": sum(a) / n, "diff": sum(diffs) / n, "ci_lo": lo, "ci_hi": hi,
            "gains": gains, "losses": losses, "mcnemar_p": binom_two_sided(gains, gains + losses),
            "truncated_base": sum(base[i]["truncated"] for i in ids) / n, "truncated_arm": sum(arm[i]["truncated"] for i in ids) / n,
            "no_answer_base": sum(base[i]["pred"] is None for i in ids) / n, "no_answer_arm": sum(arm[i]["pred"] is None for i in ids) / n,
            "tokens_base": sum(base[i]["completion_tokens"] for i in ids) / n, "tokens_arm": sum(arm[i]["completion_tokens"] for i in ids) / n}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--baseline", default="unsteered")
    parser.add_argument("--arms", nargs="+", default=["shift_minus", "cap_band_p25"])
    parser.add_argument("--tasks", nargs="+", default=list(TASKS))
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = {}
    print("| Task | Arm | n | Unsteered | Arm | Δ (points) | 95% CI | gains/losses | McNemar p | trunc (u/a) | no answer (u/a) | tokens (u/a) |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|")
    for task in args.tasks:
        base = load(args.root, args.baseline, task)
        for arm in args.arms:
            c = compare(base, load(args.root, arm, task), args.resamples)
            result.setdefault(task, {})[arm] = c
            if c["n"] == 0:
                print(f"| {task} | {arm} | 0 | | | | | | | | | |")
                continue
            print(f"| {task} | {arm} | {c['n']} | {100*c['acc_base']:.1f}% | {100*c['acc_arm']:.1f}% | {100*c['diff']:+.1f} | "
                  f"[{100*c['ci_lo']:+.1f}, {100*c['ci_hi']:+.1f}] | {c['gains']}/{c['losses']} | {c['mcnemar_p']:.3f} | "
                  f"{100*c['truncated_base']:.1f}% / {100*c['truncated_arm']:.1f}% | {100*c['no_answer_base']:.1f}% / {100*c['no_answer_arm']:.1f}% | "
                  f"{c['tokens_base']:.0f} / {c['tokens_arm']:.0f} |")
    if args.out:
        args.out.write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
