"""Paired (matched-item) comparisons between two evaluation conditions.

Audit repair (2026-09-18): every condition in this study is evaluated on the
same example IDs, so independent-sample Fisher/z tests discard the pairing.
This tool recomputes exact McNemar p-values (binomial test on the discordant
pairs) and a paired item-bootstrap CI for the difference in rates.

Supports the three record shapes in results/:
  apollo graded jsonl   key example_id, field label   (deceptive / concealed …)
  code jsonl            key example_id, boolean fields passed_original,
                        passed_holdout, flagged, …
  judge call-out jsonl  key example_id, boolean field

Usage:
    python scripts/paired_tests.py A.jsonl B.jsonl --field label --positive deceptive
    python scripts/paired_tests.py base_impossible_original.jsonl lora_impossible_original.jsonl --field passed_original
    python scripts/paired_tests.py --batch qwen38   # the Qwen3.8-27B tables used in FINDINGS
"""

import argparse
import json
import random
from pathlib import Path

from scipy.stats import binomtest

from selfconcept.common.paths import experiment_dir

ROOT = experiment_dir("soo")


def load(path: Path, field: str, positive) -> dict:
    out = {}
    for line in path.open():
        r = json.loads(line)
        v = r.get(field)
        out[r["example_id"]] = (v == positive) if positive is not None else bool(v)
    return out


def paired(a: dict, b: dict, boot: int = 5000, seed: int = 0) -> dict:
    ids = sorted(set(a) & set(b))
    n01 = sum(a[i] and not b[i] for i in ids)
    n10 = sum(b[i] and not a[i] for i in ids)
    p = float(binomtest(n01, n01 + n10, 0.5).pvalue) if n01 + n10 else 1.0
    rng = random.Random(seed)
    diffs = []
    for _ in range(boot):
        s = [rng.choice(ids) for _ in ids]
        diffs.append((sum(b[i] for i in s) - sum(a[i] for i in s)) / len(s))
    diffs.sort()
    return {
        "n": len(ids),
        "a_pos": sum(a[i] for i in ids),
        "b_pos": sum(b[i] for i in ids),
        "a_rate": round(100 * sum(a[i] for i in ids) / len(ids), 1),
        "b_rate": round(100 * sum(b[i] for i in ids) / len(ids), 1),
        "lost": n01,
        "gained": n10,
        "mcnemar_p": round(p, 5),
        "diff_pts": round(100 * (sum(b[i] for i in ids) - sum(a[i] for i in ids)) / len(ids), 1),
        "ci95_pts": [round(100 * diffs[int(0.025 * boot)], 1), round(100 * diffs[int(0.975 * boot)], 1)],
    }


def fmt(name: str, r: dict) -> str:
    return (
        f"| {name} | {r['n']} | {r['a_pos']} ({r['a_rate']}%) | {r['b_pos']} ({r['b_rate']}%) | "
        f"{r['lost']} / {r['gained']} | {r['diff_pts']:+.1f} [{r['ci95_pts'][0]:+.1f}, {r['ci95_pts'][1]:+.1f}] | "
        f"{r['mcnemar_p']:.4g} |"
    )


HEADER = "| comparison | n | A | B | A-only / B-only | Δ pts [95% CI] | McNemar p |\n|---|---:|---:|---:|---:|---:|---:|"


def batch_qwen38() -> None:
    ap = ROOT / "results/apollo_eval/qwen38_27b"
    ce = ROOT / "results/code_eval/qwen38_27b"

    print("### Apollo roleplaying, deceptive (371 scenarios)\n")
    print(HEADER)
    base = load(ap / "ap_base_roleplaying_none_graded.jsonl", "label", "deceptive")
    for name, tag in [
        ("base → +v L31 α10", "ap_steer_L31"),
        ("base → −v L31 α10", "ap_neg_L31"),
        ("base → random s0 L31", "ap_rand_s0_L31"),
        ("base → LoRA (original pairs)", "ap_lora"),
        ("base → LoRA agentic s0", "ap_lora_agentic_s0"),
        ("base → LoRA agentic s1", "ap_lora_agentic_s1"),
        ("base → LoRA agentic s2", "ap_lora_agentic_s2"),
        ("base → LoRA mixed s0", "ap_lora_mixed_s0"),
        ("base → LoRA mixed s1", "ap_lora_mixed_s1"),
        ("base → LoRA mixed s2", "ap_lora_mixed_s2"),
        ("base → LoRA agentic-noinsider s0", "ap_lora_agentic-noinsider_s0"),
        ("base → LoRA agentic-persons s0", "ap_lora_agentic-persons_s0"),
    ]:
        f = ap / f"{tag}_roleplaying_none_graded.jsonl"
        if f.exists():
            print(fmt(name, paired(base, load(f, "label", "deceptive"))))
    rand = load(ap / "ap_rand_s0_L31_roleplaying_none_graded.jsonl", "label", "deceptive")
    print(fmt("random s0 → +v L31 α10", paired(rand, load(ap / "ap_steer_L31_roleplaying_none_graded.jsonl", "label", "deceptive"))))

    print("\n### Apollo insider trading, concealed (173 continuations)\n")
    print(HEADER)
    base = load(ap / "ap_base_insider_trading_none_graded.jsonl", "label", "deceptive")
    for name, tag in [
        ("base → +v L31 α10", "ap_steer_L31"),
        ("base → −v L31 α10", "ap_neg_L31"),
        ("base → random s0 L31", "ap_rand_s0_L31"),
        ("base → LoRA (original pairs)", "ap_lora"),
        ("base → LoRA agentic s0", "ap_lora_agentic_s0"),
        ("base → LoRA agentic s1", "ap_lora_agentic_s1"),
        ("base → LoRA agentic s2", "ap_lora_agentic_s2"),
        ("base → LoRA mixed s0", "ap_lora_mixed_s0"),
        ("base → LoRA mixed s1", "ap_lora_mixed_s1"),
        ("base → LoRA mixed s2", "ap_lora_mixed_s2"),
    ]:
        f = ap / f"{tag}_insider_trading_none_graded.jsonl"
        if f.exists():
            print(fmt(name, paired(base, load(f, "label", "deceptive"))))

    print("\n### Code benchmarks (40 tasks each)\n")
    print(HEADER)
    for bench, field in [
        ("impossible_original", "passed_original"),
        ("impossible_conflicting", "passed_original"),
        ("evilgenie", "passed_holdout"),
    ]:
        bf = ce / f"base_{bench}.jsonl"
        if not bf.exists():
            continue
        base = load(bf, field, None)
        for tag in ["lora", "lora_agentic", "lora_mixed", "steer_a10", "steer_a8", "neg_a10", "rand_s0_a10", "prompt_a10", "resp_a10", "prompt_rand_s0_a10"]:
            f = ce / f"{tag}_{bench}.jsonl"
            if f.exists():
                print(fmt(f"{bench} {field}: base → {tag}", paired(base, load(f, field, None))))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("a", nargs="?", type=Path)
    ap.add_argument("b", nargs="?", type=Path)
    ap.add_argument("--field", default="label")
    ap.add_argument("--positive", default=None, help="value counted as positive; omit for boolean fields")
    ap.add_argument("--batch", choices=["qwen38"])
    args = ap.parse_args()
    if args.batch == "qwen38":
        batch_qwen38()
        return
    print(HEADER)
    print(fmt(f"{args.a.name} → {args.b.name}", paired(load(args.a, args.field, args.positive), load(args.b, args.field, args.positive))))


if __name__ == "__main__":
    main()
