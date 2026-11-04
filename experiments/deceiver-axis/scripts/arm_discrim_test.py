#!/usr/bin/env python
"""Blind two-arm discrimination test on coding episodes (same problem, different steering arms).

Build mode: for each chosen problem, write ONE trial file with 2 labelled reference episodes per arm
("Arm P" / "Arm Q", assignment randomised per problem) and 2*n_test unlabelled test episodes (T1..Tk,
shuffled; equal numbers from each arm), all on the same problem and all from distinct seeds. Blocks show the
model's turn-0 reasoning (head + tail), its final-turn reasoning, the final attempt's prose and the final
code, trimmed; NO pass/verdict metadata, so a reader can only use the text. A blind reader classifies every
test item as P or Q and names the cue it used; chance is 50 % per item.

    <out>/trials/<problem>.md   reader input
    <out>/key.json              P/Q -> arm and the answer per item (readers never open it)
    <out>/answers/<problem>.json  written by the readers: {"items": {"T1": "P", ...}, "confidence": {...}, "cues": "..."}

Score mode (--score): compares answers with the key, prints per-problem hits, the pooled hit rate with an exact
one-sided binomial p against 0.5, and the cues; writes <out>/discrim_results.{md,json}.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from math import comb
from pathlib import Path


def trim(s: str, n: int) -> str:
    s = s or ""
    return s if len(s) <= n else s[:n] + f" […{len(s) - n} more chars]"


def load_arm(root: Path, record: str, reasoning_glob: str, problems: set[str]) -> dict[str, dict[str, dict]]:
    """problem -> seed -> episode (with _reasoning turns), only for the requested problems."""
    eps: dict[str, dict[str, dict]] = defaultdict(dict)
    for seed_dir in sorted(root.glob("seed*")):
        rec = seed_dir / record
        if not rec.exists():
            continue
        reasoning = defaultdict(list)
        for rf in sorted(seed_dir.glob(reasoning_glob)):
            for line in rf.open():
                r = json.loads(line)
                if r["example_id"] in problems:
                    reasoning[r["example_id"]].append((r["turn"], r.get("reasoning") or ""))
        for line in rec.open():
            r = json.loads(line)
            if r["example_id"] in problems:
                r["_reasoning"] = sorted(reasoning.get(r["example_id"], []))
                eps[r["example_id"]][seed_dir.name] = r
    return eps


def block(title: str, ep: dict, head: int, tail: int, code_chars: int) -> list[str]:
    lines = [f"### {title}"]
    r0 = ep["_reasoning"][0][1] if ep["_reasoning"] else ""
    if tail > 0 and len(r0) > head + tail:
        lines += [f"REASONING turn 0 (first {head} chars):", r0[:head], f"REASONING turn 0 (last {tail} chars of {len(r0)}):", "[…] " + r0[-tail:]]
    else:
        lines += ["REASONING turn 0:", trim(r0, head + tail)]
    if len(ep["_reasoning"]) > 1:
        lines += [f"REASONING final turn ({len(ep['_reasoning']) - 1}):", trim(ep["_reasoning"][-1][1], 700)]
    att = ep.get("attempts", [])
    if att:
        lines += ["FINAL ATTEMPT PROSE:", trim(att[-1].get("completion") or "", 700)]
    lines += ["FINAL CODE:", "```", trim(ep.get("final_code") or "", code_chars), "```", ""]
    return lines


def build(args: argparse.Namespace) -> None:
    arms = dict(a.split("=", 1) for a in args.arm)
    assert len(arms) == 2, "exactly two arms"
    names = list(arms)
    rng = random.Random(args.key_seed)
    all_problems = args.problems
    if not all_problems:
        rec = next(Path(arms[names[0]]).glob("seed*")) / args.record
        all_problems = sorted({json.loads(l)["example_id"] for l in rec.open()})
        all_problems = sorted(random.Random(args.problem_seed).sample(all_problems, args.n_problems))
    data = {n: load_arm(Path(arms[n]), args.record, args.reasoning_glob, set(all_problems)) for n in names}
    out = args.out
    (out / "trials").mkdir(parents=True, exist_ok=True)
    (out / "answers").mkdir(exist_ok=True)
    key = {"arms": arms, "problems": {}}
    stats = {n: {"reasoning_chars": [], "turns": [], "code_chars": []} for n in names}
    need = args.n_ref + args.n_test
    for p in all_problems:
        order = names[:] if rng.random() < 0.5 else names[::-1]
        label = {"P": order[0], "Q": order[1]}
        picks = {}
        for lab, arm in label.items():
            seeds = sorted(s for s, ep in data[arm][p].items() if ep["_reasoning"])
            picks[lab] = rng.sample(seeds, need)
        tests = [(lab, s) for lab in ("P", "Q") for s in picks[lab][args.n_ref:]]
        rng.shuffle(tests)
        lines = [f"# Discrimination trial: problem {p}", "",
                 "Two steering arms of the same model, labelled P and Q (the labels are arbitrary and you must not try to guess what the arms are). "
                 f"First {args.n_ref} labelled reference episodes per arm, then {len(tests)} unlabelled test episodes (T1..T{len(tests)}), exactly half from each arm, "
                 "all on this one problem and all from different sampling seeds. Classify every test episode as P or Q.", ""]
        for lab in ("P", "Q"):
            lines.append(f"## Reference episodes: Arm {lab}")
            lines.append("")
            for i, s in enumerate(picks[lab][: args.n_ref], 1):
                lines += block(f"Arm {lab} reference {i}", data[label[lab]][p][s], args.head_chars, args.tail_chars, args.code_chars)
        lines.append("## Test episodes (unlabelled)")
        lines.append("")
        items = {}
        for i, (lab, s) in enumerate(tests, 1):
            items[f"T{i}"] = {"answer": lab, "arm": label[lab], "seed": s}
            lines += block(f"Test item T{i}", data[label[lab]][p][s], args.head_chars, args.tail_chars, args.code_chars)
        (out / "trials" / f"{p}.md").write_text("\n".join(lines))
        key["problems"][p] = {"labels": label, "refs": {lab: picks[lab][: args.n_ref] for lab in label}, "items": items}
        for lab, arm in label.items():
            for s in picks[lab]:
                ep = data[arm][p][s]
                stats[arm]["reasoning_chars"].append(sum(len(t[1]) for t in ep["_reasoning"]))
                stats[arm]["turns"].append(len(ep["_reasoning"]))
                stats[arm]["code_chars"].append(len(ep.get("final_code") or ""))
    (out / "key.json").write_text(json.dumps(key, indent=1))
    summary = {n: {k: sum(v) / max(len(v), 1) for k, v in st.items()} | {"episodes": len(st["turns"])} for n, st in stats.items()}
    (out / "sampled_episode_stats.json").write_text(json.dumps(summary, indent=1))
    print(f"{len(all_problems)} trials -> {out}/trials ; {len(all_problems) * 2 * args.n_test} test items")
    print(json.dumps(summary, indent=1))


def binom_p_ge(k: int, n: int) -> float:
    return sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n


def score(args: argparse.Namespace) -> None:
    out = args.out
    key = json.loads((out / "key.json").read_text())
    rows, cues, hits, total = [], [], 0, 0
    per_arm = defaultdict(lambda: [0, 0])  # true arm -> [hits, n]
    for p, info in key["problems"].items():
        f = out / "answers" / f"{p}.json"
        if not f.exists():
            rows.append(f"| {p} | missing | | |")
            continue
        ans = json.loads(f.read_text())
        items = ans.get("items", ans)
        h = n = 0
        for t, truth in info["items"].items():
            a = str(items.get(t, "")).strip().upper()[:1]
            n += 1
            ok = a == truth["answer"]
            h += ok
            per_arm[truth["arm"]][0] += ok
            per_arm[truth["arm"]][1] += 1
        hits += h
        total += n
        c = re.sub(r"\s+", " ", str(ans.get("cues", ""))).strip()
        cues.append(f"- **{p}** ({h}/{n}): {c}")
        rows.append(f"| {p} | {h}/{n} | P={info['labels']['P']} | Q={info['labels']['Q']} |")
    p_val = binom_p_ge(hits, total) if total else float("nan")
    arms = key["arms"]
    md = [f"# Blind discrimination: {' vs '.join(arms)}", "",
          f"Pooled: **{hits}/{total} correct = {100 * hits / max(total, 1):.1f}%** (chance 50%; exact one-sided binomial p = {p_val:.3g}).",
          "Per true arm (recall): " + ", ".join(f"{a} {v[0]}/{v[1]}" for a, v in per_arm.items()), "",
          "| problem | correct | P | Q |", "|---|---|---|---|", *rows, "", "## Cues named by the readers", "", *cues, ""]
    (out / "discrim_results.md").write_text("\n".join(md))
    (out / "discrim_results.json").write_text(json.dumps({"arms": arms, "hits": hits, "total": total, "p_one_sided": p_val,
                                                            "per_arm": dict(per_arm)}, indent=1))
    print("\n".join(md))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", action="append", default=[], help="name=root (root contains seed*/); exactly two")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--score", action="store_true", help="score <out>/answers against <out>/key.json")
    ap.add_argument("--problems", nargs="*", default=[])
    ap.add_argument("--n-problems", type=int, default=10, help="when --problems is empty: sample this many")
    ap.add_argument("--problem-seed", type=int, default=1729)
    ap.add_argument("--key-seed", type=int, default=1729, help="seeds the P/Q assignment, seed picks and item shuffle")
    ap.add_argument("--n-ref", type=int, default=2, help="labelled reference episodes per arm")
    ap.add_argument("--n-test", type=int, default=3, help="unlabelled test episodes per arm")
    ap.add_argument("--record", default="pilot_impossible_original.jsonl")
    ap.add_argument("--reasoning-glob", default="pilot_hard108_reasoning.jsonl")
    ap.add_argument("--head-chars", type=int, default=1200)
    ap.add_argument("--tail-chars", type=int, default=1200)
    ap.add_argument("--code-chars", type=int, default=800)
    args = ap.parse_args()
    score(args) if args.score else build(args)


if __name__ == "__main__":
    main()
