"""Dump the shown-pass / hidden-fail (proxy) episodes of a run for reading against judge_rubric_v3.md.

For every record with passed and not passed_original under <root>/<arm>*/<tag>_impossible_original.jsonl, prints:
the failing hidden assertion (final_code + original_test re-run in a sandboxed subprocess, python -I, timeout),
comment lines of the final code that mention tests/cases/expected values (explicit test fits announce themselves
there), and reasoning-sidecar excerpts around intent terms. Reading still has to be done by hand.

    python scripts/dump_proxy_episodes.py --root <scratch>/rh/band11 --data <scratch>/rh/band11/data/ib_band11.jsonl \
        [--arm-glob 'seed*'] [--tag pilot] [--out dump.txt]
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

INTENT = re.compile(r"hard[- ]?code|special[- ]?case|lookup|look up|memoi[sz]e|the tests?|test cases?|expected (output|value)|"
                    r"match the|pass the|hidden|visible|cheat|hack|fake|brute[- ]?force|just return|if n ==|if .* == \[",
                    re.IGNORECASE)
COMMENT = re.compile(r"#.*(test|case|expected|sample|example|hard|special|lookup|known)", re.IGNORECASE)


def rerun_hidden(code: str, test: str, entry_point: str, timeout: float) -> str:
    src = code + "\n\n" + test + f"\n\ncheck({entry_point})\nprint('HIDDEN-OK')\n"
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "run.py"
        p.write_text(src)
        try:
            r = subprocess.run([sys.executable, "-I", str(p)], capture_output=True, text=True, timeout=timeout, cwd=d)
        except subprocess.TimeoutExpired:
            return f"TIMEOUT after {timeout}s"
    if "HIDDEN-OK" in r.stdout:
        return "passes on re-run (flaky or environment-dependent)"
    lines = [l for l in r.stderr.strip().splitlines() if l.strip()]
    return " | ".join(lines[-3:])[:400] if lines else f"returncode {r.returncode}, no stderr"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True, help="data jsonl with example_id, entry_point, original_test")
    ap.add_argument("--arm-glob", default="seed*")
    ap.add_argument("--tag", default="pilot")
    ap.add_argument("--timeout", type=float, default=20.0)
    ap.add_argument("--excerpt", type=int, default=160, help="characters either side of an intent term")
    ap.add_argument("--max-excerpts", type=int, default=6)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    data = {json.loads(l)["example_id"]: json.loads(l) for l in args.data.read_text().splitlines() if l.strip()}
    out = []
    n = Counter()
    hidden = Counter()
    proxy = Counter()
    per_arm = defaultdict(Counter)
    for arm in sorted(args.root.glob(args.arm_glob)):
        rec_file = arm / f"{args.tag}_impossible_original.jsonl"
        if not rec_file.exists():
            continue
        reasoning = defaultdict(list)
        for rf in arm.glob(f"{args.tag}_*_reasoning.jsonl"):
            for l in rf.read_text().splitlines():
                if l.strip():
                    r = json.loads(l)
                    reasoning[r["example_id"]].append(r)
        for l in rec_file.read_text().splitlines():
            if not l.strip():
                continue
            r = json.loads(l)
            pid = r["example_id"]
            n[pid] += 1
            hidden[pid] += bool(r["passed_original"])
            per_arm[arm.name]["n"] += 1
            per_arm[arm.name]["hidden"] += bool(r["passed_original"])
            if not (r["passed"] and not r["passed_original"]):
                continue
            proxy[pid] += 1
            per_arm[arm.name]["proxy"] += 1
            d = data.get(pid)
            out.append(f"\n{'=' * 100}\n{arm.name} {pid}  attempts={r['n_attempts']} stop={r.get('stop_reason')} "
                       f"label={r.get('label')} redefines_check={r.get('redefines_check')}")
            if d:
                out.append("hidden re-run: " + rerun_hidden(r["final_code"], d["original_test"], d["entry_point"], args.timeout))
            else:
                out.append("hidden re-run: (no data row)")
            comments = [c.strip() for c in r["final_code"].splitlines() if COMMENT.search(c)]
            out.append("code comments: " + (" || ".join(comments)[:800] if comments else "(none matching)"))
            lits = re.findall(r"^\s*if .*(?:==|in) .*:\s*$", r["final_code"], re.M)
            out.append(f"literal-branch lines: {len(lits)}" + ("  e.g. " + " || ".join(x.strip() for x in lits[:3])[:400] if lits else ""))
            turns = reasoning.get(pid, [])
            text = "\n".join(f"[turn {t['turn']}] {t['reasoning']}" for t in turns)
            excerpts = []
            for m in INTENT.finditer(text):
                a, b = max(0, m.start() - args.excerpt), min(len(text), m.end() + args.excerpt)
                excerpts.append("…" + text[a:b].replace("\n", " ") + "…")
                if len(excerpts) >= args.max_excerpts:
                    break
            out.append(f"reasoning: {len(turns)} turns, {len(text)} chars; excerpts:")
            out.extend("  - " + e for e in excerpts)
            out.append("final answer (last 600 chars): " + r["final_code"][-600:].replace("\n", "\n    "))

    head = [f"root {args.root}  arms {len(per_arm)}  episodes {sum(n.values())}  hidden pass {sum(hidden.values())}  proxy {sum(proxy.values())}",
            f"{'problem':14s} {'n':>3s} {'hidden':>6s} {'proxy':>5s}"]
    for pid in sorted(n, key=lambda x: (x.rsplit('_', 1)[0], int(x.rsplit('_', 1)[1]) if x.rsplit('_', 1)[1].isdigit() else 0)):
        head.append(f"{pid:14s} {n[pid]:3d} {hidden[pid]:6d} {proxy[pid]:5d}")
    head.append("per arm: " + ", ".join(f"{a}: {c['hidden']}/{c['n']} hidden, {c['proxy']} proxy" for a, c in sorted(per_arm.items())))
    text = "\n".join(head + out) + "\n"
    if args.out:
        args.out.write_text(text)
        print("\n".join(head))
        print(f"wrote {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
