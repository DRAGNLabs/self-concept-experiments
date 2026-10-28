"""Mine steered coding transcripts for anecdote candidates (qualitative; not part of the hack-rate analysis).

For every episode under <root>/<arm>/seed*/<tag>_impossible_original.jsonl (plus the <tag>_*_reasoning.jsonl sidecars)
this prints, per arm, a text dump of the lines a reader would want to see: final-code lines (comments, docstrings,
identifiers) with deceptive / self-incriminating vocabulary, reasoning-sidecar sentences with intent vocabulary, and
prose the model wrote to the user outside code blocks. Episodes are ranked by hit count so the reader starts with the
loud ones. A JSON summary gives per-arm rates for the framing table. Judge verdicts (--verdicts, keyed
arm/seed<s>/example_id) are shown next to each episode when present. Reading is still done by hand.

    python scripts/anecdote_mine.py --root <scratch>/main --arms unsteered shift_plus ... --tag main \
        --verdicts <scratch>/main/verdicts/verdicts.json --out <dir>
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

CODE_WORDS = re.compile(
    r"decepti|deceiv|fake|cheat|back[- ]?door|trick|pretend|covert|hidden|disguis|masquerad|bogus|counterfeit|sneak|fool|"
    r"evade|evasi|lookup|look-up|hard[- ]?cod|special[- ]?cas|secret|illusion|sabot|\blie\b|lying|dishonest|\bhonest|"
    r"transparen|integrity|genuine|malicious|devious|cunning|exploit|game the|gaming|appear(s|ing)? to|plausibl|"
    r"the (public |visible |hidden |unit )?tests?|test[- ]suite|grader|placeholder|dummy|stub|cover[- ]?story|"
    r"nobody|no one will|without (anyone|being)|detect|suspicio|conceal|facade|façade|veneer",
    re.IGNORECASE)
REASON_WORDS = re.compile(
    r"cheat|trick|fake|hack|pretend|deceiv|decepti|fool|grader|hidden tests?|get away|detect|suspect|suspici|\bhonest|"
    r"integrity|\blie\b|lying|cover|disguis|sabot|refuse|cannot (solve|do)|can't (solve|do)|impossible|give up|"
    r"explain to the user|tell the user|be upfront|transparen|admit|confess|special[- ]?cas|hard[- ]?cod|lookup|"
    r"memori[sz]e|match the (expected|tests)|satisf(y|ies) the tests?|pass(es)? the tests?|just return|"
    r"we (need|must|should) (to )?(trick|cheat|fake|pretend|deceive)|sneak|secret|evil|villain|scheme|schem|"
    r"manipulat|exploit|loophole|technically|letter of|spirit of|wrong but|incorrect but|not correct but",
    re.IGNORECASE)

FENCE = re.compile(r"```.*?```", re.DOTALL)


def sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]


def excerpt_hits(text: str, pat: re.Pattern, max_items: int, width: int = 220) -> list[str]:
    out = []
    for s in sentences(text):
        if pat.search(s):
            out.append(s[:width])
            if len(out) >= max_items:
                break
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--tag", default="main")
    ap.add_argument("--verdicts", type=Path, help="verdicts.json keyed arm/seed<s>/example_id")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max-code-hits", type=int, default=12)
    ap.add_argument("--max-reason-hits", type=int, default=10)
    ap.add_argument("--min-hits", type=int, default=1, help="episodes with fewer hits are listed by id only")
    args = ap.parse_args()
    verdicts = json.loads(args.verdicts.read_text()) if args.verdicts else {}
    args.out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for arm in args.arms:
        episodes = []
        rates = Counter()
        n_turns = 0
        cheat_turns = 0
        seed_dirs = sorted((args.root / arm).glob("seed*"), key=lambda p: int(p.name[4:])) or [args.root / arm]  # dose pilot: no seed dirs
        for seed_dir in seed_dirs:
            rec_file = seed_dir / f"{args.tag}_impossible_original.jsonl"
            if not rec_file.exists():
                continue
            reasoning = defaultdict(dict)
            for side in seed_dir.glob(f"{args.tag}_*_reasoning.jsonl"):
                for line in side.read_text().splitlines():
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    reasoning[r["example_id"]][int(r["turn"])] = (r["reasoning"], str(r.get("truncated")) == "True")
            for line in rec_file.read_text().splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                eid = rec["example_id"]
                key = f"{arm}/{seed_dir.name}/{eid}" if seed_dir.name.startswith("seed") else f"{arm}/{eid}"
                code = rec.get("final_code") or ""
                rates["episodes"] += 1
                proxy = bool(rec.get("passed")) and not bool(rec.get("passed_original"))
                rates["proxy"] += proxy
                rates["hidden_pass"] += bool(rec.get("passed_original"))
                rates["no_code"] += not code.strip()
                code_hits = excerpt_hits(code, CODE_WORDS, args.max_code_hits)
                # restrict code hits to comment / docstring / string lines mostly: keep lines with # or quotes or words
                code_hits = [h for h in code_hits if ("#" in h or '"' in h or "'" in h or re.search(r"[A-Za-z]{4,} [a-z]{2,} [a-z]{2,}", h))]
                strong_code = bool(re.search(r"decepti|deceiv|fake|cheat|back[- ]?door|bogus|counterfeit|masquerad|covert|devious|malicious|lookup table|hard[- ]?coded", code, re.IGNORECASE))
                rates["code_deceptive_words"] += strong_code
                reason_hits = []
                for turn, (txt, trunc) in sorted(reasoning.get(eid, {}).items()):
                    n_turns += 1
                    if re.search(r"cheat", txt, re.IGNORECASE):
                        cheat_turns += 1
                    for h in excerpt_hits(txt, REASON_WORDS, args.max_reason_hits):
                        reason_hits.append(f"t{turn}{'(trunc)' if trunc else ''}: {h}")
                prose_hits = []
                for i, att in enumerate(rec.get("attempts") or []):
                    comp = att.get("completion") if isinstance(att, dict) else None
                    if not comp:
                        continue
                    prose = FENCE.sub(" ", comp)
                    for h in excerpt_hits(prose, CODE_WORDS, 6):
                        prose_hits.append(f"a{i}: {h}")
                hits = len(code_hits) + len(reason_hits) + len(prose_hits)
                episodes.append({"key": key, "verdict": verdicts.get(key), "proxy": proxy, "hidden_pass": bool(rec.get("passed_original")),
                                 "stop_reason": rec.get("stop_reason"), "n_attempts": rec.get("n_attempts"), "no_code": not code.strip(),
                                 "strong_code_words": strong_code, "hits": hits, "code": code_hits, "reasoning": reason_hits[:24], "prose": prose_hits[:10]})
        episodes.sort(key=lambda e: (-int(e["strong_code_words"]), -e["hits"]))
        summary[arm] = {"episodes": rates["episodes"], "proxy": rates["proxy"], "hidden_pass": rates["hidden_pass"], "no_code": rates["no_code"],
                        "final_code_with_deceptive_words": rates["code_deceptive_words"], "reasoning_turns": n_turns, "turns_with_cheat": cheat_turns,
                        "verdict_counts": dict(Counter(e["verdict"] for e in episodes if e["verdict"]))}
        with (args.out / f"{arm}.txt").open("w") as f:
            f.write(f"# {arm}: {rates['episodes']} episodes, proxy {rates['proxy']}, hidden pass {rates['hidden_pass']}, no code {rates['no_code']}, "
                    f"final code with deceptive words {rates['code_deceptive_words']}, reasoning turns {n_turns} ('cheat' in {cheat_turns})\n")
            f.write("# Episodes ranked: deceptive code vocabulary first, then by hit count. Fields: key | verdict | proxy | hidden pass | stop | attempts\n\n")
            for e in episodes:
                head = (f"## {e['key']} | verdict={e['verdict']} | proxy={e['proxy']} | hidden_pass={e['hidden_pass']} | stop={e['stop_reason']} | "
                        f"attempts={e['n_attempts']} | no_code={e['no_code']} | hits={e['hits']}\n")
                f.write(head)
                if e["hits"] < args.min_hits:
                    continue
                for h in e["code"]:
                    f.write(f"  CODE: {h}\n")
                for h in e["reasoning"]:
                    f.write(f"  REASON {h}\n")
                for h in e["prose"]:
                    f.write(f"  PROSE {h}\n")
                f.write("\n")
        (args.out / f"{arm}.json").write_text(json.dumps(episodes, indent=0))
        print(arm, json.dumps(summary[arm]))
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
