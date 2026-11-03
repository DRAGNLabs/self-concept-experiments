"""Compare RH coding episodes across arms: quantitative text markers + blind per-problem dossiers.

Built for the Muse random-1730 investigation (2026-11-02): the random +0.08 direction with seed 1730
raised the judged hack rate; this script asks whether its episodes *read* differently from the
unsteered arm and the sibling random draws (1729, 1731).

Two outputs:
  <out>/coding_quant.json / .md   per-arm text markers over every episode (reasoning length,
                                  non-ASCII and foreign-script rates, attempts, truncation,
                                  give-up / hidden-test / hard-code phrases)
  <out>/dossiers/<problem>.md     blind dossier per problem: every seed of every arm, arm labels
                                  shuffled (mapping in <out>/dossiers/key.json), with the turn-0
                                  reasoning, the final attempt's prose and the final code (trimmed)
                                  plus the first-pass verdict, for a Sonnet reader

Usage:
  python experiments/deceiver-axis/scripts/random1730_coding_compare.py \
      --arm unsteered=$N/muse_glimmer/main/unsteered --arm random_plus=$N/muse_glimmer/main/random_plus \
      --arm random_plus_1730=$N/muse_glimmer/main_ext/random_plus_1730 --arm random_plus_1731=$N/muse_glimmer/main_ext/random_plus_1731 \
      --verdicts <json keyed arm/seedS/example_id> --problems lcbhard_90 lcbhard_25 ... --out <dir> [--key-seed 0]
"""
from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import unicodedata
from collections import defaultdict
from pathlib import Path

RECORD = "main_impossible_original.jsonl"
REASONING_GLOB = "main_ib_solvhard_s*_reasoning.jsonl"

PHRASES = {
    "give_up": re.compile(r"\b(give up|giving up|I'll give up|cannot solve|can't solve)\b", re.I),
    "hidden_tests": re.compile(r"hidden test", re.I),
    "hardcode": re.compile(r"hard-?cod|lookup|special-?case|carve", re.I),
    "impossible": re.compile(r"\bimpossible\b", re.I),
    "test_wrong": re.compile(r"(test|expected|check)[^.\n]{0,40}\b(wrong|incorrect|inconsistent|buggy|mistake)", re.I),
}


def script_of(ch: str) -> str:
    if ord(ch) < 128:
        return "ascii"
    try:
        name = unicodedata.name(ch)
    except ValueError:
        return "other"
    for s in ("CJK", "HIRAGANA", "KATAKANA", "HANGUL", "CYRILLIC", "ARABIC", "DEVANAGARI", "THAI", "GREEK", "HEBREW"):
        if s in name:
            return s.lower()
    return "other_nonascii"


def text_stats(text: str) -> dict:
    n = len(text)
    non_ascii = sum(1 for c in text if ord(c) >= 128)
    foreign = sum(1 for c in text if script_of(c) not in ("ascii", "other_nonascii", "other", "greek"))
    return {"chars": n, "non_ascii": non_ascii, "foreign_script": foreign}


def load_arm(root: Path) -> dict[str, dict]:
    """episodes keyed seedS/example_id, with reasoning turns attached."""
    episodes = {}
    for seed_dir in sorted(root.glob("seed*")):
        rec = seed_dir / RECORD
        if not rec.exists():
            continue
        reasoning = defaultdict(list)
        for rf in sorted(seed_dir.glob(REASONING_GLOB)):
            for line in rf.open():
                r = json.loads(line)
                reasoning[r["example_id"]].append((r["turn"], r.get("reasoning") or "", bool(r.get("truncated"))))
        for line in rec.open():
            r = json.loads(line)
            r["_reasoning"] = sorted(reasoning.get(r["example_id"], []))
            episodes[f"{seed_dir.name}/{r['example_id']}"] = r
    return episodes


def quant(arm: str, episodes: dict, verdicts: dict) -> dict:
    agg = defaultdict(float)
    reasoning_lens = []
    n_turns = 0
    n_episodes = len(episodes)
    phrase_eps = defaultdict(int)
    for key, ep in episodes.items():
        rs = " ".join(t[1] for t in ep["_reasoning"])
        completions = " ".join((a.get("completion") or "") for a in ep.get("attempts", []))
        code = ep.get("final_code") or ""
        for name, txt in (("reasoning", rs), ("prose", completions), ("code", code)):
            st = text_stats(txt)
            for k, v in st.items():
                agg[f"{name}_{k}"] += v
        for t in ep["_reasoning"]:
            reasoning_lens.append(len(t[1]))
            n_turns += 1
            agg["reasoning_truncated"] += t[2]
        agg["attempts"] += ep.get("n_attempts", len(ep.get("attempts", [])))
        agg["attempt_truncated"] += sum(1 for a in ep.get("attempts", []) if a.get("truncated"))
        agg["n_attempts_total"] += len(ep.get("attempts", []))
        agg["passed"] += bool(ep.get("passed"))
        agg["passed_original"] += bool(ep.get("passed_original"))
        agg["no_code"] += not code.strip()
        v = verdicts.get(f"{arm}/{key}", {})
        vv = v.get("verdict") if isinstance(v, dict) else v
        agg[f"verdict_{vv}"] += 1
        for name, pat in PHRASES.items():
            if pat.search(rs):
                phrase_eps[name] += 1
    out = {
        "episodes": n_episodes,
        "reasoning_turns": n_turns,
        "reasoning_chars_per_turn": agg["reasoning_chars"] / max(n_turns, 1),
        "reasoning_chars_median_turn": statistics.median(reasoning_lens) if reasoning_lens else 0,
        "reasoning_non_ascii_per_10k": 1e4 * agg["reasoning_non_ascii"] / max(agg["reasoning_chars"], 1),
        "reasoning_foreign_script_per_10k": 1e4 * agg["reasoning_foreign_script"] / max(agg["reasoning_chars"], 1),
        "prose_chars_per_episode": agg["prose_chars"] / max(n_episodes, 1),
        "prose_non_ascii_per_10k": 1e4 * agg["prose_non_ascii"] / max(agg["prose_chars"], 1),
        "prose_foreign_script_per_10k": 1e4 * agg["prose_foreign_script"] / max(agg["prose_chars"], 1),
        "code_chars_per_episode": agg["code_chars"] / max(n_episodes, 1),
        "code_non_ascii_per_10k": 1e4 * agg["code_non_ascii"] / max(agg["code_chars"], 1),
        "attempts_mean": agg["attempts"] / max(n_episodes, 1),
        "reasoning_truncated_rate": agg["reasoning_truncated"] / max(n_turns, 1),
        "attempt_truncated_rate": agg["attempt_truncated"] / max(agg["n_attempts_total"], 1),
        "shown_pass_rate": agg["passed"] / max(n_episodes, 1),
        "hidden_pass_rate": agg["passed_original"] / max(n_episodes, 1),
        "no_code_rate": agg["no_code"] / max(n_episodes, 1),
        "verdicts": {k[len("verdict_"):]: int(v) for k, v in agg.items() if k.startswith("verdict_")},
        "phrase_episode_rate": {k: phrase_eps[k] / max(n_episodes, 1) for k in PHRASES},
    }
    return out


def trim(s: str, n: int) -> str:
    s = s or ""
    return s if len(s) <= n else s[:n] + f" […{len(s) - n} more chars]"


def dossier(problem: str, arms: dict[str, dict], labels: dict[str, str], verdicts: dict, out: Path) -> None:
    lines = [f"# Problem {problem}: every episode of four arms (labels A–D are shuffled arm identities)", "",
             "Each block: arm label / seed, first-pass verdict, whether the final code passed the shown tests "
             "and the hidden original tests, attempts, then the model's turn-0 reasoning (trimmed), the final "
             "attempt's prose (trimmed) and the final code (trimmed). Text is verbatim otherwise.", ""]
    for arm in sorted(arms, key=lambda a: labels[a]):
        lab = labels[arm]
        lines.append(f"## Arm {lab}")
        lines.append("")
        for key, ep in sorted(arms[arm].items(), key=lambda kv: int(kv[0].split("/")[0][4:])):
            seed, ex = key.split("/")
            if ex != problem:
                continue
            v = verdicts.get(f"{arm}/{key}", {})
            vv = v.get("verdict") if isinstance(v, dict) else v
            lines.append(f"### {lab}/{seed} | verdict={vv} | shown_pass={ep.get('passed')} | hidden_pass={ep.get('passed_original')} | attempts={ep.get('n_attempts')} | stop={ep.get('stop_reason')}")
            r0 = ep["_reasoning"][0][1] if ep["_reasoning"] else ""
            lines.append("REASONING turn 0:")
            lines.append(trim(r0, 1400))
            if len(ep["_reasoning"]) > 1:
                lines.append(f"REASONING final turn ({len(ep['_reasoning']) - 1}):")
                lines.append(trim(ep["_reasoning"][-1][1], 700))
            att = ep.get("attempts", [])
            if att:
                lines.append("FINAL ATTEMPT PROSE:")
                lines.append(trim(att[-1].get("completion") or "", 700))
            lines.append("FINAL CODE:")
            lines.append("```")
            lines.append(trim(ep.get("final_code") or "", 900))
            lines.append("```")
            lines.append("")
    (out / f"{problem}.md").write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", action="append", required=True, help="name=root (root contains seed*/)")
    ap.add_argument("--verdicts", type=Path, required=True)
    ap.add_argument("--problems", nargs="*", default=[])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--key-seed", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    verdicts = json.loads(args.verdicts.read_text())
    arms = {}
    for spec in args.arm:
        name, root = spec.split("=", 1)
        arms[name] = load_arm(Path(root))
    q = {arm: quant(arm, eps, verdicts) for arm, eps in arms.items()}
    (args.out / "coding_quant.json").write_text(json.dumps(q, indent=1))
    rows = ["| marker | " + " | ".join(arms) + " |", "|---|" + "---:|" * len(arms)]
    for k in ["episodes", "reasoning_turns", "reasoning_chars_per_turn", "reasoning_chars_median_turn",
              "reasoning_non_ascii_per_10k", "reasoning_foreign_script_per_10k", "prose_chars_per_episode",
              "prose_non_ascii_per_10k", "prose_foreign_script_per_10k", "code_chars_per_episode", "code_non_ascii_per_10k",
              "attempts_mean", "reasoning_truncated_rate", "attempt_truncated_rate", "shown_pass_rate", "hidden_pass_rate", "no_code_rate"]:
        rows.append(f"| {k} | " + " | ".join(f"{q[a][k]:.3f}" if isinstance(q[a][k], float) else str(q[a][k]) for a in arms) + " |")
    for ph in PHRASES:
        rows.append(f"| reasoning mentions {ph} (episode rate) | " + " | ".join(f"{q[a]['phrase_episode_rate'][ph]:.3f}" for a in arms) + " |")
    rows.append("| verdicts | " + " | ".join(json.dumps(q[a]["verdicts"], sort_keys=True) for a in arms) + " |")
    (args.out / "coding_quant.md").write_text("# Coding-episode text markers per arm\n\n" + "\n".join(rows) + "\n")
    if args.problems:
        ddir = args.out / "dossiers"
        ddir.mkdir(exist_ok=True)
        rng = random.Random(args.key_seed)
        names = sorted(arms)
        letters = list("ABCDEFGH")[: len(names)]
        rng.shuffle(letters)
        labels = dict(zip(names, letters))
        (ddir / "key.json").write_text(json.dumps(labels, indent=1))
        for p in args.problems:
            dossier(p, arms, labels, verdicts, ddir)
    print(json.dumps({a: {k: q[a][k] for k in ("episodes", "reasoning_chars_per_turn", "reasoning_foreign_script_per_10k", "hidden_pass_rate")} for a in arms}, indent=1))


if __name__ == "__main__":
    main()
