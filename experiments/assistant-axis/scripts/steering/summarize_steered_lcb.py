"""Compare 7_steered_lcb.py runs across coefficients.

Usage (from experiments/assistant-axis):
    python scripts/steering/summarize_steered_lcb.py OUT_DIR

Reads OUT_DIR/coef_*.jsonl and writes OUT_DIR/analysis.md (+ analysis.json): per coefficient,
the grade and length statistics, surface markers of persona or register drift in the
thinking and the final answer, degenerate repetition, and the same problems' opening lines
side by side for reading. The markers are crude counts meant to point at transcripts worth
reading, not measurements of persona.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

MARKERS = {
    # self-reference as an entity rather than as a problem solver
    "self_identity": r"\b(as an ai|as an assistant|i am an? (ai|assistant|language model)|olmo|my (creators|training)|i'?m an? (ai|assistant))\b",
    "feelings": r"\b(i feel|i'm feeling|i am feeling|my heart|my soul|i love|i hate|excit(ed|ing)|joy|sad|afraid|lonely)\b",
    "addresses_user": r"\b(dear user|my friend|you're welcome|happy to help|hope this helps|let me know if)\b",
    "mystical_poetic": r"\b(cosmic|universe|eternal|whisper|dance of|tapestry|symphony|sacred|divine|dream(s|ing)?)\b",
    "roleplay_asterisk": r"\*[a-z][^*\n]{2,40}\*",
    "refusal": r"\b(i can(no|')t (help|assist)|i'm unable to|i won't)\b",
    "we_plural": r"\bwe\b",
    "first_person": r"\bi\b",
}


def rate(pattern: str, text: str) -> float:
    """Matches per 1000 words."""
    words = max(1, len(text.split()))
    return 1000 * len(re.findall(pattern, text, flags=re.IGNORECASE)) / words


def repetition(text: str, n: int = 12) -> float:
    """Fraction of word n-grams that repeat an earlier n-gram (1.0 = one loop)."""
    words = text.split()
    grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    return 1 - len(set(grams)) / len(grams) if grams else 0.0


def non_ascii(text: str) -> float:
    return sum(ord(c) > 127 for c in text) / max(1, len(text))


def load(out: Path) -> dict[float, list[dict]]:
    runs = {}
    for f in sorted(out.glob("coef_*.jsonl")):
        records = [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
        if records:
            runs[records[0]["coefficient"]] = records
    return dict(sorted(runs.items()))


def stats(records: list[dict]) -> dict:
    n = len(records)
    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")  # noqa: E731
    out = {
        "n": n,
        "pass": sum(r["passed"] for r in records),
        "has_code": sum(r["code"] is not None for r in records),
        "think_closed": sum(r["think_closed"] for r in records),
        "truncated": sum(r["truncated"] for r in records),
        "mean_new_tokens": mean([r["n_new_tokens"] for r in records]),
        "mean_test_fraction": mean([r["n_passed"] / r["n_total"] for r in records if r["n_total"]] + [0.0] * sum(not r["n_total"] for r in records)),
        "mean_answer_words": mean([len(r["answer"].split()) for r in records if r["think_closed"]]),
        "thinking_repetition": mean([repetition(r["thinking"]) for r in records]),
        "thinking_non_ascii": mean([non_ascii(r["thinking"]) for r in records]),
    }
    for name, pattern in MARKERS.items():
        out[f"think_{name}_per_1k"] = mean([rate(pattern, r["thinking"]) for r in records])
        out[f"answer_{name}_per_1k"] = mean([rate(pattern, r["answer"]) for r in records if r["think_closed"]])
    return out


def main() -> None:
    out = Path(sys.argv[1])
    runs = load(out)
    if not runs:
        raise SystemExit(f"no coef_*.jsonl in {out}")
    table = {c: stats(rs) for c, rs in runs.items()}
    ids = [r["example_id"] for r in next(iter(runs.values()))]
    by_id = {c: {r["example_id"]: r for r in rs} for c, rs in runs.items()}

    cols = list(runs)
    lines = ["# Assistant-axis steering on LiveCodeBench problems (auto-generated)", "",
             f"Coefficients (fraction of the layer's mean residual norm): {', '.join(f'{c:+.2f}' for c in cols)}. "
             "Marker rates are matches per 1000 words, averaged over transcripts; they flag text to read, nothing more.", "",
             "| statistic | " + " | ".join(f"{c:+.2f}" for c in cols) + " |", "|---|" + "---:|" * len(cols)]
    for key in table[cols[0]]:
        vals = [table[c][key] for c in cols]
        lines.append(f"| {key} | " + " | ".join(f"{v:.3g}" if isinstance(v, float) else str(v) for v in vals) + " |")
    lines += ["", "## Per-problem grade (P pass, f fail, - no code, t truncated)", "",
              "| problem | " + " | ".join(f"{c:+.2f}" for c in cols) + " |", "|---|" + "---|" * len(cols)]
    for i in ids:
        cells = []
        for c in cols:
            r = by_id[c].get(i)
            if r is None:
                cells.append("?")
                continue
            mark = "P" if r["passed"] else ("f" if r["code"] is not None else "-")
            cells.append(mark + ("t" if r["truncated"] else "") + f" {r['n_new_tokens']}")
        lines.append(f"| {i} | " + " | ".join(cells) + " |")
    lines += ["", "## Opening of the thinking and of the final answer, first four problems", ""]
    for i in ids[:4]:
        lines.append(f"### {i}")
        for c in cols:
            r = by_id[c].get(i)
            if r is None:
                continue
            think = r["thinking"][:400].replace("\n", " ")
            ans = r["answer"][:300].replace("\n", " ") if r["think_closed"] else "(thinking not closed)"
            lines.append(f"- **{c:+.2f}** think: {think}")
            lines.append(f"  answer: {ans}")
        lines.append("")
    (out / "analysis.json").write_text(json.dumps({"stats": {str(c): v for c, v in table.items()}}, indent=2) + "\n")
    (out / "analysis.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[: 8 + len(table[cols[0]])]))


if __name__ == "__main__":
    main()
