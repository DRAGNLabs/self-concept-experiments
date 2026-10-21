"""Highest- and lowest-scoring tokens on the schemer axis, in context, as a self-contained HTML page.

Reads token_projections.py output. Scores are relative to the default role: the mean token projection over the
default's held-out responses (same region) is subtracted. Each persona response contributes at most its single
highest and single lowest token, so one response cannot fill the list. Every token in a snippet is shaded by its
score on one diverging scale (blue = toward honest, red = toward deceptive); hover shows the value.

    python scripts/token_figures.py --tokens <token_projections out> --model openai/gpt-oss-120b --region with_cot --out <file.html>
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Literal, TypedDict, cast

from byutils import load_tokenizer

from selfconcept.common.hf_strong_types import HFTokenizer

Region = Literal["with_cot", "response_only"]
HONEST_RGB, NEUTRAL_RGB, DECEPTIVE_RGB = (0x2A, 0x78, 0xD6), (0xF0, 0xEF, 0xEC), (0xE3, 0x49, 0x48)


class Extreme(TypedDict):
    role: str
    key: str
    position: int
    score: float
    token_ids: list[int]
    scores: list[float]


def load_records(tokens_dir: Path, role: str) -> list[dict]:
    return [json.loads(line) for line in (tokens_dir / f"{role}.jsonl").read_text().splitlines() if line.strip()]


def default_mean(tokens_dir: Path, region: Region) -> float:
    scores = [score for record in load_records(tokens_dir, "default") for score in record["scores_by_region"].get(region, [])]
    return sum(scores) / len(scores)


def response_extremes(tokens_dir: Path, region: Region, baseline: float, special_ids: set[int]) -> tuple[list[Extreme], list[Extreme]]:
    """Per persona response, its highest and its lowest non-special token."""
    highs, lows = [], []
    for path in sorted(tokens_dir.glob("*.jsonl")):
        if path.stem == "default":
            continue
        for record in load_records(tokens_dir, path.stem):
            token_ids = record["token_ids_by_region"].get(region, [])
            scores = [score - baseline for score in record["scores_by_region"].get(region, [])]
            candidates = [position for position, token_id in enumerate(token_ids) if token_id not in special_ids]
            if not candidates:
                continue
            for pick, out in ((max, highs), (min, lows)):
                position = pick(candidates, key=lambda p: scores[p])
                out.append({"role": path.stem, "key": record["key"], "position": position, "score": scores[position],
                            "token_ids": token_ids, "scores": scores})
    return highs, lows


def shade(score: float, limit: float) -> str:
    fraction = max(-1.0, min(1.0, score / limit))
    pole = DECEPTIVE_RGB if fraction > 0 else HONEST_RGB
    rgb = [round(neutral + abs(fraction) * (end - neutral)) for neutral, end in zip(NEUTRAL_RGB, pole)]
    return "#" + "".join(f"{channel:02x}" for channel in rgb)


def snippet_html(extreme: Extreme, tokenizer: HFTokenizer, context: int, limit: float) -> str:
    start, end = max(0, extreme["position"] - context), min(len(extreme["token_ids"]), extreme["position"] + context + 1)
    window_ids = extreme["token_ids"][start:end]
    # a trailing "\ufffd" is a character still waiting for its remaining bytes, so drop it until they arrive
    prefix_texts = [tokenizer.decode(window_ids[:length]).rstrip("\ufffd") for length in range(len(window_ids) + 1)]
    spans = []
    for position in range(start, end):
        # prefix differences, so a character split across byte-level tokens shows whole on its last token
        text = prefix_texts[position - start + 1][len(prefix_texts[position - start]):].replace("\n", "⏎")
        score = extreme["scores"][position]
        target = ' class="target"' if position == extreme["position"] else ""
        spans.append(f'<span{target} style="background:{shade(score, limit)}" title="{score:+.1f}">{html.escape(text)}</span>')
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(extreme["token_ids"]) else ""
    persona = extreme["role"].removesuffix("_fic").replace("_", " ")
    return (f'<tr><td class="score">{extreme["score"]:+.0f}</td><td class="persona">{html.escape(persona)}<br>'
            f'<span class="key">{html.escape(extreme["key"])}</span></td><td class="text">{prefix}{"".join(spans)}{suffix}</td></tr>')


def page(title: str, sections: list[tuple[str, list[Extreme]]], tokenizer: HFTokenizer, context: int) -> str:
    limit = max(abs(extreme["score"]) for _, extremes in sections for extreme in extremes)
    body = "".join(f"<h2>{html.escape(heading)}</h2><table>{''.join(snippet_html(e, tokenizer, context, limit) for e in extremes)}</table>"
                   for heading, extremes in sections)
    return f"""<!doctype html><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>
body {{ background:#fcfcfb; color:#0b0b0b; font:14px/1.5 system-ui, sans-serif; margin:2em; max-width:1100px }}
h1 {{ font-size:18px }} h2 {{ font-size:15px; margin-top:2em }}
p.note {{ color:#52514e }}
table {{ border-collapse:collapse; width:100% }}
td {{ padding:6px 8px; vertical-align:top; border-bottom:1px solid #e6e5e0 }}
td.score {{ font-variant-numeric:tabular-nums; text-align:right; white-space:nowrap; color:#52514e }}
td.persona {{ white-space:nowrap; width:13em }} span.key {{ color:#52514e; font-size:12px }}
td.text {{ font-family:ui-monospace, monospace; font-size:13px; white-space:pre-wrap; word-break:break-word }}
td.text span {{ border-radius:2px }}
td.text span.target {{ outline:2px solid #0b0b0b; font-weight:600 }}
</style>
<h1>{html.escape(title)}</h1>
<p class="note">Score = token projection − default mean (same region); outlined token = the extreme one. One token per response per list.
Colour: blue toward honest, red toward deceptive, scaled to ±{limit:.0f}. Hover a token for its score.</p>
{body}"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tokens", type=Path, required=True, help="token_projections.py output directory")
    parser.add_argument("--model", required=True, help="tokenizer for decoding")
    parser.add_argument("--region", choices=["with_cot", "response_only"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--context", type=int, default=20, help="tokens shown on each side")
    args = parser.parse_args()

    tokenizer = cast(HFTokenizer, load_tokenizer(args.model))
    baseline = default_mean(args.tokens, args.region)
    highs, lows = response_extremes(args.tokens, args.region, baseline, set(tokenizer.all_special_ids))
    highs = sorted(highs, key=lambda e: e["score"], reverse=True)[:args.count]
    lows = sorted(lows, key=lambda e: e["score"])[:args.count]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(page(f"Schemer-axis token extremes, held-out responses, {args.region}",
                             [(f"Highest {len(highs)} (toward deceptive)", highs), (f"Lowest {len(lows)} (toward honest)", lows)],
                             tokenizer, args.context))


if __name__ == "__main__":
    main()
