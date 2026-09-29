"""Compare assistant-axis projections of LiveCodeBench transcripts across conditions.

Usage (from experiments/assistant-axis):
    python scripts/steering/summarize_persona_lcb.py OUT_DIR

Reads OUT_DIR/*_axisproj.pt (8_lcb_axis_projection.py) and the transcripts they came from,
and writes OUT_DIR/persona_analysis.md (+ .json). The baseline is the unsteered run under
the model's default system prompt (coef_+0.00); every other condition is compared with it
on the problems they share, as a paired mean difference with its standard error. Shifts are
also given in units of the axis's raw norm at the layer (the Assistant-vs-average-role gap
on the axis-extraction data).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

from summarize_steered_lcb import stats

BASELINE = "coef_+0.00"
SEGMENTS = ("system", "user", "last_prompt", "thinking", "answer")
TRACE_BINS = ((0, 256), (256, 1024), (1024, 4096), (4096, 1 << 30))


def value(record: dict, segment: str, layer: int) -> float | None:
    if segment == "last_prompt":
        return float(record["last_prompt"][layer])
    v = record["segment_means"].get(segment)
    return None if v is None else float(v[layer])


def paired(cond: dict, base: dict, segment: str, layer: int) -> tuple[float, float, int]:
    diffs = [value(cond[i], segment, layer) - value(base[i], segment, layer) for i in cond.keys() & base.keys()
             if value(cond[i], segment, layer) is not None and value(base[i], segment, layer) is not None]
    if not diffs:
        return float("nan"), float("nan"), 0
    return float(np.mean(diffs)), float(np.std(diffs, ddof=1) / np.sqrt(len(diffs))) if len(diffs) > 1 else float("nan"), len(diffs)


def mean_of(records: dict, segment: str, layer: int) -> float:
    vals = [v for r in records.values() if (v := value(r, segment, layer)) is not None]
    return float(np.mean(vals)) if vals else float("nan")


def main() -> None:
    out = Path(sys.argv[1])
    files = sorted(out.glob("*_axisproj.pt"))
    if not files:
        raise SystemExit(f"no *_axisproj.pt in {out}")
    conds, transcripts, meta = {}, {}, None
    for f in files:
        d = torch.load(f, map_location="cpu", weights_only=False)
        meta = meta or d
        name = f.name.removesuffix("_axisproj.pt")
        conds[name] = {r["example_id"]: r for r in d["records"]}
        src = Path(d["source"])
        src = src if src.exists() else out / src.name
        transcripts[name] = [json.loads(line) for line in src.read_text().splitlines() if line.strip()]
    layer, norms = meta["target_layer"], meta["axis_raw_norm"]
    profile_layers = sorted({*range(len(norms) // 8, len(norms), len(norms) // 8), len(norms) - 1})
    gap = float(norms[layer])
    base = conds.get(BASELINE)
    if base is None:
        raise SystemExit(f"baseline {BASELINE}_axisproj.pt missing")
    order = [BASELINE] + sorted((c for c in conds if c.startswith("coef_") and c != BASELINE), key=lambda c: float(c[5:])) \
        + sorted((c for c in conds if c.startswith("persona_")), key=lambda c: mean_of(conds[c], "thinking", layer), reverse=True)

    lines = ["# Assistant-axis projection of LiveCodeBench transcripts (auto-generated)", "",
             f"Layer {layer}. Projection = residual output · unit axis. Axis raw norm here = {gap:.2f}: the Assistant-minus-average-role "
             "gap on the axis-extraction data, so Δ/gap = 1 means a shift as large as default-Assistant vs a typical role there. "
             f"Δ is paired against `{BASELINE}` (default system prompt, unsteered) on shared problems, ± standard error. "
             "Steered runs (`coef_±x`) are projected unsteered: they measure the text produced, not the added vector.", "",
             "| condition | n | pass | mean new tok | last prompt tok | thinking | answer | Δ last prompt | Δ thinking (/gap) | Δ answer (/gap) |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    result = {}
    for c in order:
        recs = conds[c]
        st = stats(transcripts[c])
        row = {"n": len(recs), "pass": st["pass"], "mean_new_tokens": st["mean_new_tokens"]}
        for seg in SEGMENTS:
            row[f"mean_{seg}"] = mean_of(recs, seg, layer)
            row[f"delta_{seg}"] = paired(recs, base, seg, layer)
        result[c] = row
        fmt = lambda t: f"{t[0]:+.2f} ± {t[1]:.2f} ({t[0] / gap:+.2f})" if t[2] else "—"  # noqa: E731
        lines.append(f"| {c} | {row['n']} | {row['pass']} | {row['mean_new_tokens']:.0f} | {row['mean_last_prompt']:.2f} | {row['mean_thinking']:.2f} | "
                     f"{row['mean_answer']:.2f} | {row['delta_last_prompt'][0]:+.2f} | {fmt(row['delta_thinking'])} | {fmt(row['delta_answer'])} |")

    lines += ["", "## Thinking projection by position (generated-token index), layer " + str(layer), "",
              "| condition | " + " | ".join(f"{a}–{b if b < 1 << 29 else '∞'}" for a, b in TRACE_BINS) + " |", "|---|" + "---:|" * len(TRACE_BINS)]
    for c in order:
        cells = []
        for a, b in TRACE_BINS:
            vals = []
            for r in conds[c].values():
                n_think = r["segment_tokens"].get("thinking", 0)
                seg = r["trace"][: n_think].astype(np.float32)[a:b]
                if len(seg):
                    vals.append(float(seg.mean()))
            cells.append(f"{np.mean(vals):.2f} (n={len(vals)})" if vals else "—")
        lines.append(f"| {c} | " + " | ".join(cells) + " |")

    lines += ["", "## Δ thinking vs baseline by layer, in units of the axis norm at that layer", "",
              "| condition | " + " | ".join(f"L{l}" for l in profile_layers) + " |", "|---|" + "---:|" * len(profile_layers)]
    for c in order[1:]:
        lines.append(f"| {c} | " + " | ".join(f"{paired(conds[c], base, 'thinking', l)[0] / float(norms[l]):+.2f}" for l in profile_layers) + " |")

    lines += ["", "## Behavior (transcript statistics; marker rates per 1000 words)", "",
              "| condition | think closed | truncated | answer words | role word in thinking | role word in answer | mystical/poetic (think) | roleplay asterisks (answer) | self-identity (think) |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for c in order:
        st, recs = stats(transcripts[c]), transcripts[c]
        role = recs[0].get("persona")
        role_re = re.compile(r"\b" + re.escape(role), re.I) if role else None
        word = lambda field: f"{np.mean([bool(role_re.search(r[field])) for r in recs]):.2f}" if role_re else "—"  # noqa: E731
        lines.append(f"| {c} | {st['think_closed']}/{st['n']} | {st['truncated']} | {st['mean_answer_words']:.0f} | {word('thinking')} | {word('answer')} | "
                     f"{st['think_mystical_poetic_per_1k']:.2f} | {st['answer_roleplay_asterisk_per_1k']:.2f} | {st['think_self_identity_per_1k']:.2f} |")

    lines += ["", "## Openings (first two problems of each persona run)", ""]
    for c in order:
        if not c.startswith("persona_"):
            continue
        lines.append(f"### {c}")
        for r in transcripts[c][:2]:
            lines.append(f"- system: {r['system_prompt']!r}")
            lines.append(f"  think: {r['thinking'][:400].replace(chr(10), ' ')}")
            lines.append(f"  answer: {r['answer'][:300].replace(chr(10), ' ') if r['think_closed'] else '(thinking not closed)'}")
        lines.append("")

    (out / "persona_analysis.json").write_text(json.dumps({"layer": layer, "axis_gap": gap, "conditions": result}, indent=2) + "\n")
    (out / "persona_analysis.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[: 7 + len(order)]))


if __name__ == "__main__":
    main()
