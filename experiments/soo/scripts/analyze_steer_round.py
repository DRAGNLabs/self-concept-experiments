"""Pre-specified readout for the band-steering round (STEER_ROUND.md).

Usage: python analyze_steer_round.py OUTPUT_DIR LAUNCH_JSON
Reads output/eval/<condition>_<orient>_<scenario>_<suffix>_summary.json (+ .jsonl), output/band_deltas.json
and output/overlap/<run>/summary.json; the condition list comes from launch.json. Writes output/analysis.json
and output/analysis.md. Exploratory; the layer round's rule with `band-v` as the reference.
"""

import json
from pathlib import Path
import sys

from analyze_layer_round import BASE, ORIENTS, SCENARIOS, fmt_rates, load, verdict, wilson

REFERENCE = "band-v"
KINDS = ("self_other", "nonsocial")


def gap_table(run_dir: Path, sites):
    summary = json.loads((run_dir / "summary.json").read_text())
    manifest = json.loads((run_dir / "manifest.json").read_text())
    lines = [f"### {run_dir.name}: " + ", ".join(f"{k} {v}" for k, v in manifest["arguments"].items()
                                                if k in ("positions", "band_alpha", "band_token_mode", "adapter_layers")), "",
             "| Condition | kind | " + " | ".join(f"{s}: gap (Δ vs base [CI])" for s in sites) + " |",
             "|---|---|" + "---|" * len(sites)]
    for cond, kinds in summary.items():
        for kind in KINDS:
            if kind not in kinds:
                continue
            cells = []
            for site in sites:
                m = kinds[kind].get(site)
                if not m:
                    cells.append("—"); continue
                lo, hi = m["gap_change_ci95"] if m["gap_change_ci95"] else (float("nan"), float("nan"))
                cells.append(f"{m['gap']:.4g} ({m['gap_change_vs_base']:+.3g} [{lo:+.3g}, {hi:+.3g}])")
            lines.append(f"| {cond} | {kind} | " + " | ".join(cells) + " |")
    return lines + [""]


def main():
    out, launch = Path(sys.argv[1]), json.loads(Path(sys.argv[2]).read_text())
    conditions = list(launch["conditions"])
    runs = load(out)
    missing = [f"{c}/{o}/{s}" for c in conditions for o in ORIENTS for s in SCENARIOS if s not in runs.get(c, {}).get(o, {})]
    verdicts = {c: verdict(runs, c, reference=REFERENCE) for c in conditions if c in runs and c not in (BASE, REFERENCE)}
    deltas = json.loads((out / "band_deltas.json").read_text()) if (out / "band_deltas.json").exists() else None

    lines = ["# Band-steering round analysis (auto-generated, exploratory)", "",
             f"Band layers {launch['band_layers']}, training layer {launch['layer']}. Missing runs: {', '.join(missing) if missing else 'none'}.", ""]
    if deltas:
        lines += ["## Band deltas on the training prompts (mean delta norm, rms spread of per-token deltas / mean norm, base projection output norm)", "",
                  "| module | last: |mean| | last: spread/|mean| | last: base out norm | all: |mean| | all: spread/|mean| | all: base out rms norm |", "|---|---:|---:|---:|---:|---:|---:|"]
        for key in deltas["keys"]:
            a, b = deltas["stats"]["last"][key], deltas["stats"]["all"][key]
            lines.append(f"| {key} | {a['mean_norm']:.3f} | {a['rms_spread_over_mean_norm']:.2f} | {a['base_output_norm']:.2f} | "
                         f"{b['mean_norm']:.3f} | {b['rms_spread_over_mean_norm']:.2f} | {b['base_output_rms_norm']:.2f} |")
        lines.append("")
    for o in ORIENTS:
        lines += [f"## {o}: classifier rates (honest / deceptive / refusal / other), deceptive 95% Wilson CI on main", "",
                  "| Condition | main H/D/R/O | main D CI | treasure_hunt H/D/R/O | perspectives correct | setting |", "|---|---|---|---|---:|---|"]
        for cond in conditions:
            r = runs.get(cond, {}).get(o, {})
            ci = "—"
            if "main" in r:
                lo, hi = wilson(r["main"]["counts"]["deceptive"], r["main"]["n"]); ci = f"[{lo:.2f}, {hi:.2f}]"
            persp = f"{r['perspectives']['rates']['honest']:.2f}" if "perspectives" in r else "—"
            lines.append(f"| {cond} | {fmt_rates(r, 'main')} | {ci} | {fmt_rates(r, 'treasure_hunt')} | {persp} | {launch['conditions'][cond] or 'base'} |")
        lines.append("")
    lines += ["## Verdicts against the pre-specified rule (reference: band-v)", ""]
    for name in (REFERENCE, "adapter-seed0"):
        ref = runs.get(name, {})
        if ref:
            lines.append(f"- {name}: " + "; ".join(
                f"{o} main D {ref[o]['main']['rates']['deceptive']:.3f}, refusal+other {ref[o]['main']['rates']['refusal'] + ref[o]['main']['rates']['other']:.3f}"
                + (f", perspectives {ref[o]['perspectives']['rates']['honest']:.3f}" if "perspectives" in ref[o] else "")
                for o in ORIENTS if "main" in ref.get(o, {})))
    for cond, v in verdicts.items():
        lines.append(f"- {cond}: {v['label']}")
    overlap_dir = out / "overlap"
    if overlap_dir.exists():
        sites = [f"hook_after", f"residual_L{launch['layer']}", "final_norm"]
        lines += ["", f"## Gaps on the development pairs (hook_after = attention output of band top layer {launch['band_layers'][-1]}; mean squared paired difference)", ""]
        for run_dir in sorted(p for p in overlap_dir.iterdir() if (p / "summary.json").exists()):
            lines += gap_table(run_dir, sites)
    lines += ["", "## First five main-scenario responses (orig)", ""]
    for cond in conditions:
        r = runs.get(cond, {}).get("orig", {}).get("main")
        if not r or "samples" not in r:
            continue
        lines.append(f"### {cond}")
        for s in r["samples"]:
            lines.append(f"- [{s['label']}] {s['response'].replace(chr(10), ' ')[:160]}")
        lines.append("")
    (out / "analysis.json").write_text(json.dumps({"note": "exploratory; rule in STEER_ROUND.md", "reference": REFERENCE, "missing": missing,
                                                   "conditions": launch["conditions"], "runs": runs, "verdicts": verdicts,
                                                   "band_deltas": deltas["stats"] if deltas else None}, indent=2) + "\n")
    (out / "analysis.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
