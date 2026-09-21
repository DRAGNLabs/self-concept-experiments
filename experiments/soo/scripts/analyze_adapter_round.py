"""Pre-specified analysis for the existing-adapter round (ADAPTER_ROUND.md).

Usage: python analyze_adapter_round.py OUTPUT_DIR
Reads every completed run under OUTPUT_DIR (a manifest.json marks completion),
writes OUTPUT_DIR/analysis.json and OUTPUT_DIR/analysis.md. Thresholds are the
exploratory ones written in ADAPTER_ROUND.md before the run was launched.
"""

import json
from pathlib import Path
import sys

import numpy as np

from selfconcept.soo.overlap import paired_interval

SITES = ()  # derived from the first completed run: hook_after, residual blocks in order, final_norm
PRIMARY = ("hook_after", "final_norm")
ADAPTER_TYPES = ("original", "agentic", "mixed")
CONTRACTION_PCT = -10.0
DAMAGE_RATIO = 0.5
VARIANCE_DROP_PCT = -5.0


def load_run(path):
    if not (path / "manifest.json").exists():
        return None
    summary = json.loads((path / "summary.json").read_text())
    records = [json.loads(l) for l in (path / "pairs.jsonl").read_text().splitlines() if l.strip()]
    return {"summary": summary, "records": records, "name": path.name}


def per_pair(records, condition, kind, site, field="gap_change_vs_base"):
    rows = sorted((r for r in records if r["condition"] == condition and r["kind"] == kind and r["site"] == site), key=lambda r: r["id"])
    return [r["id"] for r in rows], [r["family"] for r in rows], np.array([r[field] for r in rows])


def cell(summary, condition, kind, site):
    base, cond = summary["base"][kind][site], summary[condition][kind][site]
    pct = 100 * cond["gap_change_vs_base"] / base["gap"]
    ci = cond["gap_change_ci95"]
    return {
        "base_gap": base["gap"], "gap": cond["gap"],
        "change": cond["gap_change_vs_base"], "change_pct": pct,
        "ci95": ci, "ci95_pct": [100 * c / base["gap"] for c in ci] if ci else None,
        "centroid_gap_base": base["centroid_gap"], "centroid_gap": cond["centroid_gap"],
        "norm_change_pct": 100 * ((cond["mean_self_norm"] + cond["mean_other_norm"]) / (base["mean_self_norm"] + base["mean_other_norm"]) - 1),
        "prompt_variance_change_pct": 100 * (cond["prompt_variance"] / base["prompt_variance"] - 1),
        "displacement": cond["mean_squared_displacement_vs_base"],
    }


def main(out_dir):
    out_dir = Path(out_dir)
    runs = {p.name: load_run(p) for p in sorted(out_dir.iterdir()) if p.is_dir()}
    missing = [k for k, v in runs.items() if v is None]
    runs = {k: v for k, v in runs.items() if v is not None}
    global SITES
    first = next(iter(runs.values()))["summary"]["base"]["self_other"]
    SITES = ("hook_after", *sorted((k for k in first if k.startswith("residual_L")), key=lambda k: int(k.removeprefix("residual_L"))), "final_norm")
    result = {"note": "Exploratory, unadjusted intervals. Percentages scale raw paired changes by the observed base gap. Seed averages are taken within each pair, not pooled as independent pairs.",
              "incomplete_runs": missing, "runs": {}, "base_consistency": {}, "adapter_types": {}, "contrasts": {}, "steering_reference": {}}

    # Base gaps should agree across runs (same model, inputs, precision).
    bases = {}
    for name, run in runs.items():
        for site in SITES:
            for kind in ("self_other", "nonsocial"):
                bases.setdefault((kind, site), {})[name] = run["summary"]["base"][kind][site]["gap"]
    for (kind, site), values in bases.items():
        vals = np.array(list(values.values()))
        result["base_consistency"][f"{kind}/{site}"] = {"min": vals.min().item(), "max": vals.max().item(), "max_rel_spread": ((vals.max() - vals.min()) / vals.mean()).item()}

    # Individual runs: adapters and the steering reference.
    for name, run in runs.items():
        conditions = [c for c in run["summary"] if c != "base"]
        result["runs"][name] = {c: {kind: {site: cell(run["summary"], c, kind, site) for site in SITES}
                                    for kind in ("self_other", "nonsocial")} for c in conditions}
        if name.startswith("steering"):
            result["steering_reference"] = {c: {site: result["runs"][name][c]["self_other"][site]["change_pct"] for site in PRIMARY}
                                            for c in conditions}

    # Seed-averaged changes per adapter type, with family-bootstrap intervals and decision flags.
    seed_avg = {}
    for atype in ADAPTER_TYPES:
        names = sorted(n for n in runs if n.startswith(f"adapter-{atype}-seed"))
        if not names:
            continue
        entry = {"seeds": names, "sites": {}}
        for kind in ("self_other", "nonsocial"):
            for site in SITES:
                stacks, ids, fams = [], None, None
                for n in names:
                    i, f, v = per_pair(runs[n]["records"], "adapter", kind, site)
                    if ids is None:
                        ids, fams = i, f
                    assert i == ids, "pair order differs between seeds"
                    stacks.append(v)
                avg = np.mean(stacks, axis=0)
                seed_avg[(atype, kind, site)] = (fams, avg)
                base_gap = np.mean([runs[n]["summary"]["base"][kind][site]["gap"] for n in names])
                ci = paired_interval(avg, fams)
                per_seed_pct = [100 * runs[n]["summary"]["adapter"][kind][site]["gap_change_vs_base"] / runs[n]["summary"]["base"][kind][site]["gap"] for n in names]
                entry["sites"].setdefault(site, {})[kind] = {
                    "base_gap": base_gap, "seed_avg_change": avg.mean().item(),
                    "seed_avg_change_pct": 100 * avg.mean().item() / base_gap,
                    "ci95_pct": [100 * c / base_gap for c in ci] if ci else None,
                    "per_seed_change_pct": per_seed_pct,
                    "mean_displacement": float(np.mean([runs[n]["summary"]["adapter"][kind][site]["mean_squared_displacement_vs_base"] for n in names])),
                    "mean_prompt_variance_change_pct": float(np.mean([result["runs"][n]["adapter"][kind][site]["prompt_variance_change_pct"] for n in names])),
                }
        flags = {}
        for site in PRIMARY:
            so, ns = entry["sites"][site]["self_other"], entry["sites"][site]["nonsocial"]
            ci = so["ci95_pct"]
            contraction = (so["seed_avg_change_pct"] <= CONTRACTION_PCT and ci is not None and ci[1] < 0
                           and all(p < 0 for p in so["per_seed_change_pct"]))
            damage = (so["seed_avg_change_pct"] < 0 and ns["seed_avg_change_pct"] <= DAMAGE_RATIO * so["seed_avg_change_pct"]) \
                or so["mean_prompt_variance_change_pct"] <= VARIANCE_DROP_PCT
            flags[site] = {"held_out_contraction": bool(contraction), "damage_flag": bool(damage)}
        entry["flags"] = flags
        result["adapter_types"][atype] = entry

    # Agentic minus original, paired on seed-averaged per-pair changes.
    for kind in ("self_other", "nonsocial"):
        for site in SITES:
            if ("agentic", kind, site) in seed_avg and ("original", kind, site) in seed_avg:
                fams, a = seed_avg[("agentic", kind, site)]
                _, o = seed_avg[("original", kind, site)]
                base_gap = result["adapter_types"]["agentic"]["sites"][site][kind]["base_gap"]
                ci = paired_interval(a - o, fams)
                result["contrasts"][f"agentic_minus_original/{kind}/{site}"] = {
                    "difference": (a - o).mean().item(), "difference_pct_of_base": 100 * (a - o).mean().item() / base_gap,
                    "ci95_pct_of_base": [100 * c / base_gap for c in ci] if ci else None}

    (out_dir / "analysis.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    lines = ["# Adapter round analysis (auto-generated, exploratory)", "", f"Incomplete runs: {missing or 'none'}", ""]
    lines += ["## Seed-averaged self/other gap change by adapter type", "", "| Adapter | Site | Change % | 95% CI % | Per-seed % | Name-control % | Displacement |", "|---|---|---:|---|---|---:|---:|"]
    for atype, entry in result["adapter_types"].items():
        for site in SITES:
            so, ns = entry["sites"][site]["self_other"], entry["sites"][site]["nonsocial"]
            ci = so["ci95_pct"]
            lines.append(f"| {atype} | {site} | {so['seed_avg_change_pct']:+.2f} | "
                         f"{'[' + ', '.join(f'{c:+.2f}' for c in ci) + ']' if ci else 'n/a'} | "
                         f"{', '.join(f'{p:+.2f}' for p in so['per_seed_change_pct'])} | {ns['seed_avg_change_pct']:+.2f} | {so['mean_displacement']:.3g} |")
    lines += ["", "## Flags at primary sites", ""]
    for atype, entry in result["adapter_types"].items():
        for site, f in entry["flags"].items():
            lines.append(f"- {atype} / {site}: held-out contraction {f['held_out_contraction']}, damage flag {f['damage_flag']}")
    lines += ["", "## Agentic minus original (seed-averaged, paired)", ""]
    for key, c in result["contrasts"].items():
        ci = c["ci95_pct_of_base"]
        lines.append(f"- {key}: {c['difference_pct_of_base']:+.2f}% of base, CI {'[' + ', '.join(f'{x:+.2f}' for x in ci) + ']' if ci else 'n/a'}")
    lines += ["", "## Steering reference at L32 (self/other gap change %, primary sites)", ""]
    for cond, sites in result["steering_reference"].items():
        lines.append(f"- {cond}: " + ", ".join(f"{s} {v:+.2f}%" for s, v in sites.items()))
    lines += ["", "## Base-gap agreement across runs (max relative spread)", ""]
    worst = max(result["base_consistency"].items(), key=lambda kv: kv[1]["max_rel_spread"])
    lines.append(f"- worst site {worst[0]}: {worst[1]['max_rel_spread']:.2e}")
    (out_dir / "analysis.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1])
