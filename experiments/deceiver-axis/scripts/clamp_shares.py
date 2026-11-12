"""Open-loop clamp shares per cap / floor arm, from the threshold sidecars (read-only, pure stdlib, CPU seconds).

For each arm: tau per band layer (from the arm's own *_summary.json when one exists, else from the thresholds file the
arm's settings line names) is located within each unsteered turn's stored quantiles of the PROMPT-token and the
GENERATED-token projection (the sidecar `*_projection.jsonl` written by the HF thresholds run), by linear interpolation
on the 0.05 .. 0.95 grid. Share moved = share of positions above tau for a cap, below tau for a floor. Shares are
averaged over turns (plain mean; a token-weighted mean is kept in the JSON), then summarised over the band layers.

Where tau lies beyond a turn's 5th / 95th percentile the grid cannot resolve the share: that turn contributes the
interval [0, 0.05] or [0.95, 1]; every mean is then carried as a [lo, hi] interval and printed as a bound or a range.

"Open-loop": the quantiles are of the UNSTEERED model at each layer. In a clamped run the clamp at earlier band layers
changes the projection at later ones, so only the first band layer's share is what a run would see; the rest is an
estimate of the clamp's reach on unsteered activations.

Usage: python3 clamp_shares.py [out_dir]   (default: the directory of this script). Writes clamp_shares.json / .md.
"""
import glob
import json
import os
import sys
import time

N = "/home/jansen05/nobackup/autodelete/self-concept-experiments/schemer-axis"
G = "/grphome/grp_mirror/nobackup/autodelete/self-concept-experiments/schemer-axis"
GF = N + "/gfar_vs_gi_v2/steer/thresholds"

# sidecar directories of the thresholds runs (each holds proj_<scenario>_projection.jsonl and a *_summary.json)
SIDE = {
    ("gpt-oss", "gi_v2"): N + "/gi_v2/thresholds/gi_contrast",
    ("gpt-oss", "gfar"): GF + "/gptoss120b/gfar",
    ("Muse", "gi_v2"): N + "/muse_glimmer/gi_v2/thresholds/gi_contrast",
    ("Muse", "gfar"): GF + "/muse/gfar",
    ("Gemma", "gi_v2"): G + "/gemma4_31b_gi_v2/thresholds/gi_v2_contrast",
    ("Qwen", "gi_v2"): G + "/qwen38_27b_gi_v2/thresholds/gi_v2_contrast",
    ("Qwen", "gfar"): GF + "/qwen38_27b/gfar",
}
# the arms' own task sets (from the run directories / reasoning-file names); budgets are read from the arm summaries
ARM_TASKS = {"gpt-oss": "ib_solvhard_s0+s1+s2 (45 problems)", "Muse": "ib_solvhard_s0+s1+s2 (45 problems)",
             "Gemma": "hard108 (108 problems)", "Qwen": "qwen_hard61 (61 problems)"}
ARM_BUDGET_FALLBACK = {"gpt-oss": 16384, "Muse": 16384, "Gemma": 32768, "Qwen": 65536}  # used only when no summary exists yet

GP, MP = N + "/gi_v2/pilot", N + "/muse_glimmer/gi_v2/pilot"
GE, QW = N + "/gemma4_31b_gi_v2/rh/hard108", N + "/qwen38_27b_gi_v2/rh/qwen_hard61"
GEF = N + "/gemma4_31b_gi_v2/floor/thresholds"
# (model, axis, mode, nominal percentile, arm directory, fallback thresholds file or None)
ARMS = [
    ("gpt-oss", "gi_v2", "cap", 5, GP + "/gi_cap_p5", None),
    ("gpt-oss", "gi_v2", "cap", 25, GP + "/gi_cap_p25", None),
    ("gpt-oss", "gi_v2", "cap", 50, GP + "/gi_cap_p50", None),
    ("gpt-oss", "gi_v2", "floor", 50, GP + "/gi_floor_p50", None),
    ("gpt-oss", "gi_v2", "floor", 75, GP + "/gi_floor_p75", None),
    ("gpt-oss", "gi_v2", "floor", 95, GP + "/gi_floor_p95", None),
    ("gpt-oss", "gfar", "cap", 25, GP + "/gf_cap_p25", None),
    ("gpt-oss", "gfar", "cap", 50, GP + "/gf_cap_p50", None),
    ("gpt-oss", "gfar", "cap", 75, GP + "/gf_cap_p75", None),
    ("gpt-oss", "gfar", "cap", 95, GP + "/gf_cap_p95", None),
    ("Muse", "gi_v2", "cap", 5, MP + "/gi_cap_p5", None),
    ("Muse", "gi_v2", "cap", 25, MP + "/gi_cap_p25", None),
    ("Muse", "gi_v2", "cap", 50, MP + "/gi_cap_p50", None),
    ("Muse", "gi_v2", "floor", 50, MP + "/gi_floor_p50", None),
    ("Muse", "gi_v2", "floor", 75, MP + "/gi_floor_p75", None),
    ("Muse", "gi_v2", "floor", 95, MP + "/gi_floor_p95", None),
    ("Muse", "gfar", "cap", 25, MP + "/gf_cap_p25", None),
    ("Muse", "gfar", "cap", 75, MP + "/gf_cap_p75", GF + "/muse/gfar_p75.json"),
    ("Muse", "gfar", "cap", 95, MP + "/gf_cap_p95", GF + "/muse/gfar_p95.json"),
    ("Gemma", "gi_v2", "cap", 25, GE + "/cap_p25_L20-38", None),
    ("Gemma", "gi_v2", "cap", 35, GE + "/cap_p35_L20-38", None),
    ("Gemma", "gi_v2", "cap", 45, GE + "/cap_p45_L20-38", None),
    ("Gemma", "gi_v2", "cap", 50, GE + "/cap_p50_L20-38", None),
    ("Gemma", "gi_v2", "cap", 65, GE + "/cap_p65_L20-38", None),
    ("Gemma", "gi_v2", "floor", 50, GE + "/floor_p50_L20-38", GEF + "/gi_v2_contrast_p50.json"),
    ("Gemma", "gi_v2", "floor", 75, GE + "/floor_p75_L20-38", GEF + "/gi_v2_contrast_p75.json"),
    ("Gemma", "gi_v2", "floor", 95, GE + "/floor_p95_L20-38", GEF + "/gi_v2_contrast_p95.json"),
    ("Qwen", "gi_v2", "cap", 25, G + "/qwen38_27b_gi_v2/rh/qwen_hard61/cap_p25_L22-42", None),
    ("Qwen", "gi_v2", "floor", 50, QW + "/floor_p50_L22-42", None),
    ("Qwen", "gi_v2", "floor", 75, QW + "/floor_p75_L22-42", None),
    ("Qwen", "gi_v2", "floor", 95, QW + "/floor_p95_L22-42", None),
    ("Qwen", "gfar", "cap", 25, QW + "/gfar_cap_p25_L22-42", None),
]


def frac_below(quantiles: dict, tau: float) -> tuple[float, float]:
    """[lo, hi] for the share of positions with projection below tau, from one turn's stored quantiles."""
    levels = sorted(quantiles, key=float)
    qs, xs = [float(k) for k in levels], [quantiles[k] for k in levels]
    if tau < xs[0]:
        return 0.0, qs[0]
    if tau > xs[-1]:
        return qs[-1], 1.0
    for i in range(len(xs) - 1):
        if xs[i] <= tau <= xs[i + 1]:
            if xs[i + 1] == xs[i]:
                return qs[i + 1], qs[i + 1]
            f = qs[i] + (qs[i + 1] - qs[i]) * (tau - xs[i]) / (xs[i + 1] - xs[i])
            return f, f
    return qs[-1], qs[-1]


def mean_iv(ivs, weights=None):
    weights = weights or [1.0] * len(ivs)
    total = sum(weights)
    return (sum(w * a for w, (a, _) in zip(weights, ivs)) / total, sum(w * b for w, (_, b) in zip(weights, ivs)) / total)


def moved(iv, mode):
    return iv if mode == "floor" else (1.0 - iv[1], 1.0 - iv[0])


def mid(iv):
    return 0.5 * (iv[0] + iv[1])


def fmt(iv) -> str:
    lo, hi = iv
    if hi - lo < 5e-4:
        return f"{100 * lo:.1f}%"
    if lo <= 1e-9 and hi <= 0.05 + 1e-9:
        return "<5%"
    if hi >= 1 - 1e-9 and lo >= 0.95 - 1e-9:
        return ">95%"
    low, high = f"{100 * lo:.1f}", f"{100 * hi:.1f}"
    return f"{low}%" if low == high else f"{low}–{high}%"


def approx(result) -> str:
    """'≈' when a band mean rests on unresolved turns (the gap columns then use interval midpoints)."""
    return "≈" if any(iv[1] - iv[0] > 5e-4 for iv in (result["prompt_share"]["mean"], result["generated_share"]["mean"])) else ""


_side_cache = {}


def load_sidecar(directory):
    if directory not in _side_cache:
        files = sorted(glob.glob(directory + "/*_projection.jsonl"))
        if len(files) != 1:
            raise SystemExit(f"expected one projection sidecar in {directory}, found {files}")
        rows = [json.loads(line) for line in open(files[0])]
        summaries = sorted(glob.glob(directory + "/*_summary.json"))
        summary = json.load(open(summaries[0])) if summaries else {}
        scenario = os.path.basename(files[0]).removeprefix("proj_").removesuffix("_projection.jsonl")
        _side_cache[directory] = (files[0], rows, summary, scenario)
    return _side_cache[directory]


def arm_thresholds(arm_dir, fallback):
    summaries = sorted(glob.glob(arm_dir + "/seed*/*_summary.json"))
    for path in summaries:
        meta = json.load(open(path))
        steering = meta.get("steering") or {}
        if "cap_thresholds" in steering:
            return ({int(k): float(v) for k, v in steering["cap_thresholds"].items()}, steering["mode"], meta.get("max_new_tokens"),
                    f"arm summary ({os.path.basename(os.path.dirname(path))})", len(summaries), steering.get("direction"))
    if fallback and os.path.exists(fallback):
        return ({int(k): float(v) for k, v in json.load(open(fallback)).items()}, None, None,
                f"thresholds file {os.path.basename(fallback)} (no arm summary yet; full path in the JSON)", 0, None)
    return None


def median(values):
    s = sorted(values)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def analyse(model, axis, mode, pct, arm_dir, fallback):
    got = arm_thresholds(arm_dir, fallback)
    name = f"{model} {axis} {mode} p{pct}"
    if got is None:
        return {"arm": name, "arm_dir": arm_dir, "error": "no arm summary and no thresholds file"}
    tau, summary_mode, arm_budget, tau_source, n_summaries, direction = got
    if summary_mode is not None and summary_mode != mode:
        return {"arm": name, "arm_dir": arm_dir, "error": f"summary mode {summary_mode} != expected {mode}"}
    side_file, rows, side_summary, scenario = load_sidecar(SIDE[(model, axis)])
    layers = sorted(tau)
    turns = [r for r in rows if all(r["layers"][str(l)]["generated_projection"].get("n", 0) > 0 for l in layers)]
    gen_n = [r["generated_tokens"] for r in turns]
    prompt_n = [r["prompt_tokens"] for r in turns]
    budget = side_summary.get("max_new_tokens")
    per_layer, below_gen_by_turn, below_prompt_by_turn = {}, [[] for _ in turns], [[] for _ in turns]
    for layer in layers:
        below_gen = [frac_below(r["layers"][str(layer)]["generated_projection"]["quantiles"], tau[layer]) for r in turns]
        below_prompt = [frac_below(r["layers"][str(layer)]["prompt_projection"]["quantiles"], tau[layer]) for r in turns]
        for t in range(len(turns)):
            below_gen_by_turn[t].append(below_gen[t])
            below_prompt_by_turn[t].append(below_prompt[t])
        gen_moved = [moved(iv, mode) for iv in below_gen]
        prompt_moved = [moved(iv, mode) for iv in below_prompt]
        per_layer[layer] = {
            "tau": tau[layer],
            "generated_share": mean_iv(gen_moved), "prompt_share": mean_iv(prompt_moved),
            "generated_share_token_weighted": mean_iv(gen_moved, gen_n), "prompt_share_token_weighted": mean_iv(prompt_moved, prompt_n),
            "turns_unresolved_generated": sum(1 for a, b in below_gen if b - a > 1e-9),
            "turns_unresolved_prompt": sum(1 for a, b in below_prompt if b - a > 1e-9),
        }

    def band(key):
        ivs = [per_layer[l][key] for l in layers]
        lo_layer = min(layers, key=lambda l: mid(per_layer[l][key]))
        hi_layer = max(layers, key=lambda l: mid(per_layer[l][key]))
        return {"mean": mean_iv(ivs), "min": per_layer[lo_layer][key], "min_layer": lo_layer,
                "max": per_layer[hi_layer][key], "max_layer": hi_layer}

    gen_pct_turn = [mean_iv(ivs) for ivs in below_gen_by_turn]      # band-mean percentile of tau within each turn's generated tokens
    prompt_pct_turn = [mean_iv(ivs) for ivs in below_prompt_by_turn]
    diffs = [mid(per_layer[l]["prompt_share"]) - mid(per_layer[l]["generated_share"]) for l in layers]
    diff_lo = [max(0.0, per_layer[l]["prompt_share"][0] - per_layer[l]["generated_share"][1],
                   per_layer[l]["generated_share"][0] - per_layer[l]["prompt_share"][1]) for l in layers]
    # check: tau recomputed from the sidecar as the median over turns of the stored quantile (only for stored percentiles)
    key = str(pct / 100)
    recompute = None
    if all(key in r["layers"][str(layers[0])]["generated_projection"]["quantiles"] for r in turns):
        recompute = max(abs(median([r["layers"][str(l)]["generated_projection"]["quantiles"][key] for r in turns]) - tau[l]) for l in layers)
    return {
        "arm": name, "model": model, "axis": axis, "mode": mode, "nominal_percentile": pct, "arm_dir": arm_dir,
        "tau_source": tau_source, "tau_fallback_file": fallback, "arm_seeds_with_summary": n_summaries, "arm_direction": direction,
        "band_layers": [layers[0], layers[-1]], "n_band_layers": len(layers),
        "sidecar": side_file, "threshold_turns": len(turns), "threshold_examples": len({r["example_id"] for r in turns}),
        "threshold_generated_tokens": gen_n, "threshold_prompt_tokens": prompt_n,
        "threshold_turns_truncated_at_budget": sum(1 for g in gen_n if budget and g >= budget),
        "threshold_task_set": f"{scenario}, {side_summary.get('n')} episodes", "threshold_budget": budget,
        "arm_task_set": ARM_TASKS[model], "arm_budget": arm_budget if arm_budget is not None else ARM_BUDGET_FALLBACK[model],
        "arm_budget_source": "arm summary" if arm_budget is not None else "recipe constant (no summary yet)",
        "tau_recompute_maxdiff": recompute,
        "prompt_share": band("prompt_share"), "generated_share": band("generated_share"),
        "prompt_share_token_weighted_band_mean": mean_iv([per_layer[l]["prompt_share_token_weighted"] for l in layers]),
        "generated_share_token_weighted_band_mean": mean_iv([per_layer[l]["generated_share_token_weighted"] for l in layers]),
        "tau_generated_percentile_by_turn_band_mean": gen_pct_turn,
        "tau_generated_percentile_min_turn": min(gen_pct_turn, key=mid), "tau_generated_percentile_max_turn": max(gen_pct_turn, key=mid),
        "tau_prompt_percentile_by_turn_band_mean": prompt_pct_turn,
        "prompt_minus_generated_band_mean_midpoint": sum(diffs) / len(diffs),
        "abs_prompt_minus_generated_band_mean_midpoint": sum(abs(d) for d in diffs) / len(diffs),
        "abs_prompt_minus_generated_band_mean_lower_bound": sum(diff_lo) / len(diff_lo),
        "max_layer_abs_diff_midpoint": max(abs(d) for d in diffs), "max_layer_abs_diff_layer": layers[max(range(len(diffs)), key=lambda i: abs(diffs[i]))],
        "per_layer": {str(l): per_layer[l] for l in layers},
    }


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
    results = [analyse(*arm) for arm in ARMS]
    stamp = time.strftime("%Y-%m-%d %I:%M %p %Z")
    json.dump({"generated": stamp, "method": __doc__, "arms": results}, open(out_dir + "/clamp_shares.json", "w"), indent=1)

    ok = [r for r in results if "error" not in r]
    md = [f"# Open-loop clamp shares per cap / floor arm ({stamp})", "",
          "Script: `clamp_shares.py` in this directory; full per-layer numbers in `clamp_shares.json`.", "",
          "**What a cell is.** Share of positions the clamp moves on UNSTEERED activations: projection above tau for a cap, below tau for a "
          "floor. Tau is located in each threshold-run turn's stored projection quantiles (0.05 .. 0.95 grid, linear interpolation), separately "
          "for prompt tokens and generated tokens; shares are the plain mean over turns, then mean / min / max over the band layers (min and max "
          "are the layers with the lowest and highest share, layer in brackets).", "",
          "**Bounds.** `<5%` / `>95%`: tau is outside the stored grid in every turn. `a–b%`: tau is outside the grid in some turns, each of which "
          "contributes [0, 5%] or [95%, 100%]; the true mean lies in the range. A single number means every turn resolved.", "",
          "**Open-loop.** Quantiles are of the unsteered model at each layer. In a clamped run the clamp at earlier band layers changes later "
          "layers' projections, so only the first band layer's share is what a run sees directly; the rest measures reach on unsteered activations.", "",
          "**Gap columns** use interval midpoints; `≈` marks a gap that rests on unresolved turns (the ranking table at the end gives its lower bound).", "",
          "**Prompt tokens** are every prompt position of a turn (system, task, and on later turns the earlier turns' text fed back).", ""]
    for model in ("gpt-oss", "Muse", "Gemma", "Qwen"):
        rows = [r for r in ok if r["model"] == model]
        if not rows:
            continue
        md += [f"## {model}", "",
               "| arm | band | PROMPT share moved: band mean | min (layer) | max (layer) | GENERATED share moved: band mean | min (layer) | max (layer) | "
               "tau's generated-token percentile, band mean: lowest turn – highest turn | prompt − generated, band mean (pp) | mean abs per-layer gap (pp) |",
               "|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            p, g = r["prompt_share"], r["generated_share"]
            md.append(f"| {r['axis']} {r['mode']} p{r['nominal_percentile']} | L{r['band_layers'][0]}–{r['band_layers'][1]} | {fmt(p['mean'])} | "
                      f"{fmt(p['min'])} (L{p['min_layer']}) | {fmt(p['max'])} (L{p['max_layer']}) | {fmt(g['mean'])} | {fmt(g['min'])} (L{g['min_layer']}) | "
                      f"{fmt(g['max'])} (L{g['max_layer']}) | {fmt(r['tau_generated_percentile_min_turn'])} – {fmt(r['tau_generated_percentile_max_turn'])} | "
                      f"{approx(r)}{100 * r['prompt_minus_generated_band_mean_midpoint']:+.1f} | "
                      f"{approx(r)}{100 * r['abs_prompt_minus_generated_band_mean_midpoint']:.1f} |")
        md += ["", "Threshold source vs the arm's own run:", "",
               "| arm | tau from | arm seeds with a summary | threshold turns (episodes) | generated tokens per threshold turn | turns cut at the budget | "
               "threshold task set, budget | arm task set, budget | max abs(tau − median of stored quantile) |",
               "|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            check = "interpolated rung, not stored" if r["tau_recompute_maxdiff"] is None else f"{r['tau_recompute_maxdiff']:.2g}"
            md.append(f"| {r['axis']} {r['mode']} p{r['nominal_percentile']} | {r['tau_source']} | {r['arm_seeds_with_summary']} | "
                      f"{r['threshold_turns']} ({r['threshold_examples']}) | {', '.join(str(x) for x in r['threshold_generated_tokens'])} | "
                      f"{r['threshold_turns_truncated_at_budget']} | {r['threshold_task_set']}, {r['threshold_budget']:,} tokens | "
                      f"{r['arm_task_set']}, {r['arm_budget']:,} tokens ({r['arm_budget_source']}) | {check} |")
        md.append("")
    md += ["## Largest prompt-vs-generated gaps (mean over band layers of the absolute per-layer gap, midpoints)", "",
           "| rank | arm | mean abs per-layer gap (pp) | lower bound of that gap given unresolved turns (pp) | prompt band mean | generated band mean | largest single-layer gap (pp, layer) |",
           "|---|---|---|---|---|---|---|"]
    ranked = sorted(ok, key=lambda r: -r["abs_prompt_minus_generated_band_mean_midpoint"])
    for i, r in enumerate(ranked, 1):
        md.append(f"| {i} | {r['arm']}{'' if r['arm_seeds_with_summary'] else ' (no arm summary yet)'} | {100 * r['abs_prompt_minus_generated_band_mean_midpoint']:.1f} | "
                  f"{100 * r['abs_prompt_minus_generated_band_mean_lower_bound']:.1f} | {fmt(r['prompt_share']['mean'])} | {fmt(r['generated_share']['mean'])} | "
                  f"{100 * r['max_layer_abs_diff_midpoint']:.1f} (L{r['max_layer_abs_diff_layer']}) |")
    failed = [r for r in results if "error" in r]
    if failed:
        md += ["", "## Not computed", ""] + [f"- {r['arm']}: {r['error']} (`{r['arm_dir']}`)" for r in failed]
    open(out_dir + "/clamp_shares.md", "w").write("\n".join(md) + "\n")
    print(f"{len(ok)} arms computed, {len(failed)} failed; wrote {out_dir}/clamp_shares.json and .md")
    for r in failed:
        print("FAILED", r["arm"], r["error"])


if __name__ == "__main__":
    main()
