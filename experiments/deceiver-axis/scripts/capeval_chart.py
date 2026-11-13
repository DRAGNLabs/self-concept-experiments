"""Chart a capability-retention comparison (FINDINGS "Capability retention with IFBench for the gi arms — 2026-10-29") from
`capability_eval_compare.py` output.

Four line panels, one per benchmark (GSM8K, MMLU-Pro, HumanEval, IFBench), each showing accuracy change vs unsteered in
points with the paired 95% CI over an equally spaced, labelled dose axis, the cap arm as a categorical point to the right, and
the ±2-point retention rule as a shaded band. Under them a real table (header row, gridlines, alternating shading) with the
accuracy, Δ, 95% CI, McNemar p and gains/losses for every benchmark × arm. Numbers come from the compare json only.

    python experiments/deceiver-axis/scripts/capeval_chart.py --model muse [--compare <compare_gi.json>] [--out <path>.png]
    python experiments/deceiver-axis/scripts/capeval_chart.py --model gptoss
    python experiments/deceiver-axis/scripts/capeval_chart.py --model gemma_v2   # Gemma 4 31B, layer 30, no cap arm
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SCRATCH = Path.home() / "nobackup/autodelete/self-concept-experiments/schemer-axis"

SURFACE, INK, INK2, INK3, RULE, SHADE = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8985", "#dedcd7", "#f1f0ec"
C_LINE = "#1f4e9e"   # the dose series
C_CAP = "#2a7f62"    # cap arm
C_BAND = "#b8651b"   # retention-rule band

TASKS = [("gsm8k", "GSM8K", "1,319 test problems"), ("mmlu_pro", "MMLU-Pro", "1,400-item stratified subsample"),
         ("humaneval", "HumanEval", "164 problems"), ("ifbench", "IFBench", "300 prompts, loose accuracy")]

PRESETS = {
    "muse": dict(
        compare=SCRATCH / "muse_glimmer/capeval/compare_gi.json",
        out=SCRATCH / "muse_glimmer/capeval/capeval_gi_chart.png",
        dose_arms=[("gi_m0.16", "−0.16"), ("gi_m0.08", "−0.08"), ("gi_m0.04", "−0.04"), ("unsteered", "0\nunsteered"), ("gi_p0.04", "+0.04"), ("gi_p0.08", "+0.08")],
        side_arms=[("gi_cap_p25", "cap p25\n(layers 17–33)")],
        col_labels=["gi −0.16", "gi −0.08", "gi −0.04", "unsteered", "gi +0.04", "gi +0.08", "cap p25"],
        title="Muse-Glimmer-30B, generic + intent deceiver axis: capability retention",
        layer="26",
        sub="Steering at layer 26 in units of the layer's mean residual norm; the cap arm clips the gi projection at the layer-wise p25 over layers 17–33.\n"
            "Δ = arm accuracy minus unsteered on the same items; error bars are 95% CIs of the paired difference; p from the exact McNemar test on discordant items.",
    ),
    "gptoss": dict(
        compare=SCRATCH / "capeval/compare_gi.json",
        out=SCRATCH / "capeval/capeval_gi_chart.png",
        dose_arms=[("gi_m0.2", "−0.2"), ("gi_m0.1", "−0.1"), ("unsteered", "0\nunsteered"), ("gi_p0.1", "+0.1"), ("gi_p0.2", "+0.2")],
        side_arms=[("gi_cap_p25", "cap p25\n(layers 12–23)")],
        col_labels=["gi −0.2", "gi −0.1", "unsteered", "gi +0.1", "gi +0.2", "cap p25"],
        title="gpt-oss-120b, generic + intent deceiver axis: capability retention",
        layer="17",
        sub="Steering at layer 17 in units of the layer's mean residual norm; the cap arm clips the gi projection at the layer-wise p25 over layers 12–23.\n"
            "Δ = arm accuracy minus unsteered on the same items; error bars are 95% CIs of the paired difference; p from the exact McNemar test on discordant items.",
    ),
    "gptoss_v2": dict(
        compare=SCRATCH / "gi_v2/capeval/compare_gi_v2.json",
        out=SCRATCH / "gi_v2/capeval/capeval_gi_v2_chart.png",
        dose_arms=[("gi_m0.4", "−0.4"), ("gi_m0.2", "−0.2"), ("gi_m0.1", "−0.1"), ("unsteered", "0\nunsteered"), ("gi_p0.1", "+0.1"), ("gi_p0.2", "+0.2"), ("gi_p0.4", "+0.4")],
        side_arms=[("gi_cap_p25", "cap p25\n(layers 12–23)")],
        col_labels=["gi −0.4", "gi −0.2", "gi −0.1", "unsteered", "gi +0.1", "gi +0.2", "gi +0.4", "cap p25"],
        title="gpt-oss-120b, generic + intent deceiver axis (role set v2): capability retention",
        layer="17",
        sub="Role set v2 axis (9 deceptive vs 9 honest personas, no cheating/test-gaming roles). Steering at layer 17 in units of the layer's mean residual norm;\n"
            "the cap arm clips the gi projection at the layer-wise p25 over layers 12–23.\n"
            "Δ = arm accuracy minus unsteered on the same items; error bars are 95% CIs of the paired difference; p from the exact McNemar test on discordant items.",
    ),
    "muse_v2": dict(
        compare=SCRATCH / "muse_glimmer/gi_v2/capeval/compare_gi_v2.json",
        out=SCRATCH / "muse_glimmer/gi_v2/capeval/capeval_gi_v2_chart.png",
        dose_arms=[("gi_m0.16", "−0.16"), ("gi_m0.08", "−0.08"), ("gi_m0.04", "−0.04"), ("unsteered", "0\nunsteered"), ("gi_p0.04", "+0.04"), ("gi_p0.08", "+0.08"), ("gi_p0.16", "+0.16")],
        side_arms=[("gi_cap_p25", "cap p25\n(layers 17–33)")],
        col_labels=["gi −0.16", "gi −0.08", "gi −0.04", "unsteered", "gi +0.04", "gi +0.08", "gi +0.16", "cap p25"],
        title="Muse-Glimmer-30B, generic + intent deceiver axis (role set v2): capability retention",
        layer="26",
        sub="Role set v2 axis (9 deceptive vs 9 honest personas, no cheating/test-gaming roles). Steering at layer 26 in units of the layer's mean residual norm;\n"
            "the cap arm clips the gi projection at the layer-wise p25 over layers 17–33.\n"
            "Δ = arm accuracy minus unsteered on the same items; error bars are 95% CIs of the paired difference; p from the exact McNemar test on discordant items.",
    ),
    "gemma_v2_rand1729": dict(
        compare=SCRATCH / "gemma4_31b_gi_v2/capeval/compare_gemma_gi_v2_rand1729.json",
        out=SCRATCH / "gemma4_31b_gi_v2/capeval/capeval_gemma_gi_v2_rand1729_chart.png",
        dose_arms=[("gi_m0.08", "−0.08"), ("gi_m0.04", "−0.04"), ("unsteered", "0\nunsteered"), ("gi_p0.04", "+0.04"), ("gi_p0.08", "+0.08")],
        side_arms=[("rand1729_a0.01", "random 1729\n+0.01")],
        side_label="random-direction arm",
        col_labels=["gi −0.08", "gi −0.04", "unsteered", "gi +0.04", "gi +0.08", "rand 1729 +0.01"],
        title="Gemma 4 31B, gi_v2 doses and the random direction 1729: capability retention",
        layer="30",
        sub="Role set v2 axis (Koby's persona_contrast.pt) at the hard108 doses, plus the norm-matched random direction seed 1729 at α 0.01 (the dose that lifts the judged hack rate).\n"
            "Add-steering at layer 30 in units of the layer's mean residual norm. Δ = arm accuracy minus unsteered on the same items; error bars are 95% CIs of the paired difference; p from the exact McNemar test on discordant items.",
    ),
    "gemma_v2": dict(
        compare=SCRATCH / "gemma4_31b_gi_v2/capeval/compare_gemma_gi_v2.json",
        out=SCRATCH / "gemma4_31b_gi_v2/capeval/capeval_gemma_gi_v2_chart.png",
        dose_arms=[("gi_m0.08", "−0.08"), ("gi_m0.04", "−0.04"), ("unsteered", "0\nunsteered"), ("gi_p0.04", "+0.04"), ("gi_p0.08", "+0.08")],
        side_arms=[],
        col_labels=["gi −0.08", "gi −0.04", "unsteered", "gi +0.04", "gi +0.08"],
        title="Gemma 4 31B, generic + intent deceiver axis (role set v2): capability retention",
        layer="30",
        sub="Role set v2 axis (Koby's persona_contrast.pt). Add-steering at layer 30 in units of the layer's mean residual norm, the same doses as the hard108 RH scan.\n"
            "Δ = arm accuracy minus unsteered on the same items; error bars are 95% CIs of the paired difference; p from the exact McNemar test on discordant items.",
    ),
    "qwen_v2": dict(
        compare=SCRATCH / "qwen38_27b_gi_v2/capeval/compare_qwen_gi_v2.json",
        out=SCRATCH / "qwen38_27b_gi_v2/capeval/capeval_qwen_gi_v2_chart.png",
        dose_arms=[("gi_m0.32", "−0.32"), ("gi_m0.16", "−0.16"), ("gi_m0.08", "−0.08"), ("unsteered", "0\nunsteered"), ("gi_p0.08", "+0.08"), ("gi_p0.16", "+0.16"), ("gi_p0.24", "+0.24"), ("gi_p0.32", "+0.32")],
        side_arms=[("gi_cap_p25", "cap p25\n(layers 22–42)")],
        col_labels=["gi −0.32", "gi −0.16", "gi −0.08", "unsteered", "gi +0.08", "gi +0.16", "gi +0.24", "gi +0.32", "cap p25"],
        title="Qwen3.8-27B, generic + intent deceiver axis (role set v2): capability retention",
        layer="32",
        sub="Role set v2 axis (Koby's Qwen vector). Add-steering at layer 32 in units of the layer's mean residual norm, the same doses as the qwen_hard61 RH scan;\n"
            "the cap arm clips the gi projection at the layer-wise p25 over layers 22–42.\n"
            "Δ = arm accuracy minus unsteered on the same items; error bars are 95% CIs of the paired difference; p from the exact McNemar test on discordant items.",
    ),
    "gptoss_v2_clamp": dict(
        compare=SCRATCH / "gi_v2/capeval/compare_clamp.json",
        out=SCRATCH / "gi_v2/capeval/capeval_clamp_chart.png",
        dose_arms=[("gi_cap_p5", "cap\np5"), ("gi_cap_p25", "cap\np25"), ("gi_cap_p50", "cap\np50"), ("unsteered", "none\n(unsteered)"), ("gi_floor_p50", "floor\np50"), ("gi_floor_p75", "floor\np75"), ("gi_floor_p95", "floor\np95")],
        side_arms=[("gf_cap_p25", "Goodfire\ncap p25"), ("gf_cap_p50", "Goodfire\ncap p50")],
        side_label="Goodfire caps",
        tick_fs=8,
        series_label="gi_v2 cap and floor rungs, Δ with 95% CI",
        xlabel="gi_v2 clamp rung, layers 12–23",
        col_labels=["cap p5", "cap p25", "cap p50", "unsteered", "floor p50", "floor p75", "floor p95", "Goodfire p25", "Goodfire p50"],
        title="gpt-oss-120b, cap and floor arms: capability retention",
        layer="12–23",
        sub="A cap clips the projection on the axis down to a threshold, a floor lifts it up to it, at every position over layers 12–23; thresholds are percentiles of the unsteered projection.\n"
            "Share of generated positions the clamp moves on unsteered activations (clamp_shares.md): gi_v2 cap p5 94–96%, p25 75%, p50 53%; floor p50 47%, p75 70%, p95 90–92%;\n"
            "Goodfire cap p25 77%, p50 56%.\n"
            "Δ = arm accuracy minus the unsteered run (2026-10-29) on the same items; error bars are 95% CIs of the paired difference; p from the exact McNemar test on discordant items.",
    ),
    "muse_v2_clamp": dict(
        compare=SCRATCH / "muse_glimmer/gi_v2/capeval/compare_clamp.json",
        out=SCRATCH / "muse_glimmer/gi_v2/capeval/capeval_clamp_chart.png",
        dose_arms=[("gi_cap_p5", "cap\np5"), ("gi_cap_p25", "cap\np25"), ("gi_cap_p50", "cap\np50"), ("unsteered", "none\n(unsteered)"), ("gi_floor_p50", "floor\np50"), ("gi_floor_p75", "floor\np75"), ("gi_floor_p95", "floor\np95")],
        side_arms=[],
        series_label="gi_v2 cap and floor rungs, Δ with 95% CI",
        xlabel="gi_v2 clamp rung, layers 17–33",
        col_labels=["cap p5", "cap p25", "cap p50", "unsteered", "floor p50", "floor p75", "floor p95"],
        title="Muse-Glimmer-30B, cap and floor arms: capability retention",
        layer="17–33",
        sub="A cap clips the projection on the gi_v2 axis down to a threshold, a floor lifts it up to it, at every position over layers 17–33; thresholds are percentiles of the unsteered projection.\n"
            "Share of generated positions the clamp moves on unsteered activations (clamp_shares.md): cap p5 94–97%, p25 75%, p50 50%; floor p50 50%, p75 75%, p95 94–97%.\n"
            "Δ = arm accuracy minus the unsteered run (2026-10-29) on the same items; error bars are 95% CIs of the paired difference; p from the exact McNemar test on discordant items.",
    ),
    "muse_gfar": dict(
        compare=SCRATCH / "muse_glimmer/gi_v2/capeval/compare_gfar.json",
        out=SCRATCH / "muse_glimmer/gi_v2/capeval/capeval_gfar_chart.png",
        dose_arms=[("gf_m0.16", "−0.16"), ("gf_m0.08", "−0.08"), ("unsteered", "0\nunsteered"), ("gf_p0.08", "+0.08"), ("gf_p0.16", "+0.16")],
        side_arms=[],
        series_label="Goodfire-axis dose arms, Δ with 95% CI",
        col_labels=["Goodfire −0.16", "Goodfire −0.08", "unsteered", "Goodfire +0.08", "Goodfire +0.16"],
        title="Muse-Glimmer-30B, Goodfire reward-hacking axis: capability retention",
        layer="26",
        sub="Koby's Goodfire reward-hacking axis (hack minus honest; positive = toward hacking). Add-steering at layer 26 in units of the layer's mean residual norm, the doses of the judged task-set arms.\n"
            "Δ = arm accuracy minus the unsteered run (2026-10-29) on the same items; error bars are 95% CIs of the paired difference; p from the exact McNemar test on discordant items.",
    ),
    # Goodfire axis, every benchmarked arm per model (doses + caps). compare_gfar_full.json = rows copied from the
    # existing compare files (doses: compare_gfar; caps: compare_clamp / compare_gfcap95 / compare_gfcap90 / compare_gfcap7595), nothing recomputed.
    "gptoss_gfar_full": dict(
        compare=SCRATCH / "gi_v2/capeval/compare_gfar_full.json",
        out=SCRATCH / "gi_v2/capeval/capeval_gfar_full_chart.png",
        dose_arms=[("gf_m0.4", "−0.4"), ("gf_m0.2", "−0.2"), ("gf_m0.1", "−0.1"), ("unsteered", "0\nunsteered"), ("gf_p0.1", "+0.1"), ("gf_p0.2", "+0.2"), ("gf_p0.4", "+0.4")],
        side_arms=[("gf_cap_p25", "cap\np25"), ("gf_cap_p50", "cap\np50"), ("gf_cap_p75", "cap\np75"), ("gf_cap_p95", "cap\np95")],
        side_label="caps, layers 12–23",
        tick_fs=8,
        series_label="Goodfire-axis dose arms, Δ with 95% CI",
        col_labels=["dose −0.4", "dose −0.2", "dose −0.1", "unsteered", "dose +0.1", "dose +0.2", "dose +0.4", "cap p25", "cap p50", "cap p75", "cap p95"],
        title="gpt-oss-120b, Goodfire reward-hacking axis: capability retention, steering doses and caps",
        layer="17",
        sub="Koby's Goodfire reward-hacking axis (hack minus honest; positive = toward hacking). Doses: add-steering at layer 17 in units of the layer's mean residual norm.\n"
            "Caps: the projection on the axis is clipped down to its unsteered 25th / 50th / 75th / 95th percentile at every position over layers 12–23; on unsteered activations the caps move\n"
            "77% (p25), 56% (p50), 36% (p75) and 9–12% (p95) of generated positions.\n"
            "Δ = arm accuracy minus the unsteered run (2026-10-29) on the same items; error bars are 95% CIs of the paired difference; p from the exact McNemar test on discordant items.",
    ),
    "muse_gfar_full": dict(
        compare=SCRATCH / "muse_glimmer/gi_v2/capeval/compare_gfar_full.json",
        out=SCRATCH / "muse_glimmer/gi_v2/capeval/capeval_gfar_full_chart.png",
        dose_arms=[("gf_m0.16", "−0.16"), ("gf_m0.08", "−0.08"), ("unsteered", "0\nunsteered"), ("gf_p0.08", "+0.08"), ("gf_p0.16", "+0.16")],
        side_arms=[("gf_cap_p90", "cap\np90"), ("gf_cap_p95", "cap\np95")],
        side_label="caps, layers 17–33",
        series_label="Goodfire-axis dose arms, Δ with 95% CI",
        col_labels=["dose −0.16", "dose −0.08", "unsteered", "dose +0.08", "dose +0.16", "cap p90", "cap p95"],
        title="Muse-Glimmer-30B, Goodfire reward-hacking axis: capability retention, steering doses and caps",
        layer="26",
        sub="Koby's Goodfire reward-hacking axis (hack minus honest; positive = toward hacking). Doses: add-steering at layer 26 in units of the layer's mean residual norm.\n"
            "Caps: the projection on the axis is clipped down to its unsteered 90th / 95th percentile at every position over layers 17–33; on unsteered activations the p90 cap moves 50.7% of prompt\n"
            "and 12.5% of generated positions, the p95 cap 37.6% and 4.7–7.2%. The p25 cap has no benchmark run (the model stops submitting code on the task set).\n"
            "Δ = arm accuracy minus the unsteered run (2026-10-29) on the same items; error bars are 95% CIs of the paired difference; p from the exact McNemar test on discordant items.",
    ),
    "gemma_v2_clamp": dict(
        compare=SCRATCH / "gemma4_31b_gi_v2/capeval/compare_clamp.json",
        out=SCRATCH / "gemma4_31b_gi_v2/capeval/capeval_gemma_clamp_chart.png",
        dose_arms=[("gi_cap_p25", "cap\np25"), ("gi_cap_p35", "cap\np35"), ("gi_cap_p45", "cap\np45"), ("gi_cap_p50", "cap\np50"), ("gi_cap_p65", "cap\np65"), ("unsteered", "none\n(unsteered)")],
        side_arms=[],
        series_label="gi_v2 cap rungs, Δ with 95% CI",
        xlabel="gi_v2 cap rung, layers 20–38",
        col_labels=["cap p25", "cap p35", "cap p45", "cap p50", "cap p65", "unsteered"],
        title="Gemma 4 31B, gi_v2 cap ladder: capability retention",
        layer="20–38",
        sub="A cap clips the projection on the gi_v2 axis down to a threshold at every position over layers 20–38; thresholds are percentiles of the unsteered projection.\n"
            "Share of generated positions the cap moves on unsteered activations (clamp_shares.md): p25 75%, p35 64%, p45 55%, p50 50%, p65 36%. One unsteered run (2026-11-02) is the reference.\n"
            "Δ = arm accuracy minus unsteered on the same items; error bars are 95% CIs of the paired difference; p from the exact McNemar test on discordant items.",
    ),
    "qwen_gfar": dict(
        compare=SCRATCH / "qwen38_27b_gi_v2/capeval/compare_gfar.json",
        out=SCRATCH / "qwen38_27b_gi_v2/capeval/capeval_qwen_gfar_chart.png",
        dose_arms=[("gf_m0.08", "−0.08"), ("unsteered", "0\nunsteered"), ("gf_p0.08", "+0.08")],
        side_arms=[],
        series_label="Goodfire-axis dose arms, Δ with 95% CI",
        col_labels=["dose −0.08", "unsteered", "dose +0.08"],
        title="Qwen3.8-27B, Goodfire reward-hacking axis: capability retention, steering doses",
        layer="32",
        sub="Goodfire reward-hacking axis built for Qwen (hack minus honest; positive = toward hacking). Doses: add-steering at layer 32 in units of the layer's mean residual norm.\n"
            "The p25 cap has no benchmark run (the model stops submitting code on the task set). One unsteered run (2026-11-10) is the reference.\n"
            "Δ = arm accuracy minus unsteered on the same items; error bars are 95% CIs of the paired difference; p from the exact McNemar test on discordant items.",
    ),
    "qwen_v2_clamp": dict(
        compare=SCRATCH / "qwen38_27b_gi_v2/capeval/compare_floor.json",
        out=SCRATCH / "qwen38_27b_gi_v2/capeval/capeval_qwen_clamp_chart.png",
        dose_arms=[("gi_cap_p25", "cap\np25"), ("unsteered", "none\n(unsteered)"), ("gi_floor_p50", "floor\np50"), ("gi_floor_p75", "floor\np75"), ("gi_floor_p95", "floor\np95")],
        side_arms=[],
        series_label="gi_v2 cap and floor rungs, Δ with 95% CI",
        xlabel="gi_v2 clamp rung, layers 22–42",
        col_labels=["cap p25", "unsteered", "floor p50", "floor p75", "floor p95"],
        title="Qwen3.8-27B, cap and floor arms: capability retention",
        layer="22–42",
        sub="A cap clips the projection on the gi_v2 axis down to a threshold, a floor lifts it up to it, at every position over layers 22–42; thresholds are percentiles of the unsteered projection.\n"
            "Share of generated positions the clamp moves on unsteered activations (clamp_shares.md): cap p25 74%; floor p50 50%, p75 75%, p95 94–96%. One unsteered run is the reference (no rerun).\n"
            "Δ = arm accuracy minus unsteered on the same items; error bars are 95% CIs of the paired difference; p from the exact McNemar test on discordant items.",
    ),
}
RULE_POINTS = 2.0
SIDE_GAP = 1.0


def fmt_p(p):
    if p is None:
        return "—"
    return "< 0.001" if p < 0.001 else f"{p:.3f}" if p < 0.01 else f"{p:.2f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", choices=sorted(PRESETS), default="muse")
    ap.add_argument("--compare", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--dpi", type=int, default=220)
    args = ap.parse_args()
    P = PRESETS[args.model]
    compare = args.compare or P["compare"]
    out = args.out or P["out"]
    cmp_ = json.load(open(compare))

    dose_arms, side_arms = P["dose_arms"], [a for a in P["side_arms"] if a[0] in next(iter(cmp_.values()))]
    dose_names = [a for a, _ in dose_arms]
    all_arms = dose_names + [a for a, _ in side_arms]
    tasks = [t for t in TASKS if t[0] in cmp_]

    def xpos(arm):
        return float(dose_names.index(arm)) if arm in dose_names else len(dose_names) + SIDE_GAP + [a for a, _ in side_arms].index(arm)

    def rec(task, arm):
        return None if arm == "unsteered" else cmp_[task][arm]

    def base(task):
        return 100 * next(iter(cmp_[task].values()))["acc_base"]

    # shared y range so the CI widths are comparable across benchmarks
    lo = min(100 * r["ci_lo"] for t, _, _ in tasks for a in all_arms if (r := rec(t, a)))
    hi = max(100 * r["ci_hi"] for t, _, _ in tasks for a in all_arms if (r := rec(t, a)))
    ylim = (min(lo, -RULE_POINTS) - 1.2, max(hi, RULE_POINTS) + 1.2)

    fig = plt.figure(figsize=(13.5, 13.0), facecolor=SURFACE)
    top = 0.895 - 0.013 * max(0, P["sub"].rstrip("\n").count("\n") - 1)  # lower the panels when the subtitle wraps onto extra lines
    gs = fig.add_gridspec(3, 2, height_ratios=[3.2, 3.2, 4.6], hspace=0.42, wspace=0.14, left=0.065, right=0.985, top=top, bottom=0.03)
    axes = [fig.add_subplot(gs[i // 2, i % 2]) for i in range(4)]
    axt = fig.add_subplot(gs[2, :])
    axt.set_axis_off()

    letters = "abcd"
    for i, ((task, label, n_desc), ax) in enumerate(zip(tasks, axes)):
        ax.set_facecolor(SURFACE)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(RULE)
        ax.tick_params(colors=INK2, labelsize=9, length=0)
        ax.grid(axis="y", color=RULE, lw=0.6)
        ax.set_axisbelow(True)
        ax.axhspan(-RULE_POINTS, RULE_POINTS, color=C_BAND, alpha=0.08, lw=0, zorder=0)
        ax.axhline(0, color=INK3, lw=0.9, ls=(0, (4, 3)), zorder=1)
        if side_arms:
            ax.axvline(len(dose_names) - 0.5 + SIDE_GAP / 2, color=RULE, lw=0.8, ls=(0, (2, 3)))
        xs, ys, el, eh = [], [], [], []
        for a in dose_names:
            r = rec(task, a)
            xs.append(xpos(a))
            if r is None:
                ys.append(0.0); el.append(0.0); eh.append(0.0)
            else:
                d = 100 * r["diff"]
                ys.append(d); el.append(d - 100 * r["ci_lo"]); eh.append(100 * r["ci_hi"] - d)
        ax.errorbar(xs, ys, yerr=[el, eh], fmt="none", ecolor=C_LINE, elinewidth=1.0, capsize=2.5, alpha=0.75, zorder=2)
        ax.plot(xs, ys, color=C_LINE, lw=2.0, marker="o", ms=6, mec=SURFACE, mew=1.2, zorder=3)
        for k, (x, y, a) in enumerate(zip(xs, ys, dose_names)):
            if a == "unsteered":
                continue
            # label sits to the right of the point, on the side the outgoing (or, for the last point, incoming) segment leaves free
            nxt = ys[k + 1] if k + 1 < len(ys) else y
            prv = ys[k - 1] if k > 0 else y
            slope = (nxt - y) if k + 1 < len(ys) else (y - prv)
            above = slope <= 0
            ax.text(x + 0.12, y + (0.35 if above else -0.35), f"{y:+.1f}", fontsize=8, color=C_LINE, ha="left", va="bottom" if above else "top", zorder=5)
        for a, _ in side_arms:
            r = rec(task, a)
            d = 100 * r["diff"]
            ax.errorbar([xpos(a)], [d], yerr=[[d - 100 * r["ci_lo"]], [100 * r["ci_hi"] - d]], fmt="none", ecolor=C_CAP, elinewidth=1.0, capsize=2.5, alpha=0.75, zorder=2)
            ax.plot([xpos(a)], [d], marker="D", ms=6.5, color=C_CAP, mec=SURFACE, mew=1.2, ls="none", zorder=3)
            ax.text(xpos(a) + 0.12, d + 0.35, f"{d:+.1f}", fontsize=8, color=C_CAP, ha="left", va="bottom", zorder=5)
        ax.set_xticks([xpos(a) for a in all_arms])
        ax.set_xticklabels([lab for _, lab in dose_arms] + [lab for _, lab in side_arms], fontsize=P.get("tick_fs", 9), color=INK)
        ax.set_xlim(-0.6, xpos(all_arms[-1]) + 0.7)
        ax.set_ylim(*ylim)
        ax.set_ylabel("Δ accuracy vs unsteered, points", color=INK2, fontsize=9.5)
        ax.set_xlabel(P.get("xlabel", f"steering dose α at layer {P['layer']}") + (f"          │  {P.get('side_label', 'cap arm')}" if side_arms else ""), color=INK2, fontsize=9)
        ax.set_title(f"{letters[i]}   {label}: unsteered {base(task):.1f}%  ({n_desc})", fontsize=10.5, color=INK, loc="left", fontweight="bold", pad=6)
        if i == 0:
            handles = [Line2D([], [], color=C_LINE, marker="o", lw=2.0, label=P.get("series_label", "gi dose arms, Δ with 95% CI"))]
            if side_arms:
                handles.append(Line2D([], [], color=C_CAP, marker="D", lw=0, label=P.get("side_label", "cap arm")))
            handles += [plt.Rectangle((0, 0), 1, 1, fc=C_BAND, alpha=0.15, ec="none", label=f"retention rule: within ±{RULE_POINTS:.0f} points")]
            ax.legend(handles=handles, loc="lower left", fontsize=8.5, frameon=False, labelcolor=INK2)

    # table
    rows = []
    for task, label, _ in tasks:
        def acc_cell(a, task=task):
            r = rec(task, a)
            if r is None:
                return f"{base(task):.1f}"
            flag = "†" if (r["ci_lo"] > 0 or r["ci_hi"] < 0) else ""
            return f"{100 * r['acc_arm']:.1f} ({100 * r['diff']:+.1f}){flag}"
        rows.append((f"{label} accuracy, % (Δ)", True, acc_cell))
        rows.append(("   95% CI of Δ", False, lambda a, task=task: "—" if rec(task, a) is None else f"{100 * rec(task, a)['ci_lo']:+.1f} to {100 * rec(task, a)['ci_hi']:+.1f}"))
        rows.append(("   McNemar p · gains / losses", False, lambda a, task=task: "—" if rec(task, a) is None else f"{fmt_p(rec(task, a)['mcnemar_p'])} · {rec(task, a)['gains']} / {rec(task, a)['losses']}"))
    cell_text = [[n] + [f(a) for a in all_arms] for n, _, f in rows]
    label_w = 0.21 if len(all_arms) <= 9 else 0.17  # wide tables (10+ arms): narrower label column, smaller font, so no cell text is clipped
    widths = [label_w] + [(1 - label_w) / len(all_arms)] * len(all_arms)
    tbl = axt.table(cellText=cell_text, colLabels=["metric"] + P["col_labels"][: len(all_arms)], colWidths=widths, loc="upper center", cellLoc="center", bbox=[0.0, 0.0, 1.0, 0.94])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5 if len(all_arms) <= 9 else 7.4)
    group = 0
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(RULE)
        cell.set_linewidth(0.6)
        if r == 0:
            cell.set_text_props(color=INK, fontweight="bold")
            cell.set_facecolor(SHADE)
            continue
        g = (r - 1) // 3
        cell.set_facecolor(SHADE if g % 2 == 1 else SURFACE)
        if c == 0:
            cell.set_text_props(color=INK, ha="left", fontweight="bold" if rows[r - 1][1] else "normal")
            cell.PAD = 0.03
        else:
            cell.set_text_props(color=INK, fontweight="bold" if "†" in cell.get_text().get_text() else "normal")
    axt.set_title("e   The numbers behind panels a–d (Δ = arm minus unsteered on the same items; † 95% CI excludes zero)", fontsize=10.5, color=INK, loc="left", fontweight="bold", pad=2)

    fig.suptitle(P["title"], x=0.065, ha="left", fontsize=13, color=INK, fontweight="bold", y=0.975)
    fig.text(0.065, 0.945, P["sub"], fontsize=8.8, color=INK2, ha="left", va="top")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, facecolor=SURFACE)
    if out.suffix == ".png":
        fig.savefig(out.with_suffix(".pdf"), facecolor=SURFACE)
    print(out)


if __name__ == "__main__":
    main()
