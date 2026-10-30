"""Chart a capability-retention comparison (FINDINGS "Capability retention with IFBench for the gi arms — 2026-10-29") from
`capability_eval_compare.py` output.

Four line panels, one per benchmark (GSM8K, MMLU-Pro, HumanEval, IFBench), each showing accuracy change vs unsteered in
points with the paired 95% CI over an equally spaced, labelled dose axis, the cap arm as a categorical point to the right, and
the ±2-point retention rule as a shaded band. Under them a real table (header row, gridlines, alternating shading) with the
accuracy, Δ, 95% CI, McNemar p and gains/losses for every benchmark × arm. Numbers come from the compare json only.

    python experiments/schemer-axis/scripts/capeval_chart.py --model muse [--compare <compare_gi.json>] [--out <path>.png]
    python experiments/schemer-axis/scripts/capeval_chart.py --model gptoss
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
        sub="Role set v2 axis (9 deceptive vs 9 honest personas, no cheating/test-gaming roles). Steering at layer 17 in units of the layer's mean residual norm; the cap arm clips the gi projection at the layer-wise p25 over layers 12–23.\n"
            "Δ = arm accuracy minus unsteered on the same items; error bars are 95% CIs of the paired difference; p from the exact McNemar test on discordant items.",
    ),
    "muse_v2": dict(
        compare=SCRATCH / "muse_glimmer/gi_v2/capeval/compare_gi_v2.json",
        out=SCRATCH / "muse_glimmer/gi_v2/capeval/capeval_gi_v2_chart.png",
        dose_arms=[("gi_m0.16", "−0.16"), ("gi_m0.08", "−0.08"), ("gi_m0.04", "−0.04"), ("unsteered", "0\nunsteered"), ("gi_p0.04", "+0.04"), ("gi_p0.08", "+0.08")],
        side_arms=[("gi_cap_p25", "cap p25\n(layers 17–33)")],
        col_labels=["gi −0.16", "gi −0.08", "gi −0.04", "unsteered", "gi +0.04", "gi +0.08", "cap p25"],
        title="Muse-Glimmer-30B, generic + intent deceiver axis (role set v2): capability retention",
        layer="26",
        sub="Role set v2 axis (9 deceptive vs 9 honest personas, no cheating/test-gaming roles). Steering at layer 26 in units of the layer's mean residual norm; the cap arm clips the gi projection at the layer-wise p25 over layers 17–33.\n"
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
    gs = fig.add_gridspec(3, 2, height_ratios=[3.2, 3.2, 4.6], hspace=0.42, wspace=0.14, left=0.065, right=0.985, top=0.895, bottom=0.03)
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
        ax.set_xticklabels([lab for _, lab in dose_arms] + [lab for _, lab in side_arms], fontsize=9, color=INK)
        ax.set_xlim(-0.6, xpos(all_arms[-1]) + 0.7)
        ax.set_ylim(*ylim)
        ax.set_ylabel("Δ accuracy vs unsteered, points", color=INK2, fontsize=9.5)
        ax.set_xlabel(f"steering dose α at layer {P['layer']}" + ("          │  cap arm" if side_arms else ""), color=INK2, fontsize=9)
        ax.set_title(f"{letters[i]}   {label}: unsteered {base(task):.1f}%  ({n_desc})", fontsize=10.5, color=INK, loc="left", fontweight="bold", pad=6)
        if i == 0:
            handles = [Line2D([], [], color=C_LINE, marker="o", lw=2.0, label="gi dose arms, Δ with 95% CI"),
                       Line2D([], [], color=C_CAP, marker="D", lw=0, label="cap arm"),
                       plt.Rectangle((0, 0), 1, 1, fc=C_BAND, alpha=0.15, ec="none", label=f"retention rule: within ±{RULE_POINTS:.0f} points")]
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
    widths = [0.21] + [0.79 / len(all_arms)] * len(all_arms)
    tbl = axt.table(cellText=cell_text, colLabels=["metric"] + P["col_labels"][: len(all_arms)], colWidths=widths, loc="upper center", cellLoc="center", bbox=[0.0, 0.0, 1.0, 0.94])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
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
