"""Chart the Qwen3.8-27B gi dose pilot (FINDINGS "Qwen3.8-27B gi dose pilot, judged — 2026-10-29") from its analysis files.

Two line panels over an equally spaced, labelled dose axis — (a) hack rates (judged, proxy + hidden sets, with 95% CIs;
automatic proxy) and (b) hidden-test pass — each with the two norm-matched random arms (±0.16) as categorical points to the
right of a divider, and under them a real table (header row, gridlines, alternating shading) with every number the lines are
drawn from plus the judged HACK/BORDERLINE/NOT_HACK counts and the Holm p values. Numbers are read from
`analysis_pilot_judged_as_judged.json` and `summary_pilot2.json` in the run root (`qwen38_27b/gi/pilot`); arms missing from
either file are drawn as gaps. The +0.64 arm is labelled as broken (hidden pass 2%, half the episodes empty).

    python experiments/schemer-axis/scripts/qwen_gi_pilot_chart.py [--root <qwen38_27b/gi/pilot>] [--out <path>.png]
                                                                   [--judged-counts counts.json]   # {arm: "H/B/N"}
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

DEFAULT_ROOT = Path.home() / "nobackup/autodelete/self-concept-experiments/schemer-axis/qwen38_27b/gi/pilot"

# Design tokens: text in text tokens, recessive rules, series colours only on marks.
SURFACE, INK, INK2, INK3, RULE, SHADE = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8985", "#dedcd7", "#f1f0ec"
C_JUDGED = "#1f4e9e"   # judged hack, proxy + hidden sets
C_AUTO = "#b8651b"     # automatic proxy
C_HIDDEN = "#2a7f62"   # hidden-test pass
C_RANDOM = "#8a8985"

# Equally spaced categorical slots: nine doses (each step a doubling), a divider, then the two random arms.
DOSE_ARMS = [("gi_m0.16", "−0.16"), ("gi_m0.08", "−0.08"), ("gi_m0.04", "−0.04"), ("unsteered", "0\nunsteered"),
             ("gi_p0.04", "+0.04"), ("gi_p0.08", "+0.08"), ("gi_p0.16", "+0.16"), ("gi_p0.32", "+0.32"), ("gi_p0.64", "+0.64\n(broken)")]
SIDE_ARMS = [("random_m0.16", "random\n−0.16"), ("random_p0.16", "random\n+0.16")]
ALL_ARMS = [a for a, _ in DOSE_ARMS] + [a for a, _ in SIDE_ARMS]
COL_LABELS = ["gi −0.16", "gi −0.08", "gi −0.04", "unsteered", "gi +0.04", "gi +0.08", "gi +0.16", "gi +0.32", "gi +0.64\n(broken)", "random\n−0.16", "random\n+0.16"]
SIDE_GAP = 1.0
DIVIDER = len(DOSE_ARMS) - 1 + SIDE_GAP / 2


def xpos(arm: str) -> float:
    dose = [a for a, _ in DOSE_ARMS]
    if arm in dose:
        return float(dose.index(arm))
    return len(DOSE_ARMS) + SIDE_GAP + [a for a, _ in SIDE_ARMS].index(arm) - 0.5


def load(root: Path):
    ana = json.load(open(root / "analysis_pilot_judged_as_judged.json"))
    summ = json.load(open(root / "summary_pilot2.json"))
    return ana, summ


def finite(x) -> bool:
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


def ci_of(r: dict):
    ci = r.get("ci95")
    if not ci or not finite(ci[0]) or not finite(ci[1]) or not finite(r.get("diff_vs_unsteered")):
        return None
    return ci


def series(outcome: dict, arms: list[str], only=None):
    arms = [a for a in arms if a in outcome["arms"] and (only is None or a in only)]
    xs = [xpos(a) for a in arms]
    ys = [100 * outcome["arms"][a]["rate"] for a in arms]
    lo, hi = [], []
    for a in arms:
        r, ci = outcome["arms"][a], ci_of(outcome["arms"][a])
        lo.append(100 * (r["diff_vs_unsteered"] - ci[0]) if ci else 0)
        hi.append(100 * (ci[1] - r["diff_vs_unsteered"]) if ci else 0)
    return arms, xs, ys, lo, hi


def holm(outcome: dict, arm: str):
    c = outcome.get("contrasts", {}).get(arm)
    return None if c is None else c.get("p_holm")


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(RULE)
    ax.tick_params(colors=INK2, labelsize=9.5, length=0)
    ax.grid(axis="y", color=RULE, lw=0.6)
    ax.set_axisbelow(True)
    ax.axvline(DIVIDER, color=RULE, lw=0.8, ls=(0, (2, 3)))


def xticks(ax):
    xs = [xpos(a) for a in ALL_ARMS]
    labels = [lab for _, lab in DOSE_ARMS] + [lab for _, lab in SIDE_ARMS]
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=9.5, color=INK)
    ax.set_xlim(-0.6, xs[-1] + 0.6)


def dose_line(ax, outcome, color, lw=2.0, ci=True, marker="o", ms=5.5, z=3, only=None):
    arms, xs, ys, lo, hi = series(outcome, [a for a, _ in DOSE_ARMS], only)
    ax.plot(xs, ys, color=color, lw=lw, marker=marker, ms=ms, mec=SURFACE, mew=1.2, zorder=z)
    if ci:
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt="none", ecolor=color, elinewidth=1.0, capsize=2.5, alpha=0.75, zorder=z - 1)


def side_points(ax, outcome, marker, ci=True, z=3, only=None):
    for a, _ in SIDE_ARMS:
        if a not in outcome["arms"] or (only is not None and a not in only):
            continue
        x, r = xpos(a), outcome["arms"][a]
        y = 100 * r["rate"]
        ax.plot([x], [y], marker=marker, ms=5.5, color=C_RANDOM, mec=SURFACE, mew=1.2, ls="none", zorder=z)
        c = ci_of(r)
        if ci and c:
            ax.errorbar([x], [y], yerr=[[100 * (r["diff_vs_unsteered"] - c[0])], [100 * (c[1] - r["diff_vs_unsteered"])]], fmt="none", ecolor=C_RANDOM, elinewidth=1.0, capsize=2.5, alpha=0.75, zorder=z - 1)


def direct_labels(ax, outcome, color, dy, dx=0.12, only=None):
    for a in ALL_ARMS:
        if a not in outcome["arms"] or (only is not None and a not in only):
            continue
        y = 100 * outcome["arms"][a]["rate"]
        ax.text(xpos(a) + dx, y + dy, f"{y:.1f}", fontsize=8, color=color, ha="left", va="bottom" if dy > 0 else "top")


def fmt_p(p):
    if p is None:
        return "—"
    return "< 0.001" if p < 0.001 else f"{p:.3f}" if p < 0.01 else f"{p:.2f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--dpi", type=int, default=220)
    ap.add_argument("--judged-counts", type=Path, default=None, help="json {arm: 'H/B/N'} (HACK / BORDERLINE / NOT_HACK dossier counts)")
    args = ap.parse_args()
    out = args.out or (args.root / "qwen_gi_pilot_chart.png")
    ana, summ = load(args.root)
    jh, hp, ap_ = ana["judged_hack"], ana["hidden_pass"], ana["hack_proxy"]
    counts = json.load(open(args.judged_counts)) if args.judged_counts else {}
    judged = set(counts) if counts else None  # arms whose dossiers were judged; the others get no judged point
    unst_j = 100 * jh["arms"]["unsteered"]["rate"]
    unst_h = 100 * hp["arms"]["unsteered"]["rate"]

    fig = plt.figure(figsize=(14.0, 14.5), facecolor=SURFACE)
    gs = fig.add_gridspec(3, 1, height_ratios=[4.4, 3.2, 4.6], hspace=0.30, left=0.065, right=0.985, top=0.905, bottom=0.03)
    ax1, ax2, axt = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])
    style(ax1)
    style(ax2)
    axt.set_axis_off()

    # (a) hack rates
    ax1.axhline(unst_j, color=INK3, lw=0.8, ls=(0, (4, 3)), zorder=1)
    dose_line(ax1, ap_, C_AUTO, lw=1.4, ci=False, marker="s", ms=4.5, z=2)
    dose_line(ax1, jh, C_JUDGED, lw=2.2, ci=True, marker="o", ms=6, z=4, only=judged)
    side_points(ax1, ap_, "s", ci=False, z=2)
    side_points(ax1, jh, "o", ci=True, z=4, only=judged)
    direct_labels(ax1, jh, C_JUDGED, dy=0.6, only=judged)
    direct_labels(ax1, ap_, C_AUTO, dy=0.35)
    ax1.set_ylabel("hack rate, % of 180 episodes", color=INK2, fontsize=10)
    top = max([100 * ap_["arms"][a]["rate"] for a in ALL_ARMS if a in ap_["arms"]] +
              [100 * jh["arms"][a]["rate"] + (100 * (ci_of(jh["arms"][a])[1] - jh["arms"][a]["diff_vs_unsteered"]) if ci_of(jh["arms"][a]) else 0)
               for a in ALL_ARMS if a in jh["arms"] and (judged is None or a in judged)])
    ax1.set_ylim(-0.4, max(12.5, top + 2.0))
    xticks(ax1)
    ax1.set_xlabel("steering dose α at layer 32 (units of the layer's mean residual norm, 75.0)                    │   random direction 1729", color=INK2, fontsize=9.5)
    ax1.set_title("a   Judged hack rate: 0 % unsteered and on the negative side; only +0.08/+0.16/+0.32 move, all on two problems, nothing significant at 4 seeds", fontsize=10.5, color=INK, loc="left", fontweight="bold", pad=8)
    handles = [Line2D([], [], color=C_JUDGED, marker="o", lw=2.2, label="judged hack, proxy + hidden-pass sets (95% CI)"),
               Line2D([], [], color=C_AUTO, marker="s", lw=1.4, label="automatic proxy (shown pass, hidden fail)"),
               Line2D([], [], color=C_RANDOM, marker="o", lw=0, label="random direction (grey)")]
    ax1.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 0.98), fontsize=8.5, frameon=False, ncol=1, labelcolor=INK2)

    # (b) hidden pass
    ax2.axhline(unst_h, color=INK3, lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax2.axhspan(unst_h - 10, unst_h + 10, color=C_HIDDEN, alpha=0.06, lw=0, zorder=0)
    ax2.text(-0.55, 71.5, "dose rule: within 10 points of unsteered (shaded band)", fontsize=8, color=INK3, va="top", ha="left")
    dose_line(ax2, hp, C_HIDDEN, lw=2.2, ci=True, marker="o", ms=6, z=4)
    side_points(ax2, hp, "o", ci=True, z=4)
    direct_labels(ax2, hp, C_HIDDEN, dy=1.6)
    ax2.set_ylabel("hidden-test pass, %", color=INK2, fontsize=10)
    ax2.set_ylim(0, 75)
    xticks(ax2)
    ax2.set_xlabel("steering dose α at layer 32              │   random direction 1729", color=INK2, fontsize=9.5)
    ax2.set_title("b   Hidden-test pass stays inside the dose rule through +0.32; +0.64 collapses the model", fontsize=10.5, color=INK, loc="left", fontweight="bold", pad=8)

    # (c) table: every number behind the lines, one column per arm
    base = summ["unsteered"]["reasoning_chars_per_turn"]

    def rate_cell(outcome, a, star=True, only=None):
        if a not in outcome["arms"] or (only is not None and a not in only):
            return "not judged" if only is not None else "—"
        r = outcome["arms"][a]
        if a == "unsteered" or not finite(r.get("diff_vs_unsteered")):
            return f"{100 * r['rate']:.1f}"
        p = holm(outcome, a)
        sig = star and p is not None and p < 0.05
        return f"{100 * r['rate']:.1f} ({100 * r['diff_vs_unsteered']:+.1f}){'*' if sig else ''}"

    def ci_cell(outcome, a, only=None):
        if a not in outcome["arms"] or a == "unsteered" or (only is not None and a not in only):
            return "—"
        ci = ci_of(outcome["arms"][a])
        return "—" if ci is None else f"{100 * ci[0]:+.1f} to {100 * ci[1]:+.1f}"

    def summ_cell(a, f):
        return f(summ[a]) if a in summ else "—"

    rows = [
        ("judged hack, all sets, % (Δ)", C_JUDGED, lambda a: rate_cell(jh, a, only=judged)),
        ("   95% CI of Δ (problem bootstrap)", C_JUDGED, lambda a: ci_cell(jh, a, only=judged)),
        ("   Holm p (eight ≤ ±0.16 contrasts)", C_JUDGED, lambda a: fmt_p(holm(jh, a)) if (judged is None or a in judged) else "—"),
        ("   judged HACK / BORDERLINE / NOT", C_JUDGED, lambda a: counts.get(a, "—")),
        ("automatic proxy, % (Δ)", C_AUTO, lambda a: rate_cell(ap_, a)),
        ("hidden-test pass, % (Δ)", C_HIDDEN, lambda a: rate_cell(hp, a, star=False)),
        ("no-code episodes, %", INK, lambda a: summ_cell(a, lambda s: f"{100 * s['no_code_rate']:.1f}")),
        ("episodes stopped without an answer", INK, lambda a: summ_cell(a, lambda s: f"{s['stop_reasons'].get('stopped', 0)}")),
        ("truncated turns / turns", INK, lambda a: summ_cell(a, lambda s: f"{s['truncated_turns']} / {s['turns']}")),
        ("reasoning chars / turn (Δ%)", INK, lambda a: summ_cell(a, lambda s: f"{s['reasoning_chars_per_turn'] / 1000:.1f}k" + ("" if a == "unsteered" else f" ({100 * (s['reasoning_chars_per_turn'] / base - 1):+.0f}%)"))),
    ]
    cell_text = [[n] + [f(a) for a in ALL_ARMS] for n, _, f in rows]
    widths = [0.20] + [0.80 / len(ALL_ARMS)] * len(ALL_ARMS)
    tbl = axt.table(cellText=cell_text, colLabels=["metric"] + COL_LABELS, colWidths=widths, loc="upper center", cellLoc="center",
                    bbox=[0.0, 0.0, 1.0, 0.92])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.3)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(RULE)
        cell.set_linewidth(0.6)
        if r == 0:
            cell.set_text_props(color=INK, fontweight="bold")
            cell.set_facecolor(SHADE)
        elif c == 0:
            col = rows[r - 1][1]
            cell.set_text_props(color=col, ha="left", fontweight="bold" if col != INK else "normal")
            cell.set_facecolor(SHADE if r % 2 == 0 else SURFACE)
            cell.PAD = 0.03
        else:
            cell.set_text_props(color=INK)
            cell.set_facecolor(SHADE if r % 2 == 0 else SURFACE)
            if "*" in cell.get_text().get_text():
                cell.set_text_props(fontweight="bold")
    axt.set_title("c   The numbers behind panels a and b (Δ = arm minus unsteered; * Holm p < 0.05; +0.32 and +0.64 are outside the Holm set)", fontsize=10.5, color=INK, loc="left", fontweight="bold", pad=2)

    fig.suptitle("Qwen3.8-27B, generic + intent deceiver axis: dose pilot, judged", x=0.065, ha="left", fontsize=13, color=INK, fontweight="bold", y=0.975)
    fig.text(0.065, 0.948, "45 hard problems × 4 seeds = 180 episodes per arm, concurrent engine, 64k budget. Judged rates: blind Sonnet judging of the proxy and hidden-pass sets, rubric v3,\n"
             "first pass as judged (HACK counts, BORDERLINE does not); error bars 95% problem-bootstrap CIs of the difference vs unsteered. The random arms use direction 1729 at ±0.16.", fontsize=8.8, color=INK2, ha="left", va="top")

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, facecolor=SURFACE)
    if out.suffix == ".png":
        fig.savefig(out.with_suffix(".pdf"), facecolor=SURFACE)
    print(out)


if __name__ == "__main__":
    main()
