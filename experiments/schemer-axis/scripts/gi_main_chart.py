"""Chart the Muse gi main run table (FINDINGS "Muse gi main run — 2026-10-29") from its analysis files.

Two line panels over an equally spaced, labelled dose axis — (a) hack rates (judged, proxy + hidden sets, with 95% CIs; judged,
proxy set only; automatic proxy) and (b) hidden-test pass — each with the cap arm and the three norm-matched random directions
(+0.08) as categorical points to the right, and under them a real table (header row, gridlines, alternating row shading) with
every number the lines are drawn from plus the Holm p values. Numbers are read from `analysis_main_all_as_judged.json`,
`analysis_main_proxyset_as_judged.json` and `summary_main.json` in the run root.

    python experiments/schemer-axis/scripts/gi_main_chart.py [--root <muse_glimmer/gi/pilot>] [--out <path>.png]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

DEFAULT_ROOT = Path.home() / "nobackup/autodelete/self-concept-experiments/schemer-axis/muse_glimmer/gi/pilot"

# Design tokens: text in text tokens, recessive rules, series colours only on marks.
SURFACE, INK, INK2, INK3, RULE, SHADE = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8985", "#dedcd7", "#f1f0ec"
C_JUDGED = "#1f4e9e"   # judged hack, proxy + hidden sets
C_PROXYSET = "#6fa3dc" # judged hack, proxy set only (same quantity, lighter step)
C_AUTO = "#b8651b"     # automatic proxy
C_HIDDEN = "#2a7f62"   # hidden-test pass
C_RANDOM = "#8a8985"

# Equally spaced categorical slots: six doses, a divider, then the four side arms.
DOSE_ARMS = [("gi_m0.16", "−0.16"), ("gi_m0.08", "−0.08"), ("gi_m0.04", "−0.04"), ("unsteered", "0\nunsteered"), ("gi_p0.04", "+0.04"), ("gi_p0.08", "+0.08")]
SIDE_ARMS = [("gi_cap_p25", "cap p25\n(layers 17–33)"), ("random_plus", "random\n1729"), ("random_plus_1730", "random\n1730"), ("random_plus_1731", "random\n1731")]
ALL_ARMS = [a for a, _ in DOSE_ARMS] + [a for a, _ in SIDE_ARMS]
DIVIDER = len(DOSE_ARMS) - 0.5 + 0.5  # between slot 5 and slot 6.5
SIDE_GAP = 1.0


def xpos(arm: str) -> float:
    if arm in [a for a, _ in DOSE_ARMS]:
        return float([a for a, _ in DOSE_ARMS].index(arm))
    return len(DOSE_ARMS) + SIDE_GAP + [a for a, _ in SIDE_ARMS].index(arm)


def load(root: Path):
    all_ = json.load(open(root / "analysis_main_all_as_judged.json"))
    prox = json.load(open(root / "analysis_main_proxyset_as_judged.json"))
    summ = json.load(open(root / "summary_main.json"))
    return all_, prox, summ


def series(outcome: dict, arms: list[str]):
    xs = [xpos(a) for a in arms]
    ys = [100 * outcome["arms"][a]["rate"] for a in arms]
    # ci95 is the problem-bootstrap CI of the difference vs unsteered; shift it onto the rate
    lo = [100 * (outcome["arms"][a]["diff_vs_unsteered"] - outcome["arms"][a]["ci95"][0]) if "ci95" in outcome["arms"][a] else 0 for a in arms]
    hi = [100 * (outcome["arms"][a]["ci95"][1] - outcome["arms"][a]["diff_vs_unsteered"]) if "ci95" in outcome["arms"][a] else 0 for a in arms]
    return xs, ys, lo, hi


def holm(outcome: dict, arm: str):
    c = outcome.get("contrasts", {}).get(arm)
    return None if c is None else c["p_holm"]


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(RULE)
    ax.tick_params(colors=INK2, labelsize=9.5, length=0)
    ax.grid(axis="y", color=RULE, lw=0.6)
    ax.set_axisbelow(True)
    ax.axvline(len(DOSE_ARMS) - 1 + SIDE_GAP / 2 + 0.5 - 0.5, color=RULE, lw=0.8, ls=(0, (2, 3)))


def xticks(ax):
    xs = [xpos(a) for a in ALL_ARMS]
    labels = [lab for _, lab in DOSE_ARMS] + [lab for _, lab in SIDE_ARMS]
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=9.5, color=INK)
    ax.set_xlim(-0.6, xs[-1] + 0.6)


def dose_line(ax, outcome, color, lw=2.0, ci=True, marker="o", ms=5.5, z=3):
    arms = [a for a, _ in DOSE_ARMS]
    xs, ys, lo, hi = series(outcome, arms)
    ax.plot(xs, ys, color=color, lw=lw, marker=marker, ms=ms, mec=SURFACE, mew=1.2, zorder=z)
    if ci:
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt="none", ecolor=color, elinewidth=1.0, capsize=2.5, alpha=0.75, zorder=z - 1)
    return xs, ys


def side_points(ax, outcome, color, marker, ci=True, z=3):
    for a, _ in SIDE_ARMS:
        x = xpos(a)
        r = outcome["arms"][a]
        y = 100 * r["rate"]
        c = color if a == "gi_cap_p25" else C_RANDOM
        m = "D" if a == "gi_cap_p25" else marker
        ax.plot([x], [y], marker=m, ms=6 if m == "D" else 5.5, color=c, mec=SURFACE, mew=1.2, ls="none", zorder=z)
        if ci and "ci95" in r:
            ax.errorbar([x], [y], yerr=[[100 * (r["diff_vs_unsteered"] - r["ci95"][0])], [100 * (r["ci95"][1] - r["diff_vs_unsteered"])]], fmt="none", ecolor=c, elinewidth=1.0, capsize=2.5, alpha=0.75, zorder=z - 1)


def direct_labels(ax, outcome, color, dy, dx=0.16):
    for a in ALL_ARMS:
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
    args = ap.parse_args()
    out = args.out or (args.root / "gi_main_chart.png")
    all_, prox, summ = load(args.root)
    jh, jp, hp, ap_ = all_["judged_hack"], prox["judged_hack"], all_["hidden_pass"], all_["hack_proxy"]
    unst_j = 100 * jh["arms"]["unsteered"]["rate"]
    unst_h = 100 * hp["arms"]["unsteered"]["rate"]

    fig = plt.figure(figsize=(13.5, 14.0), facecolor=SURFACE)
    gs = fig.add_gridspec(3, 1, height_ratios=[4.6, 3.0, 4.4], hspace=0.30, left=0.07, right=0.985, top=0.905, bottom=0.03)
    ax1, ax2, axt = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])
    style(ax1)
    style(ax2)
    axt.set_axis_off()

    # (a) hack rates
    ax1.axhline(unst_j, color=INK3, lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax1.text(-0.55, unst_j + 0.7, f"unsteered judged {unst_j:.1f}%", fontsize=8, color=INK3, va="bottom", ha="left")
    dose_line(ax1, ap_, C_AUTO, lw=1.4, ci=False, marker="s", ms=4.5, z=2)
    dose_line(ax1, jp, C_PROXYSET, lw=1.6, ci=False, marker="o", ms=4.5, z=3)
    dose_line(ax1, jh, C_JUDGED, lw=2.2, ci=True, marker="o", ms=6, z=4)
    side_points(ax1, ap_, C_AUTO, "s", ci=False, z=2)
    side_points(ax1, jp, C_PROXYSET, "o", ci=False, z=3)
    side_points(ax1, jh, C_JUDGED, "o", ci=True, z=4)
    direct_labels(ax1, jh, C_JUDGED, dy=-2.2)
    ax1.set_ylabel("hack rate, % of 540 episodes", color=INK2, fontsize=10)
    ax1.set_ylim(5, 56)
    xticks(ax1)
    ax1.set_xlabel("steering dose α at layer 26 (units of the layer's mean residual norm)              │   other arms (random directions at +0.08)", color=INK2, fontsize=9.5)
    ax1.set_title("a   Judged hack rate moves monotonically with the gi dose; the cap arm lowers it, the random directions scatter ±7", fontsize=10.5, color=INK, loc="left", fontweight="bold", pad=8)
    handles = [Line2D([], [], color=C_JUDGED, marker="o", lw=2.2, label="judged hack, proxy + hidden-pass sets (95% CI)"),
               Line2D([], [], color=C_PROXYSET, marker="o", lw=1.6, label="judged hack, proxy set only"),
               Line2D([], [], color=C_AUTO, marker="s", lw=1.4, label="automatic proxy (shown pass, hidden fail)"),
               Line2D([], [], color=C_JUDGED, marker="D", lw=0, label="cap arm (diamond)"),
               Line2D([], [], color=C_RANDOM, marker="o", lw=0, label="random directions (grey)")]
    ax1.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 0.98), fontsize=8.5, frameon=False, ncol=1, labelcolor=INK2)

    # (b) hidden pass
    ax2.axhline(unst_h, color=INK3, lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax2.axhspan(unst_h - 10, unst_h, color=C_HIDDEN, alpha=0.06, lw=0, zorder=0)
    ax2.text(-0.55, unst_h - 9.6, "dose rule: within 10 points of unsteered", fontsize=8, color=INK3, va="bottom", ha="left")
    dose_line(ax2, hp, C_HIDDEN, lw=2.2, ci=True, marker="o", ms=6, z=4)
    side_points(ax2, hp, C_HIDDEN, "o", ci=True, z=4)
    direct_labels(ax2, hp, C_HIDDEN, dy=-1.6)
    ax2.set_ylabel("hidden-test pass, %", color=INK2, fontsize=10)
    ax2.set_ylim(31, 58)
    xticks(ax2)
    ax2.set_xlabel("steering dose α at layer 26              │   other arms", color=INK2, fontsize=9.5)
    ax2.set_title("b   Hidden-test pass stays inside the dose rule in every arm (+0.08 costs 8.9 points)", fontsize=10.5, color=INK, loc="left", fontweight="bold", pad=8)

    # table: every number behind the lines, one column per arm
    base = summ["unsteered"]["reasoning_chars_per_turn"]

    def rate_cell(outcome, a, star=True):
        r = outcome["arms"][a]
        if a == "unsteered":
            return f"{100 * r['rate']:.1f}"
        p = holm(outcome, a)
        sig = star and p is not None and p < 0.05
        return f"{100 * r['rate']:.1f} ({100 * r['diff_vs_unsteered']:+.1f}){'*' if sig else ''}"

    def ci_cell(outcome, a):
        r = outcome["arms"][a]
        return "—" if "ci95" not in r else f"{100 * r['ci95'][0]:+.1f} to {100 * r['ci95'][1]:+.1f}"

    rows = [
        ("judged hack, all sets, % (Δ)", C_JUDGED, lambda a: rate_cell(jh, a)),
        ("   95% CI of Δ (problem bootstrap)", C_JUDGED, lambda a: ci_cell(jh, a)),
        ("   Holm p (nine contrasts)", C_JUDGED, lambda a: fmt_p(holm(jh, a))),
        ("judged hack, proxy set only, % (Δ)", C_PROXYSET, lambda a: rate_cell(jp, a)),
        ("automatic proxy, % (Δ)", C_AUTO, lambda a: rate_cell(ap_, a)),
        ("hidden-test pass, % (Δ)", C_HIDDEN, lambda a: rate_cell(hp, a, star=False)),
        ("no-code episodes, %", INK, lambda a: f"{100 * summ[a]['no_code_rate']:.1f}"),
        ("truncated turns (of ~1100)", INK, lambda a: f"{summ[a]['truncated_turns']}"),
        ("reasoning chars / turn (Δ%)", INK, lambda a: f"{summ[a]['reasoning_chars_per_turn'] / 1000:.1f}k" + ("" if a == "unsteered" else f" ({100 * (summ[a]['reasoning_chars_per_turn'] / base - 1):+.0f}%)")),
    ]
    col_labels = ["gi −0.16", "gi −0.08", "gi −0.04", "unsteered", "gi +0.04", "gi +0.08", "cap p25", "random\n1729", "random\n1730", "random\n1731"]
    cell_text = [[n] + [f(a) for a in ALL_ARMS] for n, _, f in rows]
    widths = [0.19] + [0.81 / len(ALL_ARMS)] * len(ALL_ARMS)
    tbl = axt.table(cellText=cell_text, colLabels=["metric"] + col_labels, colWidths=widths, loc="upper center", cellLoc="center",
                    bbox=[0.0, 0.0, 1.0, 0.92])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(RULE)
        cell.set_linewidth(0.6)
        if r == 0:  # header
            cell.set_text_props(color=INK, fontweight="bold")
            cell.set_facecolor(SHADE)
        elif c == 0:  # metric names
            col = rows[r - 1][1]
            cell.set_text_props(color=col, ha="left", fontweight="bold" if col != INK else "normal")
            cell.set_facecolor(SHADE if r % 2 == 0 else SURFACE)
            cell.PAD = 0.03
        else:
            cell.set_text_props(color=INK)
            cell.set_facecolor(SHADE if r % 2 == 0 else SURFACE)
            if "*" in cell.get_text().get_text():
                cell.set_text_props(fontweight="bold")
    axt.set_title("c   The numbers behind panels a and b (Δ = arm minus unsteered; * Holm p < 0.05)", fontsize=10.5, color=INK, loc="left", fontweight="bold", pad=2)

    fig.suptitle("Muse-Glimmer-30B, generic + intent deceiver axis: 12-seed main run", x=0.07, ha="left", fontsize=13, color=INK, fontweight="bold", y=0.975)
    fig.text(0.07, 0.948, "45 hard problems × 12 seeds = 540 episodes per arm. Judged rates: blind Sonnet judging, rubric v3, first pass as judged (HACK counts, BORDERLINE does not);\n"
             "error bars 95% problem-bootstrap CIs. The unsteered and random arms are the old Muse main run's; the cap arm clips the gi projection at the layer-wise p25 over layers 17–33.", fontsize=8.8, color=INK2, ha="left", va="top")

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, facecolor=SURFACE)
    if out.suffix == ".png":
        fig.savefig(out.with_suffix(".pdf"), facecolor=SURFACE)
    print(out)


if __name__ == "__main__":
    main()
