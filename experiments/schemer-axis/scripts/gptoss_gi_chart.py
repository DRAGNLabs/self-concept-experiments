"""Chart the gpt-oss gi dose scans: role-set v2 vector (left) next to the superseded 17-role vector (right).

Two columns on identical axes. Each column has (a) hack rates over the dose slots −0.2 / −0.1 / 0 / +0.1 / +0.2 at layer 17
(judged, proxy + hidden-pass sets, with 95% problem-bootstrap CIs; judged, proxy set only; automatic proxy), the cap arm
and the three norm-matched random directions (α 0.07) as categorical points to the right, (b) hidden-test pass, and (c) a
table with every number the lines are drawn from plus the Holm p values. Arms missing from an analysis file (e.g. the v2
±0.1 and cap arms while they run) leave their slot empty and read "pending" in the table.

Inputs (per column): main_analysis.py outputs on the all-set verdicts and on the proxy-set-only verdicts.

    python experiments/schemer-axis/scripts/gptoss_gi_chart.py [--out <path>.png]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

N = Path.home() / "nobackup/autodelete/self-concept-experiments/schemer-axis"

# Design tokens (same as gi_main_chart.py): text in ink tokens, recessive rules, series colours only on marks.
SURFACE, INK, INK2, INK3, RULE, SHADE = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8985", "#dedcd7", "#f1f0ec"
C_JUDGED = "#1f4e9e"    # judged hack, proxy + hidden-pass sets
C_PROXYSET = "#6fa3dc"  # judged hack, proxy set only (same quantity, lighter step)
C_AUTO = "#b8651b"      # automatic proxy
C_HIDDEN = "#2a7f62"    # hidden-test pass
C_RANDOM = "#8a8985"

DOSES = ["−0.4\n(end point)", "−0.2", "−0.1", "0\nunsteered", "+0.1", "+0.2", "+0.4\n(end point)"]
SIDES = ["cap p25\n(layers 12–23)", "random\n1729", "random\n1730", "random\n1731"]
SIDE_GAP = 1.0

COLUMNS = [
    dict(
        key="v2",
        title="New vector: role set v2 (18 roles, no cheating / test-gaming personas), 2026-10-29",
        all=N / "gi_v2/pilot/analysis_mixed_all_as_judged.json",
        prox=N / "gi_v2/pilot/analysis_mixed_proxy_as_judged.json",
        arms=["gi_m0.4", "gi_m0.2", "gi_m0.1", "unsteered", "gi_p0.1", "gi_p0.2", "gi_p0.4", "gi_cap_p25", "random_a0.07", "random_a0.07_s1730", "random_a0.07_s1731"],
        note="all seven arms judged as one mixed set (proxy 908 + hidden-pass 178 dossiers, 2026-10-30); reference arms reuse the 2026-10-26 verdicts",
    ),
    dict(
        key="17role",
        title="Old vector: 17-role answer-region (incl. cheater / rule-follower), 2026-10-26, superseded",
        all=N / "answer17_anecdotes/coding/verdicts/analysis_all_half_as_judged.json",
        prox=N / "answer17_anecdotes/coding/verdicts/analysis_proxy_half_as_judged.json",
        arms=["ans17_raw_m0.4", "ans17_raw_m0.2", "ans17_raw_m0.1", "unsteered", "ans17_raw_p0.1", "ans17_raw_p0.2", "ans17_raw_p0.4", "ans17_raw_cap_p25", "random_a0.07", "random_a0.07_s1730", "random_a0.07_s1731"],
        note="all nine arms judged as one set shuffled across arms; ±0.4 = 1-seed probes (45 episodes, automatic metrics only, open markers)",
        missing={"ans17_raw_m0.4": "1-seed probe", "ans17_raw_p0.4": "1-seed probe"},
        probe=N / "answer17_anecdotes/coding/pm04_probe_summary.json",  # open markers: automatic metrics from the 1-seed ±0.4 probes
    ),
]
COL_LABELS = ["gi −0.4", "gi −0.2", "gi −0.1", "unsteered", "gi +0.1", "gi +0.2", "gi +0.4", "cap p25", "random\n1729", "random\n1730", "random\n1731"]


def xpos(i: int) -> float:
    return float(i) if i < len(DOSES) else len(DOSES) + SIDE_GAP + (i - len(DOSES))


def present(outcome: dict, arm: str) -> bool:
    return arm in outcome["arms"]


def ci_err(r: dict):
    if "ci95" not in r:
        return 0.0, 0.0
    return 100 * (r["diff_vs_unsteered"] - r["ci95"][0]), 100 * (r["ci95"][1] - r["diff_vs_unsteered"])


def holm(outcome: dict, arm: str):
    c = outcome.get("contrasts", {}).get(arm)
    return None if c is None else c["p_holm"]


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(RULE)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    ax.grid(axis="y", color=RULE, lw=0.6)
    ax.set_axisbelow(True)
    ax.axvline(len(DOSES) - 1 + SIDE_GAP / 2 + 0.5, color=RULE, lw=0.8, ls=(0, (2, 3)))


def xticks(ax):
    xs = [xpos(i) for i in range(len(DOSES) + len(SIDES))]
    ax.set_xticks(xs)
    ax.set_xticklabels(DOSES + SIDES, fontsize=9, color=INK)
    ax.set_xlim(-0.6, xs[-1] + 0.6)


def dose_line(ax, outcome, arms, color, lw=2.0, ci=True, marker="o", ms=5.5, z=3):
    pts = [(xpos(i), outcome["arms"][a]) for i, a in enumerate(arms[: len(DOSES)]) if present(outcome, a)]
    if not pts:
        return
    xs = [x for x, _ in pts]
    ys = [100 * r["rate"] for _, r in pts]
    ax.plot(xs, ys, color=color, lw=lw, marker=marker, ms=ms, mec=SURFACE, mew=1.2, zorder=z)
    if ci:
        lo = [ci_err(r)[0] for _, r in pts]
        hi = [ci_err(r)[1] for _, r in pts]
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt="none", ecolor=color, elinewidth=1.0, capsize=2.5, alpha=0.75, zorder=z - 1)


def side_points(ax, outcome, arms, color, marker, ci=True, z=3):
    for j, a in enumerate(arms[len(DOSES):]):
        if not present(outcome, a):
            continue
        x = xpos(len(DOSES) + j)
        r = outcome["arms"][a]
        y = 100 * r["rate"]
        is_cap = j == 0
        c = color if is_cap else C_RANDOM
        m = "D" if is_cap else marker
        ax.plot([x], [y], marker=m, ms=6 if is_cap else 5.5, color=c, mec=SURFACE, mew=1.2, ls="none", zorder=z)
        if ci and "ci95" in r:
            lo, hi = ci_err(r)
            ax.errorbar([x], [y], yerr=[[lo], [hi]], fmt="none", ecolor=c, elinewidth=1.0, capsize=2.5, alpha=0.75, zorder=z - 1)


def direct_labels(ax, outcome, arms, color, dy, dx=0.14):
    for i, a in enumerate(arms):
        if present(outcome, a):
            y = 100 * outcome["arms"][a]["rate"]
            d = abs(dy) if (dy < 0 and y + dy < 0.5) else dy  # keep labels inside the axes near zero
            ax.text(xpos(i) + dx, y + d, f"{y:.1f}", fontsize=8, color=color, ha="left", va="bottom" if d > 0 else "top")


def probe_points(ax, spec, arms, key, color, marker):
    pr = spec.get("_probe", {})
    for i, a in enumerate(arms):
        if a in pr:
            y = 100 * pr[a][key]
            ax.plot([xpos(i)], [y], marker=marker, ms=7, mfc=SURFACE, mec=color, mew=1.6, ls="none", zorder=3)
            ax.text(xpos(i) + 0.14, y, f"{y:.1f}\n(1 seed)", fontsize=7.5, color=color, ha="left", va="center")


def missing_label(spec, a):
    return spec.get("missing", {}).get(a, "pending")


def pending_marks(ax, outcome, arms, y, spec):
    for i, a in enumerate(arms):
        if not present(outcome, a) and a not in spec.get("_probe", {}):
            ax.text(xpos(i), y, missing_label(spec, a), fontsize=7.5, color=INK3, ha="center", va="center", rotation=90)


def fmt_p(p):
    if p is None:
        return "—"
    return "< 0.001" if p < 0.001 else f"{p:.3f}" if p < 0.01 else f"{p:.2f}"


def table(axt, jh, jp, hp, ap_, nc, rc, arms, title, note, spec):
    missing = set(spec.get("missing", {}).values()) | {"pending"}
    pr = spec.get("_probe", {})

    def probe_cell(a, key, scale=100.0, fmt="{:.1f}"):
        return None if a not in pr else fmt.format(scale * pr[a][key]) + " (1 seed)"

    def rate_cell(outcome, a, star=True):
        if not present(outcome, a):
            return missing_label(spec, a)
        r = outcome["arms"][a]
        if a == "unsteered":
            return f"{100 * r['rate']:.1f}"
        p = holm(outcome, a)
        sig = star and p is not None and p < 0.05
        return f"{100 * r['rate']:.1f} ({100 * r['diff_vs_unsteered']:+.1f}){'*' if sig else ''}"

    def ci_cell(outcome, a):
        if not present(outcome, a):
            return missing_label(spec, a)
        r = outcome["arms"][a]
        return "—" if "ci95" not in r else f"[{100 * r['ci95'][0]:+.1f}, {100 * r['ci95'][1]:+.1f}]"

    def p_cell(outcome, a):
        return missing_label(spec, a) if not present(outcome, a) else fmt_p(holm(outcome, a))

    def plain_cell(outcome, a, scale=100.0, fmt="{:.1f}"):
        return missing_label(spec, a) if not present(outcome, a) else fmt.format(scale * outcome["arms"][a]["rate"])

    def chars_cell(a):
        if not present(rc, a):
            return missing_label(spec, a)
        v = rc["arms"][a]["rate"]
        base = rc["arms"]["unsteered"]["rate"]
        return f"{v / 1000:.1f}k" + ("" if a == "unsteered" else f" ({100 * (v / base - 1):+.0f}%)")

    rows = [
        ("judged hack, all sets, % (Δ)", C_JUDGED, lambda a: rate_cell(jh, a)),
        ("   95% CI of Δ", C_JUDGED, lambda a: ci_cell(jh, a)),
        ("   Holm p", C_JUDGED, lambda a: p_cell(jh, a)),
        ("judged hack, proxy set, % (Δ)", C_PROXYSET, lambda a: rate_cell(jp, a)),
        ("automatic proxy, % (Δ)", C_AUTO, lambda a: probe_cell(a, "shown_pass_hidden_fail_rate") or rate_cell(ap_, a)),
        ("hidden-test pass, % (Δ)", C_HIDDEN, lambda a: probe_cell(a, "hidden_pass_rate") or rate_cell(hp, a, star=False)),
        ("no-code episodes, %", INK, lambda a: probe_cell(a, "no_code_rate") or plain_cell(nc, a)),
        ("reasoning chars / turn (Δ%)", INK, lambda a: probe_cell(a, "reasoning_chars_per_turn", 0.001, "{:.1f}k") or chars_cell(a)),
    ]
    cell_text = [[n] + [f(a) for a in arms] for n, _, f in rows]
    widths = [0.16] + [0.84 / len(arms)] * len(arms)
    tbl = axt.table(cellText=cell_text, colLabels=["metric"] + COL_LABELS, colWidths=widths, loc="upper center", cellLoc="center",
                    bbox=[0.0, 0.0, 1.0, 0.88])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(6.6)
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
            txt = cell.get_text().get_text()
            cell.set_text_props(color=INK3 if txt in missing else INK, fontweight="bold" if "*" in txt else "normal")
            cell.set_facecolor(SHADE if r % 2 == 0 else SURFACE)
    axt.set_title(title, fontsize=10, color=INK, loc="left", fontweight="bold", pad=2)
    axt.text(0.0, 0.955, note, transform=axt.transAxes, fontsize=8.3, color=INK2, ha="left", va="bottom")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=N / "gi_v2/pilot/gptoss_gi_v2_vs_17role_chart.png")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    fig = plt.figure(figsize=(22, 14.5), facecolor=SURFACE)
    gs = fig.add_gridspec(3, 2, height_ratios=[4.4, 2.8, 4.0], hspace=0.32, wspace=0.10, left=0.045, right=0.99, top=0.895, bottom=0.025)

    for spec in COLUMNS:  # load first so both columns share y-limits
        spec["_all"] = json.load(open(spec["all"]))
        spec["_prox"] = json.load(open(spec["prox"]))
        spec["_probe"] = {k: v for k, v in json.load(open(spec["probe"])).items() if not k.startswith("_")} if "probe" in spec else {}
    top = max([100 * sp["_all"][k]["arms"][a]["rate"] for sp in COLUMNS for k in ("judged_hack", "hack_proxy") for a in sp["arms"] if present(sp["_all"][k], a)]
              + [100 * v["shown_pass_hidden_fail_rate"] for sp in COLUMNS for v in sp["_probe"].values()] + [44])
    lo = min([100 * sp["_all"]["hidden_pass"]["arms"][a]["rate"] for sp in COLUMNS for a in sp["arms"] if present(sp["_all"]["hidden_pass"], a)]
             + [100 * v["hidden_pass_rate"] for sp in COLUMNS for v in sp["_probe"].values()] + [14])

    for col, spec in enumerate(COLUMNS):
        all_, prox, arms = spec["_all"], spec["_prox"], spec["arms"]
        jh, jp, hp, ap_, nc, rc = all_["judged_hack"], prox["judged_hack"], all_["hidden_pass"], all_["hack_proxy"], all_["no_code"], all_["reasoning_chars_per_turn"]
        unst_j = 100 * jh["arms"]["unsteered"]["rate"]
        unst_h = 100 * hp["arms"]["unsteered"]["rate"]

        ax1, ax2, axt = fig.add_subplot(gs[0, col]), fig.add_subplot(gs[1, col]), fig.add_subplot(gs[2, col])
        style(ax1)
        style(ax2)
        axt.set_axis_off()

        # (a) hack rates
        ax1.axhline(unst_j, color=INK3, lw=0.8, ls=(0, (4, 3)), zorder=1)
        ax1.text(len(DOSES) - 0.6, unst_j - 0.6, f"unsteered judged {unst_j:.1f}%", fontsize=8, color=INK3, va="top", ha="right")
        dose_line(ax1, ap_, arms, C_AUTO, lw=1.4, ci=False, marker="s", ms=4.5, z=2)
        dose_line(ax1, jp, arms, C_PROXYSET, lw=1.6, ci=False, marker="o", ms=4.5, z=3)
        dose_line(ax1, jh, arms, C_JUDGED, lw=2.2, ci=True, marker="o", ms=6, z=4)
        side_points(ax1, ap_, arms, C_AUTO, "s", ci=False, z=2)
        side_points(ax1, jp, arms, C_PROXYSET, "o", ci=False, z=3)
        side_points(ax1, jh, arms, C_JUDGED, "o", ci=True, z=4)
        direct_labels(ax1, jh, arms, C_JUDGED, dy=-2.0)
        probe_points(ax1, spec, arms, "shown_pass_hidden_fail_rate", C_AUTO, "s")
        pending_marks(ax1, jh, arms, y=25, spec=spec)
        ax1.set_ylim(0, top + 8)
        xticks(ax1)
        if col == 0:
            ax1.set_ylabel("hack rate, % of 540 episodes", color=INK2, fontsize=10)
        ax1.set_xlabel("steering dose α at layer 17 (units of the layer's mean residual norm)       │   other arms (random directions at α 0.07)", color=INK2, fontsize=9)
        ax1.set_title(f"a   {spec['title']}", fontsize=10, color=INK, loc="left", fontweight="bold", pad=8)
        if col == 0:
            handles = [Line2D([], [], color=C_JUDGED, marker="o", lw=2.2, label="judged hack, proxy + hidden-pass sets (95% CI)"),
                       Line2D([], [], color=C_PROXYSET, marker="o", lw=1.6, label="judged hack, proxy set only"),
                       Line2D([], [], color=C_AUTO, marker="s", lw=1.4, label="automatic proxy (shown pass, hidden fail)"),
                       Line2D([], [], color=C_JUDGED, marker="D", lw=0, label="cap arm (diamond)"),
                       Line2D([], [], color=C_RANDOM, marker="o", lw=0, label="random directions (grey)"),
                       Line2D([], [], color=C_AUTO, marker="s", mfc=SURFACE, mew=1.6, lw=0, label="open marker: 1-seed probe, automatic metric only")]
            ax1.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 0.99), fontsize=8.5, frameon=False, ncol=1, labelcolor=INK2)

        # (b) hidden pass
        ax2.axhline(unst_h, color=INK3, lw=0.8, ls=(0, (4, 3)), zorder=1)
        ax2.axhspan(unst_h - 10, unst_h, color=C_HIDDEN, alpha=0.06, lw=0, zorder=0)
        ax2.text(-0.55, unst_h - 9.6, "dose rule: within 10 points of unsteered", fontsize=8, color=INK3, va="bottom", ha="left")
        dose_line(ax2, hp, arms, C_HIDDEN, lw=2.2, ci=True, marker="o", ms=6, z=4)
        side_points(ax2, hp, arms, C_HIDDEN, "o", ci=True, z=4)
        direct_labels(ax2, hp, arms, C_HIDDEN, dy=-1.4)
        probe_points(ax2, spec, arms, "hidden_pass_rate", C_HIDDEN, "o")
        pending_marks(ax2, hp, arms, y=25, spec=spec)
        ax2.set_ylim(lo - 4, 38)
        xticks(ax2)
        if col == 0:
            ax2.set_ylabel("hidden-test pass, %", color=INK2, fontsize=10)
        ax2.set_xlabel("steering dose α at layer 17       │   other arms", color=INK2, fontsize=9)
        ax2.set_title("b   Hidden-test pass (capability side of the dose rule)", fontsize=10, color=INK, loc="left", fontweight="bold", pad=8)

        table(axt, jh, jp, hp, ap_, nc, rc, arms, "c   Numbers behind a and b (Δ = arm minus unsteered; * Holm p < 0.05)", f"Judging note: {spec['note']}.", spec)

    fig.suptitle("gpt-oss-120b, gi deceiver axis at layer 17: dose scan with the role-set v2 vector (left) and the superseded 17-role vector (right)",
                 x=0.045, ha="left", fontsize=13, color=INK, fontweight="bold", y=0.972)
    fig.text(0.045, 0.945, "45 hard problems × 12 seeds = 540 episodes per arm. Judged rates: blind Sonnet judging, rubric v3, first pass as judged (HACK counts, BORDERLINE does not); error bars 95% problem-bootstrap CIs.\n"
             "Both columns share the same unsteered arm and the same three random-direction arms (reused reference arms, judged on 2026-10-26). Hidden-pass sets: episodes passing the hidden suite with a lookup-table / hard-code signature.",
             fontsize=8.8, color=INK2, ha="left", va="top")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, facecolor=SURFACE)
    if args.out.suffix == ".png":
        fig.savefig(args.out.with_suffix(".pdf"), facecolor=SURFACE)
    print(args.out)


if __name__ == "__main__":
    main()
