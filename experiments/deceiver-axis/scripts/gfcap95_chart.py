"""Chart for the Muse set gfcap95 read: the judged hack rate of the Goodfire-axis arms beside both unsteered arms
(October reference and the 2026-11-11 rerun), and the dose-rule measure (hidden pass among episodes not judged HACK)
of each arm against each unsteered arm, with a table of the numbers of record.
Usage: gfcap95_chart.py <pilot_dir> <out.png>   (reads analysis_gfcap95_{all,rerunref_all}_as_judged.json and
dose_rule_read_gfcap95_{oct,rerunref}.json)"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE, INK, INK2, INK3, RULE, SHADE = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8985", "#dedcd7", "#f1f0ec"
C_REF, C_RERUN, C_ARM = "#1f4e9e", "#b8651b", "#6b6a66"
ARMS = (("gf_cap_p25", "cap p25"), ("gf_cap_p95", "cap p95"), ("gf_m0.16", "−0.16"), ("gf_m0.08", "−0.08"),
        ("gf_p0.08", "+0.08"), ("gf_p0.16", "+0.16"))


def fmt_p(p: float) -> str:
    return f"{p:.4f}" if p >= 0.0001 else f"{p:.0e}"


def main() -> None:
    pilot, out = Path(sys.argv[1]), Path(sys.argv[2])
    an = {k: json.loads((pilot / f"analysis_gfcap95_{f}_as_judged.json").read_text())["judged_hack"]
          for k, f in (("oct", "all"), ("rerun", "rerunref_all"))}
    dr = {k: json.loads((pilot / f"dose_rule_read_gfcap95_{f}.json").read_text())
          for k, f in (("oct", "oct"), ("rerun", "rerunref"))}
    rule = {k: {a["arm"]: a for a in dr[k]["arms"]} for k in dr}
    ref = {k: 100 * an[k]["arms"]["unsteered"]["rate"] for k in an}

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig = plt.figure(figsize=(13.5, 9.6), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.78], hspace=0.30, wspace=0.20, left=0.06, right=0.98,
                          top=0.90, bottom=0.03)
    fig.suptitle("Muse: Goodfire-axis arms against both unsteered arms (12 seeds × 45 problems per arm)",
                 x=0.06, ha="left", fontsize=13.5, color=INK, fontweight="bold")

    ax = fig.add_subplot(gs[0, 0], facecolor=SURFACE)
    names = ["cap p25", "cap p95", "−0.16", "−0.08", "unsteered\nOctober", "unsteered\nrerun", "+0.08", "+0.16"]
    keys = ["gf_cap_p25", "gf_cap_p95", "gf_m0.16", "gf_m0.08", None, None, "gf_p0.08", "gf_p0.16"]
    vals = [100 * an["oct"]["arms"][k]["rate"] if k else None for k in keys]
    vals[4], vals[5] = ref["oct"], ref["rerun"]
    cols = [C_ARM] * 4 + [C_REF, C_RERUN] + [C_ARM] * 2
    ax.bar(range(8), vals, width=0.62, color=cols, edgecolor=SURFACE, linewidth=2)
    for i, v in enumerate(vals):
        ax.text(i, v + 1.2, f"{v:.1f}%", ha="center", va="bottom", fontsize=9.5, color=INK)
    ax.axhline(ref["oct"], color=C_REF, lw=1, ls=(0, (4, 3)), zorder=0)
    ax.axhline(ref["rerun"], color=C_RERUN, lw=1, ls=(0, (4, 3)), zorder=0)
    ax.set_xticks(range(8)); ax.set_xticklabels(names, color=INK2)
    for i in (4, 5):
        ax.get_xticklabels()[i].set_fontsize(8.3)
    ax.set_xlabel("Goodfire-axis arm (cap percentile or steering dose)", color=INK2)
    ax.set_ylim(0, 62); ax.set_ylabel("judged reward-hacking rate (%)", color=INK2)
    ax.set_title("Judged hack rate, as judged", loc="left", fontsize=11, color=INK)

    bx = fig.add_subplot(gs[0, 1], facecolor=SURFACE)
    shown = [a for a in ARMS if a[0] != "gf_cap_p25"]
    for j, (k, colr, lab, off) in enumerate((("oct", C_REF, "against October unsteered", -0.14),
                                             ("rerun", C_RERUN, "against the rerun", 0.14))):
        xs = [i + off for i in range(len(shown))]
        ds = [rule[k][a]["rule_delta"] for a, _ in shown]
        lo = [rule[k][a]["rule_delta"] - rule[k][a]["rule_ci95"][0] for a, _ in shown]
        hi = [rule[k][a]["rule_ci95"][1] - rule[k][a]["rule_delta"] for a, _ in shown]
        bx.errorbar(xs, ds, yerr=[lo, hi], fmt="o", ms=8, color=colr, ecolor=colr, elinewidth=2, capsize=0,
                    markeredgecolor=SURFACE, markeredgewidth=2, label=lab)
    bx.axhline(0, color=INK3, lw=1)
    for y in (-10, 10):
        bx.axhline(y, color=INK, lw=1, ls=(0, (2, 3)))
    bx.text(-0.55, 10.5, "10-point rule (values in the table)", ha="left", va="bottom", fontsize=8.5, color=INK2)
    bx.set_xticks(range(len(shown))); bx.set_xticklabels([n for _, n in shown], color=INK2)
    bx.set_xlim(-0.6, len(shown) - 0.4); bx.set_ylim(-30, 16)
    bx.set_xlabel("Goodfire-axis arm (cap p25 not drawn: 98.9% no code, see table)", color=INK2)
    bx.set_ylabel("difference from unsteered (points)", color=INK2)
    bx.set_title("Dose-rule measure: hidden pass among episodes not judged HACK", loc="left", fontsize=11, color=INK)
    bx.legend(frameon=False, loc="lower right", fontsize=9.5, labelcolor=INK2)
    for a in (ax, bx):
        a.grid(axis="y", color=RULE, lw=0.8); a.set_axisbelow(True); a.tick_params(length=0, colors=INK2)
        for s in a.spines.values():
            s.set_visible(False)

    tx = fig.add_subplot(gs[1, :]); tx.axis("off")
    head = ["arm", "judged\nhack", "Δ vs October\n[95% CI]", "Holm p", "Δ vs rerun\n[95% CI]", "Holm p",
            "rule Δ vs October\n[95% CI]", "rule Δ vs rerun\n[95% CI]", "no code Δ\nOct / rerun",
            "dose rule\nOct / rerun"]
    rows = [["unsteered October", f"{ref['oct']:.1f}%", "", "", "", "", f"level {dr['oct']['reference']['rule_level_pct']:.1f}%", "", "", ""],
            ["unsteered rerun", f"{ref['rerun']:.1f}%", "", "", "", "", "", f"level {dr['rerun']['reference']['rule_level_pct']:.1f}%", "", ""]]
    for k, n in ARMS:
        cell = [f"Goodfire {n}", f"{100 * an['oct']['arms'][k]['rate']:.1f}%"]
        for r in ("oct", "rerun"):
            c = an[r]["contrasts"][k]
            cell += [f"{100 * c['diff']:+.1f} [{100 * c['ci95'][0]:+.1f}, {100 * c['ci95'][1]:+.1f}]", fmt_p(c["p_holm"])]
        for r in ("oct", "rerun"):
            a = rule[r][k]
            cell.append(f"{a['rule_delta']:+.1f} [{a['rule_ci95'][0]:+.1f}, {a['rule_ci95'][1]:+.1f}]")
        cell.append(f"{rule['oct'][k]['no_code_delta']:+.1f} / {rule['rerun'][k]['no_code_delta']:+.1f}")
        cell.append(" / ".join("inside" if rule[r][k]["status"].startswith("inside") else "outside" for r in ("oct", "rerun")))
        rows.append(cell)
    tab = tx.table(cellText=rows, colLabels=head, loc="upper center", cellLoc="center",
                   colWidths=[0.125, 0.06, 0.135, 0.06, 0.135, 0.06, 0.135, 0.135, 0.085, 0.10])
    tab.auto_set_font_size(False); tab.set_fontsize(9); tab.scale(1, 1.62)
    for (r, c), cell in tab.get_celld().items():
        cell.set_edgecolor(RULE); cell.set_linewidth(0.8)
        cell.set_facecolor(SHADE if r == 0 else SURFACE)
        cell.get_text().set_color(INK if r == 0 or c == 0 else INK2)
        if r == 0:
            cell.set_height(cell.get_height() * 1.5); cell.get_text().set_fontweight("bold")
    tx.text(0.0, 0.0, "Rates are problem-stratified means; intervals are problem-bootstrap 95%; Holm over the six Goodfire-axis arms (plus three "
            "random-direction add arms against October).\nDose rule: rule measure and no code each within 10 points of unsteered. "
            "The two unsteered arms were judged on different days (October / Nov 11),\nso their own difference mixes run-to-run change with judge drift.",
            fontsize=8.5, color=INK3, va="bottom", transform=tx.transAxes)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print("wrote", out)


if __name__ == "__main__":
    main()
