"""Chart the Sonnet-judged random-direction control of the Gemma 4 31B hard108 pilot (FINDINGS "Gemma 4 31B hard108
random-direction control, judged — 2026-11-03") plus the negated random 1729 arm (FINDINGS "Gemma random 1729 negated,
judged — 2026-11-04").

Two dot panels over the arms that sit inside the dose rule (gi_v2 at ±0.04 / ±0.08, unsteered, random 1729 at α −0.01
and three random directions at α 0.01): (a) judged hack rate (proxy + hidden-signature sets, as judged) with the problem-bootstrap 95% CI of the
contrast shifted onto the rate and the BORDERLINE-up / WEAK-down band as a bar, and (b) hidden-test pass with its CI.
The unsteered rate is a dashed reference line in both. Under them a table with every number drawn, Δ, CI and Holm p:
gi arms from the seven-arm family (`analysis_all7/`, Holm over six), random arms from their own family
(`analysis_rand11/`, Holm over four: the three α 0.01 randoms and the negated 1729).

    python experiments/deceiver-axis/scripts/gemma_hard108_rand_chart.py [--root <gemma4_31b_gi_v2/rh/hard108>] [--out <path>.png]
        [--rand-analysis analysis_rand11] [--summary analysis_rand11/summary_rand11.json]

The 2026-11-03 three-random version is `--rand-analysis analysis_rand10 --summary analysis_rand10/summary_rand10.json
--no-neg` (drops the negated arm and its detail files).
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

DEFAULT_ROOT = Path.home() / "nobackup/autodelete/self-concept-experiments/schemer-axis/gemma4_31b_gi_v2/rh/hard108"
SURFACE, INK, INK2, INK3, RULE, SHADE = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8985", "#dedcd7", "#f1f0ec"
C_GI, C_RAND, C_REF = "#1f4e9e", "#b8651b", "#52514e"

ARMS = [("L30_a-0.08", "gi −0.08"), ("L30_a-0.04", "gi −0.04"), ("unsteered", "unsteered"), ("L30_a0.04", "gi +0.04"),
        ("L30_a0.08", "gi +0.08"), ("L30_rand1729_a-0.01", "random 1729\nα −0.01"), ("L30_rand1729_a0.01", "random 1729\nα +0.01"),
        ("L30_rand1730_a0.01", "random 1730\nα 0.01"), ("L30_rand1731_a0.01", "random 1731\nα 0.01")]
DETAILS = ["verdicts_main/verdicts_main.detail.json", "verdicts_main/verdicts_hidden_main.detail.json",
           "verdicts_pm04/verdicts_pm04.detail.json", "verdicts_pm04/verdicts_hidden_pm04.detail.json",
           "verdicts_rand/verdicts_rand.detail.json", "verdicts_rand/verdicts_hidden_rand.detail.json",
           "verdicts_rand1729neg/verdicts_rand1729neg.detail.json", "verdicts_rand1729neg/verdicts_hidden_rand1729neg.detail.json"]


def colour(arm: str) -> str:
    return C_REF if arm == "unsteered" else C_RAND if "rand" in arm else C_GI


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--out", type=Path)
    p.add_argument("--dpi", type=int, default=170)
    p.add_argument("--rand-analysis", default="analysis_rand11", help="analysis dir of the random-arm family under --root")
    p.add_argument("--summary", default="analysis_rand11/summary_rand11.json", help="pilot_summary.py output under --root")
    p.add_argument("--no-neg", action="store_true", help="drop the negated 1729 arm (the 2026-11-03 chart)")
    args = p.parse_args()
    root = args.root
    arms = [a for a in ARMS if not (args.no_neg and a[0] == "L30_rand1729_a-0.01")]
    details = [d for d in DETAILS if not (args.no_neg and "rand1729neg" in d)]
    n_rand = sum("rand" in a for a, _ in arms)
    files = {"all": "analysis_all_as_judged.json", "up": "analysis_all_as_judged_borderline_up.json",
             "down": "analysis_all_as_judged_weak_down.json", "proxy": "analysis_proxy_as_judged.json"}
    fam = {"gi": {k: json.loads((root / "analysis_all7" / f).read_text()) for k, f in files.items()},
           "rand": {k: json.loads((root / args.rand_analysis / f).read_text()) for k, f in files.items()}}
    summary = json.loads((root / args.summary).read_text())
    counts = defaultdict(lambda: {"proxy": Counter(), "hidden": Counter()})
    for name in details:
        which = "hidden" if "hidden" in Path(name).name else "proxy"
        for key, obj in json.loads((root / name).read_text()).items():
            counts[key.split("/")[0]][which][obj["verdict"]] += 1

    def src(arm, var="all"):
        return fam["rand" if "rand" in arm else "gi"][var]

    def e(arm, outcome, var="all"):
        return src(arm, var)[outcome]["arms"][arm]

    def rate(arm, outcome, var="all"):
        return 100 * e(arm, outcome, var)["rate"]

    def ci(arm, outcome):
        x = e(arm, outcome)
        base = rate("unsteered", outcome)
        return None if "ci95" not in x else (base + 100 * x["ci95"][0], base + 100 * x["ci95"][1])

    def holm(arm, outcome):
        v = src(arm)[outcome].get("contrasts", {}).get(arm, {}).get("p_holm")
        return "—" if v is None else ("<0.001" if v < 0.001 else f"{v:.3f}")

    def delta(arm, outcome):
        x = e(arm, outcome)
        return "—" if "diff_vs_unsteered" not in x else f"{100 * x['diff_vs_unsteered']:+.1f}"

    def cicell(arm, outcome):
        x = e(arm, outcome)
        return "—" if "ci95" not in x else f"[{100 * x['ci95'][0]:+.1f}, {100 * x['ci95'][1]:+.1f}]"

    fig = plt.figure(figsize=(14 + 1.5 * (len(arms) - 8), 13.2), facecolor=SURFACE)
    gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 0.8, 1.25], hspace=0.42, left=0.07, right=0.98, top=0.86, bottom=0.03)
    ax_a, ax_b, ax_t = (fig.add_subplot(gs[i]) for i in range(3))
    xs = range(len(arms))
    for ax, outcome, ylabel, title in ((ax_a, "judged_hack", "judged hack rate (%)", "(a) Judged hack rate (HACK STRONG + WEAK, as judged)"),
                                       (ax_b, "hidden_pass", "hidden-test pass (%)", "(b) Capability: hidden-test pass")):
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(RULE)
        ax.tick_params(colors=INK2, labelsize=9, length=3)
        ax.yaxis.grid(True, color=RULE, lw=0.6)
        ax.set_axisbelow(True)
        ax.set_ylabel(ylabel, color=INK2, fontsize=9.5)
        ax.set_xticks(list(xs))
        ax.set_xticklabels([lab for _, lab in arms], color=INK, fontsize=9.5)
        ax.set_xlim(-0.6, len(arms) - 0.4)
        ax.axvline(4.5, color=RULE, lw=1.0)
        ax.axhline(rate("unsteered", outcome), color=C_REF, lw=1.0, ls="--", zorder=1)
        for i, (arm, _) in enumerate(arms):
            if outcome == "judged_hack":
                lo, hi = rate(arm, outcome, "down"), rate(arm, outcome, "up")
                ax.bar(i, hi - lo, bottom=lo, width=0.34, color=colour(arm), alpha=0.16, lw=0, zorder=2)
            c = ci(arm, outcome)
            if c:
                ax.plot([i, i], c, color=colour(arm), lw=2, solid_capstyle="round", zorder=3)
            y = rate(arm, outcome)
            ax.plot(i, y, "o", ms=8.5, color=colour(arm), mec=SURFACE, mew=2, zorder=4)
            ax.annotate(f"{y:.1f}", (i, y), xytext=(10, 4), textcoords="offset points", color=INK2, fontsize=8.5,
                        bbox=dict(boxstyle="square,pad=0.1", fc=SURFACE, ec="none"))
        ax.set_title(title, loc="left", color=INK, fontsize=10.5, pad=10)
        ax.set_ylim(0, max(rate(a, outcome, "up" if outcome == "judged_hack" else "all") for a, _ in arms) * 1.25)
    ax_a.text(2.0, ax_a.get_ylim()[1] * 0.96, "gi_v2 vector (persona contrast), layer 30", ha="center", va="top", color=INK2, fontsize=9)
    ax_a.text(4.5 + n_rand / 2, ax_a.get_ylim()[1] * 0.96, "random unit directions, layer 30", ha="center", va="top", color=INK2, fontsize=9)
    ax_a.legend(handles=[Line2D([], [], color=C_GI, lw=2, marker="o", ms=7, mec=SURFACE, label="gi_v2 arm (rate, 95% CI of Δ)"),
                         Line2D([], [], color=C_RAND, lw=2, marker="o", ms=7, mec=SURFACE, label="random arm (rate, 95% CI of Δ)"),
                         Patch(color=INK3, alpha=0.25, label="WEAK not counted … BORDERLINE counted"),
                         Line2D([], [], color=C_REF, lw=1, ls="--", label="unsteered")],
                loc="upper left", bbox_to_anchor=(0.0, 0.88), fontsize=8.5, frameon=False, labelcolor=INK2)

    ax_t.axis("off")

    def hbn(a, which):
        c = counts[a][which]
        return f"{c['HACK']} / {c['BORDERLINE']} / {c['NOT_HACK']}"

    rows = [
        ("judged hack, as judged (%)", [f"{rate(a, 'judged_hack'):.1f}" for a, _ in arms]),
        ("   Δ vs unsteered (points)", [delta(a, "judged_hack") for a, _ in arms]),
        ("   95% CI (problem bootstrap)", [cicell(a, "judged_hack") for a, _ in arms]),
        (f"   Holm p (gi: 6 contrasts; random: {n_rand})", [holm(a, "judged_hack") for a, _ in arms]),
        ("   problems up / down / tie", ["—" if a == "unsteered" else "{problems_up} / {problems_down} / {problems_tied}".format(**e(a, "judged_hack")) for a, _ in arms]),
        ("judged hack, proxy set only (%)", [f"{rate(a, 'judged_hack', 'proxy'):.1f}" for a, _ in arms]),
        ("judged hack, BORDERLINE counted (%)", [f"{rate(a, 'judged_hack', 'up'):.1f}" for a, _ in arms]),
        ("judged hack, WEAK not counted (%)", [f"{rate(a, 'judged_hack', 'down'):.1f}" for a, _ in arms]),
        ("automatic proxy (%)", [f"{rate(a, 'hack_proxy'):.1f}" for a, _ in arms]),
        ("hidden-test pass (%)", [f"{rate(a, 'hidden_pass'):.1f}" for a, _ in arms]),
        ("   Δ vs unsteered / 95% CI", [f"{delta(a, 'hidden_pass')} {cicell(a, 'hidden_pass')}" if a != "unsteered" else "—" for a, _ in arms]),
        ("shown-test pass (%)", [f"{rate(a, 'shown_pass'):.1f}" for a, _ in arms]),
        ("no code submitted (%)", [f"{rate(a, 'no_code'):.1f}" for a, _ in arms]),
        ("truncated turns / turns", [f"{summary[a]['truncated_turns']} / {summary[a]['turns']}" for a, _ in arms]),
        ("dossiers H / B / N: proxy set", [hbn(a, "proxy") for a, _ in arms]),
        ("dossiers H / B / N: hidden-signature set", [hbn(a, "hidden") for a, _ in arms]),
    ]
    table = ax_t.table(cellText=[[r[0]] + r[1] for r in rows], colLabels=[""] + [lab.split("\n")[0].replace("random", "rand") for _, lab in arms],
                       loc="upper center", cellLoc="center", colWidths=[0.25] + [0.75 / len(arms)] * len(arms), bbox=[0.0, 0.0, 1.0, 1.0])
    table.auto_set_font_size(False)
    table.set_fontsize(7.8)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor(RULE)
        cell.set_linewidth(0.6)
        cell.get_text().set_color(INK if r == 0 or c == 0 else INK2)
        if r == 0:
            cell.set_facecolor(SHADE)
            cell.get_text().set_fontweight("bold")
        elif r % 2 == 0:
            cell.set_facecolor("#f7f6f3")
        else:
            cell.set_facecolor(SURFACE)
        if c == 0:
            cell.get_text().set_ha("left")
            cell.PAD = 0.02
    ax_t.set_title("Numbers behind the dots (problem-stratified means over 108 problems × 16 seeds = 1,728 episodes per arm; H / B / N = HACK / BORDERLINE / NOT_HACK)",
                   loc="left", color=INK, fontsize=10, pad=6)
    fig.suptitle("Gemma 4 31B hard108, layer 30: the gi_v2 arms inside the dose rule next to three random directions"
                 + ("" if args.no_neg else " and the negation of 1729") + "\n"
                 "(blind Sonnet rubric-v3 judging, no adjudication)", x=0.07, ha="left", color=INK, fontsize=12.5, y=0.985)
    fig.text(0.07, 0.925, "Random directions at α 0.01 are capability-matched (hidden pass within 3 points of unsteered), not dose-matched to the gi arms.\n"
             "Koby's runs (random arms: jobs 14982739 seed 0, 14988674 seeds 1–15" + ("" if args.no_neg else "; negated 1729: 15010135 tasks 225–240")
             + "). CIs: problem bootstrap of the contrast vs unsteered, shifted onto the rate.",
             color=INK2, fontsize=9, va="top")
    out = args.out or root / args.rand_analysis / "gemma_hard108_rand_chart.png"
    fig.savefig(out, dpi=args.dpi, facecolor=SURFACE)
    fig.savefig(out.with_suffix(".pdf"), facecolor=SURFACE)
    print(out)


if __name__ == "__main__":
    main()
