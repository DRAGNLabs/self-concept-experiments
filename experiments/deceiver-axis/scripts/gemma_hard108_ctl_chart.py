"""Chart the random-1729 control family of the Gemma 4 31B hard108 pilot (FINDINGS "Gemma random 1729 controls:
dose, on-manifold part and per-token noise, 16 seeds — 2026-11-05").

Four panels and a table. Left column: the five signed doses of random direction 1729 (α −0.01, −0.005, 0 = unsteered,
+0.005, +0.01) as a line over a labelled dose axis, (a) judged hack rate with the problem-bootstrap 95% CI of the contrast
shifted onto the rate and the BORDERLINE-up / WEAK-down band as a bar, (b) hidden-test pass with its CI. Right column:
the same two outcomes for the four same-norm controls at α 0.01 (1729's top-50-PC part, per-token isotropic noise, random
directions 1730 and 1731) next to the full 1729 arm. Under them a table with every number drawn, Δ, CI, Holm p and the
seed-paired direction count. Numbers: full-dose and 1730 / 1731 arms from `analysis_rand11/` (Holm over four), the four
new control arms from `analysis_ctl15/` (Holm over four), unsteered from either (identical).

    python experiments/deceiver-axis/scripts/gemma_hard108_ctl_chart.py [--root <gemma4_31b_gi_v2/rh/hard108>] [--out <path>.png]
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
C_1729, C_NOISE, C_OTHER, C_REF = "#b8651b", "#1f7a6a", "#6a5aa8", "#52514e"

# (arm, table label, family, dose for the line panel or None, control-panel label or None, colour)
ARMS = [
    ("L30_rand1729_a-0.01", "1729 α −0.01", "rand", -0.01, None, C_1729),
    ("L30_rand1729_a-0.005", "1729 α −0.005", "ctl", -0.005, None, C_1729),
    ("unsteered", "unsteered", "rand", 0.0, None, C_REF),
    ("L30_rand1729_a0.005", "1729 α +0.005", "ctl", 0.005, None, C_1729),
    ("L30_rand1729_a0.01", "1729 α +0.01", "rand", 0.01, "random 1729\nα +0.01 (full)", C_1729),
    ("L30_rand1729pc50_a0.01", "1729 top-50-PC", "ctl", None, "1729 top-50-PC\npart, α 0.01", C_1729),
    ("L30_noise_a0.01", "per-token noise", "ctl", None, "per-token\nnoise, α 0.01", C_NOISE),
    ("L30_rand1730_a0.01", "random 1730", "rand", None, "random 1730\nα 0.01", C_OTHER),
    ("L30_rand1731_a0.01", "random 1731", "rand", None, "random 1731\nα 0.01", C_OTHER),
]
DETAILS = ["verdicts_main/verdicts_main.detail.json", "verdicts_main/verdicts_hidden_main.detail.json",
           "verdicts_rand/verdicts_rand.detail.json", "verdicts_rand/verdicts_hidden_rand.detail.json",
           "verdicts_rand1729neg/verdicts_rand1729neg.detail.json", "verdicts_rand1729neg/verdicts_hidden_rand1729neg.detail.json"]
# noise seeds 8-15 were judged as two sets (s8-13 while seeds 14-15 still ran, then s14-15)
_CTL_SETS = {"a0.005": ("s0-7", "s8-15"), "a-0.005": ("s0-7", "s8-15"),
             "pc50": ("s0-7", "s8-15"), "noise": ("s0-7", "s8-13", "s14-15")}
for _tag, _seedsets in _CTL_SETS.items():
    for _seeds in _seedsets:
        DETAILS += [f"verdicts_ctl_{_tag}_{_seeds}/verdicts_ctl_{_tag}_{_seeds}.detail.json",
                    f"verdicts_ctl_{_tag}_{_seeds}/verdicts_hidden_ctl_{_tag}_{_seeds}.detail.json"]
FILES = {"all": "analysis_all_as_judged.json", "up": "analysis_all_as_judged_borderline_up.json",
         "down": "analysis_all_as_judged_weak_down.json", "proxy": "analysis_proxy_as_judged.json"}


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(RULE)
    ax.tick_params(colors=INK2, labelsize=9, length=3)
    ax.yaxis.grid(True, color=RULE, lw=0.6)
    ax.set_axisbelow(True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--out", type=Path)
    p.add_argument("--dpi", type=int, default=170)
    p.add_argument("--rand-analysis", default="analysis_rand11")
    p.add_argument("--ctl-analysis", default="analysis_ctl15")
    p.add_argument("--summary", default="analysis_ctl15/summary_ctl15.json", help="pilot_summary.py output under --root")
    args = p.parse_args()
    root = args.root
    fam = {"rand": {k: json.loads((root / args.rand_analysis / f).read_text()) for k, f in FILES.items()},
           "ctl": {k: json.loads((root / args.ctl_analysis / f).read_text()) for k, f in FILES.items()}}
    summary = json.loads((root / args.summary).read_text())
    family = {a: f for a, _, f, _, _, _ in ARMS}
    colour = {a: c for a, _, _, _, _, c in ARMS}
    counts = defaultdict(lambda: {"proxy": Counter(), "hidden": Counter()})
    per_seed = defaultdict(Counter)
    for name in DETAILS:
        which = "hidden" if "hidden" in Path(name).name else "proxy"
        for key, obj in json.loads((root / name).read_text()).items():
            arm, seed, _ = key.split("/")
            counts[arm][which][obj["verdict"]] += 1
            if obj["verdict"] == "HACK":
                per_seed[arm][seed] += 1

    def src(arm, var="all"):
        return fam[family[arm]][var]

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

    def seeds_updown(arm):
        if arm == "unsteered":
            return "—"
        seeds = sorted({*per_seed["unsteered"], *per_seed[arm]}, key=lambda s: int(s[4:]))
        d = [per_seed[arm][s] - per_seed["unsteered"][s] for s in seeds]
        return f"{sum(x > 0 for x in d)} / {sum(x < 0 for x in d)} / {sum(x == 0 for x in d)}"

    dose_arms = [(a, d) for a, _, _, d, _, _ in ARMS if d is not None]
    ctl_arms = [(a, lab) for a, _, _, _, lab, _ in ARMS if lab is not None]

    fig = plt.figure(figsize=(15.5, 14.2), facecolor=SURFACE)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 0.8, 1.35], width_ratios=[1.0, 1.0], hspace=0.5, wspace=0.18,
                          left=0.06, right=0.985, top=0.865, bottom=0.03)
    panels = {("dose", "judged_hack"): fig.add_subplot(gs[0, 0]), ("ctl", "judged_hack"): fig.add_subplot(gs[0, 1]),
              ("dose", "hidden_pass"): fig.add_subplot(gs[1, 0]), ("ctl", "hidden_pass"): fig.add_subplot(gs[1, 1])}
    ax_t = fig.add_subplot(gs[2, :])
    ylabels = {"judged_hack": "judged hack rate (%)", "hidden_pass": "hidden-test pass (%)"}
    titles = {("dose", "judged_hack"): "(a) Judged hack rate along the 1729 dose axis",
              ("ctl", "judged_hack"): "(c) Judged hack rate, same-norm controls at α 0.01",
              ("dose", "hidden_pass"): "(b) Hidden-test pass along the 1729 dose axis",
              ("ctl", "hidden_pass"): "(d) Hidden-test pass, same-norm controls at α 0.01"}
    ymax = {o: max(rate(a, o, "up" if o == "judged_hack" else "all") for a, *_ in ARMS) * 1.3 for o in ylabels}

    def draw_point(ax, x, arm, outcome, width):
        if outcome == "judged_hack":
            lo, hi = rate(arm, outcome, "down"), rate(arm, outcome, "up")
            ax.bar(x, hi - lo, bottom=lo, width=width, color=colour[arm], alpha=0.16, lw=0, zorder=2)
        c = ci(arm, outcome)
        if c:
            ax.plot([x, x], c, color=colour[arm], lw=2, solid_capstyle="round", zorder=3)
        y = rate(arm, outcome)
        ax.plot(x, y, "o", ms=8.5, color=colour[arm], mec=SURFACE, mew=2, zorder=4)
        ax.annotate(f"{y:.1f}", (x, y), xytext=(10, 4), textcoords="offset points", color=INK2, fontsize=8.5,
                    bbox=dict(boxstyle="square,pad=0.1", fc=SURFACE, ec="none"))

    for outcome in ylabels:
        ax = panels[("dose", outcome)]
        style(ax)
        xs = [d for _, d in dose_arms]
        ax.plot(xs, [rate(a, outcome) for a, _ in dose_arms], color=C_1729, lw=2, zorder=3)
        for a, d in dose_arms:
            draw_point(ax, d, a, outcome, 0.0011)
        ax.axhline(rate("unsteered", outcome), color=C_REF, lw=1.0, ls="--", zorder=1)
        ax.set_xticks(xs)
        ax.set_xticklabels(["−0.01", "−0.005", "0\n(unsteered)", "+0.005", "+0.01"], color=INK, fontsize=9.5)
        ax.set_xlabel("steering coefficient α of random direction 1729 at layer 30 (injected norm α × 285.0)", color=INK2, fontsize=9.5)
        ax.set_xlim(-0.0135, 0.0135)
        ax.set_ylim(0, ymax[outcome])
        ax.set_ylabel(ylabels[outcome], color=INK2, fontsize=9.5)
        ax.set_title(titles[("dose", outcome)], loc="left", color=INK, fontsize=10.5, pad=10)

        ax = panels[("ctl", outcome)]
        style(ax)
        for i, (a, _) in enumerate(ctl_arms):
            draw_point(ax, i, a, outcome, 0.34)
        ax.axhline(rate("unsteered", outcome), color=C_REF, lw=1.0, ls="--", zorder=1)
        ax.axvline(0.5, color=RULE, lw=1.0)
        ax.set_xticks(range(len(ctl_arms)))
        ax.set_xticklabels([lab for _, lab in ctl_arms], color=INK, fontsize=9)
        ax.set_xlabel("control arm (every arm injects norm 0.01 × 285.0 = 2.85 per token at layer 30)", color=INK2, fontsize=9.5)
        ax.set_xlim(-0.6, len(ctl_arms) - 0.4)
        ax.set_ylim(0, ymax[outcome])
        ax.set_ylabel(ylabels[outcome], color=INK2, fontsize=9.5)
        ax.set_title(titles[("ctl", outcome)], loc="left", color=INK, fontsize=10.5, pad=10)

    panels[("dose", "judged_hack")].legend(
        handles=[Line2D([], [], color=C_1729, lw=2, marker="o", ms=7, mec=SURFACE, label="random 1729 and its top-50-PC part (rate, 95% CI of Δ)"),
                 Line2D([], [], color=C_NOISE, lw=0, marker="o", ms=7, mec=SURFACE, label="per-token isotropic noise, 1729's norm"),
                 Line2D([], [], color=C_OTHER, lw=0, marker="o", ms=7, mec=SURFACE, label="other fixed random directions (1730, 1731)"),
                 Patch(color=INK3, alpha=0.25, label="WEAK not counted … BORDERLINE counted"),
                 Line2D([], [], color=C_REF, lw=1, ls="--", label="unsteered")],
        loc="upper left", fontsize=8.5, frameon=False, labelcolor=INK2)

    ax_t.axis("off")

    def hbn(a, which):
        c = counts[a][which]
        return f"{c['HACK']} / {c['BORDERLINE']} / {c['NOT_HACK']}"

    cols = [a for a, *_ in ARMS]
    rows = [
        ("judged hack, as judged (%)", [f"{rate(a, 'judged_hack'):.1f}" for a in cols]),
        ("   Δ vs unsteered (points)", [delta(a, "judged_hack") for a in cols]),
        ("   95% CI (problem bootstrap)", [cicell(a, "judged_hack") for a in cols]),
        ("   Holm p (four contrasts per family)", [holm(a, "judged_hack") for a in cols]),
        ("   problems up / down / tie", ["—" if a == "unsteered" else "{problems_up} / {problems_down} / {problems_tied}".format(**e(a, "judged_hack")) for a in cols]),
        ("   seeds up / down / tie (16)", [seeds_updown(a) for a in cols]),
        ("judged hack, proxy set only (%)", [f"{rate(a, 'judged_hack', 'proxy'):.1f}" for a in cols]),
        ("judged hack, BORDERLINE counted (%)", [f"{rate(a, 'judged_hack', 'up'):.1f}" for a in cols]),
        ("judged hack, WEAK not counted (%)", [f"{rate(a, 'judged_hack', 'down'):.1f}" for a in cols]),
        ("automatic proxy (%)", [f"{rate(a, 'hack_proxy'):.1f}" for a in cols]),
        ("hidden-test pass (%)", [f"{rate(a, 'hidden_pass'):.1f}" for a in cols]),
        ("   Δ vs unsteered / 95% CI", [f"{delta(a, 'hidden_pass')} {cicell(a, 'hidden_pass')}" if a != "unsteered" else "—" for a in cols]),
        ("shown-test pass (%)", [f"{rate(a, 'shown_pass'):.1f}" for a in cols]),
        ("no code submitted (%)", [f"{rate(a, 'no_code'):.1f}" for a in cols]),
        ("truncated turns / turns", [f"{summary[a]['truncated_turns']} / {summary[a]['turns']}" for a in cols]),
        ("dossiers H / B / N: proxy set", [hbn(a, "proxy") for a in cols]),
        ("dossiers H / B / N: hidden-signature set", [hbn(a, "hidden") for a in cols]),
        ("analysis family (Holm)", ["rand11" if family[a] == "rand" else "ctl15" for a in cols]),
    ]
    table = ax_t.table(cellText=[[r[0]] + r[1] for r in rows], colLabels=[""] + [lab for _, lab, *_ in ARMS],
                       loc="upper center", cellLoc="center", colWidths=[0.22] + [0.78 / len(cols)] * len(cols), bbox=[0.0, 0.0, 1.0, 1.0])
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
    fig.suptitle("Gemma 4 31B hard108, layer 30: random direction 1729 along its dose axis and against four same-norm controls\n"
                 "(blind Sonnet rubric-v3 judging, no adjudication, 16 seeds per arm)", x=0.06, ha="left", color=INK, fontsize=12.5, y=0.985)
    fig.text(0.06, 0.935, "Top-50-PC part = 1729 projected onto the top 50 principal components of the layer-30 residual (0.9% of its norm) and rescaled to the full injected norm.\n"
             "Per-token noise = a fresh isotropic unit vector per token position (seeded bank of 8192) at the same norm.\n"
             "Runs: Koby's for ±0.01 and 1730 / 1731 (jobs 14982739, 14988674, 15010135), ours for ±0.005 / pc50 / noise (array 15049924).\n"
             "CIs: problem bootstrap of the contrast vs unsteered, shifted onto the rate; Holm within each four-contrast family.",
             color=INK2, fontsize=8.8, va="top", linespacing=1.35)
    out = args.out or root / args.ctl_analysis / "gemma_hard108_ctl_chart.png"
    fig.savefig(out, dpi=args.dpi, facecolor=SURFACE)
    fig.savefig(out.with_suffix(".pdf"), facecolor=SURFACE)
    print(out)


if __name__ == "__main__":
    main()
