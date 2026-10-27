"""α-scan figure for the answer-region 17-role vector (gpt-oss-120b and Muse-Glimmer-30B).

Reads the main_analysis.py outputs of the answer-region main runs and half-dose arms and draws, per model, the judged hack
rate (proxy + hidden-pass sets, as judged) against α, with the norm-matched random directions as reference points, and below
it hidden pass and no-code against α with the ±10-point dose-rule band. Error bars are 95% problem-bootstrap intervals of
each arm's paired difference from unsteered (main_analysis.py: 45 problems resampled, seeds kept with their
problem), drawn around the arm's rate; the unsteered arm has none. Writes PNG + PDF and a CSV table view.

    python scripts/plot_alpha_scan.py --out results/alpha_scan_2026-10-26
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path.home() / "nobackup/autodelete/self-concept-experiments/schemer-axis"
GPT = ROOT / "answer17_anecdotes/coding/verdicts"
MUSE = ROOT / "muse_glimmer/answer17_pilot/verdicts_main"

MODELS = {
    "gpt-oss-120b (layer 17)": {
        "judged": GPT / "analysis_all_half_as_judged.json",
        "capability": GPT / "analysis_all_half_as_judged.json",
        "arms": {"ans17_raw_m0.2": -0.2, "ans17_raw_m0.1": -0.1, "unsteered": 0.0,
                 "ans17_raw_p0.1": 0.1, "ans17_raw_p0.2": 0.2},
        "random": {"random_a0.07": 0.07, "random_a0.07_s1730": 0.07, "random_a0.07_s1731": 0.07},
        "cap": "ans17_raw_cap_p25",
    },
    "Muse-Glimmer-30B (layer 26)": {
        "judged": MUSE / "analysis_main_all_as_judged.json",
        "capability": MUSE / "analysis_main_proxy_as_judged.json",
        "arms": {"ans17_raw_m0.16": -0.16, "ans17_raw_m0.08": -0.08, "ans17_raw_m0.04": -0.04, "unsteered": 0.0,
                 "ans17_raw_p0.04": 0.04, "ans17_raw_p0.08": 0.08},
        "random": {"random_plus": 0.08, "random_plus_1730": 0.08, "random_plus_1731": 0.08},
        "cap": "ans17_raw_cap_p25",
    },
}

INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SERIES1, SERIES2 = "#2a78d6", "#eb6834"


def rate_ci(arm):
    """Arm rate with the 95% problem-bootstrap CI of its paired difference from unsteered, placed around the rate."""
    r = 100 * arm["rate"]
    if "ci95" not in arm:
        return r, r, r
    lo, hi = arm["ci95"]
    return r, r - 100 * arm["diff_vs_unsteered"] + 100 * lo, r - 100 * arm["diff_vs_unsteered"] + 100 * hi


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK2, length=0, labelsize=8.5)


def line_with_ci(ax, xs, vals, color, label):
    r, lo, hi = (np.array(v) for v in zip(*vals))
    ax.errorbar(xs, r, yerr=[r - lo, hi - r], color=color, linewidth=2, elinewidth=1, capsize=0,
                marker="o", markersize=6.5, markeredgecolor=SURFACE, markeredgewidth=1.5, label=label, zorder=3)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/alpha_scan_2026-10-26")
    args = ap.parse_args()

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9})
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.4), height_ratios=[1.25, 1], facecolor=SURFACE)
    rows = []
    for col, (model, cfg) in enumerate(MODELS.items()):
        jd = json.load(open(cfg["judged"]))["judged_hack"]["arms"]
        cd = json.load(open(cfg["capability"]))
        xs = list(cfg["arms"].values())

        ax = axes[0, col]
        style(ax)
        hack = [rate_ci(jd[a]) for a in cfg["arms"]]
        r = line_with_ci(ax, xs, hack, SERIES1, "answer-region 17-role vector")
        rand = [rate_ci(jd[a]) for a in cfg["random"]]
        rx = list(cfg["random"].values())
        offs = np.array([-1, 0, 1]) * (max(xs) - min(xs)) * 0.012
        for (v, lo, hi), x0, o in zip(rand, rx, offs):
            ax.errorbar(x0 + o, v, yerr=[[v - lo], [hi - v]], color=MUTED, elinewidth=1, capsize=0, marker="o",
                        markersize=6, markerfacecolor=SURFACE, markeredgecolor=MUTED, markeredgewidth=1.5, zorder=2)
        ax.plot([], [], "o", color=MUTED, markerfacecolor=SURFACE, markeredgewidth=1.5,
                label="random directions (3 seeds)")
        ax.annotate(f"{r[0]:.1f}%", (xs[0], r[0]), xytext=(7, -3), textcoords="offset points", ha="left",
                    va="top", color=INK2, fontsize=8.5)
        ax.annotate(f"{r[-1]:.1f}%", (xs[-1], r[-1]), xytext=(-7, 2), textcoords="offset points", ha="right",
                    va="bottom", color=INK2, fontsize=8.5)
        i0 = xs.index(0.0)
        ax.annotate(f"unsteered {r[i0]:.1f}%", (0, r[i0]), xytext=(6, -14), textcoords="offset points",
                    color=INK2, fontsize=8.5)
        ax.set_ylim(0, 60)
        ax.set_title(model, loc="left", color=INK, fontsize=10.5, fontweight="bold")
        ax.set_ylabel("judged hack rate (% of episodes)", color=INK2, fontsize=9)
        if col == 0:
            ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="upper left")

        ax = axes[1, col]
        style(ax)
        hp = [rate_ci(cd["hidden_pass"]["arms"][a]) for a in cfg["arms"]]
        nc = [rate_ci(cd["no_code"]["arms"][a]) for a in cfg["arms"]]
        base = hp[i0][0]
        ax.axhspan(base - 10, base + 10, color=GRID, alpha=0.55, linewidth=0, zorder=0)
        ax.text(xs[0], base + 10.8, "dose rule: hidden pass within ±10 points of unsteered", color=MUTED,
                fontsize=7.5, va="bottom")
        h = line_with_ci(ax, xs, hp, SERIES1, "hidden-test pass")
        n = line_with_ci(ax, xs, nc, SERIES2, "no code submitted")
        ax.annotate("hidden pass", (xs[0], h[0]), xytext=(7, 5), textcoords="offset points", color=INK2,
                    fontsize=8.5, ha="left", va="bottom")
        ax.annotate("no code", (xs[0], n[0]), xytext=(7, 5), textcoords="offset points", color=INK2,
                    fontsize=8.5, ha="left", va="bottom")
        ax.set_ylim(0, 70)
        ax.set_xlabel("α (fraction of the layer's mean residual norm)", color=INK2, fontsize=9)
        ax.set_ylabel("% of episodes", color=INK2, fontsize=9)
        if col == 0:
            ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="upper left", ncol=2)
        for a in axes[:, col]:
            a.set_xticks(xs)
            a.set_xticklabels([f"{x:+g}" if x else "0" for x in xs])
            a.set_xlim(min(xs) - 0.025 * (max(xs) - min(xs)) * 2, max(xs) + 0.05 * (max(xs) - min(xs)))

        for arm, x in list(cfg["arms"].items()) + list(cfg["random"].items()) + [(cfg["cap"], None)]:
            hk = rate_ci(jd[arm])
            rows.append({"model": model, "arm": arm, "alpha": "" if x is None else x,
                         "judged_hack_pct": round(hk[0], 1),
                         "diff_pts": round(100 * jd[arm].get("diff_vs_unsteered", 0), 1),
                         "diff_ci_lo": round(100 * jd[arm]["ci95"][0], 1) if "ci95" in jd[arm] else "",
                         "diff_ci_hi": round(100 * jd[arm]["ci95"][1], 1) if "ci95" in jd[arm] else "",
                         "hidden_pass_pct": round(100 * cd["hidden_pass"]["arms"][arm]["rate"], 1),
                         "no_code_pct": round(100 * cd["no_code"]["arms"][arm]["rate"], 1)})

    fig.text(0.01, 0.005, "Judged hack: first-pass blind Sonnet verdicts (HACK STRONG/WEAK), proxy + hidden-pass "
             "signature sets, 45 problems × 12 seeds per arm.\nBars: 95% problem-bootstrap CI of the paired difference "
             "from unsteered. Random directions: gpt-oss at α 0.07 (capability-matched), Muse at α 0.08 "
             "(norm-matched to +0.08).\nCap arms (not on the α axis): gpt-oss 4.4%, Muse 20.4%.",
             color=MUTED, fontsize=7.5, linespacing=1.5)
    fig.tight_layout(rect=(0, 0.065, 1, 1))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=200, facecolor=SURFACE)
    with open(out.with_suffix(".csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(out.with_suffix(".png"))


if __name__ == "__main__":
    main()
