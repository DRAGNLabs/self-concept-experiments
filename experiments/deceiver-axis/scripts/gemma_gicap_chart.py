"""Chart the Sonnet-judged Gemma 4 31B hard108 gi_v2 doses with Koby's cap arm (FINDINGS "Gemma gi_v2 cap arm (Koby's cap_p25_L20-38), judged — 2026-11-06").

Adapted from gemma_hard108_chart.py: seven layer-32 doses on the dose axis (−0.32 … +0.32, unsteered at 0) and the cap arm
(cap_p25_L20-38, Koby's run) as a side point after a gap on every line panel; eight table columns. Six layer-30 doses.

Two line panels over an equally spaced, labelled dose axis (layer-32 doses of the gi_v2 vector with unsteered in the
middle; the arms are read from the analysis file, so the three-arm ±0.08 set and the five-arm ±0.04/±0.08 set both work):
(a) hack rates — judged (proxy + hidden-signature sets, as judged, with the problem-bootstrap 95% CI of the contrast
shifted onto the arm rate), the BORDERLINE-up / WEAK-down sensitivity band, and the automatic proxy — and (b) hidden-test
and shown-test pass. Under them a real table (header row, gridlines, alternating shading) with every number the lines are
drawn from plus the judged HACK/BORDERLINE/NOT_HACK dossier counts, Δ vs unsteered, CI and Holm p. Numbers are read from
`analysis_all_as_judged.json`, `analysis_all_as_judged_borderline_up.json`, `analysis_all_as_judged_weak_down.json`,
`analysis_proxy_as_judged.json` and `summary_main.json` in `--analysis-dir` (default: the judging root) and the verdict
detail files named by `--details` (default: `verdicts_main/verdicts_{main,hidden_main}.detail.json` under the root).

    python experiments/deceiver-axis/scripts/gemma_gicap_chart.py [--root <gemma4_31b_gi_v2/rh/hard108>] \
        [--analysis-dir <root>/analysis_all8 --summary summary_all8.json] [--out <path>.png]
"""
from __future__ import annotations

import argparse
import json
import textwrap
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

DEFAULT_ROOT = Path.home() / "nobackup/autodelete/self-concept-experiments/schemer-axis/gemma4_31b_gi_v2/rh/hard108"

# Design tokens: text in text tokens, recessive rules, series colours only on marks. The first three series colours are the
# validated ones of qwen_gi_pilot_chart.py; the purple for shown pass is new and is also distinguished by marker and dash.
SURFACE, INK, INK2, INK3, RULE, SHADE = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8985", "#dedcd7", "#f1f0ec"
C_JUDGED = "#1f4e9e"   # judged hack, proxy + hidden sets
C_AUTO = "#b8651b"     # automatic proxy
C_HIDDEN = "#2a7f62"   # hidden-test pass
C_SHOWN = "#7a3e9d"    # shown-test pass

# Filled by set_arms() from the analysis file: (arm, x tick label) in dose order, column labels and x positions.
ARMS: list = []
COL_LABELS: list = []
X: dict = {}


# Cap arms as side points off the dose axis, in rung order (p25 = Koby's arm; p35 = our lighter rung, array 15104078, 2026-11-06;
# p45 = Koby's cap-ladder run, job 15125968, judged 2026-11-09). Only the arms present in the analysis file are drawn.
CAPS = [("cap_p25_L20-38", "cap\np25", "cap p25 L20–38"), ("cap_p35_L20-38", "cap\np35", "cap p35 L20–38"), ("cap_p45_L20-38", "cap\np45", "cap p45 L20–38")]  # (arm, tick, table label); the band is named in the xlabel
CAP_LABEL = {a: col for a, _, col in CAPS}
CAP_ARMS: list = []  # the cap arms present in the analysis file, in rung order (filled by set_arms)
SIDE_GAP = 0.8  # x gap between the last dose and the first side point


def dose(arm: str) -> float:
    return 0.0 if arm == "unsteered" else float(arm.split("_a")[1])


def set_arms(arms: list) -> None:
    """Dose arms in dose order, then the cap arm as a side point off the dose axis."""
    global ARMS, COL_LABELS, X, CAP_ARMS
    doses = sorted([a for a in arms if a not in CAP_LABEL], key=dose)
    ARMS = [(a, "0\nunsteered" if a == "unsteered" else f"{dose(a):+.2f}".replace("-", "−")) for a in doses]
    X = {arm: float(i) for i, (arm, _) in enumerate(ARMS)}
    CAP_ARMS = [a for a, _, _ in CAPS if a in arms]
    for i, a in enumerate(CAP_ARMS):
        ARMS.append((a, next(tick for arm, tick, _ in CAPS if arm == a)))
        X[a] = len(doses) - 1 + SIDE_GAP + 0.6 + i
    COL_LABELS = ["unsteered" if a == "unsteered" else CAP_LABEL.get(a, f"L30 {lab}") for a, lab in ARMS]


def dose_arms():
    return [(a, lab) for a, lab in ARMS if a not in CAP_LABEL]


def load(root: Path, analysis_dir: Path, summary: str, details: list) -> dict:
    files = {"all": "analysis_all_as_judged.json", "up": "analysis_all_as_judged_borderline_up.json",
             "down": "analysis_all_as_judged_weak_down.json", "proxy": "analysis_proxy_as_judged.json"}
    data = {k: json.loads((analysis_dir / f).read_text()) for k, f in files.items()}
    data["summary"] = json.loads((analysis_dir / summary).read_text())
    counts = defaultdict(lambda: {"proxy": Counter(), "hidden": Counter()})
    for name in details:
        which = "hidden" if "hidden" in Path(name).name else "proxy"
        if not (root / name).exists():
            print(f"note: no detail file {name}, H/B/N counts for its arm will be empty")
            continue
        for key, obj in json.loads((root / name).read_text()).items():
            counts[key.split("/")[0]][which][obj["verdict"]] += 1
    data["counts"] = counts
    set_arms(list(data["all"]["judged_hack"]["arms"]))
    return data


def rate(outcome: dict, arm: str) -> float:
    return 100 * outcome["arms"][arm]["rate"]


def ci(outcome: dict, arm: str):
    """95% CI of the contrast vs unsteered, shifted onto the arm's rate; None for the reference."""
    e = outcome["arms"][arm]
    if "ci95" not in e:
        return None
    base = rate(outcome, "unsteered")
    return base + 100 * e["ci95"][0], base + 100 * e["ci95"][1]


def style(ax, ylabel: str):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(RULE)
    ax.tick_params(colors=INK2, labelsize=9, length=3)
    ax.yaxis.grid(True, color=RULE, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9.5)
    ax.set_xticks([X[a] for a, _ in ARMS])
    ax.set_xticklabels([label for _, label in ARMS], color=INK, fontsize=9.5)
    ax.set_xlim(-0.5, max(X.values()) + 0.5)
    if CAP_ARMS:
        ax.axvline(X[CAP_ARMS[0]] - SIDE_GAP / 2 - 0.3, color=RULE, lw=0.8, ls=":", zorder=1)
    ax.set_xlabel("dose α at layer 30 (units of the mean residual norm), gi_v2 vector;\ncap rungs beside: τ at the unsteered percentile over layers 20–38",
                  color=INK2, fontsize=8.5)


def line(ax, outcome, color, label, marker="o", ls="-", lw=2.0, with_ci=True, dy=1.6):
    xs = [X[a] for a, _ in dose_arms()]
    ys = [rate(outcome, a) for a, _ in dose_arms()]
    ax.plot(xs, ys, color=color, lw=lw, ls=ls, marker=marker, ms=6, mec=SURFACE, mew=1.2, zorder=4, label=label)
    if CAP_ARMS:
        cx = [X[a] for a in CAP_ARMS]; cy = [rate(outcome, a) for a in CAP_ARMS]
        xs += cx; ys += cy
        ax.plot(cx, cy, color=color, marker=marker, ms=6, mec=SURFACE, mew=1.2, ls=":" if len(cx) > 1 else "none", lw=1.2, zorder=4)
    if with_ci:
        for a, _ in ARMS:
            c = ci(outcome, a)
            if c:
                ax.plot([X[a], X[a]], c, color=color, lw=1.2, alpha=0.75, zorder=3)
                ax.plot([X[a] - 0.06, X[a] + 0.06], [c[0], c[0]], color=color, lw=1.0, alpha=0.75, zorder=3)
                ax.plot([X[a] - 0.06, X[a] + 0.06], [c[1], c[1]], color=color, lw=1.0, alpha=0.75, zorder=3)
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(9, dy), fontsize=8.5, color=INK2,
                    ha="left", va="center", zorder=6)


def fmt_p(p):
    return "" if p is None else ("< 0.0001" if p < 1e-4 else f"{p:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--dpi", type=int, default=220)
    ap.add_argument("--analysis-dir", type=Path, default=None, help="dir with analysis_*_as_judged*.json (default: --root)")
    ap.add_argument("--summary", default="summary_main.json", help="pilot_summary output in --analysis-dir")
    ap.add_argument("--details", nargs="+", default=[f"verdicts_{t}/verdicts_{h}{t}.detail.json" for t in ("main", "pm04", "pm16", "ctl_gicap_s0-15", "ctl_gicap35_s0-15", "ctl_gicap45_s0-15") for h in ("", "hidden_")],
                    help="verdict .detail.json files relative to --root ('hidden' in the name = hidden-pass sets, signature + supplementary)")
    ap.set_defaults(analysis_dir=None)
    ap.add_argument("--jobs", default="Koby's hard108 gi_v2 runs (share gemma4_31b_gi_v2/rh/hard108; cap thresholds job 14991849; cap p45 job 15125968), our cap p35 array 15104078", help="run named in the caption")
    args = ap.parse_args()
    d = load(args.root, args.analysis_dir or args.root / "analysis_main", args.summary, args.details)
    n_contrasts = len(ARMS) - 1
    J, UP, DOWN, PX = d["all"]["judged_hack"], d["up"]["judged_hack"], d["down"]["judged_hack"], d["proxy"]["judged_hack"]
    HP, SP, NC = d["all"]["hidden_pass"], d["all"]["shown_pass"], d["all"]["no_code"]
    AUTO = d["all"]["hack_proxy"]

    fig = plt.figure(figsize=(12.4 + 1.2 * max(0, len(CAP_ARMS) - 1), 11.6), facecolor=SURFACE)  # a rung wider per extra cap rung
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15], hspace=0.34, wspace=0.30, left=0.07, right=0.985, top=0.825, bottom=0.03)  # room for the title and a three-line caption above the panel titles
    ax_a, ax_b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    ax_t = fig.add_subplot(gs[1, :])

    # (a) hack rates
    style(ax_a, "share of episodes (%)")
    xs = [X[a] for a, _ in dose_arms()]
    ax_a.fill_between(xs, [rate(DOWN, a) for a, _ in dose_arms()], [rate(UP, a) for a, _ in dose_arms()], color=C_JUDGED, alpha=0.12, lw=0, zorder=1)
    for a in CAP_ARMS:
        ax_a.plot([X[a], X[a]], [rate(DOWN, a), rate(UP, a)], color=C_JUDGED, alpha=0.25, lw=6, solid_capstyle="butt", zorder=1)
    line(ax_a, AUTO, C_AUTO, "automatic proxy (shown pass, hidden fail)", marker="s", ls="--", lw=1.6, with_ci=False)
    line(ax_a, J, C_JUDGED, "judged hack (proxy + hidden sets, as judged)")
    ax_a.set_ylim(0, max(rate(AUTO, a) for a, _ in ARMS) * 1.25)
    ax_a.set_title("(a) Hack rates: judged vs the automatic proxy", loc="left", color=INK, fontsize=11, pad=10)
    ax_a.legend(handles=[Line2D([], [], color=C_JUDGED, lw=2, marker="o", ms=6, mec=SURFACE, label="judged hack, as judged (95% CI of Δ)"),
                         Patch(facecolor=C_JUDGED, alpha=0.12, label="sensitivity band: WEAK↓ … BORDERLINE↑"),
                         Line2D([], [], color=C_AUTO, lw=1.6, ls="--", marker="s", ms=6, mec=SURFACE, label="automatic proxy")],
                loc="upper left", fontsize=8.5, frameon=False, labelcolor=INK2)

    # (b) capability
    style(ax_b, "share of episodes (%)")
    line(ax_b, SP, C_SHOWN, "shown-test pass", marker="D", ls="-.", lw=1.6)
    line(ax_b, HP, C_HIDDEN, "hidden-test pass", marker="o")
    ax_b.set_ylim(0, max(rate(SP, a) for a, _ in ARMS) * 1.3)
    ax_b.set_title("(b) Capability: hidden-test and shown-test pass per dose", loc="left", color=INK, fontsize=10.5, pad=10)
    ax_b.legend(handles=[Line2D([], [], color=C_HIDDEN, lw=2, marker="o", ms=6, mec=SURFACE, label="hidden-test pass (95% CI of Δ)"),
                         Line2D([], [], color=C_SHOWN, lw=1.6, ls="-.", marker="D", ms=6, mec=SURFACE, label="shown-test pass (95% CI of Δ)")],
                loc="upper left", fontsize=8.5, frameon=False, labelcolor=INK2)

    # table
    ax_t.axis("off")

    def delta(outcome, a):
        e = outcome["arms"][a]
        return "—" if "diff_vs_unsteered" not in e else f"{100 * e['diff_vs_unsteered']:+.1f}"

    def cicell(outcome, a):
        e = outcome["arms"][a]
        return "—" if "ci95" not in e else f"[{100 * e['ci95'][0]:+.1f}, {100 * e['ci95'][1]:+.1f}]"

    def holm(outcome, a):
        return fmt_p(outcome["contrasts"].get(a, {}).get("p_holm")) or "—"

    def hbn(a, which):
        c = d["counts"][a][which]
        return f"{c['HACK']} / {c['BORDERLINE']} / {c['NOT_HACK']}"

    rows = [
        ("judged hack, as judged (%)", [f"{rate(J, a):.1f}" for a, _ in ARMS]),
        ("   Δ vs unsteered (points)", [delta(J, a) for a, _ in ARMS]),
        ("   95% CI (problem bootstrap)", [cicell(J, a) for a, _ in ARMS]),
        (f"   Holm p ({n_contrasts} contrasts)", [holm(J, a) for a, _ in ARMS]),
        ("judged hack, proxy set only (%)", [f"{rate(PX, a):.1f}" for a, _ in ARMS]),
        ("judged hack, BORDERLINE counted (%)", [f"{rate(UP, a):.1f}" for a, _ in ARMS]),
        ("judged hack, WEAK not counted (%)", [f"{rate(DOWN, a):.1f}" for a, _ in ARMS]),
        ("automatic proxy (%)", [f"{rate(AUTO, a):.1f}" for a, _ in ARMS]),
        ("   Δ vs unsteered / Holm p", [f"{delta(AUTO, a)} / {holm(AUTO, a)}" if a != 'unsteered' else '—' for a, _ in ARMS]),
        ("hidden-test pass (%)", [f"{rate(HP, a):.1f}" for a, _ in ARMS]),
        ("   Δ vs unsteered / 95% CI", [f"{delta(HP, a)} {cicell(HP, a)}" if a != 'unsteered' else '—' for a, _ in ARMS]),
        ("shown-test pass (%)", [f"{rate(SP, a):.1f}" for a, _ in ARMS]),
        ("no code submitted (%)", [f"{rate(NC, a):.1f}" for a, _ in ARMS]),
        ("truncated turns / turns", [f"{d['summary'][a]['truncated_turns']} / {d['summary'][a]['turns']}" for a, _ in ARMS]),
        ("dossiers judged H / B / N: proxy set", [hbn(a, "proxy") for a, _ in ARMS]),
        ("dossiers judged H / B / N: hidden-pass set (hard-code signature)", [hbn(a, "hidden") for a, _ in ARMS]),
        ("episodes (108 problems × 16 seeds)", [str(J["arms"][a]["episodes"]) for a, _ in ARMS]),
    ]
    table = ax_t.table(cellText=[r[1] for r in rows], rowLabels=[r[0] for r in rows], colLabels=COL_LABELS,
                       loc="upper center", cellLoc="center", rowLoc="left", colWidths=[0.7 / len(ARMS)] * len(ARMS),
                       bbox=[0.26, 0.0, 0.72, 1.0])
    table.auto_set_font_size(False)
    table.set_fontsize(7.2)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor(RULE)
        cell.set_linewidth(0.6)
        cell.get_text().set_color(INK if r == 0 or c == -1 else INK2)
        if r == 0:
            cell.set_facecolor(SHADE)
            cell.get_text().set_fontweight("bold")
        elif r % 2 == 0:
            cell.set_facecolor("#f7f6f3")
        else:
            cell.set_facecolor(SURFACE)
        if c == -1:
            cell.get_text().set_ha("left")
            cell.PAD = 0.02
    ax_t.set_title("Numbers behind the lines (rates are problem-stratified means over 108 problems; H / B / N = HACK / BORDERLINE / NOT_HACK)",
                   loc="left", color=INK, fontsize=10, pad=6)

    cap_title = "the p25 cap" if CAP_ARMS == ["cap_p25_L20-38"] else "the cap dose ladder (τ at the unsteered " + " / ".join(a.split("_")[1] for a in CAP_ARMS) + ")"
    fig.suptitle(f"Gemma 4 31B, hard108 dose scan at layer 30 (gi_v2 vector) and {cap_title} over layers 20–38:\n"
                 "blind Sonnet rubric-v3 judging of every proxy episode and every hidden-pass episode with a hard-code signature",
                 x=0.07, ha="left", color=INK, fontsize=12.5, y=0.99)
    fig.text(0.07, 0.935, textwrap.fill(f"{args.jobs}; seeds 0–15 on every arm; 1728 episodes per arm.", 170) + "\n"
             "Judged rate = HACK (STRONG + WEAK) as judged, no adjudication. CIs: problem bootstrap of the contrast vs unsteered, shifted onto the rate.",
             color=INK2, fontsize=9, va="top")
    out = args.out or args.root / "analysis_all8" / "gemma_hard108_judged_chart_8arm.png"
    fig.savefig(out, dpi=args.dpi, facecolor=SURFACE)
    fig.savefig(out.with_suffix(".pdf"), facecolor=SURFACE)
    print(out)


if __name__ == "__main__":
    main()
