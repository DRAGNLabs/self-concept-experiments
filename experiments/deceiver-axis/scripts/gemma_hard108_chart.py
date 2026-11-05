"""Chart the Sonnet-judged Gemma 4 31B hard108 dose pilot (FINDINGS "Gemma 4 31B hard108 dose pilot, judged — 2026-11-02").

Two line panels over an equally spaced, labelled dose axis (layer-30 doses of the gi_v2 vector with unsteered in the
middle; the arms are read from the analysis file, so the three-arm ±0.08 set and the five-arm ±0.04/±0.08 set both work):
(a) hack rates — judged (proxy + hidden-signature sets, as judged, with the problem-bootstrap 95% CI of the contrast
shifted onto the arm rate), the BORDERLINE-up / WEAK-down sensitivity band, and the automatic proxy — and (b) hidden-test
and shown-test pass. Under them a real table (header row, gridlines, alternating shading) with every number the lines are
drawn from plus the judged HACK/BORDERLINE/NOT_HACK dossier counts, Δ vs unsteered, CI and Holm p. Numbers are read from
`analysis_all_as_judged.json`, `analysis_all_as_judged_borderline_up.json`, `analysis_all_as_judged_weak_down.json`,
`analysis_proxy_as_judged.json` and `summary_main.json` in `--analysis-dir` (default: the judging root) and the verdict
detail files named by `--details` (default: `verdicts_main/verdicts_{main,hidden_main}.detail.json` under the root).

    python experiments/deceiver-axis/scripts/gemma_hard108_chart.py [--root <gemma4_31b_gi_v2/rh/hard108>] [--out <path>.png]
    python experiments/deceiver-axis/scripts/gemma_hard108_chart.py --analysis-dir <root>/analysis_all5 --summary summary_all5.json \
        --details verdicts_main/verdicts_main.detail.json verdicts_main/verdicts_hidden_main.detail.json \
                  verdicts_pm04/verdicts_pm04.detail.json verdicts_pm04/verdicts_hidden_pm04.detail.json \
        --jobs "jobs 14938958 / 14940277 / 14978114" --out <root>/analysis_all5/gemma_hard108_judged_chart_5arm.png
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


def dose(arm: str) -> float:
    return 0.0 if arm == "unsteered" else float(arm.split("_a")[1])


def set_arms(arms: list) -> None:
    global ARMS, COL_LABELS, X
    ordered = sorted(arms, key=dose)
    ARMS = [(a, "0\nunsteered" if a == "unsteered" else f"{dose(a):+.2f}".replace("-", "−")) for a in ordered]
    COL_LABELS = ["unsteered" if a == "unsteered" else f"L30 {lab}" for a, lab in ARMS]
    X = {arm: i for i, (arm, _) in enumerate(ARMS)}


def load(root: Path, analysis_dir: Path, summary: str, details: list) -> dict:
    files = {"all": "analysis_all_as_judged.json", "up": "analysis_all_as_judged_borderline_up.json",
             "down": "analysis_all_as_judged_weak_down.json", "proxy": "analysis_proxy_as_judged.json"}
    data = {k: json.loads((analysis_dir / f).read_text()) for k, f in files.items()}
    data["summary"] = json.loads((analysis_dir / summary).read_text())
    counts = defaultdict(lambda: {"proxy": Counter(), "hidden": Counter()})
    for name in details:
        which = "hidden" if "hidden" in Path(name).name else "proxy"
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
    ax.set_xticks(range(len(ARMS)))
    ax.set_xticklabels([label for _, label in ARMS], color=INK, fontsize=9.5)
    ax.set_xlim(-0.5, len(ARMS) - 0.5)
    ax.set_xlabel("steering dose α at layer 30 (units of the layer's mean residual norm), gi_v2 vector", color=INK2, fontsize=9)


def line(ax, outcome, color, label, marker="o", ls="-", lw=2.0, with_ci=True, dy=1.6):
    xs = [X[a] for a, _ in ARMS]
    ys = [rate(outcome, a) for a, _ in ARMS]
    ax.plot(xs, ys, color=color, lw=lw, ls=ls, marker=marker, ms=6, mec=SURFACE, mew=1.2, zorder=4, label=label)
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
    ap.add_argument("--details", nargs="+", default=["verdicts_main/verdicts_main.detail.json", "verdicts_main/verdicts_hidden_main.detail.json"],
                    help="verdict .detail.json files relative to --root ('hidden' in the name = hidden-signature set)")
    ap.add_argument("--jobs", default="jobs 14938958 / 14940277", help="run jobs named in the caption")
    args = ap.parse_args()
    d = load(args.root, args.analysis_dir or args.root, args.summary, args.details)
    n_contrasts = len(ARMS) - 1
    J, UP, DOWN, PX = d["all"]["judged_hack"], d["up"]["judged_hack"], d["down"]["judged_hack"], d["proxy"]["judged_hack"]
    HP, SP, NC = d["all"]["hidden_pass"], d["all"]["shown_pass"], d["all"]["no_code"]
    AUTO = d["all"]["hack_proxy"]

    fig = plt.figure(figsize=(12.4, 11.2), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15], hspace=0.30, wspace=0.26, left=0.07, right=0.985, top=0.87, bottom=0.03)
    ax_a, ax_b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    ax_t = fig.add_subplot(gs[1, :])

    # (a) hack rates
    style(ax_a, "share of episodes (%)")
    xs = [X[a] for a, _ in ARMS]
    ax_a.fill_between(xs, [rate(DOWN, a) for a, _ in ARMS], [rate(UP, a) for a, _ in ARMS], color=C_JUDGED, alpha=0.12, lw=0, zorder=1)
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
        ("dossiers judged H / B / N: hidden-signature set", [hbn(a, "hidden") for a, _ in ARMS]),
        ("episodes (108 problems × 16 seeds)", [str(J["arms"][a]["episodes"]) for a, _ in ARMS]),
    ]
    table = ax_t.table(cellText=[r[1] for r in rows], rowLabels=[r[0] for r in rows], colLabels=COL_LABELS,
                       loc="upper center", cellLoc="center", rowLoc="left", colWidths=[0.6 / len(ARMS)] * len(ARMS),
                       bbox=[0.30, 0.0, 0.66, 1.0])
    table.auto_set_font_size(False)
    table.set_fontsize(8.6 if len(ARMS) <= 3 else 7.6)
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

    fig.suptitle("Gemma 4 31B, hard108 dose pilot at layer 30 (gi_v2 vector): blind Sonnet rubric-v3 judging\n"
                 "of every proxy episode and every hidden-pass episode with a hard-code signature", x=0.07, ha="left", color=INK, fontsize=12.5, y=0.985)
    fig.text(0.07, 0.925, f"Koby's run ({args.jobs}; unsteered seeds 0–7 from the calibration pools), 1,728 episodes per arm.\n"
             "Judged rate = HACK (STRONG + WEAK) as judged, no adjudication. CIs: problem bootstrap of the contrast vs unsteered, shifted onto the rate.",
             color=INK2, fontsize=9, va="top")
    out = args.out or args.root / "gemma_hard108_judged_chart.png"
    fig.savefig(out, dpi=args.dpi, facecolor=SURFACE)
    fig.savefig(out.with_suffix(".pdf"), facecolor=SURFACE)
    print(out)


if __name__ == "__main__":
    main()
