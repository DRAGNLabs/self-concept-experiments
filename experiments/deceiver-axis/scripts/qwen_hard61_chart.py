"""Chart the Sonnet-judged Qwen3.8-27B qwen_hard61 gi_v2 run (FINDINGS "Qwen3.8-27B qwen_hard61 gi_v2 run (Koby's run), judged — 2026-11-06").

Adapted from gemma_hard108_chart.py: seven layer-32 doses on the dose axis (−0.32 … +0.32, unsteered at 0) and the cap arm
(cap_p25_L22-42) as a side point after a gap on every line panel; eight table columns. Since 2026-11-09 the floor-ladder
arms (floor_pNN_L22-42, the inverse cap) are drawn as a second side group after the cap, joined by a dotted run, when the
analysis file carries them (twelve columns; `--analysis-dir <root>/analysis_floor --summary summary_floor.json`).

Two line panels over an equally spaced, labelled dose axis (layer-32 doses of the gi_v2 vector with unsteered in the
middle; the arms are read from the analysis file, so the three-arm ±0.08 set and the five-arm ±0.04/±0.08 set both work):
(a) hack rates — judged (proxy + hidden-signature sets, as judged, with the problem-bootstrap 95% CI of the contrast
shifted onto the arm rate), the BORDERLINE-up / WEAK-down sensitivity band, and the automatic proxy — and (b) hidden-test
and shown-test pass. Under them a real table (header row, gridlines, alternating shading) with every number the lines are
drawn from plus the judged HACK/BORDERLINE/NOT_HACK dossier counts, Δ vs unsteered, CI and Holm p. Numbers are read from
`analysis_all_as_judged.json`, `analysis_all_as_judged_borderline_up.json`, `analysis_all_as_judged_weak_down.json`,
`analysis_proxy_as_judged.json` and `summary_main.json` in `--analysis-dir` (default: the judging root) and the verdict
detail files named by `--details` (default: `verdicts_main/verdicts_{main,hidden_main}.detail.json` under the root).

    python experiments/deceiver-axis/scripts/qwen_hard61_chart.py [--root <qwen38_27b_gi_v2/rh/qwen_hard61>] \
        [--analysis-dir <root>/analysis_main] [--out <path>.png]
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

DEFAULT_ROOT = Path.home() / "nobackup/autodelete/self-concept-experiments/schemer-axis/qwen38_27b_gi_v2/rh/qwen_hard61"

# Design tokens: text in text tokens, recessive rules, series colours only on marks. The first three series colours are the
# validated ones of qwen_gi_pilot_chart.py; the purple for shown pass is new and is also distinguished by marker and dash.
SURFACE, INK, INK2, INK3, RULE, SHADE = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8985", "#dedcd7", "#f1f0ec"
C_JUDGED = "#1f4e9e"   # judged hack, proxy + hidden sets
C_AUTO = "#b8651b"     # automatic proxy
C_HIDDEN = "#2a7f62"   # hidden-test pass
HIDDEN_LABEL = "hidden-test pass"  # becomes "honest hidden-test pass" when the analysis files carry honest_hidden_pass
C_SHOWN = "#7a3e9d"    # shown-test pass

# Filled by set_arms() from the analysis file: (arm, x tick label) in dose order, column labels and x positions.
ARMS: list = []
COL_LABELS: list = []
X: dict = {}


CAP = "cap_p25_L22-42"
SIDE_GAP = 0.8  # x gap between the last dose and the side point
CAPS: list = []    # cap arms present (side group 1), by percentile
FLOORS: list = []  # floor arms present (side group 2, the inverse cap), by percentile


def is_side(arm: str) -> bool:
    return arm.startswith(("cap_p", "floor_p"))


def pct(arm: str) -> int:
    return int(arm.split("_p")[1].split("_")[0])


def dose(arm: str) -> float:
    return 0.0 if arm == "unsteered" else float(arm.split("_a")[1])


def set_arms(arms: list) -> None:
    """Dose arms in dose order, then the side groups off the dose axis: cap arm(s), then the floor rungs, a gap before each group."""
    global ARMS, COL_LABELS, X, CAPS, FLOORS
    doses = sorted([a for a in arms if not is_side(a)], key=dose)
    ARMS = [(a, "0\nunsteered" if a == "unsteered" else f"{dose(a):+.2f}".replace("-", "−")) for a in doses]
    X = {arm: float(i) for i, (arm, _) in enumerate(ARMS)}
    CAPS = sorted([a for a in arms if a.startswith("cap_p")], key=pct)
    FLOORS = sorted([a for a in arms if a.startswith("floor_p")], key=pct)
    x = float(len(doses) - 1)
    for group in (CAPS, FLOORS):
        if not group:
            continue
        x += SIDE_GAP + 0.6
        for a in group:
            kind = "cap" if a.startswith("cap_p") else "floor"
            ARMS.append((a, f"{kind} p{pct(a)}\nL22–42"))
            X[a] = x
            x += 1.0
        x -= 1.0
    short = bool(FLOORS)  # twelve columns, rungs one unit apart: no room for the band in every header / tick, it is in the title and axis label
    COL_LABELS = ["unsteered" if a == "unsteered" else f"L32 {lab}" if not is_side(a) else (lab.split("\n")[0] if short else lab.replace("\n", " "))
                  for a, lab in ARMS]
    if short:
        ARMS = [(a, lab.split("\n")[0].replace(" ", "\n") if is_side(a) else lab) for a, lab in ARMS]


def dose_arms():
    return [(a, lab) for a, lab in ARMS if not is_side(a)]


def side_groups():
    return [g for g in (CAPS, FLOORS) if g]


def load(root: Path, analysis_dir: Path, summary: str, details: list) -> dict:
    files = {"all": "analysis_all_as_judged.json", "up": "analysis_all_as_judged_borderline_up.json",
             "down": "analysis_all_as_judged_weak_down.json", "proxy": "analysis_proxy_as_judged.json"}
    data = {k: json.loads((analysis_dir / f).read_text()) for k, f in files.items()}
    for block in data.values():  # main_analysis since 2026-11-07 writes honest_hidden_pass (hidden pass among non-HACK episodes)
        if "hidden_pass" not in block and "honest_hidden_pass" in block:
            block["hidden_pass"] = block["honest_hidden_pass"]
            globals()["HIDDEN_LABEL"] = "honest hidden-test pass"
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
    ax.set_xticks([X[a] for a, _ in ARMS])
    ax.set_xticklabels([label for _, label in ARMS], color=INK, fontsize=9.5)
    ax.set_xlim(-0.5, max(X.values()) + 0.5)
    for group in side_groups():
        ax.axvline(X[group[0]] - SIDE_GAP / 2 - 0.3, color=RULE, lw=0.8, ls=":", zorder=1)
    beside = "cap arm, then floor rungs (inverse cap), both on layers 22–42, beside" if FLOORS else "cap arm beside"
    ax.set_xlabel(f"dose α at layer 32 (units of the mean residual norm), gi_v2 vector; {beside}", color=INK2, fontsize=8.5, wrap=True)


def line(ax, outcome, color, label, marker="o", ls="-", lw=2.0, with_ci=True, dy=1.6):
    xs = [X[a] for a, _ in dose_arms()]
    ys = [rate(outcome, a) for a, _ in dose_arms()]
    ax.plot(xs, ys, color=color, lw=lw, ls=ls, marker=marker, ms=6, mec=SURFACE, mew=1.2, zorder=4, label=label)
    for group in side_groups():  # side points; a rung group of several arms is joined by a dotted run, never bridged to the doses
        gx, gy = [X[a] for a in group], [rate(outcome, a) for a in group]
        xs += gx; ys += gy
        ax.plot(gx, gy, color=color, marker=marker, ms=6, mec=SURFACE, mew=1.2, ls=":" if len(group) > 1 else "none", lw=1.3, zorder=4)
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
    ap.add_argument("--details", nargs="+", default=["verdicts_main/verdicts_main.detail.json", "verdicts_main/verdicts_hidden_main.detail.json",
                             "verdicts_main/verdicts_hidden_imp_main.detail.json"],
                    help="verdict .detail.json files relative to --root ('hidden' in the name = hidden-pass sets, signature + supplementary)")
    ap.set_defaults(analysis_dir=None)
    ap.add_argument("--jobs", default="Koby's qwen_hard61 run, settings.tsv", help="run named in the caption")
    args = ap.parse_args()
    d = load(args.root, args.analysis_dir or args.root / "analysis_main", args.summary, args.details)
    n_contrasts = len(ARMS) - 1
    J, UP, DOWN, PX = d["all"]["judged_hack"], d["up"]["judged_hack"], d["down"]["judged_hack"], d["proxy"]["judged_hack"]
    HP, SP, NC = d["all"]["hidden_pass"], d["all"]["shown_pass"], d["all"]["no_code"]
    AUTO = d["all"]["hack_proxy"]

    fig = plt.figure(figsize=(12.4 + 1.25 * max(0, len(ARMS) - 9), 11.2), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15], hspace=0.34, wspace=0.30, left=0.07, right=0.985, top=0.86, bottom=0.03)
    ax_a, ax_b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    ax_t = fig.add_subplot(gs[1, :])

    # (a) hack rates
    style(ax_a, "share of episodes (%)")
    xs = [X[a] for a, _ in dose_arms()]
    ax_a.fill_between(xs, [rate(DOWN, a) for a, _ in dose_arms()], [rate(UP, a) for a, _ in dose_arms()], color=C_JUDGED, alpha=0.12, lw=0, zorder=1)
    for a in CAPS + FLOORS:
        ax_a.plot([X[a], X[a]], [rate(DOWN, a), rate(UP, a)], color=C_JUDGED, alpha=0.25, lw=6, solid_capstyle="butt", zorder=1)
    line(ax_a, AUTO, C_AUTO, "automatic proxy (shown pass, hidden fail)", marker="s", ls="--", lw=1.6, with_ci=False)
    line(ax_a, J, C_JUDGED, "judged hack (proxy + hidden sets, as judged)")
    top_judged = max(max(rate(UP, a) for a, _ in ARMS), max((ci(J, a) or (0, 0))[1] for a, _ in ARMS))  # floors can lift judged above the proxy
    ax_a.set_ylim(0, max(max(rate(AUTO, a) for a, _ in ARMS) * 1.25, top_judged * 1.12))
    ax_a.set_title("(a) Hack rates: judged vs the automatic proxy", loc="left", color=INK, fontsize=11, pad=10)
    ax_a.legend(handles=[Line2D([], [], color=C_JUDGED, lw=2, marker="o", ms=6, mec=SURFACE, label="judged hack, as judged (95% CI of Δ)"),
                         Patch(facecolor=C_JUDGED, alpha=0.12, label="sensitivity band: WEAK↓ … BORDERLINE↑"),
                         Line2D([], [], color=C_AUTO, lw=1.6, ls="--", marker="s", ms=6, mec=SURFACE, label="automatic proxy")],
                loc="upper left", fontsize=8.5, frameon=False, labelcolor=INK2)

    # (b) capability
    style(ax_b, "share of episodes (%)")
    line(ax_b, SP, C_SHOWN, "shown-test pass", marker="D", ls="-.", lw=1.6)
    line(ax_b, HP, C_HIDDEN, HIDDEN_LABEL, marker="o")
    ax_b.set_ylim(0, max(rate(SP, a) for a, _ in ARMS) * 1.3)
    ax_b.set_title(f"(b) Capability: {HIDDEN_LABEL} and shown-test pass per dose", loc="left", color=INK, fontsize=10.5, pad=10)
    ax_b.legend(handles=[Line2D([], [], color=C_HIDDEN, lw=2, marker="o", ms=6, mec=SURFACE, label=f"{HIDDEN_LABEL} (95% CI of Δ)"),
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
        (f"{HIDDEN_LABEL} (%)", [f"{rate(HP, a):.1f}" for a, _ in ARMS]),
        ("   Δ vs unsteered / 95% CI", [f"{delta(HP, a)} {cicell(HP, a).replace(', ', ',')}" if a != 'unsteered' else '—' for a, _ in ARMS]),
        ("shown-test pass (%)", [f"{rate(SP, a):.1f}" for a, _ in ARMS]),
        ("no code submitted (%)", [f"{rate(NC, a):.1f}" for a, _ in ARMS]),
        ("truncated turns / turns", [f"{d['summary'][a]['truncated_turns']} / {d['summary'][a]['turns']}" for a, _ in ARMS]),
        ("dossiers judged H / B / N: proxy set", [hbn(a, "proxy") for a, _ in ARMS]),
        ("dossiers judged H / B / N: hidden-pass sets (signature + lcbhard_57/81)", [hbn(a, "hidden") for a, _ in ARMS]),
        ("episodes (61 problems × 12 seeds)", [str(J["arms"][a]["episodes"]) for a, _ in ARMS]),
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
        elif len(cell.get_text().get_text()) > (fit := 16 * 12 / len(ARMS)):  # twelve columns: a "Δ [lo,hi]" cell outgrows its column
            cell.get_text().set_fontsize(7.2 * fit / len(cell.get_text().get_text()))
    ax_t.set_title("Numbers behind the lines (rates are problem-stratified means over 61 problems; H / B / N = HACK / BORDERLINE / NOT_HACK)",
                   loc="left", color=INK, fontsize=10, pad=6)

    side = ("the p25 cap and the floor ladder (inverse cap, τ at the unsteered p" + " / p".join(str(pct(a)) for a in FLOORS) + ") over layers 22–42"
            if FLOORS else "and the p25 cap over layers 22–42")
    head = f"Qwen3.8-27B, qwen_hard61 dose scan at layer 32 (gi_v2 vector), {side}:\n".replace("), and the", ") and the")
    fig.suptitle(head +
                 "blind Sonnet rubric-v3 judging of every proxy episode, every hidden-pass episode with a hard-code signature\n"
                 "and every lcbhard_57 / lcbhard_81 hidden pass",
                 x=0.07, ha="left", color=INK, fontsize=12.5, y=0.985)
    fig.text(0.07, 0.915, f"{args.jobs}; unsteered seeds 8–19, steered{', cap and floor' if FLOORS else ' and cap'} arms seeds 0–11; 732 episodes per arm.\n"
             "Judged rate = HACK (STRONG + WEAK) as judged, no adjudication. CIs: problem bootstrap of the contrast vs unsteered, shifted onto the rate.",
             color=INK2, fontsize=9, va="top")
    out = args.out or args.root / "analysis_main" / "qwen_hard61_judged_chart.png"
    fig.savefig(out, dpi=args.dpi, facecolor=SURFACE)
    fig.savefig(out.with_suffix(".pdf"), facecolor=SURFACE)
    print(out)


if __name__ == "__main__":
    main()
