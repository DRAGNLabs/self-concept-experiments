"""Chart the Sonnet-judged Gemma 4 31B hard108 gi_v2 doses with Koby's cap arm (FINDINGS "Gemma gi_v2 cap arm (Koby's cap_p25_L20-38), judged — 2026-11-06").

Adapted from gemma_hard108_chart.py: seven layer-32 doses on the dose axis (−0.32 … +0.32, unsteered at 0) and the cap arm
(cap_p25_L20-38, Koby's run) as a side point after a gap on every line panel; eight table columns. Six layer-30 doses.

Two line panels over an equally spaced, labelled dose axis (layer-30 doses of the gi_v2 vector with unsteered in the
middle; the arms are read from the analysis file, so the three-arm ±0.08 set and the five-arm ±0.04/±0.08 set both work),
drawn by the helpers of capladder_chart.py (see its docstring) so that the three charts agree:
(a) hack rates — judged (proxy + hidden-signature sets, as judged) with the 95% problem-bootstrap CI of each arm's own
rate, the unsteered arm included (the CI of Δ is in the table), the BORDERLINE-up / WEAK-down sensitivity band, the
automatic proxy, and the random-direction steering arms of the analysis file as a grey band (none in the cap-ladder
families; the footnote then says so) — and (b) the dose rule's measures, honest_hidden_pass (hidden pass among episodes not
judged a hack; raw hidden pass, labelled as such, from an analysis file without it) and no-code, each with its unsteered
± 10-point band and a ring on every value outside it. Under them a real table (header row, gridlines, alternating shading)
with every number the lines are drawn from plus the judged HACK/BORDERLINE/NOT_HACK dossier counts, Δ vs unsteered, CI,
Holm p, the dose rule and episodes / seeds per arm, and a footnote on the bars, bands, marks and p-values. Numbers are read from
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

from capladder_chart import (capability, capability_panel, chart_notes, draw_table, fewer_seeds, fit_titles, hack_panel, heaviest_rungs, is_random,
                             make_figure, n_note, random_arms, table_rows, wrapped_xlabel)

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
# p45 / p50 / p65 = Koby's cap-ladder run, job 15125968, judged 2026-11-09). Only the arms present in the analysis file are drawn.
CAPS = [("cap_p25_L20-38", "cap\np25", "cap p25 L20–38"), ("cap_p35_L20-38", "cap\np35", "cap p35 L20–38"), ("cap_p45_L20-38", "cap\np45", "cap p45 L20–38"), ("cap_p50_L20-38", "cap\np50", "cap p50 L20–38"), ("cap_p65_L20-38", "cap\np65", "cap p65 L20–38")]  # (arm, tick, table label); the band is named in the xlabel
CAP_LABEL = {a: col for a, _, col in CAPS}
CAP_ARMS: list = []  # the cap arms present in the analysis file, in rung order (filled by set_arms)
SIDE_GAP = 0.8  # x gap between the last dose and the first side point


def dose(arm: str) -> float:
    return 0.0 if arm == "unsteered" else float(arm.split("_a")[1])


def set_arms(arms: list) -> None:
    """Dose arms in dose order, then the cap arm as a side point off the dose axis."""
    global ARMS, COL_LABELS, X, CAP_ARMS
    doses = sorted([a for a in arms if a not in CAP_LABEL and not is_random(a)], key=dose)  # random arms are not columns: steering ones are the grey band of (a)
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
    heaviest = heaviest_rungs(CAP_ARMS, [])
    ax.set_xticks([X[a] for a, _ in ARMS])
    ax.set_xticklabels([label + ("\nheaviest" if a in heaviest else "") for a, label in ARMS], color=INK, fontsize=9.5)
    ax.set_xlim(-0.5, max(X.values()) + 0.5)
    if CAP_ARMS:
        ax.axvline(X[CAP_ARMS[0]] - SIDE_GAP / 2 - 0.3, color=RULE, lw=0.8, ls=":", zorder=1)
    label = "dose α at layer 30 (units of the mean residual norm), gi_v2 vector; doses equally spaced, not to scale"
    if CAP_ARMS:
        label += "; cap rungs beside, τ at the unsteered percentile over layers 20–38 (lower percentile = heavier clamp)"
    wrapped_xlabel(ax, label)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--dpi", type=int, default=220)
    ap.add_argument("--analysis-dir", type=Path, default=None, help="dir with analysis_*_as_judged*.json (default: --root)")
    ap.add_argument("--summary", default="summary_main.json", help="pilot_summary output in --analysis-dir")
    ap.add_argument("--details", nargs="+", default=[f"verdicts_{t}/verdicts_{h}{t}.detail.json" for t in ("main", "pm04", "pm16", "ctl_gicap_s0-15", "ctl_gicap35_s0-15", "ctl_gicap45_s0-15", "ctl_gicap50_s0-15", "ctl_gicap65_s0-15") for h in ("", "hidden_")],
                    help="verdict .detail.json files relative to --root ('hidden' in the name = hidden-pass sets, signature + supplementary)")
    ap.set_defaults(analysis_dir=None)
    ap.add_argument("--jobs", default="Koby's hard108 gi_v2 runs (share gemma4_31b_gi_v2/rh/hard108; cap thresholds job 14991849; cap p45 / p50 / p65 job 15125968), our cap p35 array 15104078", help="run named in the caption")
    args = ap.parse_args()
    d = load(args.root, args.analysis_dir or args.root / "analysis_main", args.summary, args.details)
    J, UP, DOWN, PX = d["all"]["judged_hack"], d["up"]["judged_hack"], d["down"]["judged_hack"], d["proxy"]["judged_hack"]
    SP, NC, AUTO = d["all"]["shown_pass"], d["all"]["no_code"], d["all"]["hack_proxy"]
    HP, hp_label, honest = capability(d["all"])  # the dose rule's measure; raw hidden pass only from a file without it
    arms, doses = [a for a, _ in ARMS], [a for a, _ in dose_arms()]
    groups = [CAP_ARMS] if CAP_ARMS else []
    labelled = list(zip(arms, COL_LABELS))
    hollow = fewer_seeds(J, arms + random_arms(J))
    axis_arms = "steered and cap gi_v2 arms" if CAP_ARMS else "steered gi_v2 arms"
    n_contrasts = len(J["contrasts"])

    fig_w = 12.4 + 1.2 * max(0, len(CAP_ARMS) - 1)  # a rung wider per extra cap rung
    rows = table_rows(d, arms, J, UP, DOWN, PX, AUTO, HP, NC, SP, hp_label, honest, f"{n_contrasts} contrasts",
                      hidden_set="hidden-pass set (hard-code signature)", clamp_model="Gemma")
    notes = chart_notes(J, HP, NC, honest, labelled, hollow, axis_arms, bool(CAP_ARMS), False, width=int(fig_w * 14.2))
    fig, ax_a, ax_b, ax_t, fig_h, notes_y = make_figure(fig_w, len(rows), len(notes))

    style(ax_a, "share of episodes (%)")
    title_a = hack_panel(ax_a, X, doses, groups, J, UP, DOWN, AUTO, hollow)
    style(ax_b, "share of episodes (%)")
    title_b = capability_panel(ax_b, X, doses, groups, HP, NC, hp_label, honest, hollow)

    draw_table(fig, ax_t, rows, COL_LABELS, size=7.2, bbox=(0.26, 0.0, 0.72, 1.0))
    ax_t.set_title(f"Numbers behind the lines (rates are problem-stratified means over {J['problems']} problems; H / B / N = HACK / BORDERLINE / NOT_HACK)",
                   loc="left", color=INK, fontsize=10, pad=6)

    cap_title = "the p25 cap" if CAP_ARMS == ["cap_p25_L20-38"] else "the cap dose ladder (τ at the unsteered " + " / ".join(a.split("_")[1] for a in CAP_ARMS) + ")"
    title = fig.suptitle(f"Gemma 4 31B, hard108 dose scan at layer 30 (gi_v2 vector) and {cap_title} over layers 20–38:\n"
                         "blind Sonnet rubric-v3 judging of every proxy episode and every hidden-pass episode with a hard-code signature",
                         x=0.07, ha="left", color=INK, fontsize=12.5, y=1 - 0.17 / fig_h)
    fig.text(0.07, 1 - 0.82 / fig_h, textwrap.fill(f"{args.jobs}; {n_note(J, labelled + [(a, a) for a in random_arms(J)])}.", int(fig_w * 13.4)) + "\n"
             "Judged rate = HACK (STRONG + WEAK) as judged, no adjudication. Notes on the bars, bands, marks and p-values are under the table.",
             color=INK2, fontsize=9, va="top")
    fig.text(0.07, notes_y, "\n".join(notes), color=INK2, fontsize=8.3, va="top", linespacing=1.25)
    fit_titles(fig, title, title_a, title_b, ax_b, dpi=args.dpi)
    out = args.out or args.root / "analysis_all8" / "gemma_hard108_judged_chart_8arm.png"
    fig.savefig(out, dpi=args.dpi, facecolor=SURFACE)
    fig.savefig(out.with_suffix(".pdf"), facecolor=SURFACE)
    print(out)


if __name__ == "__main__":
    main()
