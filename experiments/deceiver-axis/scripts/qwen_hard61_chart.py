"""Chart the Sonnet-judged Qwen3.8-27B qwen_hard61 gi_v2 run (FINDINGS "Qwen3.8-27B qwen_hard61 gi_v2 run (Koby's run), judged — 2026-11-06").

Adapted from gemma_hard108_chart.py: seven layer-32 doses on the dose axis (−0.32 … +0.32, unsteered at 0) and the cap arm
(cap_p25_L22-42) as a side point after a gap on every line panel; eight table columns. Since 2026-11-09 the floor-ladder
arms (floor_pNN_L22-42, the inverse cap) are drawn as a second side group after the cap, joined by a dotted run, when the
analysis file carries them (twelve columns; `--analysis-dir <root>/analysis_floor --summary summary_floor.json`).

Two line panels over an equally spaced, labelled dose axis (layer-32 doses of the gi_v2 vector with unsteered in the
middle; the arms are read from the analysis file, so the three-arm ±0.08 set and the five-arm ±0.04/±0.08 set both work),
drawn by the helpers of capladder_chart.py (see its docstring) so that the three charts agree:
(a) hack rates — judged (proxy + hidden-signature sets, as judged) with the 95% problem-bootstrap CI of each arm's own
rate, the unsteered arm included (the CI of Δ is in the table), the BORDERLINE-up / WEAK-down sensitivity band, the
automatic proxy, and the random-direction steering arms of the analysis file as a grey band (Qwen has none; the footnote
says so) — and (b) the dose rule's measures (hidden_pass since the 2026-11-12 merge; before: honest_hidden_pass, hidden pass among episodes not judged a hack) and
no-code, each with its unsteered ± 10-point band and a ring on every value outside it. An arm on fewer seeds than the rest
(L32_a0.24: 4 seeds) is drawn hollow; episodes and seeds per arm come from the analysis file. Under the panels a real table
(header row, gridlines, alternating shading) with every number the lines are drawn from plus the judged
HACK/BORDERLINE/NOT_HACK dossier counts, Δ vs unsteered, CI, Holm p, the dose rule and episodes / seeds per arm, and a
footnote on the bars, bands, marks and p-values. Numbers are read from
`analysis_all_as_judged.json`, `analysis_all_as_judged_borderline_up.json`, `analysis_all_as_judged_weak_down.json`,
`analysis_proxy_as_judged.json` and `summary_main.json` in `--analysis-dir` (default: the judging root) and the verdict
detail files named by `--details` (default: `verdicts_main/verdicts_{main,hidden_main}.detail.json` under the root).

    python experiments/deceiver-axis/scripts/qwen_hard61_chart.py [--root <qwen38_27b_gi_v2/rh/qwen_hard61>] \
        [--analysis-dir <root>/analysis_main] [--out <path>.png]
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

DEFAULT_ROOT = Path.home() / "nobackup/autodelete/self-concept-experiments/schemer-axis/qwen38_27b_gi_v2/rh/qwen_hard61"

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


CAP = "cap_p25_L22-42"
SIDE_GAP = 0.8  # x gap between the last dose and the side point
CAPS: list = []    # cap arms present (side group 1), by percentile
FLOORS: list = []  # floor arms present (side group 2, the inverse cap), by percentile


VECTOR = "gi_v2 vector"  # named in the axis label and title; --axis gfar swaps in the Goodfire axis


def core(arm: str) -> str:
    """Arm name without the Goodfire-axis prefix (gfar_cap_p25_L22-42 -> cap_p25_L22-42)."""
    return arm.removeprefix("gfar_")


def is_side(arm: str) -> bool:
    return core(arm).startswith(("cap_p", "floor_p"))


def pct(arm: str) -> int:
    return int(arm.split("_p")[1].split("_")[0])


def dose(arm: str) -> float:
    return 0.0 if arm == "unsteered" else float(arm.split("_a")[1])


def set_arms(arms: list) -> None:
    """Dose arms in dose order, then the side groups off the dose axis: cap arm(s), then the floor rungs, a gap before each group."""
    global ARMS, COL_LABELS, X, CAPS, FLOORS
    arms = [a for a in arms if not is_random(a)]  # random arms are not columns: steering ones are the grey band of (a), clamp ones are not drawn
    doses = sorted([a for a in arms if not is_side(a)], key=dose)
    ARMS = [(a, "0\nunsteered" if a == "unsteered" else f"{dose(a):+.2f}".replace("-", "−")) for a in doses]
    X = {arm: float(i) for i, (arm, _) in enumerate(ARMS)}
    CAPS = sorted([a for a in arms if core(a).startswith("cap_p")], key=pct)
    FLOORS = sorted([a for a in arms if core(a).startswith("floor_p")], key=pct)
    x = float(len(doses) - 1)
    for group in (CAPS, FLOORS):
        if not group:
            continue
        x += SIDE_GAP + 0.6
        for a in group:
            kind = "cap" if core(a).startswith("cap_p") else "floor"
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
    data["summary"] = json.loads((analysis_dir / summary).read_text())
    counts = defaultdict(lambda: {"proxy": Counter(), "hidden": Counter()})
    for name in details:
        which = "hidden" if "hidden" in Path(name).name else "proxy"
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
    heaviest = heaviest_rungs(CAPS, FLOORS)
    ax.set_xticks([X[a] for a, _ in ARMS])
    ax.set_xticklabels([label + ("\nheaviest" if a in heaviest else "") for a, label in ARMS], color=INK, fontsize=9.5)
    ax.set_xlim(-0.5, max(X.values()) + 0.5)
    for group in side_groups():
        ax.axvline(X[group[0]] - SIDE_GAP / 2 - 0.3, color=RULE, lw=0.8, ls=":", zorder=1)
    label = f"dose α at layer 32 (units of the mean residual norm), {VECTOR}; doses equally spaced, not to scale"
    if FLOORS:
        label += ("; cap arm, then floor rungs (inverse cap), both on layers 22–42, beside, τ at the unsteered percentile "
                  "(cap: lower percentile = heavier clamp; floor: higher percentile = heavier clamp)")
    elif CAPS:
        label += "; cap arm on layers 22–42 beside, τ at the unsteered percentile (lower percentile = heavier clamp)"
    wrapped_xlabel(ax, label)


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
    ap.add_argument("--axis", choices=["gi_v2", "gfar"], default="gi_v2",
                    help="gfar: arms named gfar_L32_a<dose> / gfar_cap_pNN_L22-42 (Goodfire reward-hacking axis); give --analysis-dir and --details")
    args = ap.parse_args()
    if args.axis == "gfar":
        globals()["VECTOR"] = "Goodfire reward-hacking axis"
    d = load(args.root, args.analysis_dir or args.root / "analysis_main", args.summary, args.details)
    J, UP, DOWN, PX = d["all"]["judged_hack"], d["up"]["judged_hack"], d["down"]["judged_hack"], d["proxy"]["judged_hack"]
    SP, NC, AUTO = d["all"]["shown_pass"], d["all"]["no_code"], d["all"]["hack_proxy"]
    HP, hp_label, honest = capability(d["all"])  # the dose rule's measure, hidden_pass; honest_hidden_pass only from a 2026-11-07 to 11-12 file without it
    arms, doses = [a for a, _ in ARMS], [a for a, _ in dose_arms()]
    labelled = list(zip(arms, COL_LABELS))
    hollow = fewer_seeds(J, arms + random_arms(J))  # an arm on fewer seeds than the rest (the 4-seed L32_a0.24 pilot arm)
    kinds = ["steered"] + (["cap"] if CAPS else []) + (["floor"] if FLOORS else [])
    axis_arms = (", ".join(kinds[:-1]) + " and " + kinds[-1] if len(kinds) > 1 else kinds[0]) + (" gfar arms" if args.axis == "gfar" else " gi_v2 arms")
    n_contrasts = len(J["contrasts"])

    fig_w = 12.4 + 1.25 * max(0, len(ARMS) - 9)
    rows = table_rows(d, arms, J, UP, DOWN, PX, AUTO, HP, NC, SP, hp_label, honest, f"{n_contrasts} contrast{'s' if n_contrasts != 1 else ''}",
                      hidden_set="hidden-pass sets (signature + lcbhard_57/81)", clamp_model="Qwen")
    notes = chart_notes(J, HP, NC, honest, labelled, hollow, axis_arms, bool(CAPS), bool(FLOORS), width=int(fig_w * 14.2))
    fig, ax_a, ax_b, ax_t, fig_h, notes_y = make_figure(fig_w, len(rows), len(notes), header_in=1.95)  # three title lines

    style(ax_a, "share of episodes (%)")
    title_a = hack_panel(ax_a, X, doses, side_groups(), J, UP, DOWN, AUTO, hollow)
    style(ax_b, "share of episodes (%)")
    title_b = capability_panel(ax_b, X, doses, side_groups(), HP, NC, hp_label, honest, hollow)

    draw_table(fig, ax_t, rows, COL_LABELS, size=7.2, bbox=(0.26, 0.0, 0.72, 1.0))
    ax_t.set_title(f"Numbers behind the lines (rates are problem-stratified means over {J['problems']} problems; H / B / N = HACK / BORDERLINE / NOT_HACK)",
                   loc="left", color=INK, fontsize=10, pad=6)

    side = ("the p25 cap and the floor ladder (inverse cap, τ at the unsteered p" + " / p".join(str(pct(a)) for a in FLOORS) + ") over layers 22–42"
            if FLOORS else "and the p25 cap over layers 22–42")
    head = f"Qwen3.8-27B, qwen_hard61 dose scan at layer 32 ({VECTOR}), {side}:\n".replace("), and the", ") and the")
    title = fig.suptitle(head +
                         "blind Sonnet rubric-v3 judging of every proxy episode, every hidden-pass episode with a hard-code signature\n"
                         "and every lcbhard_57 / lcbhard_81 hidden pass",
                         x=0.07, ha="left", color=INK, fontsize=12.5, y=1 - 0.17 / fig_h)
    fig.text(0.07, 1 - 1.0 / fig_h, textwrap.fill(f"{args.jobs}; {n_note(J, labelled + [(a, a) for a in random_arms(J)])}.", int(fig_w * 13.4)) + "\n"
             "Judged rate = HACK (STRONG + WEAK) as judged, no adjudication. Notes on the bars, bands, marks and p-values are under the table.",
             color=INK2, fontsize=9, va="top")
    fig.text(0.07, notes_y, "\n".join(notes), color=INK2, fontsize=8.3, va="top", linespacing=1.25)
    fit_titles(fig, title, title_a, title_b, ax_b, dpi=args.dpi)
    out = args.out or (args.root / "analysis_main" / "qwen_hard61_judged_chart.png" if args.axis == "gi_v2"
                       else (args.analysis_dir or args.root) / "qwen_hard61_gfar_chart.png")
    fig.savefig(out, dpi=args.dpi, facecolor=SURFACE)
    fig.savefig(out.with_suffix(".pdf"), facecolor=SURFACE)
    print(out)


if __name__ == "__main__":
    main()
