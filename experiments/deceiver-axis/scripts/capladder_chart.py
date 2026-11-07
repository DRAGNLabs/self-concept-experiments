"""Chart a model's gi_v2 dose scan with the cap dose ladder (p5 / p25 / p50 caps) as side points (PLAN "Capping dose ladder on
every model"; FINDINGS "gpt-oss cap dose ladder, judged — 2026-11-06").

gemma_gicap_chart.py's layout on the 45-problem × 12-seed main-run layout of gpt-oss and Muse: two line panels over an equally
spaced, labelled dose axis (the layer-17 / layer-26 doses of the gi_v2 vector, unsteered at 0) with the cap rungs as side points
after a gap, ordered strongest first (p5, p25, p50); under them a real table (header, gridlines, shading) with every number the
lines are drawn from, the judged H / B / N dossier counts, Δ vs unsteered, CI and Holm p. Panel (a) judged hack (as judged, with
the problem-bootstrap 95% CI of the contrast shifted onto the rate), the BORDERLINE-up / WEAK-down sensitivity band and the
automatic proxy; panel (b) hidden-test and shown-test pass. Numbers come from `analysis_<tag>_all_as_judged*.json`,
`analysis_<tag>_proxy_as_judged.json` and the pilot_summary file in --root, and the verdict detail files named by --details.

    python experiments/deceiver-axis/scripts/capladder_chart.py --model gptoss [--tag capladder] [--out <path>.png]
    python experiments/deceiver-axis/scripts/capladder_chart.py --model muse --tag capladder
    python experiments/deceiver-axis/scripts/capladder_chart.py --model gptoss --tag floorladder   # cap and floor rungs side by side
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

N = Path.home() / "nobackup/autodelete/self-concept-experiments/schemer-axis"
MODELS = {
    "gptoss": dict(root=N / "gi_v2/pilot", name="gpt-oss-120b", layer=17, band="12–23", summary="summary_ladder.json",
                   details=["verdicts_main/verdicts_mixed.detail.json", "verdicts_main/verdicts_hidden_mixed.detail.json",
                            "verdicts_main/ref_proxy.detail.json", "verdicts_main/ref_hidden.detail.json",
                            "verdicts_capladder/verdicts_capladder.detail.json", "verdicts_capladder/verdicts_hidden_capladder.detail.json",
                            "verdicts_floorladder/verdicts_floorladder.detail.json", "verdicts_floorladder/verdicts_hidden_floorladder.detail.json"],
                   jobs="main run 14898345 / 14898350 (doses, p25 cap; judged 2026-10-30), cap ladder 15104508 (p5 / p50; judged 2026-11-06)",
                   jobs_floor="main run 14898345 / 14898350 (doses, p25 cap; judged 2026-10-30), cap ladder 15104508 (p5 / p50) and floor ladder 15107232 (p95 / p75 / p50), both judged 2026-11-06"),
    "muse": dict(root=N / "muse_glimmer/gi_v2/pilot", name="Muse-Glimmer-30B", layer=26, band="17–33", summary="summary_ladder.json",
                 details=["verdicts_main/verdicts_main.detail.json", "verdicts_main/verdicts_hidden_main.detail.json",
                          "verdicts_main/verdicts_p016.detail.json", "verdicts_main/verdicts_hidden_p016.detail.json",
                          "verdicts_main/ref_all.detail.json",
                          "verdicts_capladder/verdicts_capladder.detail.json", "verdicts_capladder/verdicts_hidden_capladder.detail.json",
                          "verdicts_floorladder/verdicts_floorladder.detail.json", "verdicts_floorladder/verdicts_hidden_floorladder.detail.json"],
                 jobs="main run 14898353 + 14917541 (doses, p25 cap; judged 2026-10-30), cap ladder 15104509 (p5 / p50)"),
}

SURFACE, INK, INK2, INK3, RULE, SHADE = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8985", "#dedcd7", "#f1f0ec"
C_JUDGED, C_AUTO, C_HIDDEN, C_SHOWN = "#1f4e9e", "#b8651b", "#2a7f62", "#7a3e9d"

CAPS = ["gi_cap_p5", "gi_cap_p25", "gi_cap_p50"]
FLOORS = ["gi_floor_p50", "gi_floor_p75", "gi_floor_p95"]  # the inverse cap (2026-11-06): projection clipped from below at tau
SIDE = CAPS + FLOORS  # side points beside the dose scan, caps then floors, a dotted rule before each group
SIDE_GAP = 0.8
ARMS: list = []
COL_LABELS: list = []
X: dict = {}


def dose(arm: str) -> float:
    if arm == "unsteered":
        return 0.0
    sign = -1 if "_m" in arm else 1
    return sign * float(arm.split("_")[-1][1:])


def set_arms(arms: list, layer: int) -> None:
    global ARMS, COL_LABELS, X
    doses = sorted([a for a in arms if a.startswith("gi_") and a not in SIDE] + ["unsteered"], key=dose)
    ARMS = [(a, "0\nunsteered" if a == "unsteered" else f"{dose(a):+.1f}".replace("-", "−")) for a in doses]
    X = {a: float(i) for i, (a, _) in enumerate(ARMS)}
    x = len(doses) - 1 + 0.6
    for group in (CAPS, FLOORS):
        present = [c for c in group if c in arms]
        if not present:
            continue
        x += SIDE_GAP
        for c in present:
            ARMS.append((c, f"{c.split('_')[1]}\n{c.split('_')[-1]}"))
            X[c] = x
            x += 1.15
    COL_LABELS = ["unsteered" if a == "unsteered" else (lab.replace("\n", " ") if a in SIDE else f"L{layer} {lab}") for a, lab in ARMS]


def dose_arms():
    return [(a, lab) for a, lab in ARMS if a not in SIDE]


def cap_arms():
    """Every side point: the cap rungs and, when present, the floor rungs."""
    return [(a, lab) for a, lab in ARMS if a in SIDE]


def load(root: Path, tag: str, summary: str, details: list, layer: int) -> dict:
    files = {"all": f"analysis_{tag}_all_as_judged.json", "up": f"analysis_{tag}_all_as_judged_borderline_up.json",
             "down": f"analysis_{tag}_all_as_judged_weak_down.json", "proxy": f"analysis_{tag}_proxy_as_judged.json"}
    data = {k: json.loads((root / f).read_text()) for k, f in files.items()}
    data["summary"] = json.loads((root / summary).read_text())
    counts = defaultdict(lambda: {"proxy": Counter(), "hidden": Counter()})
    for name in details:
        if not (root / name).exists():  # a ladder set not judged yet on this model (the floor ladder before it ran)
            continue
        which = "hidden" if "hidden" in Path(name).name else "proxy"
        for key, obj in json.loads((root / name).read_text()).items():
            counts[key.split("/")[0]][which][obj["verdict"]] += 1
    data["counts"] = counts
    set_arms(list(data["all"]["judged_hack"]["arms"]), layer)
    return data


def rate(outcome, arm):
    return 100 * outcome["arms"][arm]["rate"]


def ci(outcome, arm):
    e = outcome["arms"][arm]
    if "ci95" not in e:
        return None
    base = rate(outcome, "unsteered")
    return base + 100 * e["ci95"][0], base + 100 * e["ci95"][1]


def style(ax, ylabel, layer):
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
    ax.set_xticklabels([lab for _, lab in ARMS], color=INK, fontsize=9.5)
    ax.set_xlim(-0.5, max(X.values()) + 0.5)
    for group in (CAPS, FLOORS):
        first = [a for a in group if a in X]
        if first:
            ax.axvline(X[first[0]] - SIDE_GAP / 2 - 0.3, color=RULE, lw=0.8, ls=":", zorder=1)
    rungs = "cap rungs" if not any(a in X for a in FLOORS) else "cap rungs (ceiling) and floor rungs (inverse cap)"
    ax.set_xlabel(f"dose α at layer {layer} (units of the mean residual norm), gi_v2 vector;\n{rungs} beside: τ at the unsteered percentile",
                  color=INK2, fontsize=8.5)


def line(ax, outcome, color, marker="o", ls="-", lw=2.0, with_ci=True, dy=1.6):
    xs = [X[a] for a, _ in dose_arms()]
    ys = [rate(outcome, a) for a, _ in dose_arms()]
    ax.plot(xs, ys, color=color, lw=lw, ls=ls, marker=marker, ms=6, mec=SURFACE, mew=1.2, zorder=4)
    cx = [X[a] for a, _ in cap_arms()]
    cy = [rate(outcome, a) for a, _ in cap_arms()]
    for group in (CAPS, FLOORS):  # one dotted run per rung group, not a line bridging the cap and floor groups
        gx = [X[a] for a in group if a in X]
        gy = [rate(outcome, a) for a in group if a in X]
        if gx:
            ax.plot(gx, gy, color=color, lw=lw * 0.6, ls=":", marker=marker, ms=6, mec=SURFACE, mew=1.2, zorder=4)
    if with_ci:
        for a, _ in ARMS:
            c = ci(outcome, a)
            if c:
                ax.plot([X[a], X[a]], c, color=color, lw=1.2, alpha=0.75, zorder=3)
                for v in c:
                    ax.plot([X[a] - 0.06, X[a] + 0.06], [v, v], color=color, lw=1.0, alpha=0.75, zorder=3)
    for x, y in zip(xs + cx, ys + cy):
        ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(9, dy), fontsize=8.5, color=INK2, ha="left", va="center", zorder=6)


def fmt_p(p):
    return "" if p is None else ("< 0.0001" if p < 1e-4 else f"{p:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", choices=MODELS, required=True)
    ap.add_argument("--tag", default="capladder", help="analysis tag: analysis_<tag>_all_as_judged.json etc. in the pilot root")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--dpi", type=int, default=220)
    args = ap.parse_args()
    M = MODELS[args.model]
    root, layer = M["root"], M["layer"]
    d = load(root, args.tag, M["summary"], M["details"], layer)
    n_contrasts = len(d["all"]["judged_hack"].get("contrasts", {}))
    J, UP, DOWN, PX = d["all"]["judged_hack"], d["up"]["judged_hack"], d["down"]["judged_hack"], d["proxy"]["judged_hack"]
    HP, SP, NC, AUTO = d["all"]["hidden_pass"], d["all"]["shown_pass"], d["all"]["no_code"], d["all"]["hack_proxy"]

    fig = plt.figure(figsize=(12.8 + 1.3 * len([a for a in FLOORS if a in X]), 11.4), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15], hspace=0.34, wspace=0.30, left=0.07, right=0.985, top=0.86, bottom=0.03)
    ax_a, ax_b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    ax_t = fig.add_subplot(gs[1, :])

    style(ax_a, "share of episodes (%)", layer)
    xs = [X[a] for a, _ in dose_arms()]
    ax_a.fill_between(xs, [rate(DOWN, a) for a, _ in dose_arms()], [rate(UP, a) for a, _ in dose_arms()], color=C_JUDGED, alpha=0.12, lw=0, zorder=1)
    for a, _ in cap_arms():
        ax_a.plot([X[a], X[a]], [rate(DOWN, a), rate(UP, a)], color=C_JUDGED, alpha=0.25, lw=6, solid_capstyle="butt", zorder=1)
    line(ax_a, AUTO, C_AUTO, marker="s", ls="--", lw=1.6, with_ci=False)
    line(ax_a, J, C_JUDGED)
    ax_a.set_ylim(0, max(rate(AUTO, a) for a, _ in ARMS) * 1.25)
    ax_a.set_title("(a) Hack rates: judged vs the automatic proxy", loc="left", color=INK, fontsize=11, pad=10)
    ax_a.legend(handles=[Line2D([], [], color=C_JUDGED, lw=2, marker="o", ms=6, mec=SURFACE, label="judged hack, as judged (95% CI of Δ)"),
                         Patch(facecolor=C_JUDGED, alpha=0.12, label="sensitivity band: WEAK↓ … BORDERLINE↑"),
                         Line2D([], [], color=C_AUTO, lw=1.6, ls="--", marker="s", ms=6, mec=SURFACE, label="automatic proxy (shown pass, hidden fail)")],
                loc="upper left", fontsize=8.5, frameon=False, labelcolor=INK2)

    style(ax_b, "share of episodes (%)", layer)
    line(ax_b, SP, C_SHOWN, marker="D", ls="-.", lw=1.6)
    line(ax_b, HP, C_HIDDEN, marker="o")
    ax_b.set_ylim(0, max(rate(SP, a) for a, _ in ARMS) * 1.3)
    ax_b.set_title("(b) Capability: hidden-test and shown-test pass per dose and rung", loc="left", color=INK, fontsize=10.5, pad=10)
    ax_b.legend(handles=[Line2D([], [], color=C_HIDDEN, lw=2, marker="o", ms=6, mec=SURFACE, label="hidden-test pass (95% CI of Δ)"),
                         Line2D([], [], color=C_SHOWN, lw=1.6, ls="-.", marker="D", ms=6, mec=SURFACE, label="shown-test pass (95% CI of Δ)")],
                loc="upper left", fontsize=8.5, frameon=False, labelcolor=INK2)

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

    def summ(a, k):
        s = d["summary"].get(a)
        return "—" if s is None else str(s.get(k, "—"))

    rows = [
        ("judged hack, as judged (%)", [f"{rate(J, a):.1f}" for a, _ in ARMS]),
        ("   Δ vs unsteered (points)", [delta(J, a) for a, _ in ARMS]),
        ("   95% CI (problem bootstrap)", [cicell(J, a) for a, _ in ARMS]),
        (f"   Holm p ({n_contrasts} contrasts: steered gi arms + 3 randoms)", [holm(J, a) for a, _ in ARMS]),
        ("judged hack, proxy set only (%)", [f"{rate(PX, a):.1f}" for a, _ in ARMS]),
        ("judged hack, BORDERLINE counted (%)", [f"{rate(UP, a):.1f}" for a, _ in ARMS]),
        ("judged hack, WEAK not counted (%)", [f"{rate(DOWN, a):.1f}" for a, _ in ARMS]),
        ("automatic proxy (%)", [f"{rate(AUTO, a):.1f}" for a, _ in ARMS]),
        ("   Δ vs unsteered / Holm p", [f"{delta(AUTO, a)} / {holm(AUTO, a)}" if a != "unsteered" else "—" for a, _ in ARMS]),
        ("hidden-test pass (%)", [f"{rate(HP, a):.1f}" for a, _ in ARMS]),
        ("   Δ vs unsteered / 95% CI", [f"{delta(HP, a)} {cicell(HP, a)}" if a != "unsteered" else "—" for a, _ in ARMS]),
        ("shown-test pass (%)", [f"{rate(SP, a):.1f}" for a, _ in ARMS]),
        ("no code submitted (%)", [f"{rate(NC, a):.1f}" for a, _ in ARMS]),
        ("   Δ vs unsteered / 95% CI", [f"{delta(NC, a)} {cicell(NC, a)}" if a != "unsteered" else "—" for a, _ in ARMS]),
        ("truncated turns / turns", [f"{summ(a, 'truncated_turns')} / {summ(a, 'turns')}" for a, _ in ARMS]),
        ("dossiers judged H / B / N: proxy set", [hbn(a, "proxy") for a, _ in ARMS]),
        ("dossiers judged H / B / N: hidden-pass set (signature)", [hbn(a, "hidden") for a, _ in ARMS]),
        ("episodes (45 problems × 12 seeds)", [str(J["arms"][a]["episodes"]) for a, _ in ARMS]),
    ]
    table = ax_t.table(cellText=[r[1] for r in rows], rowLabels=[r[0] for r in rows], colLabels=COL_LABELS,
                       loc="upper center", cellLoc="center", rowLoc="left", colWidths=[0.72 / len(ARMS)] * len(ARMS), bbox=[0.25, 0.0, 0.74, 1.0])
    table.auto_set_font_size(False)
    table.set_fontsize(7.0)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor(RULE)
        cell.set_linewidth(0.6)
        cell.get_text().set_color(INK if r == 0 or c == -1 else INK2)
        if r == 0:
            cell.set_facecolor(SHADE)
            cell.get_text().set_fontweight("bold")
        else:
            cell.set_facecolor("#f7f6f3" if r % 2 == 0 else SURFACE)
        if c == -1:
            cell.get_text().set_ha("left")
            cell.PAD = 0.02
    ax_t.set_title("Numbers behind the lines (rates are problem-stratified means over 45 problems; H / B / N = HACK / BORDERLINE / NOT_HACK)",
                   loc="left", color=INK, fontsize=10, pad=6)
    floors = any(a in X for a in FLOORS)
    ladder = ("the cap ladder (τ at the unsteered p5 / p25 / p50) and the floor ladder, its inverse (τ at p50 / p75 / p95)" if floors
              else "the cap dose ladder (τ at the unsteered p5 / p25 / p50)")
    fig.suptitle(f"{M['name']}, gi_v2 dose scan at layer {layer} and {ladder} over layers {M['band']}:\n"
                 "blind Sonnet rubric-v3 judging of every proxy episode and every hidden-pass episode with a hard-code signature",
                 x=0.07, ha="left", color=INK, fontsize=12.5, y=0.985)
    fig.text(0.07, 0.915, f"{M.get('jobs_floor', M['jobs']) if floors else M['jobs']}; seeds 0–11 on every arm; 540 episodes per arm.\n"
             "Judged rate = HACK (STRONG + WEAK) as judged, no adjudication. CIs: problem bootstrap of the contrast vs unsteered, shifted onto the rate. "
             f"Holm over the {n_contrasts} contrasts: every steered gi arm and the three random-direction controls (the main-run family).", color=INK2, fontsize=9, va="top")
    out = args.out or root / f"{args.model}_gi_v2_{'floorladder' if floors else 'capladder'}_chart.png"
    fig.savefig(out, dpi=args.dpi, facecolor=SURFACE)
    fig.savefig(out.with_suffix(".pdf"), facecolor=SURFACE)
    print(out)


if __name__ == "__main__":
    main()
