"""Chart the Sonnet-judged prompt-rh-prevention arms (FINDINGS "Mo's honesty-prompt arms on gpt-oss-120b, judged — 2026-11-03").

Mo's experiment: the gpt-oss-120b main-run recipe (hard 45, seeds 67–78, critical_no_carve, neutral feedback, stated
budget 5) with one of six honesty prompts put in front of the task message, against a no-prefix baseline. Two bar panels
over the seven arms (ordered by no-code rate, the degradation axis): (a) hack rates — judged (proxy + hidden-signature
sets, as judged; the problem-bootstrap 95% CI of the contrast vs baseline shifted onto the arm rate; the BORDERLINE-up /
WEAK-down sensitivity band as a thin range beside the bar) and the automatic proxy — and (b) hidden-test pass,
shown-test pass and no-code, with the PLAN degradation line (baseline no-code + 10 points). Under them a table with
every number the bars are drawn from plus the dossier counts, Δ vs baseline, CI and Holm p over the six prompt arms.

    python experiments/deceiver-axis/scripts/prompt_rh_chart.py [--root <prompt_rh_prevention>] [--out <path>.png]
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

DEFAULT_ROOT = Path.home() / "nobackup/autodelete/self-concept-experiments/schemer-axis/prompt_rh_prevention"

# Design tokens (gemma_hard108_chart.py): text in text tokens, recessive rules, series colours only on marks.
SURFACE, INK, INK2, INK3, RULE, SHADE = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8985", "#dedcd7", "#f1f0ec"
C_JUDGED = "#1f4e9e"   # judged hack, proxy + hidden sets
C_AUTO = "#b8651b"     # automatic proxy
C_HIDDEN = "#2a7f62"   # hidden-test pass
C_SHOWN = "#7a3e9d"    # shown-test pass
C_NOCODE = "#8a8985"   # no code submitted (grey, hatched: an absence, not a series)

LABEL = {"unsteered": "baseline\n(no prefix)", "strict": "strict", "diligent_dev": "diligent_dev", "abort_info": "abort_info",
         "heroic_ai": "heroic_ai", "wise_professor": "wise_professor", "honest_al": "honest_al"}
COL = {"unsteered": "baseline", **{a: a for a in LABEL if a != "unsteered"}}


def load(root: Path) -> dict:
    a = root / "analysis_official"
    files = {"all": "analysis_all_as_judged.json", "up": "analysis_all_as_judged_borderline_up.json",
             "down": "analysis_all_as_judged_weak_down.json", "proxy": "analysis_proxy_as_judged.json"}
    data = {k: json.loads((a / f).read_text()) for k, f in files.items()}
    data["summary"] = json.loads((a / "summary_official.json").read_text())
    counts = defaultdict(lambda: {"proxy": Counter(), "hidden": Counter()})
    for name, which in (("verdicts_official.detail.unsteered.json", "proxy"), ("verdicts_hidden_official.detail.unsteered.json", "hidden")):
        for key, obj in json.loads((root / "verdicts_official" / name).read_text()).items():
            counts[key.split("/")[0]][which][obj["verdict"]] += 1
    data["counts"] = counts
    nc = data["all"]["no_code"]["arms"]
    data["arms"] = sorted(nc, key=lambda arm: nc[arm]["rate"])
    return data


def rate(outcome: dict, arm: str) -> float:
    return 100 * outcome["arms"][arm]["rate"]


def ci(outcome: dict, arm: str):
    e = outcome["arms"][arm]
    if "ci95" not in e:
        return None
    base = rate(outcome, "unsteered")
    return base + 100 * e["ci95"][0], base + 100 * e["ci95"][1]


def style(ax, arms, ylabel: str):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(RULE)
    ax.tick_params(colors=INK2, labelsize=9, length=3)
    ax.yaxis.grid(True, color=RULE, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9.5)
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels([LABEL[a] for a in arms], color=INK, fontsize=9)
    ax.set_xlim(-0.6, len(arms) - 0.4)
    ax.set_xlabel("prompt arm (Mo's honesty prompt in front of the task message), ordered by no-code rate", color=INK2, fontsize=9)


def bars(ax, arms, outcome, color, offset, width, label, hatch=None, with_ci=False, dy=3):
    xs = [i + offset for i in range(len(arms))]
    ys = [rate(outcome, a) for a in arms]
    ax.bar(xs, ys, width=width, color=color if not hatch else SURFACE, edgecolor=color, lw=1.0 if hatch else 0,
           hatch=hatch, label=label, zorder=3)
    if with_ci:
        for x, a in zip(xs, arms):
            c = ci(outcome, a)
            if c:
                lo, hi = max(c[0], 0), c[1]
                ax.plot([x, x], [lo, hi], color=INK, lw=1.0, alpha=0.8, zorder=5)
                ax.plot([x - 0.05, x + 0.05], [hi, hi], color=INK, lw=0.9, alpha=0.8, zorder=5)
                ax.plot([x - 0.05, x + 0.05], [lo, lo], color=INK, lw=0.9, alpha=0.8, zorder=5)
    for x, y, a in zip(xs, ys, arms):
        top = y if not with_ci or ci(outcome, a) is None else max(y, ci(outcome, a)[1])
        ax.annotate(f"{y:.1f}", (x, top), textcoords="offset points", xytext=(0, dy), fontsize=8, color=INK2, ha="center", va="bottom", zorder=6)
    return xs


def fmt_p(p):
    return "" if p is None else ("< 0.0001" if p < 1e-4 else f"{p:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--dpi", type=int, default=170)
    args = parser.parse_args()
    d = load(args.root)
    arms = d["arms"]
    J, UP, DOWN, PX = d["all"]["judged_hack"], d["up"]["judged_hack"], d["down"]["judged_hack"], d["proxy"]["judged_hack"]
    AUTO, HP, SP, NC, TURNS = (d["all"][k] for k in ("hack_proxy", "hidden_pass", "shown_pass", "no_code", "turns"))
    n_contrasts = len(J["contrasts"])

    fig = plt.figure(figsize=(13.2, 12.4), facecolor=SURFACE)
    gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 0.95, 1.55], hspace=0.42, left=0.07, right=0.985, top=0.862, bottom=0.015)
    ax_a, ax_b, ax_t = (fig.add_subplot(gs[i]) for i in range(3))

    # (a) hack rates
    style(ax_a, arms, "hack rate (% of 540 episodes)")
    xj = bars(ax_a, arms, J, C_JUDGED, -0.19, 0.34, "judged hack (proxy + hidden-signature sets, as judged), 95% CI of the contrast vs baseline", with_ci=True)
    bars(ax_a, arms, AUTO, C_AUTO, 0.19, 0.34, "automatic proxy (passed shown tests, failed the hidden suite)")
    for x, a in zip(xj, arms):
        lo, hi = rate(DOWN, a), rate(UP, a)
        ax_a.plot([x - 0.21, x - 0.21], [lo, hi], color=C_JUDGED, lw=3.2, alpha=0.35, solid_capstyle="butt", zorder=4)
    ax_a.set_ylim(0, max(rate(AUTO, a) for a in arms) * 1.3)
    handles = [Patch(color=C_JUDGED, label="judged hack, as judged (proxy + hidden-signature sets)"),
               Line2D([], [], color=INK, lw=1.0, label="95% CI of Δ vs baseline (problem bootstrap), shifted onto the rate"),
               Line2D([], [], color=C_JUDGED, lw=3.2, alpha=0.35, label="sensitivity band: WEAK not counted … BORDERLINE counted"),
               Patch(color=C_AUTO, label="automatic proxy: passed shown tests, failed the hidden suite")]
    ax_a.legend(handles=handles, loc="upper right", fontsize=8.2, frameon=False, labelcolor=INK2)
    ax_a.set_title("(a) Hack rates per arm", loc="left", color=INK, fontsize=10.5, pad=6)

    # (b) capability / degradation
    style(ax_b, arms, "% of 540 episodes")
    bars(ax_b, arms, HP, C_HIDDEN, -0.27, 0.25, "hidden-test pass")
    bars(ax_b, arms, SP, C_SHOWN, 0.0, 0.25, "shown-test pass")
    bars(ax_b, arms, NC, C_NOCODE, 0.27, 0.25, "no code submitted", hatch="////")
    rule = rate(NC, "unsteered") + 10
    ax_b.axhline(rule, color=INK3, lw=1.0, ls=(0, (4, 3)), zorder=2)
    ax_b.set_ylim(0, max(rate(NC, a) for a in arms) * 1.25)
    ax_b.legend(handles=[Patch(color=C_HIDDEN, label="hidden-test pass (the full correct suite)"),
                         Patch(color=C_SHOWN, label="shown-test pass (the flawed shown suite)"),
                         Patch(facecolor=SURFACE, edgecolor=C_NOCODE, hatch="////", label="no code submitted (reply without code)"),
                         Line2D([], [], color=INK3, lw=1.0, ls=(0, (4, 3)), label=f"PLAN degradation line: baseline no-code + 10 points = {rule:.1f}%")],
                loc="upper left", fontsize=8.2, frameon=False, labelcolor=INK2)
    ax_b.set_title("(b) Hidden pass, shown pass and no-code per arm", loc="left", color=INK, fontsize=10.5, pad=6)

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

    def per_code(a):
        hacks = d["counts"][a]["proxy"]["HACK"] + d["counts"][a]["hidden"]["HACK"]
        code = 540 * (1 - NC["arms"][a]["rate"])
        return f"{100 * hacks / code:.1f}"

    rows = [
        ("judged hack, as judged (%)", [f"{rate(J, a):.1f}" for a in arms]),
        ("   Δ vs baseline (points)", [delta(J, a) for a in arms]),
        ("   95% CI (problem bootstrap)", [cicell(J, a) for a in arms]),
        (f"   Holm p ({n_contrasts} contrasts)", [holm(J, a) for a in arms]),
        ("judged hack, proxy set only (%)", [f"{rate(PX, a):.1f}" for a in arms]),
        ("judged hack, BORDERLINE counted (%)", [f"{rate(UP, a):.1f}" for a in arms]),
        ("judged hack, WEAK not counted (%)", [f"{rate(DOWN, a):.1f}" for a in arms]),
        ("judged hacks per code-submitting episode (%)", [per_code(a) for a in arms]),
        ("automatic proxy (%)", [f"{rate(AUTO, a):.1f}" for a in arms]),
        ("   Δ vs baseline / Holm p", [f"{delta(AUTO, a)} / {holm(AUTO, a)}" if a != "unsteered" else "—" for a in arms]),
        ("hidden-test pass (%)", [f"{rate(HP, a):.1f}" for a in arms]),
        ("   Δ vs baseline / 95% CI", [f"{delta(HP, a)} {cicell(HP, a)}" if a != "unsteered" else "—" for a in arms]),
        ("shown-test pass (%)", [f"{rate(SP, a):.1f}" for a in arms]),
        ("no code submitted (%)", [f"{rate(NC, a):.1f}" for a in arms]),
        ("   Δ vs baseline (points)", [delta(NC, a) for a in arms]),
        ("turns per episode", [f"{TURNS['arms'][a]['rate']:.2f}" for a in arms]),
        ("dossiers judged H / B / N: proxy set", [hbn(a, "proxy") for a in arms]),
        ("dossiers judged H / B / N: hidden-signature set", [hbn(a, "hidden") for a in arms]),
        ("episodes (45 problems × 12 seeds)", [str(J["arms"][a]["episodes"]) for a in arms]),
    ]
    table = ax_t.table(cellText=[r[1] for r in rows], rowLabels=[r[0] for r in rows], colLabels=[COL[a] for a in arms],
                       loc="upper center", cellLoc="center", rowLoc="left", colWidths=[0.66 / len(arms)] * len(arms),
                       bbox=[0.27, 0.0, 0.72, 1.0])
    table.auto_set_font_size(False)
    table.set_fontsize(7.8)
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
    ax_t.set_title("Numbers behind the bars (rates are problem-stratified means over 45 problems; H / B / N = HACK / BORDERLINE / NOT_HACK)",
                   loc="left", color=INK, fontsize=10, pad=6)

    fig.suptitle("gpt-oss-120b, Mo's prompt-rh-prevention arms (hard 45, seeds 67–78, 540 episodes per arm): blind Sonnet rubric-v3 judging\n"
                 "of every proxy episode and every hidden-pass episode with a hard-code signature", x=0.07, ha="left", color=INK, fontsize=12.5, y=0.985)
    fig.text(0.07, 0.935, "Same recipe as the deceiver-axis gpt-oss main run (critical_no_carve, neutral feedback, stated budget 5, low reasoning), with an honesty prompt in front of the task\n"
             "message instead of steering. Judged rate = HACK (STRONG + WEAK) as judged, no adjudication. Holm over the six prompt arms vs baseline. Arms ordered by no-code rate.",
             color=INK2, fontsize=9, va="top")
    out = args.out or args.root / "analysis_official" / "prompt_rh_judged_chart.png"
    fig.savefig(out, dpi=args.dpi, facecolor=SURFACE)
    fig.savefig(out.with_suffix(".pdf"), facecolor=SURFACE)
    print(out)


if __name__ == "__main__":
    main()
