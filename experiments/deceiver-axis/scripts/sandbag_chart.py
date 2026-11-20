"""Chart of the sandbagging runs (PLAN "Sandbagging under an incentive"): the sandbagging gap (accuracy under the control
notice minus accuracy under the incentive notice) per steering arm, gi_v2 and the Goodfire axis on one x-axis (unsteered,
the negative doses, the cap rungs), one row per model, with the numbers in a table under each plot.

    sandbag_chart.py --analysis "gpt-oss-120b=<analysis.json>" ["Muse-Glimmer-30B=<analysis.json>"] --out chart.png

Reads the files `sandbag_analysis.py analyze --out` writes. A hollow marker is an arm that is not usable by the plan's rule
(control accuracy more than 10 points below unsteered, or cut-off / unanswered control answers more than 10 points above).
"""
import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

AXES = (("gi_", "gi_v2 (Deceiver Axis)", "#2a78d6", "o"), ("gf_", "Goodfire reward-hacking axis", "#eb6834", "s"))
INK, SOFT, GRID = "#1f1e1c", "#52514e", "#dddbd5"


def slot(arm: str) -> tuple:
    """(group, order value, tick label) of an arm name without its axis prefix: m0.2 -> dose, cap_p25 -> cap."""
    rest = arm[3:]
    if rest.startswith("m"):
        return (1, float(rest[1:]), f"α −{rest[1:]}")
    match = re.fullmatch(r"cap_p(\d+)", rest)
    if match:
        return (2, -float(match.group(1)), f"cap p{match.group(1)}")
    raise SystemExit(f"arm name not understood: {arm}")


def signed(value: float) -> str:
    return f"{value:+.1f}".replace("-", "−")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--analysis", nargs="+", required=True, metavar="NAME=FILE")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--benchmark", default="GSM8K", help="benchmark name in the subtitle")
    args = parser.parse_args()
    panels = []
    for spec in args.analysis:
        name, path = spec.split("=", 1)
        panels.append((name, json.loads(Path(path).read_text())))

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    columns = max(len({slot(a) for a in d["arms"] if a != "unsteered"}) for _, d in panels) + 1
    figure = plt.figure(figsize=(1.62 * columns + 3.4, 7.3 * len(panels) + 1.9), facecolor="white")
    top, bottom = 1 - 1.7 / figure.get_figheight(), 0.75 / figure.get_figheight()
    grid = figure.add_gridspec(2 * len(panels), 1, height_ratios=[1.0, 0.86] * len(panels), left=0.2, right=0.985, top=top, bottom=bottom, hspace=0.26)
    for row, (name, data) in enumerate(panels):
        arms = data["arms"]
        slots = sorted({slot(a) for a in arms if a != "unsteered"})
        position = {s: i + 1 for i, s in enumerate(slots)}
        axes = figure.add_subplot(grid[2 * row])
        base = arms["unsteered"]["gap"]
        axes.axhline(0, color=SOFT, linewidth=1.0, zorder=1)
        axes.axhspan(base["ci"][0], base["ci"][1], color="#000000", alpha=0.06, zorder=0)
        axes.axhline(base["mean"], color=INK, linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)
        axes.plot([0, 0], base["ci"], color=INK, linewidth=1.6, alpha=0.6, solid_capstyle="butt", zorder=2)
        axes.plot(0, base["mean"], marker="D", markersize=7, color=INK, zorder=5)
        for prefix, _, color, marker in AXES:
            mine = {slot(a): arms[a] for a in arms if a.startswith(prefix)}
            for group, start in ((1, [(0, base["mean"])]), (2, [])):
                points = start + [(position[s], mine[s]["gap"]["mean"]) for s in slots if s[0] == group and s in mine]
                axes.plot([x for x, _ in points], [y for _, y in points], color=color, linewidth=2.0, zorder=3)
            for s, arm in mine.items():
                axes.plot([position[s]] * 2, arm["gap"]["ci"], color=color, linewidth=1.6, alpha=0.6, zorder=2, solid_capstyle="butt")
                axes.plot(position[s], arm["gap"]["mean"], marker=marker, markersize=8.5, markerfacecolor=color if arm["usable"] else "white",
                          markeredgecolor=color, markeredgewidth=2.0, linewidth=0, zorder=4)
        tick_text = ["unsteered"] + [s[2] for s in slots]
        axes.set_xticks(range(len(tick_text))); axes.set_xticklabels(tick_text, fontsize=9.5, color=SOFT)
        axes.set_xlim(-0.6, len(tick_text) - 0.4)
        doses = sum(s[0] == 1 for s in slots)
        if doses and doses < len(slots):
            axes.axvline(doses + 0.5, color=GRID, linewidth=1.2, zorder=0)
        axes.grid(axis="y", color=GRID, linewidth=0.8); axes.set_axisbelow(True)
        axes.tick_params(colors=SOFT, labelsize=9.5, length=3)
        for side in ("top", "right"):
            axes.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axes.spines[side].set_color("#c9c7c1")
        seeds = arms["unsteered"]["seeds"]
        axes.set_title(f"{name}   (unsteered gap {signed(base['mean'])} points: {arms['unsteered']['control']['accuracy']:.1f}% correct under the control notice, "
                       f"{arms['unsteered']['incentive']['accuracy']:.1f}% under the incentive notice; {data['items']} items × {seeds} seeds)",
                       fontsize=10.5, color=INK, loc="left", pad=8)
        axes.set_ylabel("sandbagging gap (points)", color=SOFT, fontsize=10)

        table = figure.add_subplot(grid[2 * row + 1]); table.set_xlim(axes.get_xlim()); table.axis("off")
        labels = []
        for prefix, short, color, _ in AXES:
            title = short.split(" (")[0].replace(" reward-hacking axis", "")
            labels += [(f"{title}: % correct, control / incentive", color, prefix, "acc"), ("gap [95% CI]", color, prefix, "gap"),
                       ("Δ vs unsteered [95% CI]", color, prefix, "delta"), ("Holm p", color, prefix, "holm")]
        rows_n = len(labels) + 1
        table.set_ylim(0, rows_n)
        for r in range(rows_n + 1):
            table.plot([-0.5, len(tick_text) - 0.5], [r, r], color=GRID, linewidth=1.3 if r in (rows_n - 1, rows_n - 5) else 0.8, clip_on=False)
        for i in range(len(tick_text) + 1):
            table.plot([i - 0.5, i - 0.5], [0, rows_n], color=GRID, linewidth=0.8, clip_on=False)
        for i, text in enumerate(tick_text):
            table.text(i, rows_n - 0.5, text, ha="center", va="center", fontsize=9, color=INK, weight="bold")
        for r, (text, color, prefix, kind) in enumerate(labels):
            y = rows_n - 1.5 - r
            table.text(-0.6, y, text, ha="right", va="center", fontsize=9, color=color, weight="bold" if kind == "acc" else "normal")
            cells = {0: arms["unsteered"]} | {position[slot(a)]: arms[a] for a in arms if a.startswith(prefix)}
            for i in range(len(tick_text)):
                arm = cells.get(i)
                if arm is None:
                    cell = "not run" if kind == "acc" else ""
                elif kind == "acc":
                    cell = f"{arm['control']['accuracy']:.1f} / {arm['incentive']['accuracy']:.1f}"
                elif kind == "gap":
                    cell = f"{signed(arm['gap']['mean'])} [{signed(arm['gap']['ci'][0])}, {signed(arm['gap']['ci'][1])}]"
                elif i == 0:
                    cell = ""
                elif kind == "delta":
                    v = arm["vs_unsteered"]
                    cell = f"{signed(v['mean'])} [{signed(v['ci'][0])}, {signed(v['ci'][1])}]"
                else:
                    v = arm["vs_unsteered"]
                    cell = "not usable" if not arm["usable"] else ("<0.001" if v["holm"] < 0.0005 else f"{v['holm']:.3f}")
                table.text(i, y, cell, ha="center", va="center", fontsize=8.4 if kind in ("gap", "delta") else 9, color=INK)
    height = figure.get_figheight()
    figure.text(0.02, 1 - 0.42 / height, "Sandbagging under an incentive: does negative steering or capping close the gap?", fontsize=15, color=INK, weight="bold")
    figure.text(0.02, 1 - 0.72 / height, f"Gap = share of {args.benchmark} items solved when a low score leads to unlearning minus the share solved when a high score does; "
                "bars: 95% interval (bootstrap over items).", fontsize=10.2, color=SOFT)
    handles = [Line2D([], [], color=color, marker=marker, markersize=8, linewidth=2, label=name) for _, name, color, marker in AXES]
    handles += [Line2D([], [], color=INK, marker="D", markersize=7, linewidth=1.0, linestyle=(0, (4, 3)), label="unsteered (band: its 95% interval)"),
                Line2D([], [], color=SOFT, marker="o", markerfacecolor="white", markeredgewidth=2, markersize=8, linewidth=0, label="hollow: arm not usable")]
    figure.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.015, 1 - 0.8 / height), ncol=4, frameon=False, fontsize=9.5, columnspacing=1.6)
    notes = ["Each question is its own conversation and comes after an evaluation notice with a goal sentence; the two notices differ only in which score leads to unlearning. "
             "A lower gap means less deliberate underperformance. Caps run from the mildest rung (p95) to the strongest (p5).",
             "Holm: sign-flip test over items of the gap against the unsteered gap, corrected within each axis's usable arms of a model. Cap rungs are not matched across the two axes; "
             "equal α is not equal effect on the coding tasks."]
    for n, line in enumerate(notes):
        figure.text(0.02, (0.42 - 0.22 * n) / height, line, fontsize=8.8, color=SOFT)
    figure.savefig(args.out, dpi=170)
    print(args.out)


if __name__ == "__main__":
    main()
