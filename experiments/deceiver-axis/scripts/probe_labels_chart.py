"""Chart of the blind general-question probe labels: change in a label's rate against unsteered by steering dose, the
gi_v2 vector and the Goodfire reward-hacking axis on one dose axis, one panel per model, with the numbers in a table
under each panel.

    probe_labels_chart.py --analysis "gpt-oss-120b=<analysis.json>" ["Muse-Glimmer-30B=<analysis.json>"] --out chart.png
                          [--label final_deception] [--label-name "final answer deceives"]

Reads the files `probe_blind_labels.py analyze --out` writes (main reading: "unclear" = no). A marker is hollow when
the arm was degraded in at least one question set and is compared on the remaining sets only.
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

AXES = (("gi_", "gi_v2 (Deceiver Axis)", "#2a78d6", "o"), ("gf_", "Goodfire reward-hacking axis", "#eb6834", "s"))
INK, SOFT, GRID = "#1f1e1c", "#52514e", "#dddbd5"
SET_NAMES = {"probes_low": "low", "probes_high": "high", "probes_striking": "striking"}


def dose(setting: str) -> float:
    sign, value = setting.split("_", 1)[1][0], setting.split("_", 1)[1][1:]
    return float(value) * (1 if sign == "p" else -1)


def panel_data(analysis: dict, label: str) -> dict:
    rows = analysis["contrasts"]["main"][label]
    counts = analysis["counts"]["main"][label]
    all_sets = sorted(analysis["usability"]["unsteered"])
    doses = sorted({dose(s) for s in rows} | {0.0})
    series = {}
    for prefix, _, _, _ in AXES:
        points = {}
        for setting, row in rows.items():
            if setting.startswith(prefix) and "delta" in row:
                points[dose(setting)] = dict(delta=100 * row["delta"], ci=[100 * row["ci"][0], 100 * row["ci"][1]], rate=100 * row["rate"],
                                             n=row["n"], holm=row.get("holm"), partial=len(row["sets"]) < len(all_sets),
                                             sets=[SET_NAMES.get(s, s) for s in row["sets"]])
        series[prefix] = points
    yes = sum(counts["unsteered"][s][0] for s in all_sets)
    total = sum(counts["unsteered"][s][1] for s in all_sets)
    return dict(doses=doses, series=series, unsteered=(int(yes), int(total)), prompts=analysis["prompts"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--analysis", nargs="+", required=True, metavar="NAME=FILE")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--label", default="final_deception")
    parser.add_argument("--label-name", default="final answer deceives or advises deceiving")
    args = parser.parse_args()
    panels = []
    for spec in args.analysis:
        name, path = spec.split("=", 1)
        panels.append((name, panel_data(json.loads(Path(path).read_text()), args.label)))

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    widths = [max(len(data["doses"]), 3) + 1.2 for _, data in panels]
    figure = plt.figure(figsize=((2.05 if len(panels) == 1 else 1.6) * sum(widths) + 1.6, 8.6), facecolor="white")
    grid = figure.add_gridspec(2, len(panels), width_ratios=widths, height_ratios=[1.0, 0.62], left=0.105 if len(panels) > 1 else 0.17,
                               right=0.985, top=0.80, bottom=0.115, hspace=0.16, wspace=0.28)
    low = min([p["ci"][0] for _, d in panels for s in d["series"].values() for p in s.values()] + [0.0])
    high = max([p["ci"][1] for _, d in panels for s in d["series"].values() for p in s.values()] + [0.0])
    pad = 0.08 * (high - low or 1.0)
    for column, (name, data) in enumerate(panels):
        axes = figure.add_subplot(grid[0, column])
        position = {d: i for i, d in enumerate(data["doses"])}
        axes.axhline(0, color=SOFT, linewidth=1.0, zorder=1)
        for prefix, _, color, marker in AXES:
            points = data["series"][prefix]
            xs = sorted(points)
            if not xs:
                continue
            line_x = sorted(xs + [0.0])
            axes.plot([position[d] for d in line_x], [points[d]["delta"] if d in points else 0.0 for d in line_x], color=color, linewidth=2.0, zorder=3)
            for d in xs:
                p = points[d]
                axes.plot([position[d]] * 2, p["ci"], color=color, linewidth=1.6, alpha=0.6, zorder=2, solid_capstyle="butt")
                axes.plot(position[d], p["delta"], marker=marker, markersize=8.5, markerfacecolor="white" if p["partial"] else color,
                          markeredgecolor=color, markeredgewidth=2.0, linewidth=0, zorder=4)
        axes.plot(position[0.0], 0, marker="o", markersize=5, color=INK, zorder=5)
        ticks = range(len(data["doses"]))
        tick_text = ["unsteered" if d == 0 else f"{d:+g}".replace("-", "−") for d in data["doses"]]
        axes.set_xticks(list(ticks)); axes.set_xticklabels(tick_text, fontsize=9.5, color=SOFT)
        axes.set_xlim(-0.6, len(data["doses"]) - 0.4); axes.set_ylim(low - pad, high + pad)
        axes.grid(axis="y", color=GRID, linewidth=0.8); axes.set_axisbelow(True)
        axes.tick_params(colors=SOFT, labelsize=9.5, length=3)
        for side in ("top", "right"):
            axes.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axes.spines[side].set_color("#c9c7c1")
        yes, total = data["unsteered"]
        axes.set_title(f"{name}\nunsteered: {yes} of {total} answers ({100 * yes / total:.1f}%)", fontsize=10.5, color=INK, loc="left", pad=8, linespacing=1.4)
        if column == 0:
            axes.set_ylabel("Δ share of answers vs. unsteered (points)", color=SOFT, fontsize=10)
        # table under the panel, columns at the dose positions
        table = figure.add_subplot(grid[1, column]); table.set_xlim(axes.get_xlim()); table.axis("off")
        labels = []
        for prefix, short, color, _ in AXES:
            labels += [(f"{short.split(' (')[0].replace(' reward-hacking axis', '')}: yes / answers", color), ("Δ points [95% CI]", color), ("Holm p", color)]
        labels.append(("sets compared", SOFT))
        rows_n = len(labels) + 1
        table.set_ylim(0, rows_n)
        for r in range(rows_n + 1):
            table.plot([-0.5, len(data["doses"]) - 0.5], [r, r], color=GRID, linewidth=0.8, clip_on=False)
        for i in range(len(data["doses"]) + 1):
            table.plot([i - 0.5, i - 0.5], [0, rows_n], color=GRID, linewidth=0.8, clip_on=False)
        for i, text in enumerate(tick_text):
            table.text(i, rows_n - 0.5, f"α {text}" if text != "unsteered" else text, ha="center", va="center", fontsize=9, color=INK, weight="bold")
        for r, (text, color) in enumerate(labels):
            y = rows_n - 1.5 - r
            table.text(-0.6, y, text, ha="right", va="center", fontsize=9, color=color if r < 6 else SOFT,
                       weight="bold" if r in (0, 3) else "normal")
            prefix = AXES[r // 3][0] if r < 6 else None
            for d, i in position.items():
                if d == 0:
                    cell = f"{yes} / {total}" if r in (0, 3) else ("all" if r == 6 else "")
                elif r == 6:
                    used = {tuple(data["series"][p][d]["sets"]) for p, *_ in AXES if d in data["series"][p]}
                    sets = [data["series"][p][d]["sets"] for p, *_ in AXES if d in data["series"][p]]
                    cell = "all" if all(len(s) == 3 for s in sets) else " / ".join(sorted({", ".join(s) for s in sets if len(s) < 3})) + (" *" if len(used) > 1 else "")
                else:
                    p = data["series"][prefix].get(d)
                    if p is None:
                        cell = "not run"
                    elif r % 3 == 0:
                        cell = f"{round(p['rate'] * p['n'] / 100)} / {p['n']}"
                    elif r % 3 == 1:
                        cell = f"{p['delta']:+.1f} [{p['ci'][0]:+.1f}, {p['ci'][1]:+.1f}]".replace("-", "−")
                    else:
                        cell = "" if p["holm"] is None else ("<0.001" if p["holm"] < 0.001 else f"{p['holm']:.3f}")
                table.text(i, y, cell, ha="center", va="center", fontsize=8.6 if r % 3 == 1 and r < 6 else 9, color=INK)
    figure.text(0.02, 0.955, f"Blind label “{args.label_name}” on general-question probes", fontsize=15, color=INK, weight="bold")
    prompts = sorted({d["prompts"] for _, d in panels})
    figure.text(0.02, 0.918, f"Change against unsteered by steering dose; {prompts[0]} prompts, two seeds, Sonnet labels blind to the setting; bars: 95% interval (bootstrap over prompts).",
                fontsize=10.2, color=SOFT)
    handles = [Line2D([], [], color=color, marker=marker, markersize=8, linewidth=2, label=name) for _, name, color, marker in AXES]
    handles.append(Line2D([], [], color=SOFT, marker="o", markerfacecolor="white", markeredgewidth=2, markersize=8, linewidth=0,
                          label="hollow: compared on fewer question sets (degraded in the others)"))
    figure.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.015, 0.905), ncol=3, frameon=False, fontsize=9.5, columnspacing=1.6)
    notes = ["A set is left out for an arm when more than 10 points more of its answers than unsteered are cut off or empty; the Δ and the count are then over the remaining sets.",
             "Holm: sign-flip test over prompts, corrected within each axis's doses of a model. “unclear” counts as no. * the two axes are compared on different sets at this dose."]
    for n, line in enumerate(notes):
        figure.text(0.02, 0.055 - 0.028 * n, line, fontsize=8.8, color=SOFT)
    figure.savefig(args.out, dpi=200)
    print(args.out)


if __name__ == "__main__":
    main()
