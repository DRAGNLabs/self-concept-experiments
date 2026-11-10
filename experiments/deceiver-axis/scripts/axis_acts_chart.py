"""Chart of axis_acts_analysis.py: one column per model, layers on x. Rows: cosine of each Goodfire reward-hacking
direction with gi_v2 (grey band: the sign-flip null of the all-response direction), then the episode-level correlation
of the two projections on unsteered episodes for the reasoning and the answer tokens, then the within-episode
token-level correlation for both (grey band: 95% of |r| between the gi_v2 projection and random directions, the level
an unrelated direction reaches; dashed lines on the episode rows: the same 95% level for the model's control
directions, real contrast directions, when the analysis ran with --control-dirs). A data table at each model's steer
layer sits under the panels.

    python scripts/axis_acts_chart.py --json <axis_compare_*.json> ... --out <png>
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SURFACE, INK, SECONDARY, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = {"gf_all": ("#2a78d6", "all response tokens"), "gf_close": ("#eb6834", "tool-call close"), "gf_last": ("#1baf7a", "last token")}
ROWS = [("cosine", None, "cosine with gi_v2"), ("episode", "reasoning", "episode r, reasoning tokens"), ("episode", "answer", "episode r, answer tokens"),
        ("token", "reasoning", "token r within episode, reasoning"), ("token", "answer", "token r within episode, answer")]


def series_values(result: dict, kind: str, segment: str | None, name: str) -> tuple[np.ndarray, np.ndarray | None]:
    """(values per layer, null half-width per layer or None)."""
    if kind == "cosine":
        entry = result["cosine"][name]
        return np.array(entry["cosine"]), (np.array(entry["sign_flip_null_abs_95"]) if "sign_flip_null_abs_95" in entry else None)
    rows = result[kind][segment][name]
    if kind == "episode":
        return np.array([row["r"] for row in rows]), np.array([row["vs_random"]["reference"]["null_abs_95"] for row in rows])
    return np.array([row["within_episode"]["r"] for row in rows]), np.array([row["within_episode"]["vs_random"]["reference"]["null_abs_95"] for row in rows])


def control_null(result: dict, segment: str, name: str) -> np.ndarray | None:
    rows = result["episode"][segment][name]
    if "vs_control" not in rows[0]:
        return None
    return np.array([row["vs_control"]["episode"]["reference"]["null_abs_95"] for row in rows])


def table_rows(result: dict) -> list[list[str]]:
    steer = result["layers"].index(result["steer_layer"])
    rows = []
    for name, (_, label) in SERIES.items():
        if name not in result["cosine"]:
            continue
        cosine = result["cosine"][name]
        cells = [result["label"], f"L{result['steer_layer']}", label,
                 f"{cosine['cosine'][steer]:+.3f}" + (f" ({cosine['sign_flip_null_abs_95'][steer]:.3f})" if "sign_flip_null_abs_95" in cosine else "")]
        for segment in ("reasoning", "answer"):
            e = result["episode"][segment][name][steer]
            if "vs_control" in e:
                c = e["vs_control"]["episode"]
                cells.append(f"{e['r']:+.2f} ({e['vs_random']['reference']['null_abs_95']:.2f} / {c['reference']['null_abs_95']:.2f}; "
                             f"{100 * c['reference']['percentile']:.0f} / {100 * c['goodfire']['percentile']:.0f})")
            else:
                cells.append(f"{e['r']:+.2f} ({e['vs_random']['reference']['null_abs_95']:.2f} / n/a; n/a)")
        e = result["episode"]["reasoning"][name][steer]
        if "vs_control" in e:
            c = e["vs_control"]["within_problem"]
            cells.append(f"{e['within_problem']['r']:+.2f} ({c['reference']['null_abs_95']:.2f}; {100 * c['reference']['percentile']:.0f} / {100 * c['goodfire']['percentile']:.0f})")
        else:
            cells.append(f"{e['within_problem']['r']:+.2f} (n/a)")
        for segment in ("reasoning", "answer"):
            t = result["token"][segment][name][steer]["within_episode"]
            cells.append(f"{t['r']:+.2f} ({t['vs_random']['reference']['null_abs_95']:.2f})")
        rows.append(cells)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    results = [json.loads(path.read_text()) for path in args.json]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": AXIS, "axes.labelcolor": SECONDARY,
                         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "axes.facecolor": SURFACE, "figure.facecolor": SURFACE})
    columns = len(results)
    table = [row for result in results for row in table_rows(result)]
    figure = plt.figure(figsize=(max(4.3 * columns, 17.2), 2.25 * len(ROWS) + 0.36 * len(table) + 3.0))  # the table needs the four-column width
    grid = figure.add_gridspec(len(ROWS) + 1, columns, height_ratios=[1.0] * len(ROWS) + [0.17 * len(table) + 0.42], hspace=0.62, wspace=0.24,
                               left=0.05, right=0.985, top=0.925, bottom=0.045)
    for ri, (kind, segment, row_title) in enumerate(ROWS):
        limit = 0.0
        axes = []
        for ci, result in enumerate(results):
            ax = figure.add_subplot(grid[ri, ci])
            axes.append(ax)
            layers = np.array(result["layers"])
            band_drawn = False
            for name, (color, label) in SERIES.items():
                if name not in result["cosine"]:
                    continue
                values, null = series_values(result, kind, segment, name)
                if null is not None and not band_drawn:
                    ax.fill_between(layers, -null, null, color=MUTED, alpha=0.18, linewidth=0, label="null, 95% of |value| (sign-flip / random directions)")
                    band_drawn = True
                    limit = max(limit, float(np.nanmax(null)))
                    control = control_null(result, segment, name) if kind == "episode" else None
                    if control is not None:
                        for sign in (-1, 1):
                            ax.plot(layers, sign * control, color=SECONDARY, linewidth=1.2, linestyle=(0, (4, 3)), label="control-direction null, 95% of |r|" if sign > 0 else None)
                        limit = max(limit, float(np.nanmax(control)))
                ax.plot(layers, values, color=color, linewidth=2, solid_capstyle="round", solid_joinstyle="round", label=label)
                limit = max(limit, float(np.nanmax(np.abs(values))))
            ax.axhline(0, color=AXIS, linewidth=1)
            ax.axvline(result["steer_layer"], color=SECONDARY, linewidth=1)
            ax.grid(axis="y", color=GRID, linewidth=1)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.set_xlabel(f"layer (line: steer layer L{result['steer_layer']}; bar: cap band L{result['band'][0]}-{result['band'][1]})", fontsize=8)
            ax.set_title(f"{result['label']}: {row_title}" if ri else f"{result['label']} ({result['episodes']} episodes): {row_title}", fontsize=9.5, loc="left", color=INK)
        limit = min(1.0, max(0.1, np.ceil(limit * 11) / 10)) if kind != "cosine" else max(0.1, np.ceil(limit * 22) / 20)
        for ax, result in zip(axes, results):
            ax.set_ylim(-limit, limit)
            ax.plot(result["band"], [-limit, -limit], color=SECONDARY, linewidth=5, solid_capstyle="butt", clip_on=False)
        if ri == 1:
            handles, labels = axes[0].get_legend_handles_labels()
            figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.962), ncol=len(labels), frameon=False, fontsize=9.5, title=None)
    figure.suptitle("Goodfire reward-hacking directions against gi_v2: direction cosine, and correlation of the two projections on unsteered episodes",
                    fontsize=13, x=0.05, ha="left", y=0.985)
    table_ax = figure.add_subplot(grid[len(ROWS), :])
    table_ax.axis("off")
    header = ["model", "layer", "Goodfire direction", "cosine with gi_v2\n(sign-flip null 95%)", "episode r, reasoning (null 95%:\nrandom / control; pct gi / gf)", "episode r, answer (null 95%:\nrandom / control; pct gi / gf)",
              "within-problem r, reasoning\n(control null 95%; pct gi / gf)", "token r within episode,\nreasoning (null 95%)", "token r within episode,\nanswer (null 95%)"]
    drawn = table_ax.table(cellText=table, colLabels=header, loc="upper center", cellLoc="center", colWidths=[0.09, 0.04, 0.105, 0.105, 0.155, 0.155, 0.15, 0.1, 0.1])
    drawn.auto_set_font_size(False)
    drawn.set_fontsize(9)
    for (row, _column), cell in drawn.get_celld().items():
        cell.set_edgecolor(AXIS)
        cell.set_linewidth(0.8)
        cell.set_height(cell.get_height() * (2.3 if row == 0 else 1.45))
        cell.set_facecolor("#f0efec" if row == 0 else SURFACE)
        cell.get_text().set_color(INK if row == 0 else SECONDARY)
    figure.text(0.05, 0.008, "Steer-layer values. Control null: the model's 109 isolated antonym contrast directions. pct gi / gf: share of control directions whose |r| with the gi_v2 projection (gi) or with the Goodfire projection (gf)\n"
                "is below the observed |r| (95+ = outside the population). Token-level null: the 32 random directions stored at extraction.", fontsize=8.5, color=SECONDARY)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
