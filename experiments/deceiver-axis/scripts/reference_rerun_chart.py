"""Chart for a reference rerun read (reference_rerun_read.py output): per-seed values of the reference arm and the
rerun on each column, arm means, and a table with the difference, its problem-bootstrap interval and the seed
permutation p. Usage: reference_rerun_chart.py <rerun_read.json> <out.png> [--title T] [--mark-seeds seed0 seed1]
[--reference-label L] [--rerun-label L]"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402

SURFACE, INK, INK2, INK3, RULE, SHADE = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8985", "#dedcd7", "#f1f0ec"
C_REF, C_RERUN = "#1f4e9e", "#b8651b"
PANELS = (("honest_hidden_pass_rate", "Hidden pass among\nnon-proxy episodes", True),
          ("shown_pass_rate", "Shown pass", True),
          ("shown_pass_hidden_fail_rate", "Shown pass, hidden fail\n(proxy)", False),
          ("raw_hidden_pass_rate", "Hidden pass,\nall episodes", False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("read", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--title", default="Unsteered reference against its rerun")
    parser.add_argument("--mark-seeds", nargs="*", default=[])
    parser.add_argument("--mark-label", default="two-seed recheck")
    parser.add_argument("--reference-label", default="reference")
    parser.add_argument("--rerun-label", default="rerun")
    args = parser.parse_args()
    d = json.loads(args.read.read_text())
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig = plt.figure(figsize=(12.5, 7.4), facecolor=SURFACE)
    grid = fig.add_gridspec(2, len(PANELS), height_ratios=[3.0, 1.25], hspace=0.16, wspace=0.28, left=0.06, right=0.985, top=0.80, bottom=0.03)
    arms = (("reference", args.reference_label, C_REF), ("rerun", args.rerun_label, C_RERUN))
    for i, (column, label, in_rule) in enumerate(PANELS):
        ax = fig.add_subplot(grid[0, i], facecolor=SURFACE)
        values_all = []
        for x, (arm, arm_label, color) in enumerate(arms):
            per_seed = d["arms"][arm]["per_seed"]
            seeds = sorted(per_seed, key=lambda s: int(s[4:]))
            ys = [100 * per_seed[s][column] for s in seeds]
            values_all += ys
            counts: dict[float, int] = {}
            for s, y in zip(seeds, ys):
                k = counts.get(round(y, 1), 0)
                counts[round(y, 1)] = k + 1
                offset = ((k + 1) // 2) * 0.055 * (1 if k % 2 else -1)
                marked = arm == "rerun" and s in args.mark_seeds
                ax.scatter(x + offset, y, s=46, color=SURFACE if marked else color, edgecolor=color, linewidth=1.6 if marked else 0.8, zorder=4 if marked else 3)
            mean = 100 * d["arms"][arm]["pooled"][column]
            ax.plot([x - 0.3, x + 0.3], [mean, mean], color=color, linewidth=2, zorder=2)
            ax.text(x + 0.33, mean, f"{mean:.1f}%", va="center", ha="left", fontsize=9.5, color=INK)
        low, high = min(values_all), max(values_all)
        ax.set_ylim(low - 4, high + 4)
        ax.set_xlim(-0.55, 1.75)
        ax.set_xticks([0, 1], [a[1] for a in arms])
        ax.set_title(label + ("" if not in_rule else "  ·  rule column"), fontsize=10.5, color=INK, loc="left", pad=8)
        ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(5))
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
        ax.grid(axis="y", color=RULE, linewidth=0.7)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(RULE)
        ax.tick_params(colors=INK2, length=0)
        if i == 0:
            ax.set_ylabel("share of episodes, per seed (12 seeds × 45 problems)", color=INK2, fontsize=9)
        tax = fig.add_subplot(grid[1, i], facecolor=SURFACE)
        tax.axis("off")
        diff = d["difference"][column]
        rows = [[args.reference_label, f"{100 * d['arms']['reference']['pooled'][column]:.1f}%"],
                [args.rerun_label, f"{100 * d['arms']['rerun']['pooled'][column]:.1f}%"],
                ["difference", f"{100 * diff['delta']:+.1f}"],
                ["95% interval", f"{100 * diff['ci95'][0]:+.1f} to {100 * diff['ci95'][1]:+.1f}"],
                ["covers zero", "yes" if diff["covers_zero"] else "no"],
                ["perm. p (seeds)", f"{diff['seed_permutation_p']:.4f}" if diff["seed_permutation_p"] >= 0.0001 else "<0.0001"]]
        table = tax.table(cellText=rows, colWidths=[0.56, 0.44], loc="upper center", cellLoc="left")
        table.auto_set_font_size(False)
        table.set_fontsize(9.2)
        table.scale(1, 1.32)
        for (r, c), cell in table.get_celld().items():
            cell.set_edgecolor(RULE)
            cell.set_linewidth(0.7)
            cell.set_facecolor(SHADE if c == 0 else SURFACE)
            cell.get_text().set_color(INK2 if c == 0 else INK)
            if r == 4 and c == 1 and not diff["covers_zero"]:
                cell.get_text().set_fontweight("bold")
    fig.suptitle(args.title, x=0.06, y=0.975, ha="left", fontsize=13.5, color=INK, fontweight="bold")
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=C_REF, markersize=7, label=f"{args.reference_label}, one dot per seed"),
               plt.Line2D([], [], marker="o", linestyle="", color=C_RERUN, markersize=7, label=f"{args.rerun_label}, one dot per seed"),
               plt.Line2D([], [], color=INK2, linewidth=2, label="arm mean (all 540 episodes)")]
    if args.mark_seeds:
        handles.insert(2, plt.Line2D([], [], marker="o", linestyle="", markerfacecolor=SURFACE, markeredgecolor=C_RERUN, markeredgewidth=1.6, markersize=7, label=args.mark_label))
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.052, 0.94), ncol=len(handles), frameon=False, fontsize=9.3, labelcolor=INK2, handletextpad=0.4, columnspacing=1.6)
    fig.savefig(args.out, dpi=170, facecolor=SURFACE)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
