"""Four-panel trade-off chart from cap_vs_steer.json: per model, change in hack rate vs change in honest hidden pass
(both against unsteered, 95% paired-bootstrap CIs), one line through unsteered and the negative steering doses and one
through unsteered and the cap thresholds (weakest to strongest). The shaded band marks honest hidden pass more than
10 points below unsteered. Writes <out>.png and <out>.pdf.

    python scripts/cap_vs_steer_chart.py <cap_vs_steer.json> <out path without suffix>
"""
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

CAPABILITY_RULE = -10.0
STEERING_COLOR, CAP_COLOR = "#2a78d6", "#eb6834"
INK, SECONDARY_INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
TITLES_BY_MODEL = {"gpt-oss-120b": "gpt-oss-120b", "Muse-Glimmer-30B": "Muse-Glimmer-30B", "Gemma 4 31B": "Gemma 4 31B",
                   "Qwen3.8-27B (hidden-test fail, not judged)": "Qwen3.8-27B (hack = hidden-test fail, not judged)"}


def first_group(pattern: str, arm: str) -> str:
    found = re.search(pattern, arm)
    if found is None:
        raise ValueError(f"arm {arm!r} does not match {pattern!r}")
    return found.group(1)


def dose(arm: str) -> float:
    return abs(float(first_group(r"a?(-?\d*\.\d+)$", arm)))


def cap_percentile(arm: str) -> int:
    return int(first_group(r"cap_p(\d+)", arm))


def label(arm: str) -> str:
    return f"cap p{cap_percentile(arm)}" if "cap" in arm else f"−{dose(arm):g}"


def annotate(axes: Axes, text: str, x: float, y: float, offset: tuple[int, int]) -> None:
    axes.annotate(text, (x, y), textcoords="offset points", xytext=offset, fontsize=8.5, color=SECONDARY_INK, zorder=6,
                  bbox={"boxstyle": "round,pad=0.15", "facecolor": SURFACE, "edgecolor": "none", "alpha": 0.85})


def draw_series(axes: Axes, comparison: dict, arms: list[str], color: str, name: str, offset: tuple[int, int]) -> None:
    changes = comparison["vs_unsteered"]
    xs = [0.0] + [changes[arm]["honest_hidden_pass"]["estimate"] for arm in arms]
    ys = [0.0] + [changes[arm]["hack"]["estimate"] for arm in arms]
    axes.plot(xs, ys, color=color, linewidth=2, label=name, zorder=3)
    for arm, x, y in zip(arms, xs[1:], ys[1:]):
        hack, capability = changes[arm]["hack"], changes[arm]["honest_hidden_pass"]
        axes.errorbar(x, y, xerr=[[x - capability["low"]], [capability["high"] - x]], yerr=[[y - hack["low"]], [hack["high"] - y]],
                      fmt="o", markersize=8, color=color, markeredgecolor=SURFACE, markeredgewidth=2, elinewidth=1.2, alpha=0.9, zorder=4)
        annotate(axes, label(arm), x, y, offset)


def draw_panel(axes: Axes, model: str, comparison: dict) -> None:
    arms = list(comparison["vs_unsteered"])
    steered = sorted((arm for arm in arms if "cap" not in arm), key=dose)
    caps = sorted((arm for arm in arms if "cap" in arm), key=cap_percentile, reverse=True)
    lows = [comparison["vs_unsteered"][arm]["honest_hidden_pass"]["low"] for arm in arms]
    left = min(min(lows), CAPABILITY_RULE) - 4
    axes.axvspan(left, CAPABILITY_RULE, color=GRID, alpha=0.7, linewidth=0, zorder=0)
    axes.text(CAPABILITY_RULE - 0.8, 0.97, "beyond the 10-pt\ncapability rule", transform=axes.get_xaxis_transform(), ha="right", va="top", fontsize=8, color=MUTED)
    axes.axhline(0, color="#c3c2b7", linewidth=1, zorder=1)
    axes.axvline(0, color="#c3c2b7", linewidth=1, zorder=1)
    draw_series(axes, comparison, steered, STEERING_COLOR, "negative steering", (-34, 7))
    draw_series(axes, comparison, caps, CAP_COLOR, "capping", (8, 6))
    axes.plot(0, 0, "o", markersize=8, color=INK, markeredgecolor=SURFACE, markeredgewidth=2, zorder=5)
    annotate(axes, "unsteered", 0, 0, (6, 8))
    levels = comparison["levels"]["unsteered"]
    axes.set_title(f"{TITLES_BY_MODEL[model]}\nunsteered: hack {levels['hack']:.1f}%, honest hidden pass {levels['honest_hidden_pass']:.1f}%", fontsize=10, color=INK, loc="left")
    axes.set_xlim(left, max(8, max(comparison["vs_unsteered"][arm]["honest_hidden_pass"]["high"] for arm in arms) + 2))
    axes.grid(color=GRID, linewidth=0.6, zorder=0)
    axes.set_axisbelow(True)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color("#c3c2b7")
    axes.tick_params(colors=MUTED, labelsize=8.5)


def main() -> None:
    comparisons = json.loads(Path(sys.argv[1]).read_text())
    out = Path(sys.argv[2])
    figure, axes_grid = plt.subplots(2, 2, figsize=(12, 9.5), facecolor=SURFACE)
    for axes, (model, comparison) in zip(axes_grid.flat, comparisons.items()):
        axes.set_facecolor(SURFACE)
        draw_panel(axes, model, comparison)
    for axes in axes_grid[1]:
        axes.set_xlabel("Δ honest hidden pass vs unsteered (pp)  →  better", fontsize=9.5, color=SECONDARY_INK)
    for axes in axes_grid[:, 0]:
        axes.set_ylabel("Δ hack rate vs unsteered (pp)  ↓  better", fontsize=9.5, color=SECONDARY_INK)
    handles, labels = axes_grid[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.015, 0.955), ncol=2, frameon=False, fontsize=10)
    figure.suptitle("Capping vs negative steering along gi_v2: hack rate vs honest capability (95% CIs, paired bootstrap over problems)",
                    x=0.02, y=0.985, ha="left", fontsize=12, color=INK)
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    for suffix in ("png", "pdf"):
        figure.savefig(out.with_suffix(f".{suffix}"), dpi=200, facecolor=SURFACE)


if __name__ == "__main__":
    main()
