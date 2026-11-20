"""EXPLORATORY (lightly reviewed when merged 2026-11-20): kept for reference, not a validated measurement.

100% stacked bars of judged-hack mechanism and pressure source per arm (unsteered, gi_v2 -, gfar -), one panel
per model, from hack_timing.py's JSON. Writes <out>_mechanism.{png,pdf} and <out>_pressure_source.{png,pdf} (shares, bars sum to 100%) and
<out>_count_mechanism / <out>_count_pressure_source (counts, x scale per model). Models
with an arm holding no judged hack are left out. (CPU)

    python experiments/deceiver-axis/scripts/hack_mix_chart.py --timing-json <hack_timing.json> --out <stem>
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch

from selfconcept.deceiver_axis.behaviour_measures.episodes import ARMS
from selfconcept.deceiver_axis.behaviour_measures.palette import ARM_LABEL, GRID, INK, MUTED

SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
LABEL_BY_CATEGORY = {"lookup_table": "lookup table", "input_branch": "input branch", "fitted_rule": "fitted rule",
                     "size_restriction": "size restriction", "test_tampering": "test tampering",
                     "always_equal": "always-equal object", "prior_failures": "prior failures",
                     "impossibility": "believes it impossible", "none_stated": "none stated",
                     "attempt_budget": "attempt budget", "pass_tests_instruction": "\"pass the tests\" instruction",
                     "grading_note": "grading note", "STRONG": "HACK, strong",
                     "WEAK": "HACK, weak", "BORDERLINE": "BORDERLINE",
                     "passed": "passed the visible tests", "attempt_limit": "used all 5 attempts",
                     "stopped": "stopped without passing", "context_window": "ran out of context",
                     "flagged": "flagged for human review"}
TITLE_BY_FIELD = {"severity": "Judged hack strength, with BORDERLINE verdicts (rubric-v3 judge)",
                  "mechanism": "Hack mechanism (rubric-v3 judge)",
                  "pressure_source": "Pressure the reasoning cites for hacking (rubric-v3 judge)",
                  "stop_reason": "How episodes end (harness stop reason, all episodes)"}
MIN_LABELED_SHARE = 0.08
ORDINAL_COLOR_BY_FIELD = {"severity": {"STRONG": "#1c5cab", "WEAK": "#5598e7", "BORDERLINE": "#9ec5f4"}}


def label_ink(color: str) -> str:
    red, green, blue = to_rgb(color)
    return "#0b0b0b" if 0.2126 * red + 0.7152 * green + 0.0722 * blue > 0.45 else "white"


def save_field_chart(summary_by_model: dict[str, dict], field: str, absolute: bool, out: Path) -> None:
    """Shares of each arm's hacks (bars sum to 100%), or counts with a per-model x scale (absolute)."""
    models = [model for model, summary in summary_by_model.items()
              if all(sum(summary[field][arm].values()) for arm in ARMS)]
    totals: Counter = Counter()
    for model in models:
        for arm in ARMS:
            totals.update(summary_by_model[model][field][arm])
    if field in ORDINAL_COLOR_BY_FIELD:
        color_by_category = ORDINAL_COLOR_BY_FIELD[field]
        categories = list(color_by_category)
    else:
        categories = [category for category, _ in totals.most_common()]
        color_by_category = dict(zip(categories, SERIES_COLORS))
    left_out = [model for model in summary_by_model if model not in models]

    figure, axes_grid = plt.subplots(len(models), 1, figsize=(9, 1.25 * len(ARMS) * len(models) + 1.2),
                                     sharex=not absolute)
    for axes, model in zip(axes_grid, models):
        counts_by_arm = summary_by_model[model][field]
        scale = max(sum(counts_by_arm[arm].values()) for arm in ARMS) if absolute else 1.0
        for row, arm in enumerate(ARMS):
            n_hacks = sum(counts_by_arm[arm].values())
            left = 0.0
            for category in categories:
                count = counts_by_arm[arm].get(category, 0)
                width = count if absolute else count / n_hacks
                if count == 0:
                    continue
                axes.barh(row, width, left=left, height=0.62, color=color_by_category[category],
                          edgecolor="white", linewidth=2)
                if width / scale >= MIN_LABELED_SHARE:
                    axes.text(left + 0.012 * scale, row, str(count) if absolute else f"{width:.0%}", va="center",
                              ha="left", fontsize=8.5, color=label_ink(color_by_category[category]))
                left += width
            axes.text(left + 0.01 * scale, row, f"n = {n_hacks}", va="center", ha="left", fontsize=8.5,
                      color=MUTED)
        axes.set_yticks(range(len(ARMS)), [ARM_LABEL[arm] for arm in ARMS], fontsize=9, color=INK)
        axes.invert_yaxis()
        axes.set_xlim(0, scale * 1.1 if absolute else 1)
        axes.set_title(model, loc="left", fontsize=10.5, color=INK)
        axes.tick_params(colors=MUTED, labelsize=9, length=0)
        for spine in axes.spines.values():
            spine.set_visible(False)
    unit = {"severity": "judged hacks and BORDERLINE verdicts", "stop_reason": "episodes"}.get(field, "judged hacks")
    if absolute:
        axes_grid[-1].set_xlabel(f"{unit} (count; x scale per model)", color=MUTED, fontsize=9)
    else:
        axes_grid[-1].xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
        axes_grid[-1].set_xlabel(f"share of {unit}", color=MUTED, fontsize=9)
    for axes in axes_grid:
        axes.grid(axis="x", color=GRID, linewidth=0.6)
        axes.set_axisbelow(True)
    handles = [Patch(color=color_by_category[category], label=LABEL_BY_CATEGORY.get(category, category))
               for category in categories]
    figure.legend(handles=handles, loc="upper center", ncol=min(len(categories), 3), frameon=False, fontsize=9,
                  bbox_to_anchor=(0.5, 0.965))
    figure.suptitle(TITLE_BY_FIELD[field], x=0.02, y=0.995, ha="left", fontsize=11, color=INK)
    episodes = "Arms of a model have the same number of episodes. " if absolute else ""
    figure.text(0.02, 0.005, f"Main dose. {episodes}Segments under {MIN_LABELED_SHARE:.0%} of the "
                f"{'model scale' if absolute else 'bar'} are unlabelled."
                + (f"\nLeft out (an arm with none): {', '.join(left_out)}." if left_out else ""),
                fontsize=8, color=MUTED)
    figure.tight_layout(rect=(0, 0.03, 0.97, 0.915))
    stem = out.parent / f"{out.name}{'_count' if absolute else ''}_{field}"
    figure.savefig(stem.with_suffix(".png"), dpi=200)
    figure.savefig(stem.with_suffix(".pdf"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--timing-json", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    summary_by_model: dict[str, dict] = json.loads(args.timing_json.read_text())
    for field in TITLE_BY_FIELD:
        for absolute in (False, True):
            save_field_chart(summary_by_model, field, absolute, args.out)


if __name__ == "__main__":
    main()
