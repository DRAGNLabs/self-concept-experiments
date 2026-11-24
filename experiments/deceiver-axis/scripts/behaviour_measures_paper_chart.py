"""Paper figure: one row of four behaviour-measure panels (honest abstention, incapability admission, ABSTAIN flag on the
unsteered transcripts, ABSTAIN-flag tokens on each arm's own transcripts), the same layout as behaviour_measures_chart.py
(x = change under the Reward Hacking Axis, y = change under the Deceiver Axis, negative steering at each model's main
dose) with the paper's names, no panel explanations (they go in the caption), the unsteered rates under each title and
panel letters. The five visible-broken problems (VISIBLE_BROKEN_PROBLEM_IDS) are left out; --flag-turns must hold
abstain_flag_turns.py output run with the same exclusion. Writes <out>.{png,pdf,json}.

    python experiments/deceiver-axis/scripts/behaviour_measures_paper_chart.py --out <dir>/behaviour_measures_paper \\
        --flag-turns ~/nobackup/autodelete/self-concept-experiments/schemer-axis/abstain_flag/turns_excl_visible_broken
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from behaviour_measures_chart import (ChangesByModel, GapByModel, Proxy, draw_gap_marker, flag_changes,
                                      model_changes)
from selfconcept.deceiver_axis.behaviour_measures.episodes import PRESETS, VISIBLE_BROKEN_PROBLEM_IDS, ModelPreset
from selfconcept.deceiver_axis.behaviour_measures.palette import COLOR_BY_MODEL, GRID, INK, MARKER_BY_MODEL, MUTED

PANEL_PROXIES: list[Proxy] = ["abstention", "incapability", "flag_on_unsteered", "flag_own"]
TITLE_BY_PROXY: dict[Proxy, str] = {"abstention": "Honest abstention", "incapability": "Incapability admission",
                                    "flag_on_unsteered": "ABSTAIN flag,\nunsteered transcripts",
                                    "flag_own": "ABSTAIN-flag tokens,\nown transcripts"}
PER_10K_TOKEN_PROXIES: set[Proxy] = {"flag_own"}
PAPER_NAME_BY_MODEL = {"gpt-oss-120b": "gpt-oss-120b", "Muse": "Muse-Glimmer-30B", "Gemma 4 31B": "Gemma 4 31B IT",
                       "Qwen3.8-27B": "Qwen3.8-27B"}
MARKER_GLYPH_BY_MODEL = {"gpt-oss-120b": "●", "Muse": "■", "Gemma 4 31B": "▲", "Qwen3.8-27B": "◆"}
DEFAULT_FLAG_TURNS = (Path.home() / "nobackup/autodelete/self-concept-experiments/schemer-axis/abstain_flag"
                      / "turns_excl_visible_broken")
TITLE_SIZE, SUBTITLE_SIZE, LABEL_SIZE, TICK_SIZE, LETTER_SIZE = 17, 15, 16, 14, 20
MARKER_SIZE, BAR_WIDTH = 13, 3.5


def unsteered_text(proxy: Proxy, changes_by_model: ChangesByModel, model: str) -> str:
    if proxy not in changes_by_model[model]:
        return "n/a"
    rate = changes_by_model[model][proxy][("gi_v2", "main")]["unsteered_rate"]
    return f"{rate:.0f}" if proxy in PER_10K_TOKEN_PROXIES else f"{100 * rate:.1f}%"


def draw_unsteered_rates(axes: Axes, proxy: Proxy, changes_by_model: ChangesByModel) -> None:
    """Under the title: a heading, then a 2x2 grid of each model's marker glyph (in its colour) and unsteered rate."""
    heading = "Unsteered; per 10k tokens" if proxy in PER_10K_TOKEN_PROXIES else "Unsteered; Δ in pp"
    axes.annotate(heading, (0, 1), xycoords="axes fraction", xytext=(0, 46), textcoords="offset points",
                  fontsize=SUBTITLE_SIZE, color=MUTED, ha="left", va="bottom")
    for index, model in enumerate(changes_by_model):
        row, column = divmod(index, 2)
        axes.annotate(f"{MARKER_GLYPH_BY_MODEL[model]} {unsteered_text(proxy, changes_by_model, model)}",
                      (0.5 * column, 1), xycoords="axes fraction", xytext=(0, 26 - 20 * row),
                      textcoords="offset points", fontsize=SUBTITLE_SIZE, color=COLOR_BY_MODEL[model], ha="left",
                      va="bottom")


def draw_panel(axes: Axes, letter: str, proxy: Proxy, changes_by_model: ChangesByModel,
               gap_by_model: GapByModel) -> None:
    extent = 1.0
    for model, changes_by_proxy in changes_by_model.items():
        if proxy in changes_by_proxy:
            changes = changes_by_proxy[proxy]
            bar_coordinates = draw_gap_marker(axes, model, changes[("gfar", "main")], changes[("gi_v2", "main")],
                                              gap_by_model[model][proxy], MARKER_SIZE, BAR_WIDTH)
            extent = max(extent, *map(abs, bar_coordinates))
    limit = extent * 1.1
    axes.plot([-limit, limit], [-limit, limit], linestyle=":", color=MUTED, linewidth=1.5, zorder=1)
    axes.axhline(0, color=GRID, linewidth=1.2, zorder=0)
    axes.axvline(0, color=GRID, linewidth=1.2, zorder=0)
    axes.plot(0, 0, marker="o", markersize=6, color=INK, zorder=5)
    axes.set_xlim(-limit, limit)
    axes.set_ylim(-limit, limit)
    axes.set_aspect("equal")
    axes.tick_params(colors=MUTED, labelsize=TICK_SIZE)
    for axis in (axes.xaxis, axes.yaxis):
        axis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 5, 10]))
    for spine in axes.spines.values():
        spine.set_color(GRID)
    title = axes.set_title(TITLE_BY_PROXY[proxy], fontsize=TITLE_SIZE, color=INK, loc="left", pad=70)
    axes.annotate(letter, (0, 1), xycoords=title, xytext=(-12, 0), textcoords="offset points", fontsize=LETTER_SIZE,
                  fontweight="bold", color=INK, ha="right", va="top")
    draw_unsteered_rates(axes, proxy, changes_by_model)
    draw_better_arrow(axes)


def draw_better_arrow(axes: Axes) -> None:
    """An arrow toward the upper left (the Deceiver Axis changes the measure more), from the empty lower-right corner."""
    tail, head = (0.95, 0.29), (0.79, 0.45)
    axes.annotate("", xy=head, xycoords="axes fraction", xytext=tail, textcoords="axes fraction",
                  arrowprops={"arrowstyle": "-|>", "color": MUTED, "linewidth": 2, "mutation_scale": 22})
    axes.text(0.97, 0.04, "Deceiver Axis\nbetter", transform=axes.transAxes, fontsize=SUBTITLE_SIZE - 1, color=MUTED,
              ha="right", va="bottom", linespacing=1.1)


def legend_handles() -> list[Line2D]:
    model_handles = [Line2D([], [], color=COLOR_BY_MODEL[preset["label"]], marker=MARKER_BY_MODEL[preset["label"]],
                            linestyle="none", markersize=MARKER_SIZE - 2,
                            label=f"{PAPER_NAME_BY_MODEL[preset['label']]} (α = −{preset['alpha_by_dose']['main']:g})")
                     for preset in PRESETS]
    return model_handles + [
        Line2D([], [], color=MUTED, marker="o", linestyle="none", markersize=MARKER_SIZE - 2, markerfacecolor=MUTED,
               label="filled: axes differ (95% CI)"),
        Line2D([], [], color=MUTED, linestyle=":", linewidth=1.5, label="equal effect")]


def save_figure(changes_by_model: ChangesByModel, gap_by_model: GapByModel, out: Path) -> None:
    figure, axes_row = plt.subplots(1, len(PANEL_PROXIES), figsize=(14, 5.6))
    figure.subplots_adjust(left=0.075, right=0.965, bottom=0.27, top=0.73, wspace=0.32)
    for letter, axes, proxy in zip("abcd", axes_row, PANEL_PROXIES):
        draw_panel(axes, letter, proxy, changes_by_model, gap_by_model)
    figure.supxlabel("Change when steering against the Reward Hacking Axis", fontsize=LABEL_SIZE, color=INK, y=0.155)
    figure.supylabel("Change when steering\nagainst the Deceiver Axis", fontsize=LABEL_SIZE, color=INK, x=0.005,
                     y=0.49)
    figure.legend(handles=legend_handles(), loc="lower center", ncol=3, frameon=False, fontsize=SUBTITLE_SIZE,
                  handletextpad=0.2, columnspacing=1.5)
    figure.savefig(out.with_suffix(".png"), dpi=200)
    figure.savefig(out.with_suffix(".pdf"))
    plt.close(figure)


def with_honest_abstention(preset: ModelPreset) -> ModelPreset:
    """Honest abstention for every model: one that never stops on an untruncated reply is plotted at 0%."""
    return {**preset, "exclusion_reason_by_proxy": {proxy: reason for proxy, reason
                                                    in preset["exclusion_reason_by_proxy"].items()
                                                    if proxy != "abstention"}}


def all_changes(n_draws: int, seed: int, flag_turns: Path) -> tuple[ChangesByModel, GapByModel]:
    rng = np.random.default_rng(seed)
    changes_by_model: ChangesByModel = {}
    gap_by_model: GapByModel = {}
    for preset in PRESETS:
        model = preset["label"]
        changes_by_model[model], gap_by_model[model] = model_changes(with_honest_abstention(preset), n_draws, rng,
                                                                     VISIBLE_BROKEN_PROBLEM_IDS)
        for proxy, (changes, gap) in flag_changes(flag_turns / f"{preset['slug']}_turns.json").items():
            changes_by_model[model][proxy], gap_by_model[model][proxy] = changes, gap
    return changes_by_model, gap_by_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True, help="output stem: <out>.{png,pdf,json}")
    parser.add_argument("--n-draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--flag-turns", type=Path, default=DEFAULT_FLAG_TURNS,
                        help="directory with abstain_flag_turns.py output <slug>_turns.json per model, run with "
                             "--exclude-problems set to the five visible-broken problems")
    args = parser.parse_args()
    changes_by_model, gap_by_model = all_changes(args.n_draws, args.seed, args.flag_turns)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_figure(changes_by_model, gap_by_model, args.out)
    args.out.with_suffix(".json").write_text(json.dumps(
        {"excluded_problems": sorted(VISIBLE_BROKEN_PROBLEM_IDS),
         **{PAPER_NAME_BY_MODEL[model]: {
             proxy: {"alpha": next(preset["alpha_by_dose"]["main"] for preset in PRESETS if preset["label"] == model),
                     "deceiver_axis": changes_by_proxy[proxy][("gi_v2", "main")],
                     "reward_hacking_axis": changes_by_proxy[proxy][("gfar", "main")],
                     "deceiver_minus_reward_hacking": gap_by_model[model][proxy]}
             for proxy in PANEL_PROXIES if proxy in changes_by_proxy}
            for model, changes_by_proxy in changes_by_model.items()}}, indent=1))


if __name__ == "__main__":
    main()
