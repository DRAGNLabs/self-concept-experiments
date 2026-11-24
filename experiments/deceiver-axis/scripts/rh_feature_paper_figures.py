"""Paper figures for the reward-hacking-feature subsections, drawn from existing analysis outputs (CPU, no recomputation).

separation: within-problem (stratified) AUROC of hack vs non-hack unsteered episodes for the Deceiver Axis (gi_v2) and the
    Reward Hacking Axis (gfar), reasoning and answer tokens, with the 100 antonym control directions' 97.5th
    percentile as a dotted tick (hack_separation.py outputs on the 41 / 103-problem sets with lcbhard_61 / 67 re-scored).
geometry: markdown table of cos(gi_v2, gfar) and episode-level r of their projections, each with a p value against the 100
    antonym control directions (gfar_gi_v2_control_null.py outputs).
lens: median shift of the per-token gfar projection under gi_v2 steering on the control runs' tokens, over the unsteered
    p10-p90 width, by layer; dotted lines at the 2.5th and 97.5th percentiles of the antonym controls' median shifts; shaded layers where gfar's within-problem
    AUROC is above the antonym controls' 97.5th percentile (steer_gfar_lens_chart.py and gfar_layer_separation.py outputs, with the five
    wrong-shown-test problems left out and the re-scored verdicts).

    python experiments/deceiver-axis/scripts/rh_feature_paper_figures.py --out-dir ~/nobackup/autodelete/self-concept-experiments/figures
"""
import argparse
import json
from pathlib import Path
from typing import Literal, TypedDict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.colors import to_rgb
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import numpy as np
from jaxtyping import Bool, Float

import pareto_chart as pc

Direction = Literal["gi_v2", "gfar"]
Segment = Literal["reasoning", "answer"]
DIRECTIONS: tuple[Direction, ...] = ("gi_v2", "gfar")
SEGMENTS: tuple[Segment, ...] = ("reasoning", "answer")

SCRATCH = Path.home() / "nobackup/autodelete/self-concept-experiments"
SEPARATION_DIR = Path("/grphome/grp_mirror/nobackup/autodelete/self-concept-experiments/schemer-axis/gfar_vs_gi_v2_jansen05/hack_separation_2026-11-23")
CONTROL_NULL_DIR = SCRATCH / "gfar_gi_v2_control_null"
LENS_CHART_DIR = SCRATCH / "schemer-axis/steer_gfar_lens/charts_2026-11-24"


class Model(TypedDict):
    key: str
    name: str


MODELS: tuple[Model, ...] = (
    {"key": "gptoss120b", "name": "gpt-oss-120b"},
    {"key": "muse", "name": "Muse-Glimmer-30B"},
    {"key": "gemma4_31b", "name": "Gemma 4 31B IT"},
    {"key": "qwen38_27b", "name": "Qwen3.8-27B"},
)
AXIS_NAMES: dict[Direction, str] = {"gi_v2": "The Deceiver Axis", "gfar": "The Reward Hacking Axis"}
SEGMENT_NAMES: dict[Segment, str] = {"reasoning": "reasoning tokens", "answer": "final answer tokens"}
COLORS_BY_DIRECTION: dict[Direction, str] = {"gi_v2": pc.GROUP_COLORS[0], "gfar": pc.GROUP_COLORS[1]}
REASONING_TINT = 0.5
LENS_NULL_PERCENTILES = (2.5, 97.5)
SEPARATING_SHADE = "#e4e2d9"


def tint(color: str, amount: float) -> tuple[float, float, float]:
    """Mix `color` toward the chart surface by `amount` (0 = unchanged, 1 = surface)."""
    (red, green, blue), (surface_red, surface_green, surface_blue) = to_rgb(color), to_rgb(pc.SURFACE)
    return (red + (surface_red - red) * amount, green + (surface_green - green) * amount, blue + (surface_blue - blue) * amount)


def bar_color(direction: Direction, segment: Segment) -> tuple[float, float, float]:
    return tint(COLORS_BY_DIRECTION[direction], REASONING_TINT if segment == "reasoning" else 0.0)


def style_axes(axes: Axes) -> None:
    axes.set_facecolor(pc.SURFACE)
    axes.spines[["top", "right"]].set_visible(False)
    axes.spines[["left", "bottom"]].set_color(pc.AXIS)
    axes.tick_params(colors=pc.INK2, labelsize=9.5)


class SeparationBar(TypedDict):
    model: str
    direction: Direction
    segment: Segment
    auroc: float
    control_p97_5: float
    hacks: int
    non_hacks: int


def load_separation_bars(model: Model) -> list[SeparationBar]:
    record = json.loads((SEPARATION_DIR / f"{model['key']}_rescored.json").read_text())
    rows_by_key = {(row["direction"], row["segment"]): row for row in record["rows"] if row["borderline"] == "counted_not_hack"}
    return [{"model": model["name"], "direction": direction, "segment": segment,
             "auroc": rows_by_key[direction, segment]["within_problem"]["auroc"],
             "control_p97_5": rows_by_key[direction, segment]["within_problem"]["control"]["p97_5"],
             "hacks": rows_by_key[direction, segment]["hacks"], "non_hacks": rows_by_key[direction, segment]["non_hacks"]}
            for segment in SEGMENTS for direction in DIRECTIONS]


def draw_separation(bars_by_model: list[list[SeparationBar]], out_stem: Path) -> None:
    bar_height, segment_gap, block_gap = 0.8, 0.35, 1.0
    figure, axes = plt.subplots(figsize=(7.5, 0.42 * sum(map(len, bars_by_model)) + 1.6), facecolor=pc.SURFACE)
    style_axes(axes)
    tick_positions: list[float] = []
    tick_labels: list[str] = []
    cursor = 0.0
    for model_bars in bars_by_model:
        positions_by_segment: dict[Segment, list[float]] = {segment: [] for segment in SEGMENTS}
        for bar in model_bars:
            position = cursor
            positions_by_segment[bar["segment"]].append(position)
            axes.barh(position, bar["auroc"], height=bar_height, color=bar_color(bar["direction"], bar["segment"]),
                      edgecolor=pc.SURFACE, linewidth=1.5, zorder=2)
            axes.text(0.012, position, f"{bar['auroc']:.2f}", va="center", ha="left", fontsize=8.5,
                      color=pc.INK if bar["segment"] == "reasoning" else pc.SURFACE, zorder=4)
            cursor += 1 + (segment_gap if bar is model_bars[len(DIRECTIONS) - 1] else 0)
        for segment, positions in positions_by_segment.items():
            control_p97_5_values = {bar["control_p97_5"] for bar in model_bars if bar["segment"] == segment}
            assert len(control_p97_5_values) == 1, "control directions are shared across directions within a segment"
            axes.plot([control_p97_5_values.pop()] * 2, [positions[0] - bar_height / 2, positions[-1] + bar_height / 2], color=pc.INK,
                      linestyle=(0, (1, 1.2)), linewidth=1.6, zorder=3)
        all_positions = [position for positions in positions_by_segment.values() for position in positions]
        tick_positions.append(float(np.mean(all_positions)))
        tick_labels.append(model_bars[0]["model"])
        cursor += block_gap
    axes.axvline(0.5, color=pc.ZERO, linewidth=0.9, zorder=1)
    axes.set_yticks(tick_positions, tick_labels, fontsize=10.5, color=pc.INK)
    axes.tick_params(axis="y", length=0)
    axes.invert_yaxis()
    axes.set_xlim(0, 1)
    axes.set_xlabel("stratified AUROC, hack vs non-hack episodes", fontsize=10, color=pc.INK2)
    axes.grid(axis="x", color=pc.GRID, linewidth=0.6, zorder=0)
    patches_by_direction = {direction: [Patch(color=bar_color(direction, segment), label=f"{AXIS_NAMES[direction]}, {SEGMENT_NAMES[segment]}")
                                        for segment in SEGMENTS] for direction in DIRECTIONS}
    control_tick = Line2D([], [], color=pc.INK, linestyle=(0, (1, 1.2)), linewidth=1.6, label="antonym control directions, 97.5th percentile")
    handles: list[Artist] = [*patches_by_direction["gi_v2"], control_tick, *patches_by_direction["gfar"]]
    axes.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.45, -0.1), ncol=2, frameon=False, fontsize=9)
    axes.set_title("Separating reward hacks from non-hacks", fontsize=12.5, color=pc.INK, loc="left", pad=10)
    figure.tight_layout()
    save(figure, out_stem, {"bars": [bar for model_bars in bars_by_model for bar in model_bars]})


class GeometryRow(TypedDict):
    model: str
    layer: int
    cosine: float
    cosine_p_against_deceiver_controls: float
    cosine_p_against_rh_feature_controls: float
    r_reasoning: float
    r_reasoning_p: float
    r_answer: float
    r_answer_p: float
    controls: int


def p_from_share_smaller(share_smaller: float, controls: int) -> float:
    """One-sided permutation p, (1 + controls at or above) / (1 + controls), from the share of controls strictly below."""
    at_or_above = round((1 - share_smaller) * controls)
    return (1 + at_or_above) / (1 + controls)


def load_geometry_row(model: Model) -> GeometryRow:
    record = json.loads((CONTROL_NULL_DIR / f"{model['key']}_antonym100.json").read_text())
    controls = len(record["control_files"])
    episode_by_segment = {segment: record["correlation_by_segment"][segment]["episode"] for segment in SEGMENTS}
    return {"model": model["name"], "layer": record["layer"], "controls": controls,
            "cosine": record["cosine"]["value"],
            "cosine_p_against_deceiver_controls": p_from_share_smaller(record["cosine"]["percentile_among_gi_v2_controls"], controls),
            "cosine_p_against_rh_feature_controls": p_from_share_smaller(record["cosine"]["percentile_among_gfar_controls"], controls),
            "r_reasoning": episode_by_segment["reasoning"]["value"],
            "r_reasoning_p": p_from_share_smaller(episode_by_segment["reasoning"]["percentile_among_gi_v2_controls"], controls),
            "r_answer": episode_by_segment["answer"]["value"],
            "r_answer_p": p_from_share_smaller(episode_by_segment["answer"]["percentile_among_gi_v2_controls"], controls)}


def geometry_markdown(rows: list[GeometryRow]) -> str:
    header = ("| Model (layer) | cosine | p (vs Deceiver Axis–antonym) | p (vs Reward Hacking Axis–antonym) "
              "| r, reasoning | p | r, final answer | p |\n|---|---|---|---|---|---|---|---|\n")
    return header + "".join(
        f"| {row['model']} (L{row['layer']}) | {row['cosine']:+.3f} | {row['cosine_p_against_deceiver_controls']:.2f} "
        f"| {row['cosine_p_against_rh_feature_controls']:.2f} | {row['r_reasoning']:+.2f} | {row['r_reasoning_p']:.2f} "
        f"| {row['r_answer']:+.2f} | {row['r_answer_p']:.2f} |\n" for row in rows)


class LensPanel(TypedDict):
    model: str
    steer_layer: int
    alpha: float
    layers: list[int]
    median_shift: list[float]
    null_low: list[float]
    null_high: list[float]
    separating: list[bool]


def separating_layers(saved: dict, layer_count: int) -> Bool[np.ndarray, "layers"]:
    separating = np.zeros(layer_count, dtype=bool)
    for entry in (saved.get("gfar_separation") or {}).get("layers", []):
        separating[entry["layer"]] = entry["auroc"] > entry["control_p97_5"]
    return separating


def load_lens_panel(model: Model) -> LensPanel:
    saved = json.loads((LENS_CHART_DIR / f"{model['key']}.json").read_text())
    median_shift: Float[np.ndarray, "layers"] = np.asarray(saved["null"]["gi_v2_normalized"])
    null_low, null_high = np.percentile(np.asarray(saved["null"]["normalized"]), LENS_NULL_PERCENTILES, axis=0)
    steer_layer = saved["steer_layer"]
    layers = np.arange(steer_layer, len(median_shift))
    return {"model": model["name"], "steer_layer": steer_layer, "alpha": saved["alpha"], "layers": layers.tolist(),
            "median_shift": median_shift[layers].tolist(), "null_low": null_low[layers].tolist(),
            "null_high": null_high[layers].tolist(), "separating": separating_layers(saved, len(median_shift))[layers].tolist()}


def draw_lens_panel(axes: Axes, panel: LensPanel) -> None:
    style_axes(axes)
    for layer, separating in zip(panel["layers"], panel["separating"]):
        if separating:
            axes.axvspan(layer - 0.5, layer + 0.5, color=SEPARATING_SHADE, linewidth=0, zorder=0)
    axes.axhline(0, color=pc.ZERO, linewidth=0.9, zorder=1)
    for bound in ("null_low", "null_high"):
        axes.plot(panel["layers"], panel[bound], color=pc.INK2, linestyle=(0, (1, 1.5)), linewidth=1.4, zorder=2)
    axes.plot(panel["layers"], panel["median_shift"], color=COLORS_BY_DIRECTION["gi_v2"], linewidth=2, zorder=3)
    axes.set_xlim(panel["layers"][0] - 0.5, panel["layers"][-1] + 0.5)
    axes.xaxis.set_major_locator(MaxNLocator(integer=True))
    axes.grid(axis="y", color=pc.GRID, linewidth=0.6, zorder=0)
    axes.set_title(f"{panel['model']} (α = +{panel['alpha']:g} at layer {panel['steer_layer']})", fontsize=10.5, color=pc.INK, loc="left")


def draw_lens(panels: list[LensPanel], out_stem: Path) -> None:
    figure, axes_grid = plt.subplots(2, 2, figsize=(10, 6.6), facecolor=pc.SURFACE)
    for axes, panel in zip(axes_grid.flat, panels):
        draw_lens_panel(axes, panel)
    for axes in axes_grid[1]:
        axes.set_xlabel("layer", fontsize=10, color=pc.INK2)
    for axes in axes_grid[:, 0]:
        axes.set_ylabel("Δ median Reward Hacking Axis projection\n(÷ unsteered p10–p90 width)", fontsize=9.5, color=pc.INK2)
    handles: list[Artist] = [Line2D([], [], color=COLORS_BY_DIRECTION["gi_v2"], linewidth=2, label="Deceiver Axis steering, median shift"),
               Line2D([], [], color=pc.INK2, linestyle=(0, (1, 1.5)), linewidth=1.4,
                      label=f"antonym control directions, {LENS_NULL_PERCENTILES[0]:g}–{LENS_NULL_PERCENTILES[1]:g}% range"),
               Patch(color=SEPARATING_SHADE, label="Reward Hacking Axis separates hacks (above antonym-control 97.5th percentile)")]
    figure.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.06), ncol=2, frameon=False, fontsize=9)
    figure.suptitle("Effect of Deceiver Axis steering on the Reward Hacking Axis", fontsize=12.5,
                    color=pc.INK, x=0.01, ha="left")
    figure.tight_layout()
    save(figure, out_stem, {"panels": panels})


def save(figure: Figure, out_stem: Path, record: dict) -> None:
    for suffix in (".png", ".pdf"):
        figure.savefig(out_stem.with_suffix(suffix), dpi=200, bbox_inches="tight", facecolor=pc.SURFACE)
    plt.close(figure)
    out_stem.with_suffix(".json").write_text(json.dumps(record, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    models_with_unsteered_hacks = [model for model in MODELS if (SEPARATION_DIR / f"{model['key']}_rescored.json").exists()]
    draw_separation([load_separation_bars(model) for model in models_with_unsteered_hacks], args.out_dir / "rh_feature_separation")
    geometry = geometry_markdown([load_geometry_row(model) for model in MODELS])
    (args.out_dir / "rh_feature_geometry.md").write_text(geometry)
    print(geometry)
    draw_lens([load_lens_panel(model) for model in MODELS], args.out_dir / "rh_feature_lens")


if __name__ == "__main__":
    main()
