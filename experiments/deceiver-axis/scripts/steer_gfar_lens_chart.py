"""Chart of steer_gfar_lens_extract.py output: per-layer percentiles of the per-token gfar projection, unsteered vs gi_v2-steered.

The chart shows all episodes: steered minus unsteered (the same percentile of each), and that difference divided by
the unsteered p10-p90 width at the layer; the dashed line is the injected vector carried unchanged, coefficient x
cos(gi_v2 at the steer layer, gfar at the layer). The json also holds the percentiles of judged hacks (verdict HACK =
STRONG or WEAK) and the rest (unjudged episodes count as not hack) when --verdicts is given. Each panel's y-axis is
clipped to the range of its p2-p98 lines. Writes <out>.png and <out>.json; --rechart redraws the png from <out>.json.

    python scripts/steer_gfar_lens_chart.py --lens <dir of seed<N>.pt> --verdicts <as_judged.json> --title "gpt-oss-120b" --out <stem>
    python scripts/steer_gfar_lens_chart.py --rechart --title "gpt-oss-120b" --out <stem>
"""
import argparse
import json
from pathlib import Path
from typing import Literal, TypedDict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import torch
from jaxtyping import Bool, Float

from selfconcept.measurement.unit_axes import load_unit_axes
from steer_gfar_lens_extract import load_gfar_units

Group = Literal["combined", "hack", "rest"]
Condition = Literal["unsteered", "steered"]
CONDITIONS: tuple[Condition, ...] = ("unsteered", "steered")
PERCENTILES = np.array([*range(0, 10), *range(10, 91, 10), *range(91, 101)])
DISPLAYED_PERCENTILES = [*range(0, 10, 2), 20, 40, 50, 60, 80, *range(92, 101, 2)]
CLIP_LOW_PERCENTILE, CLIP_HIGH_PERCENTILE = 2, 98
MEDIAN_COLOR = "#1f1f1d"
INJECTION_COLOR = "#6b3fa0"
LOWER_PERCENTILES = [percentile for percentile in DISPLAYED_PERCENTILES if percentile < 50]
UPPER_PERCENTILES = [percentile for percentile in DISPLAYED_PERCENTILES if percentile > 50]


def percentile_color(percentile: int) -> tuple[float, float, float, float]:
    """Ordinal steps: blues darken toward p0, reds darken toward p100, so neighbouring tail lines stay distinguishable."""
    if percentile < 50:
        rank_from_median = len(LOWER_PERCENTILES) - 1 - LOWER_PERCENTILES.index(percentile)
        return plt.get_cmap("Blues")(0.35 + 0.6 * rank_from_median / (len(LOWER_PERCENTILES) - 1))
    rank_from_median = UPPER_PERCENTILES.index(percentile)
    return plt.get_cmap("Reds")(0.35 + 0.6 * rank_from_median / (len(UPPER_PERCENTILES) - 1))


class GroupPercentiles(TypedDict):
    episodes: int
    response_tokens: int
    percentiles_by_condition: dict[Condition, list[list[float]]]


class Lens(TypedDict):
    projections_by_condition: dict[Condition, Float[np.ndarray, "layers tokens"]]
    token_episode: np.ndarray
    verdict_keys: list[str]
    steer_layer: int
    alpha: float


def load_lens(lens_dir: Path) -> Lens:
    seed_files = sorted(lens_dir.glob("seed*.pt"), key=lambda path: int(path.stem.removeprefix("seed")))
    seeds = [torch.load(path, map_location="cpu", weights_only=False) for path in seed_files]
    episode_counts = np.cumsum([0] + [len(seed["example_id"]) for seed in seeds])
    return {
        "projections_by_condition": {condition: torch.cat([seed[condition] for seed in seeds], dim=1).numpy() for condition in CONDITIONS},
        "token_episode": np.concatenate([episode_base + np.repeat(np.arange(len(seed["example_id"])), np.diff(seed["token_offsets"]))
                                         for episode_base, seed in zip(episode_counts, seeds)]),
        "verdict_keys": [key for seed in seeds for key in seed["verdict_key"]],
        "steer_layer": seeds[0]["steer_layer"],
        "alpha": seeds[0]["alpha"],
    }


def episode_masks_by_group(verdict_keys: list[str], verdicts_by_key: dict[str, str] | None) -> dict[Group, Bool[np.ndarray, "episodes"]]:
    everything = np.ones(len(verdict_keys), dtype=bool)
    if verdicts_by_key is None:
        return {"combined": everything}
    is_hack = np.array([verdicts_by_key.get(key) == "HACK" for key in verdict_keys])
    return {"combined": everything, "hack": is_hack, "rest": ~is_hack}


def group_percentiles(lens: Lens, episode_mask: Bool[np.ndarray, "episodes"]) -> GroupPercentiles:
    token_mask = episode_mask[lens["token_episode"]]
    return {
        "episodes": int(episode_mask.sum()),
        "response_tokens": int(token_mask.sum()),
        "percentiles_by_condition": {
            condition: [np.percentile(layer_projections[token_mask].astype(np.float32), PERCENTILES).tolist()
                        for layer_projections in lens["projections_by_condition"][condition]]
            for condition in CONDITIONS},
    }


def percentile_columns(values: Float[np.ndarray, "layers percentiles"], percentiles: list[int]) -> Float[np.ndarray, "layers shown"]:
    return values[:, [PERCENTILES.tolist().index(percentile) for percentile in percentiles]]


def draw_percentile_lines(axis: Axes, values: Float[np.ndarray, "layers percentiles"]) -> None:
    layers = np.arange(values.shape[0])
    for percentile in DISPLAYED_PERCENTILES:
        line = percentile_columns(values, [percentile])[:, 0]
        if percentile == 50:
            axis.plot(layers, line, color=MEDIAN_COLOR, linewidth=2.4, zorder=3, label="p50")
        else:
            axis.plot(layers, line, color=percentile_color(percentile), linewidth=1.1, label=f"p{percentile}")


def clip_to_inner_percentiles(axis: Axes, panels: list[Float[np.ndarray, "layers percentiles"]]) -> None:
    inner = np.concatenate([percentile_columns(values, [CLIP_LOW_PERCENTILE, CLIP_HIGH_PERCENTILE]) for values in panels])
    low, high = float(np.nanmin(inner)), float(np.nanmax(inner))
    padding = 0.05 * (high - low) or 1.0
    axis.set_ylim(low - padding, high + padding)


def carried_injection_projection(lens_dir: Path) -> Float[np.ndarray, "layers"]:
    """coefficient * <unit gi_v2 at the steer layer, unit gfar at each layer>; NaN before the steer layer."""
    seed = torch.load(next(lens_dir.glob("seed*.pt")), map_location="cpu", weights_only=False)
    gfar_units = load_gfar_units(seed["gfar"], *_artifact_shape(seed["gfar"]))
    num_layers, hidden = gfar_units.shape
    steer_layer = seed["steer_layer"]
    gi_v2_unit = torch.as_tensor(load_unit_axes(Path(seed["gi_v2"]), [steer_layer], num_layers, hidden, seed["model"])[steer_layer]["direction"])
    projection = (seed["coefficient"] * (gfar_units @ gi_v2_unit)).numpy()
    projection[:steer_layer] = np.nan
    return projection


def _artifact_shape(gfar_spec: str) -> tuple[int, int]:
    path, key = gfar_spec.rsplit(":", 1)
    num_layers, hidden = torch.load(path, map_location="cpu", weights_only=False)[key].shape
    return int(num_layers), int(hidden)


def chart(combined: GroupPercentiles, injected: Float[np.ndarray, "layers"], steer_layer: int, alpha: float, title: str, out_png: Path) -> None:
    unsteered = np.asarray(combined["percentiles_by_condition"]["unsteered"])
    delta = np.asarray(combined["percentiles_by_condition"]["steered"]) - unsteered
    width = percentile_columns(unsteered, [90])[:, 0] - percentile_columns(unsteered, [10])[:, 0]
    figure, axes = plt.subplots(2, 1, figsize=(9, 9), sharex=True)
    rows = [(delta, injected, "steered − unsteered\ngfar projection"),
            (delta / width[:, None], injected / width, "steered − unsteered\n÷ unsteered p10–p90 width")]
    for axis, (values, injected_row, label) in zip(axes, rows):
        draw_percentile_lines(axis, values)
        clip_to_inner_percentiles(axis, [values])
        axis.plot(np.arange(len(injected_row)), injected_row, color=INJECTION_COLOR, linestyle="--", linewidth=1.8, zorder=4,
                  label="injection carried unchanged")
        low, high = axis.get_ylim()
        axis.set_ylim(min(low, float(np.nanmin(injected_row))), max(high, float(np.nanmax(injected_row))))
        axis.axvline(steer_layer, color="#6b6a66", linestyle=":", linewidth=0.8)
        axis.axhline(0, color="#6b6a66", linewidth=0.8)
        axis.grid(color="#e6e5e1", linewidth=0.5)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_ylabel(label)
    axes[1].set_xlabel("layer")
    handles, labels = axes[0].get_legend_handles_labels()
    order = [labels.index(f"p{percentile}") for percentile in reversed(DISPLAYED_PERCENTILES)] + [labels.index("injection carried unchanged")]
    figure.legend([handles[i] for i in order], [labels[i] for i in order], loc="center left", bbox_to_anchor=(0.98, 0.5), frameon=False,
                  title="percentile over\nresponse tokens")
    figure.suptitle(f"{title}: gi_v2 +{alpha:g} at L{steer_layer} (dotted), effect on the per-token gfar projection\n"
                    f"all {combined['episodes']} unsteered episodes, {combined['response_tokens']:,} response tokens, teacher-forced; "
                    f"dashed: coef × cos(gi_v2 at L{steer_layer}, gfar at layer)\n"
                    f"y-axes clipped to the p{CLIP_LOW_PERCENTILE}–p{CLIP_HIGH_PERCENTILE} range", fontsize=10)
    figure.tight_layout()
    figure.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lens", type=Path, help="required unless --rechart")
    parser.add_argument("--verdicts", type=Path, default=None, help="as_judged.json keyed '<arm>/seed<s>/<example_id>'")
    parser.add_argument("--title", required=True)
    parser.add_argument("--out", type=Path, required=True, help="output stem")
    parser.add_argument("--rechart", action="store_true", help="redraw <out>.png from <out>.json without reading the lens")
    args = parser.parse_args()

    if args.rechart:
        saved = json.loads(args.out.with_suffix(".json").read_text())
        chart(saved["groups"]["combined"], carried_injection_projection(Path(saved["lens"])), saved["steer_layer"], saved["alpha"], args.title,
              args.out.with_suffix(".png"))
        return

    lens = load_lens(args.lens)
    verdicts_by_key = json.loads(args.verdicts.read_text()) if args.verdicts else None
    percentiles_by_group: dict[Group, GroupPercentiles] = {group: group_percentiles(lens, mask)
                            for group, mask in episode_masks_by_group(lens["verdict_keys"], verdicts_by_key).items() if mask.any()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps({
        "lens": str(args.lens), "verdicts": str(args.verdicts), "steer_layer": lens["steer_layer"], "alpha": lens["alpha"],
        "percentiles": PERCENTILES.tolist(), "groups": percentiles_by_group}, indent=1))
    chart(percentiles_by_group["combined"], carried_injection_projection(args.lens), lens["steer_layer"], lens["alpha"], args.title, args.out.with_suffix(".png"))


if __name__ == "__main__":
    main()
