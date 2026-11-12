"""Chart of steer_gfar_lens_extract.py output: per-layer percentiles of the per-token gfar projection, unsteered vs gi_v2-steered.

Rows: unsteered, steered, steered minus unsteered (the same percentile of each), and that difference divided by the
unsteered p10-p90 width at the layer. Columns: all episodes, judged hacks (verdict HACK = STRONG or WEAK) and the rest
(unjudged episodes count as not hack); without --verdicts only the combined column. Each panel's y-axis is clipped to
the range of its p2-p98 lines. Writes <out>.png and <out>.json; --rechart redraws the png from an existing <out>.json.

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
from matplotlib.colors import LinearSegmentedColormap, Normalize

Group = Literal["combined", "hack", "rest"]
Condition = Literal["unsteered", "steered"]
CONDITIONS: tuple[Condition, ...] = ("unsteered", "steered")
PERCENTILES = np.array([*range(0, 10), *range(10, 91, 10), *range(91, 101)])
DISPLAYED_PERCENTILES = [*range(0, 10, 2), 20, 40, 50, 60, 80, *range(92, 101, 2)]
MIDDLE_PERCENTILES = {20, 40, 60, 80}
CLIP_LOW_PERCENTILE, CLIP_HIGH_PERCENTILE = 2, 98
MEDIAN_COLOR = "#1f1f1d"
PERCENTILE_COLORMAP = LinearSegmentedColormap.from_list("blue_gray_red", ["#2a78d6", "#b9b8b4", "#e34948"])
GROUP_TITLES: dict[Group, str] = {"combined": "all episodes", "hack": "judged hack (STRONG / WEAK)", "rest": "not judged hack"}


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
            axis.plot(layers, line, color=MEDIAN_COLOR, linewidth=2.4, zorder=3)
        else:
            axis.plot(layers, line, color=PERCENTILE_COLORMAP(percentile / 100), linewidth=1.3 if percentile in MIDDLE_PERCENTILES else 0.8)


def clip_to_inner_percentiles(axis: Axes, panels: list[Float[np.ndarray, "layers percentiles"]]) -> None:
    inner = np.concatenate([percentile_columns(values, [CLIP_LOW_PERCENTILE, CLIP_HIGH_PERCENTILE]) for values in panels])
    low, high = float(np.nanmin(inner)), float(np.nanmax(inner))
    padding = 0.05 * (high - low) or 1.0
    axis.set_ylim(low - padding, high + padding)


def chart(percentiles_by_group: dict[Group, GroupPercentiles], steer_layer: int, alpha: float, title: str, out_png: Path) -> None:
    groups = list(percentiles_by_group)
    figure, axes = plt.subplots(4, len(groups), figsize=(5.2 * len(groups) + 1.2, 14), squeeze=False, sharex=True)
    row_labels = ["unsteered\ngfar projection", f"gi_v2 +{alpha:g} at L{steer_layer}\ngfar projection",
                  "steered − unsteered\ngfar projection", "steered − unsteered\n÷ unsteered p10–p90 width"]
    for column, group in enumerate(groups):
        result = percentiles_by_group[group]
        unsteered = np.asarray(result["percentiles_by_condition"]["unsteered"])
        steered = np.asarray(result["percentiles_by_condition"]["steered"])
        delta = steered - unsteered
        width = percentile_columns(unsteered, [90])[:, 0] - percentile_columns(unsteered, [10])[:, 0]
        normalized_delta = delta / width[:, None]
        for row, (values, clip_panels) in enumerate(((unsteered, [unsteered, steered]), (steered, [unsteered, steered]),
                                                     (delta, [delta]), (normalized_delta, [normalized_delta]))):
            axis = axes[row, column]
            draw_percentile_lines(axis, values)
            clip_to_inner_percentiles(axis, clip_panels)
            axis.axvline(steer_layer, color="#6b6a66", linestyle="--", linewidth=0.8)
            axis.grid(color="#e6e5e1", linewidth=0.5)
            axis.spines[["top", "right"]].set_visible(False)
            if row >= 2:
                axis.axhline(0, color="#6b6a66", linewidth=0.8)
            if column == 0:
                axis.set_ylabel(row_labels[row])
        axes[3, column].set_xlabel("layer")
        axes[0, column].set_title(f"{GROUP_TITLES[group]}\n{result['episodes']} episodes, {result['response_tokens']:,} response tokens", fontsize=10)
    colorbar = figure.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, 100), cmap=PERCENTILE_COLORMAP), ax=axes, fraction=0.02, pad=0.02)
    colorbar.set_label("percentile over response tokens: 0–8 and 92–100 every 2 (thin), 20 / 40 / 60 / 80 (medium), 50 in black")
    figure.suptitle(f"{title}: per-token gfar projection, unsteered episodes teacher-forced with and without gi_v2\n"
                    f"(dashed: steer layer; y-axes clipped to the p{CLIP_LOW_PERCENTILE}–p{CLIP_HIGH_PERCENTILE} range, so p0 / p100 can run off-scale)")
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
        chart(saved["groups"], saved["steer_layer"], saved["alpha"], args.title, args.out.with_suffix(".png"))
        return

    lens = load_lens(args.lens)
    verdicts_by_key = json.loads(args.verdicts.read_text()) if args.verdicts else None
    percentiles_by_group: dict[Group, GroupPercentiles] = {group: group_percentiles(lens, mask)
                            for group, mask in episode_masks_by_group(lens["verdict_keys"], verdicts_by_key).items() if mask.any()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps({
        "lens": str(args.lens), "verdicts": str(args.verdicts), "steer_layer": lens["steer_layer"], "alpha": lens["alpha"],
        "percentiles": PERCENTILES.tolist(), "groups": percentiles_by_group}, indent=1))
    chart(percentiles_by_group, lens["steer_layer"], lens["alpha"], args.title, args.out.with_suffix(".png"))


if __name__ == "__main__":
    main()
