"""Chart of steer_gfar_lens_extract.py output: per-layer percentiles of the per-token gfar projection, unsteered vs gi_v2-steered.

Rows: unsteered, steered, steered minus unsteered (the same percentile of each). Columns: all episodes, judged hacks
(verdict HACK = STRONG or WEAK) and the rest (unjudged episodes count as not hack); without --verdicts only the
combined column. Writes <out>.png and <out>.json.

    python scripts/steer_gfar_lens_chart.py --lens <dir of seed<N>.pt> --verdicts <as_judged.json> --title "gpt-oss-120b" --out <stem>
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
DECILES = set(range(10, 91, 10))
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


def draw_percentile_lines(axis: Axes, values: Float[np.ndarray, "layers percentiles"]) -> None:
    layers = np.arange(values.shape[0])
    for column, percentile in enumerate(PERCENTILES):
        axis.plot(layers, values[:, column], color=PERCENTILE_COLORMAP(percentile / 100),
                  linewidth=1.6 if percentile in DECILES or percentile in (0, 100) else 0.6)


def chart(percentiles_by_group: dict[Group, GroupPercentiles], steer_layer: int, alpha: float, title: str, out_png: Path) -> None:
    groups = list(percentiles_by_group)
    figure, axes = plt.subplots(3, len(groups), figsize=(5.2 * len(groups) + 1.2, 11), squeeze=False, sharex=True)
    row_labels = ["unsteered", f"gi_v2 +{alpha:g} at L{steer_layer}", "steered − unsteered"]
    for column, group in enumerate(groups):
        result = percentiles_by_group[group]
        unsteered = np.asarray(result["percentiles_by_condition"]["unsteered"])
        steered = np.asarray(result["percentiles_by_condition"]["steered"])
        for row, values in enumerate((unsteered, steered, steered - unsteered)):
            axis = axes[row, column]
            draw_percentile_lines(axis, values)
            axis.axvline(steer_layer, color="#6b6a66", linestyle="--", linewidth=0.8)
            axis.grid(color="#e6e5e1", linewidth=0.5)
            axis.spines[["top", "right"]].set_visible(False)
            if column == 0:
                axis.set_ylabel(f"{row_labels[row]}\ngfar projection")
        axes[2, column].axhline(0, color="#6b6a66", linewidth=0.8)
        axes[2, column].set_xlabel("layer")
        axes[0, column].set_title(f"{GROUP_TITLES[group]}\n{result['episodes']} episodes, {result['response_tokens']:,} response tokens", fontsize=10)
    for column in range(len(groups)):
        axes[1, column].sharey(axes[0, column])
    colorbar = figure.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, 100), cmap=PERCENTILE_COLORMAP), ax=axes, fraction=0.02, pad=0.02)
    colorbar.set_label("percentile over response tokens (thick: 0, 10, 20 … 90, 100; thin: 1–9, 91–99)")
    figure.suptitle(f"{title}: per-token gfar projection, unsteered episodes teacher-forced with and without gi_v2 (dashed: steer layer)")
    figure.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lens", type=Path, required=True)
    parser.add_argument("--verdicts", type=Path, default=None, help="as_judged.json keyed '<arm>/seed<s>/<example_id>'")
    parser.add_argument("--title", required=True)
    parser.add_argument("--out", type=Path, required=True, help="output stem")
    args = parser.parse_args()

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
