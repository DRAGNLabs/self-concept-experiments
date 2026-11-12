"""Chart of steer_gfar_lens_extract.py output: per-layer percentiles of the per-token gfar projection, unsteered vs gi_v2-steered.

The chart shows all episodes: steered minus unsteered (the same percentile of each), and that difference divided by
the unsteered p10-p90 width at the layer; with --component-lens (the steered_gfar_component arm: only gi_v2's
component along gfar at the steer layer is added) also that arm's normalised difference and the full arm minus
the component arm. Each dashed line is the arm's injected vector carried unchanged, coefficient x (injected unit vector .
gfar at the layer). With --null-lens (chunk<i>/seed<N>.pt holding control-direction arms) rows 1-2 add the 5-95
percentile band of the control directions' median shifts and gi_v2's median shift on the same tokens; with
--null-component-lens (the same controls run with --null-mode gfar_component and the steered_gfar_component arm) rows
3-4 add the same band for the controls' gfar components and for full minus component, matched per control. The json also holds the percentiles of judged hacks (verdict HACK =
STRONG or WEAK) and the rest (unjudged episodes count as not hack) when --verdicts is given. Each panel's y-axis is
clipped to the range of its p2-p98 lines. Writes <out>.png and <out>.json; --rechart redraws the png from <out>.json.

    python scripts/steer_gfar_lens_chart.py --lens <dir of seed<N>.pt> [--component-lens <dir>] --verdicts <as_judged.json> --title "gpt-oss-120b" --out <stem>
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
from torch import Tensor
from jaxtyping import Bool, Float

from selfconcept.measurement.unit_axes import load_unit_axes
from steer_gfar_lens_extract import load_gfar_units

Group = Literal["combined", "hack", "rest"]
Condition = Literal["unsteered", "steered"]
Arm = Literal["steered", "steered_gfar_component"]
CONDITIONS: tuple[Condition, ...] = ("unsteered", "steered")
PERCENTILES = np.array([*range(0, 10), *range(10, 91, 10), *range(91, 101)])
DISPLAYED_PERCENTILES = [*range(0, 10, 2), 20, 40, 50, 60, 80, *range(92, 101, 2)]
CLIP_LOW_PERCENTILE, CLIP_HIGH_PERCENTILE = 2, 98
NULL_BAND_PERCENTILES = (5, 95)
MEDIAN_COLOR = "#1f1f1d"
INJECTION_COLOR = "#6b3fa0"
LOWER_PERCENTILES = [percentile for percentile in DISPLAYED_PERCENTILES if percentile < 50]
UPPER_PERCENTILES = [percentile for percentile in DISPLAYED_PERCENTILES if percentile > 50]


def percentile_color(percentile: int) -> tuple[float, float, float, float]:
    """Ordinal steps, darkest next to the (black) median and lightening toward p0 (blues) and p100 (reds)."""
    if percentile < 50:
        rank_from_median = len(LOWER_PERCENTILES) - 1 - LOWER_PERCENTILES.index(percentile)
        return plt.get_cmap("Blues")(0.95 - 0.65 * rank_from_median / (len(LOWER_PERCENTILES) - 1))
    rank_from_median = UPPER_PERCENTILES.index(percentile)
    return plt.get_cmap("Reds")(0.95 - 0.65 * rank_from_median / (len(UPPER_PERCENTILES) - 1))


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


def load_lens(lens_dir: Path, arm: Arm) -> Lens:
    """The unsteered projections and the arm's, the latter under the "steered" condition key."""
    seed_files = sorted(lens_dir.glob("seed*.pt"), key=lambda path: int(path.stem.removeprefix("seed")))
    seeds = [torch.load(path, map_location="cpu", weights_only=False) for path in seed_files]
    episode_counts = np.cumsum([0] + [len(seed["example_id"]) for seed in seeds])
    return {
        "projections_by_condition": {"unsteered": torch.cat([seed["unsteered"] for seed in seeds], dim=1).numpy(),
                                     "steered": torch.cat([seed[arm] for seed in seeds], dim=1).numpy()},
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


def carried_injection_projection(lens_dir: Path, arm: Arm) -> Float[np.ndarray, "layers"]:
    """coefficient * <injected vector, unit gfar at each layer>, the injected vector being unit gi_v2 at the steer layer
    (steered) or its component along gfar there (steered_gfar_component); NaN before the steer layer."""
    seed = torch.load(next(lens_dir.glob("seed*.pt")), map_location="cpu", weights_only=False)
    gfar_units = load_gfar_units(seed["gfar"], *_artifact_shape(seed["gfar"]))
    num_layers, hidden = gfar_units.shape
    steer_layer = seed["steer_layer"]
    gi_v2_unit = torch.as_tensor(load_unit_axes(Path(seed["gi_v2"]), [steer_layer], num_layers, hidden, seed["model"])[steer_layer]["direction"])
    gfar_at_steer_layer = gfar_units[steer_layer]
    injected = gi_v2_unit if arm == "steered" else (gi_v2_unit @ gfar_at_steer_layer) * gfar_at_steer_layer
    projection = (seed["coefficient"] * (gfar_units @ injected)).numpy()
    projection[:steer_layer] = np.nan
    return projection


def _artifact_shape(gfar_spec: str) -> tuple[int, int]:
    path, key = gfar_spec.rsplit(":", 1)
    num_layers, hidden = torch.load(path, map_location="cpu", weights_only=False)[key].shape
    return int(num_layers), int(hidden)


class NullMedians(TypedDict):
    names: list[str]
    raw: list[list[float]]
    normalized: list[list[float]]
    gi_v2_raw: list[float]
    gi_v2_normalized: list[float]
    episodes: int
    response_tokens: int


def layer_medians(projections: Float[Tensor, "layers tokens"]) -> Float[np.ndarray, "layers"]:
    return np.median(projections.float().numpy(), axis=1)


def null_medians(null_lens_dir: Path, gi_v2_arm: Arm) -> NullMedians:
    """Per control direction, p50(steered) - p50(unsteered) per layer, raw and over the chunk's unsteered p10-p90 width."""
    raw_by_name: dict[str, Float[np.ndarray, "layers"]] = {}
    normalized_by_name: dict[str, Float[np.ndarray, "layers"]] = {}
    gi_v2_raw: Float[np.ndarray, "layers"] | None = None
    gi_v2_normalized: Float[np.ndarray, "layers"] | None = None
    episodes = response_tokens = 0
    for chunk_dir in sorted(null_lens_dir.glob("chunk*")):
        seeds = [torch.load(path, map_location="cpu", weights_only=False) for path in sorted(chunk_dir.glob("seed*.pt"))]
        unsteered = torch.cat([seed["unsteered"] for seed in seeds], dim=1).float().numpy()
        unsteered_median = np.median(unsteered, axis=1)
        low, high = np.percentile(unsteered, [10, 90], axis=1)
        for name in seeds[0]["null_projections"]:
            raw = layer_medians(torch.cat([seed["null_projections"][name] for seed in seeds], dim=1)) - unsteered_median
            raw_by_name[name], normalized_by_name[name] = raw, raw / (high - low)
        if gi_v2_raw is None:
            gi_v2_raw = layer_medians(torch.cat([seed[gi_v2_arm] for seed in seeds], dim=1)) - unsteered_median
            gi_v2_normalized = gi_v2_raw / (high - low)
            episodes, response_tokens = sum(len(seed["example_id"]) for seed in seeds), unsteered.shape[1]
    assert gi_v2_raw is not None and gi_v2_normalized is not None
    names = sorted(raw_by_name)
    return {"names": names, "raw": [raw_by_name[name].tolist() for name in names],
            "normalized": [normalized_by_name[name].tolist() for name in names],
            "gi_v2_raw": gi_v2_raw.tolist(), "gi_v2_normalized": gi_v2_normalized.tolist(),
            "episodes": episodes, "response_tokens": response_tokens}


def draw_null_band(axis: Axes, medians: Float[np.ndarray, "directions layers"], gi_v2_subset: Float[np.ndarray, "layers"], count: int) -> None:
    layers = np.arange(medians.shape[1])
    low, high = np.percentile(medians, NULL_BAND_PERCENTILES, axis=0)
    axis.fill_between(layers, low, high, color="#b9b8b4", alpha=0.45, linewidth=0, zorder=1,
                      label=f"antonym controls: {NULL_BAND_PERCENTILES[0]}–{NULL_BAND_PERCENTILES[1]}% of median shifts (n={count})")
    axis.plot(layers, gi_v2_subset, color=MEDIAN_COLOR, linestyle=(0, (1, 1.5)), linewidth=1.4, zorder=3,
              label="gi_v2 median shift on the controls' tokens")
    bottom, top = axis.get_ylim()
    axis.set_ylim(min(bottom, float(np.nanmin(low))), max(top, float(np.nanmax(high))))


class ChartRow(TypedDict):
    values: Float[np.ndarray, "layers percentiles"]
    injected: Float[np.ndarray, "layers"]
    label: str


type NullBand = tuple[Float[np.ndarray, "directions layers"], Float[np.ndarray, "layers"], int]


def null_bands_by_row(null: NullMedians | None, null_component: NullMedians | None) -> dict[int, NullBand]:
    """Row index -> (control medians, gi_v2 subset median, control count); row 3 matches each control's full and component run."""
    bands: dict[int, NullBand] = {}
    if null is not None:
        bands[0] = (np.asarray(null["raw"]), np.asarray(null["gi_v2_raw"]), len(null["names"]))
        bands[1] = (np.asarray(null["normalized"]), np.asarray(null["gi_v2_normalized"]), len(null["names"]))
    if null_component is not None:
        bands[2] = (np.asarray(null_component["normalized"]), np.asarray(null_component["gi_v2_normalized"]), len(null_component["names"]))
    if null is not None and null_component is not None:
        full_by_name = dict(zip(null["names"], null["normalized"]))
        component_by_name = dict(zip(null_component["names"], null_component["normalized"]))
        names = sorted(set(full_by_name) & set(component_by_name))
        bands[3] = (np.asarray([np.subtract(full_by_name[name], component_by_name[name]) for name in names]),
                    np.asarray(null["gi_v2_normalized"]) - np.asarray(null_component["gi_v2_normalized"]), len(names))
    return bands


def chart(full: GroupPercentiles, component: GroupPercentiles | None, null: NullMedians | None, null_component: NullMedians | None,
          injected_by_arm: dict[Arm, Float[np.ndarray, "layers"]], steer_layer: int, alpha: float, title: str, out_png: Path) -> None:
    unsteered = np.asarray(full["percentiles_by_condition"]["unsteered"])
    width = percentile_columns(unsteered, [90])[:, 0] - percentile_columns(unsteered, [10])[:, 0]
    full_delta = np.asarray(full["percentiles_by_condition"]["steered"]) - unsteered
    rows: list[ChartRow] = [
        {"values": full_delta, "injected": injected_by_arm["steered"], "label": "full gi_v2: steered − unsteered\ngfar projection"},
        {"values": full_delta / width[:, None], "injected": injected_by_arm["steered"] / width,
         "label": "full gi_v2: steered − unsteered\n÷ unsteered p10–p90 width"},
    ]
    if component is not None:
        component_delta = (np.asarray(component["percentiles_by_condition"]["steered"])
                           - np.asarray(component["percentiles_by_condition"]["unsteered"]))
        component_rows: list[ChartRow] = [
            {"values": component_delta / width[:, None], "injected": injected_by_arm["steered_gfar_component"] / width,
             "label": "gfar component of gi_v2 only:\nsteered − unsteered ÷ width"},
            {"values": (full_delta - component_delta) / width[:, None],
             "injected": (injected_by_arm["steered"] - injected_by_arm["steered_gfar_component"]) / width,
             "label": "row 2 − row 3\n(full − component only) ÷ width"},
        ]
        rows += component_rows
    figure, axes = plt.subplots(len(rows), 1, figsize=(9, 4.5 * len(rows)), sharex=True, squeeze=False)
    bands = null_bands_by_row(null, null_component)
    for row_index, (axis, row) in enumerate(zip(axes[:, 0], rows)):
        draw_percentile_lines(axis, row["values"])
        clip_to_inner_percentiles(axis, [row["values"]])
        if row_index in bands:
            draw_null_band(axis, *bands[row_index])
        axis.plot(np.arange(len(row["injected"])), row["injected"], color=INJECTION_COLOR, linestyle="--", linewidth=1.8, zorder=4,
                  label="injection carried unchanged")
        low, high = axis.get_ylim()
        axis.set_ylim(min(low, float(np.nanmin(row["injected"]))), max(high, float(np.nanmax(row["injected"]))))
        axis.axvline(steer_layer, color="#6b6a66", linestyle=":", linewidth=0.8)
        axis.axhline(0, color="#6b6a66", linewidth=0.8)
        axis.grid(color="#e6e5e1", linewidth=0.5)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_ylabel(row["label"])
    axes[-1, 0].set_xlabel("layer")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    order = ([labels.index(f"p{percentile}") for percentile in reversed(DISPLAYED_PERCENTILES)] + [labels.index("injection carried unchanged")]
             + [index for index, label in enumerate(labels) if label.startswith(("antonym", "gi_v2 median"))])
    figure.legend([handles[i] for i in order], [labels[i] for i in order], loc="center left", bbox_to_anchor=(0.98, 0.5), frameon=False,
                  title="percentile over\nresponse tokens")
    figure.suptitle(f"{title}: gi_v2 +{alpha:g} at L{steer_layer} (dotted), effect on the per-token gfar projection\n"
                    f"all {full['episodes']} unsteered episodes, {full['response_tokens']:,} response tokens, teacher-forced; "
                    f"dashed: the injected vector carried unchanged\n"
                    f"width = unsteered p10–p90 width at the layer; y-axes clipped to the p{CLIP_LOW_PERCENTILE}–p{CLIP_HIGH_PERCENTILE} range", fontsize=10)
    figure.tight_layout()
    figure.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(figure)


def injected_by_arm(lens_dir: Path, component_lens_dir: Path | None) -> dict[Arm, Float[np.ndarray, "layers"]]:
    injected: dict[Arm, Float[np.ndarray, "layers"]] = {"steered": carried_injection_projection(lens_dir, "steered")}
    if component_lens_dir is not None:
        injected["steered_gfar_component"] = carried_injection_projection(component_lens_dir, "steered_gfar_component")
    return injected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lens", type=Path, help="required unless --rechart")
    parser.add_argument("--component-lens", type=Path, default=None, help="dir of seed<N>.pt holding the steered_gfar_component arm")
    parser.add_argument("--null-lens", type=Path, default=None, help="dir of chunk<i>/seed<N>.pt holding control-direction arms")
    parser.add_argument("--null-component-lens", type=Path, default=None, help="the same controls run with --null-mode gfar_component")
    parser.add_argument("--verdicts", type=Path, default=None, help="as_judged.json keyed '<arm>/seed<s>/<example_id>'")
    parser.add_argument("--title", required=True)
    parser.add_argument("--out", type=Path, required=True, help="output stem")
    parser.add_argument("--rechart", action="store_true", help="redraw <out>.png from <out>.json without reading the lens")
    args = parser.parse_args()

    if args.rechart:
        saved = json.loads(args.out.with_suffix(".json").read_text())
        component_lens = Path(saved["component_lens"]) if saved.get("component_lens") else None
        null = null_medians(args.null_lens, "steered") if args.null_lens is not None else saved.get("null")
        null_component = (null_medians(args.null_component_lens, "steered_gfar_component") if args.null_component_lens is not None
                          else saved.get("null_component"))
        saved["null"], saved["null_component"] = null, null_component
        args.out.with_suffix(".json").write_text(json.dumps(saved, indent=1))
        chart(saved["groups"]["combined"], saved.get("component"), null, null_component, injected_by_arm(Path(saved["lens"]), component_lens), saved["steer_layer"],
              saved["alpha"], args.title, args.out.with_suffix(".png"))
        return

    lens = load_lens(args.lens, "steered")
    verdicts_by_key = json.loads(args.verdicts.read_text()) if args.verdicts else None
    percentiles_by_group: dict[Group, GroupPercentiles] = {group: group_percentiles(lens, mask)
                            for group, mask in episode_masks_by_group(lens["verdict_keys"], verdicts_by_key).items() if mask.any()}
    steer_layer, alpha = lens["steer_layer"], lens["alpha"]
    del lens
    component = None
    if args.component_lens is not None:
        component_lens = load_lens(args.component_lens, "steered_gfar_component")
        component = group_percentiles(component_lens, np.ones(len(component_lens["verdict_keys"]), dtype=bool))
        del component_lens
    null = null_medians(args.null_lens, "steered") if args.null_lens is not None else None
    null_component = null_medians(args.null_component_lens, "steered_gfar_component") if args.null_component_lens is not None else None
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps({
        "lens": str(args.lens), "component_lens": str(args.component_lens) if args.component_lens else None, "verdicts": str(args.verdicts),
        "steer_layer": steer_layer, "alpha": alpha,
        "percentiles": PERCENTILES.tolist(), "groups": percentiles_by_group, "component": component, "null": null,
        "null_component": null_component}, indent=1))
    chart(percentiles_by_group["combined"], component, null, null_component, injected_by_arm(args.lens, args.component_lens), steer_layer, alpha, args.title, args.out.with_suffix(".png"))


if __name__ == "__main__":
    main()
