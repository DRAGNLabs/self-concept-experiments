"""EXPLORATORY (lightly reviewed when merged 2026-11-20): kept for reference, not a validated measurement.

Scatter of unsteered episodes' gi_v2 projection against their gfar projection, colored by judged HACK. (CPU)

Takes `hack_separation.py` output records and replots their episodes (same caches, layer, verdicts and directions):
one row per record, one column per token segment. `@<arm>` after a record path plots that arm's caches instead of the
record's arm (the record's acts directory and verdicts must cover it). HACK = judged STRONG or WEAK; everything else is
not hack. Episodes with no tokens in a segment are left out of that segment's panel. --share-axes puts every row of a
column on the same axes, for comparing arms of one model.

    python experiments/deceiver-axis/scripts/plot_hack_projections.py \
        --record gpt-oss=<hack_separation>/gptoss120b.json --record Muse=<hack_separation>/muse.json --out <dir>/plot.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TypedDict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from jaxtyping import Bool, Float
from matplotlib.axes import Axes

from hack_separation import SEGMENTS, Segment, load_episodes, load_unit_direction, pooled_auroc

NON_HACK_COLOR = "#2a78d6"
HACK_COLOR = "#eb6834"
X_DIRECTION = "gi_v2"
Y_DIRECTION = "gfar"


class ModelProjections(TypedDict):
    label: str
    layer: int
    projections_by_segment_by_direction: dict[str, dict[Segment, Float[np.ndarray, "episodes"]]]
    has_tokens_by_segment: dict[Segment, Bool[np.ndarray, "episodes"]]
    is_hack: Bool[np.ndarray, "episodes"]


def model_projections(label: str, record_spec: str) -> ModelProjections:
    record_path, _, arm_override = record_spec.partition("@")
    record = json.loads(Path(record_path).read_text())
    episodes = load_episodes(Path(record["acts"]), arm_override or record.get("arm", "unsteered"), record["layer"])
    verdicts_by_key: dict[str, str] = json.loads(Path(record["verdicts"]).read_text())
    unit_directions_by_name = dict(load_unit_direction(spec, record["layer"]) for spec in record["directions"])
    return {"label": label, "layer": record["layer"],
            "projections_by_segment_by_direction": {
                name: {segment: episodes["means_by_segment"][segment] @ unit_direction for segment in SEGMENTS}
                for name, unit_direction in unit_directions_by_name.items()},
            "has_tokens_by_segment": {segment: episodes["token_counts_by_segment"][segment] > 0 for segment in SEGMENTS},
            "is_hack": np.array([verdicts_by_key.get(key) == "HACK" for key in episodes["verdict_keys"]])}


def auroc(projections: Float[np.ndarray, "episodes"], is_hack: Bool[np.ndarray, "episodes"]) -> float:
    return float(pooled_auroc(projections[:, None], is_hack)[0])


def draw_panel(axes: Axes, model: ModelProjections, segment: Segment) -> None:
    has_tokens = model["has_tokens_by_segment"][segment]
    x_projections = model["projections_by_segment_by_direction"][X_DIRECTION][segment][has_tokens]
    y_projections = model["projections_by_segment_by_direction"][Y_DIRECTION][segment][has_tokens]
    is_hack = model["is_hack"][has_tokens]
    empty_note = f", {int((~has_tokens).sum())} empty left out" if not has_tokens.all() else ""
    axes.scatter(x_projections[~is_hack], y_projections[~is_hack], s=14, color=NON_HACK_COLOR, alpha=0.45, linewidths=0,
                 label=f"not hack ({int((~is_hack).sum())})")
    axes.scatter(x_projections[is_hack], y_projections[is_hack], s=18, color=HACK_COLOR, alpha=0.85,
                 edgecolors="white", linewidths=0.6, label=f"hack ({int(is_hack.sum())})")
    axes.set_title(f"{model['label']} L{model['layer']}, {segment} tokens{empty_note}\n"
                   f"pooled AUROC {X_DIRECTION} {auroc(x_projections, is_hack):.2f}, {Y_DIRECTION} {auroc(y_projections, is_hack):.2f}",
                   fontsize=9)
    axes.set_xlabel(f"{X_DIRECTION} projection", fontsize=8)
    axes.set_ylabel(f"{Y_DIRECTION} projection", fontsize=8)
    axes.tick_params(labelsize=7)
    axes.grid(color="#e4e3df", linewidth=0.5)
    axes.set_axisbelow(True)
    for spine in ("top", "right"):
        axes.spines[spine].set_visible(False)
    axes.legend(fontsize=7, frameon=False, loc="best")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--record", action="append", required=True, help="label=hack_separation output .json")
    parser.add_argument("--share-axes", action="store_true")
    parser.add_argument("--title", default="Unsteered episodes: gi_v2 vs gfar projection (episode token mean), judged hack vs not")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    models = [model_projections(label, record_spec) for label, record_spec in (spec.split("=", 1) for spec in args.record)]
    figure, axes_grid = plt.subplots(len(models), len(SEGMENTS), figsize=(4.2 * len(SEGMENTS), 3.6 * len(models)), squeeze=False,
                                     sharex="col" if args.share_axes else False, sharey="col" if args.share_axes else False)
    for model, axes_row in zip(models, axes_grid):
        for segment, axes in zip(SEGMENTS, axes_row):
            draw_panel(axes, model, segment)
    figure.suptitle(args.title, fontsize=10)
    figure.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out, dpi=160)


if __name__ == "__main__":
    main()
