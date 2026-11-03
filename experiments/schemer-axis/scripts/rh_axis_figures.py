"""Figures for the reward-hacking direction comparisons, per model and across models.

    python scripts/rh_axis_figures.py --manifest <manifest.json> --out <dir>

The manifest lists, per model: {"label", "layer", "directions": [directions.pt, ...], "pair_vectors": <pair .pt>,
"chris_axis": <direction name>, "geometry": <chris_rh_axis_geometry.json>, "logit_lens": <logit_lens.json> | null}.
directions.pt files map name -> (layer, hidden); later files add directions (e.g. Goodfire's) without code changes.

Writes, as png and pdf:
- cosine_heatmap_<model>: every pair of directions at the model's layer; cells of the reference reply-pair axis
  inside its sign-flip null (95th percentile |cosine|) are greyed;
- cosine_by_layer_<reference>: cosine with gi_v2 and with the reply-pair axis at every layer, one panel per model;
- pair_projections: each real hack/honest pair's projection difference on each direction, with k/n;
- reply_auroc: AUROC over all replies with a bootstrap-over-pairs 95% interval, models side by side;
- pair_gi_vs_accuracy: each letter/spirit pair's cosine with gi_v2 against its pair accuracy;
- logit_lens_<model>: top / bottom 8 tokens by z for every lens direction and layer, with the random-direction
  95th-percentile |z|.
"""
import argparse
import json
from pathlib import Path
from typing import TypedDict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from jaxtyping import Float
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

plt.rcParams["font.family"] = ["DejaVu Sans", "Droid Sans Fallback"]

Directions = dict[str, Float[torch.Tensor, "layer hidden"]]

LABEL_BY_DIRECTION = {
    "gi_v2": "gi_v2 (deceiver)",
    "gi_17role": "gi 17-role",
    "letter_vs_spirit": "letter − spirit",
    "jobsworth_vs_discretionary": "jobsworth control",
    "rh_personas_7v3": "RH personas 7v3",
    "rh_narrow": "RH narrow",
    "nonrh_holdout": "non-RH deceivers",
    "table_rules_lawyer_fic-table_spirit_player_fic": "pair: tabletop",
    "kpi_chaser_fic-mission_driven_employee_fic": "pair: KPI",
    "test_studier_fic-deep_learner_fic": "pair: test studier",
    "literal_genie_fic-faithful_genie_fic": "pair: genie",
    "spec_literal_contractor_fic-purpose_minded_builder_fic": "pair: contractor",
    "test_passing_programmer_fic-working_code_programmer_fic": "pair: programmer",
    "chris_same_prompt_pairs": "Chris same-prompt",
    "chris_stagec_gptoss_pairs": "Chris stage C",
    "chris_stagec_muse_pairs": "Chris stage C",
    "chris_same_prompt": "Chris same-prompt",
    "chris_stagec": "Chris stage C",
    "chris_stagec_muse": "Chris stage C",
    "letter_minus_default": "letter − default",
    "spirit_minus_default": "spirit − default",
}
DUPLICATE_DIRECTIONS = {"jobsworth_clerk_fic-discretionary_clerk_fic"}  # the single-pair jobsworth contrast itself
SURFACE, INK, SECONDARY_INK, MUTED_INK, GRIDLINE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
NULL_GREY = "#e1e0d9"
MODEL_COLORS = ["#2a78d6", "#eb6834"]
DIVERGING = LinearSegmentedColormap.from_list("blue_grey_red", ["#104281", "#3987e5", "#f0efec", "#e34948", "#8f2323"])
N_BOOTSTRAP = 2000
TOP_TOKENS = 8


class ModelEntry(TypedDict):
    label: str
    layer: int
    directions: list[str]
    pair_vectors: str
    chris_axis: str
    geometry: str
    logit_lens: str | None


def label(name: str) -> str:
    return LABEL_BY_DIRECTION.get(name, name)


def load_directions(paths: list[str]) -> Directions:
    merged: Directions = {}
    for path in paths:
        merged |= {name: vector.float() for name, vector in torch.load(path, map_location="cpu").items() if name not in DUPLICATE_DIRECTIONS}
    known = [name for name in LABEL_BY_DIRECTION if name in merged]
    return {name: merged[name] for name in known + [name for name in merged if name not in LABEL_BY_DIRECTION]}


def unit(vectors: torch.Tensor) -> torch.Tensor:
    return vectors / vectors.norm(dim=-1, keepdim=True)


def style(axes: Axes) -> None:
    axes.set_facecolor(SURFACE)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color("#c3c2b7")
    axes.tick_params(colors=SECONDARY_INK, labelsize=8)


def save(figure: Figure, out: Path, stem: str) -> None:
    figure.patch.set_facecolor(SURFACE)
    for suffix in ("png", "pdf"):
        figure.savefig(out / f"{stem}.{suffix}", dpi=200, facecolor=SURFACE)
    plt.close(figure)


def pair_set_name(entry: ModelEntry) -> str:
    return Path(entry["pair_vectors"]).stem


def sign_flip_null(entry: ModelEntry, geometry: dict, direction: str) -> float | None:
    by_axis = geometry["pair_sets"][pair_set_name(entry)]["sign_flip_null_abs_cosine_95th_with_persona_axis"]
    return by_axis[direction][entry["layer"]] if direction in by_axis else None


def cosine_heatmap(entry: ModelEntry, directions: Directions, geometry: dict, out: Path) -> None:
    names = list(directions)
    at_layer = unit(torch.stack([directions[name][entry["layer"]] for name in names]))
    cosines = (at_layer @ at_layer.T).numpy()
    size = len(names)
    figure, axes = plt.subplots(figsize=(0.55 * size + 3, 0.5 * size + 1.5))
    lower = np.tril(np.ones((size, size), dtype=bool), k=-1)
    axes.imshow(np.where(lower, cosines, np.nan)[1:, :-1], cmap=DIVERGING, vmin=-1, vmax=1, extent=(-0.5, size - 1.5, size - 0.5, 0.5))
    chris_index = names.index(entry["chris_axis"]) if entry["chris_axis"] in names else None
    for row in range(size):
        for column in range(row):
            null = None
            if chris_index in (row, column):
                other = names[column if row == chris_index else row]
                null = sign_flip_null(entry, geometry, other)
            inside_null = null is not None and abs(cosines[row, column]) < null
            if inside_null:
                axes.add_patch(Rectangle((column - 0.5, row - 0.5), 1, 1, color=NULL_GREY, zorder=2))
            axes.text(column, row, f"{cosines[row, column]:.2f}", ha="center", va="center", fontsize=7, zorder=3,
                      color=MUTED_INK if inside_null else (SURFACE if abs(cosines[row, column]) > 0.55 else INK))
    axes.set_xticks(range(size - 1), [label(name) for name in names[:-1]], rotation=60, ha="right")
    axes.set_yticks(range(1, size), [label(name) for name in names[1:]])
    for spine in axes.spines.values():
        spine.set_visible(False)
    axes.set_facecolor(SURFACE)
    axes.tick_params(colors=SECONDARY_INK, labelsize=8, length=0)
    axes.set_title(f"{entry['label']}, layer {entry['layer']}: cosine between directions\n"
                   f"grey: |cosine| with {label(entry['chris_axis'])} inside its sign-flip null (95th pct)",
                   fontsize=9, color=INK, loc="left")
    colorbar = figure.colorbar(axes.images[0], ax=axes, shrink=0.6, ticks=[-1, -0.5, 0, 0.5, 1])
    colorbar.outline.set_visible(False)  # type: ignore[union-attr]
    colorbar.ax.tick_params(colors=SECONDARY_INK, labelsize=8)
    figure.tight_layout()
    save(figure, out, f"cosine_heatmap_{entry['label']}")


def cosine_by_layer(entries: list[ModelEntry], directions_by_model: dict[str, Directions], reference_of: str, out: Path) -> None:
    figure, panels = plt.subplots(1, len(entries), figsize=(7 * len(entries), 6), squeeze=False)
    for panel, entry in zip(panels[0], entries):
        directions = directions_by_model[entry["label"]]
        reference_name = entry["chris_axis"] if reference_of == "chris" else "gi_v2"
        reference = unit(directions[reference_name])
        names = [name for name in directions if name != reference_name]
        cosines = torch.stack([(unit(directions[name]) * reference).sum(-1) for name in names]).numpy()
        image = panel.imshow(cosines, cmap=DIVERGING, vmin=-1, vmax=1, aspect="auto", interpolation="nearest")
        panel.set_yticks(range(len(names)), [label(name) for name in names])
        panel.set_xlabel("layer", color=SECONDARY_INK, fontsize=8)
        panel.axvline(entry["layer"], color=INK, linewidth=0.8)
        panel.set_title(f"{entry['label']}: cosine with {label(reference_name)} (line: layer {entry['layer']})",
                        fontsize=9, color=INK, loc="left")
        for spine in panel.spines.values():
            spine.set_visible(False)
        panel.tick_params(colors=SECONDARY_INK, labelsize=8, length=0)
        colorbar = figure.colorbar(image, ax=panel, shrink=0.6, ticks=[-1, -0.5, 0, 0.5, 1])
        colorbar.outline.set_visible(False)  # type: ignore[union-attr]
        colorbar.ax.tick_params(colors=SECONDARY_INK, labelsize=8)
    figure.tight_layout()
    save(figure, out, f"cosine_by_layer_{reference_of}")


class PairData(TypedDict):
    differences: Float[torch.Tensor, "pair hidden"]
    hack: Float[torch.Tensor, "pair hidden"]
    honest: Float[torch.Tensor, "pair hidden"]
    is_holdout: torch.Tensor


def load_pairs(entry: ModelEntry) -> PairData:
    pair_vectors = torch.load(entry["pair_vectors"], map_location="cpu", weights_only=False)
    rows, activations = pair_vectors["rows"], pair_vectors["activations"][:, entry["layer"]].float()
    index_by_key = {(row["pair_id"], row["condition"]): i for i, row in enumerate(rows)}
    pair_ids = sorted({row["pair_id"] for row in rows})
    split_by_pair = {row["pair_id"]: row["split"] for row in rows}
    hack = torch.stack([activations[index_by_key[(pair_id, "hack")]] for pair_id in pair_ids])
    honest = torch.stack([activations[index_by_key[(pair_id, "honest")]] for pair_id in pair_ids])
    return {"differences": hack - honest, "hack": hack, "honest": honest,
            "is_holdout": torch.tensor([split_by_pair[pair_id] == "holdout" for pair_id in pair_ids])}


def evaluated_pairs(entry: ModelEntry, name: str, pairs: PairData) -> torch.Tensor:
    """Every pair, except that the reply-pair axis is only scored on the pairs it was not built from."""
    return pairs["is_holdout"] if name == entry["chris_axis"] else torch.ones_like(pairs["is_holdout"])


def auroc(hack_projections: torch.Tensor, honest_projections: torch.Tensor) -> float:
    margins = hack_projections[:, None] - honest_projections[None, :]
    return ((margins > 0).float() + 0.5 * (margins == 0).float()).mean().item()


def bootstrap_auroc(hack_projections: torch.Tensor, honest_projections: torch.Tensor, generator: torch.Generator) -> tuple[float, float]:
    count = hack_projections.shape[0]
    resampled = [auroc(hack_projections[picks], honest_projections[picks])
                 for picks in torch.randint(0, count, (N_BOOTSTRAP, count), generator=generator)]
    low, high = np.quantile(resampled, [0.025, 0.975])
    return float(low), float(high)


def scored_label(entry: ModelEntry, name: str) -> str:
    return label(name) + (" (holdout pairs)" if name == entry["chris_axis"] else "")


def pair_projections(entries: list[ModelEntry], directions_by_model: dict[str, Directions], pairs_by_model: dict[str, PairData], out: Path) -> None:
    height = max(len(directions_by_model[entry["label"]]) for entry in entries)
    figure, panels = plt.subplots(1, len(entries), figsize=(6.5 * len(entries), 0.42 * height + 1.5), squeeze=False)
    generator = torch.Generator().manual_seed(0)
    for panel, entry in zip(panels[0], entries):
        directions, pairs = directions_by_model[entry["label"]], pairs_by_model[entry["label"]]
        names = list(directions)
        for row, name in enumerate(names):
            keep = evaluated_pairs(entry, name, pairs)
            axis = unit(directions[name][entry["layer"]])
            reply_spread = torch.cat([pairs["hack"], pairs["honest"]]) @ axis
            projections = pairs["differences"][keep] @ axis / reply_spread.std()
            jitter = (torch.rand(projections.shape[0], generator=generator) - 0.5) * 0.35
            panel.scatter(projections.numpy(), (row + jitter).numpy(), s=14, color=MODEL_COLORS[0], alpha=0.75, linewidths=0)
            panel.text(1.01, row, f"{int((projections > 0).sum())}/{projections.shape[0]}", transform=panel.get_yaxis_transform(),
                       va="center", fontsize=8, color=SECONDARY_INK)
        panel.axvline(0, color=MUTED_INK, linewidth=0.8)
        panel.set_yticks(range(len(names)), [scored_label(entry, name) for name in names])
        panel.invert_yaxis()
        panel.grid(axis="x", color=GRIDLINE, linewidth=0.5)
        panel.set_axisbelow(True)
        style(panel)
        panel.set_xlabel("hack − honest projection (SD units of single replies; right of 0: hack higher)", fontsize=8, color=SECONDARY_INK)
        panel.set_title(f"{entry['label']}, layer {entry['layer']}: {pair_set_name(entry)} (k/n: pairs right of 0)",
                        fontsize=9, color=INK, loc="left")
    figure.tight_layout()
    save(figure, out, "pair_projections")


def reply_auroc(entries: list[ModelEntry], directions_by_model: dict[str, Directions], pairs_by_model: dict[str, PairData], out: Path) -> None:
    rows = list(dict.fromkeys(scored_label(entry, name) for entry in entries for name in directions_by_model[entry["label"]]))
    figure, axes = plt.subplots(figsize=(7.5, 0.4 * len(rows) + 1.5))
    generator = torch.Generator().manual_seed(0)
    offsets = np.linspace(-0.15, 0.15, len(entries))
    for model_index, entry in enumerate(entries):
        directions, pairs = directions_by_model[entry["label"]], pairs_by_model[entry["label"]]
        for name, direction in directions.items():
            keep = evaluated_pairs(entry, name, pairs)
            axis = unit(direction[entry["layer"]])
            hack, honest = pairs["hack"][keep] @ axis, pairs["honest"][keep] @ axis
            low, high = bootstrap_auroc(hack, honest, generator)
            y = rows.index(scored_label(entry, name)) + offsets[model_index]
            axes.plot([low, high], [y, y], color=MODEL_COLORS[model_index], linewidth=2, solid_capstyle="round")
            axes.scatter([auroc(hack, honest)], [y], s=36, color=MODEL_COLORS[model_index], edgecolors=SURFACE, linewidths=1.5, zorder=3)
        axes.scatter([], [], s=36, color=MODEL_COLORS[model_index], label=f"{entry['label']} (layer {entry['layer']}, {pair_set_name(entry)})")
    axes.axvline(0.5, color=MUTED_INK, linewidth=0.8)
    axes.set_xlim(0, 1)
    axes.set_yticks(range(len(rows)), rows)
    axes.invert_yaxis()
    axes.grid(axis="x", color=GRIDLINE, linewidth=0.5)
    axes.set_axisbelow(True)
    style(axes)
    axes.set_xlabel("AUROC: hack replies vs honest replies (bootstrap over pairs, 95%)", fontsize=8, color=SECONDARY_INK)
    axes.set_title("Separating real hack replies from honest ones", fontsize=9, color=INK, loc="left")
    axes.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0, -0.09))
    figure.tight_layout()
    save(figure, out, "reply_auroc")


def pair_gi_vs_accuracy(entries: list[ModelEntry], geometries: dict[str, dict], out: Path) -> None:
    figure, axes = plt.subplots(figsize=(7, 5))
    placed: list[tuple[float, float, str]] = []
    for model_index, entry in enumerate(entries):
        geometry, layer = geometries[entry["label"]], entry["layer"]
        accuracy_by_pair = geometry["pair_sets"][pair_set_name(entry)]["letter_spirit_pairs"]
        for name, pair in geometry["letter_spirit_pairs"].items():
            x = pair["cosine_with_gi_v2"][layer]
            y = accuracy_by_pair[name]["all_pair_accuracy"][str(layer)]
            axes.scatter([x], [y], s=48, color=MODEL_COLORS[model_index], edgecolors=SURFACE, linewidths=1.5, zorder=3)
            short = "jobsworth" if name.startswith("jobsworth") else label(name).removeprefix("pair: ")
            placed.append((x, y, short))
        axes.scatter([], [], s=48, color=MODEL_COLORS[model_index], label=f"{entry['label']} (layer {layer}, {pair_set_name(entry)})")
    axes.axhline(0.5, color=MUTED_INK, linewidth=0.8)
    axes.axvline(0, color=MUTED_INK, linewidth=0.8)
    axes.set_ylim(0, 1.04)
    annotate_without_overlap(axes, figure, placed)
    axes.grid(color=GRIDLINE, linewidth=0.5)
    axes.set_axisbelow(True)
    style(axes)
    axes.set_xlabel("pair direction's cosine with gi_v2", fontsize=8, color=SECONDARY_INK)
    axes.set_ylabel("share of real pairs where the hack reply projects higher", fontsize=8, color=SECONDARY_INK)
    axes.set_title("Letter/spirit pairs: deception loading vs separating real hacks", fontsize=9, color=INK, loc="left")
    axes.legend(frameon=False, fontsize=8, loc="lower left")
    figure.tight_layout()
    save(figure, out, "pair_gi_vs_accuracy")


def annotate_without_overlap(axes: Axes, figure: Figure, points: list[tuple[float, float, str]]) -> None:
    """Each label takes the first nearby slot (right, left, then further above / below) that clears the points and the
    labels placed so far; a label away from its first slot gets a leader line."""
    figure.canvas.draw()
    points_per_pixel = 72 / figure.dpi
    anchors = [tuple(axes.transData.transform((x, y)) * points_per_pixel) for x, y, _ in points]
    occupied: list[tuple[float, float, float, float]] = [(anchor_x - 4, anchor_y - 4, 8, 8) for anchor_x, anchor_y in anchors]
    for (x, y, text), (anchor_x, anchor_y) in sorted(zip(points, anchors), key=lambda item: (-item[0][1], item[0][0])):
        width, height = 4.4 * len(text), 8.0
        slots = [(5.0, 2.0), (-5.0 - width, 2.0), (5.0, -10.0), (-5.0 - width, -10.0)] + [
            (dx, dy) for step in (1, 2, 3, 4) for dx in (5.0, -5.0 - width) for dy in (2.0 + 10 * step, -10.0 - 10 * step)]
        def clear(dx: float, dy: float) -> bool:
            left, bottom = anchor_x + dx, anchor_y + dy
            return not any(left < box_x + box_width and box_x < left + width and bottom < box_y + box_height and box_y < bottom + height
                           for box_x, box_y, box_width, box_height in occupied)
        dx, dy = next((slot for slot in slots if clear(*slot)), slots[0])
        occupied.append((anchor_x + dx, anchor_y + dy, width, height))
        leader = {"arrowstyle": "-", "color": MUTED_INK, "linewidth": 0.5} if (dx, dy) not in slots[:2] else None
        axes.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=7, color=SECONDARY_INK,
                      ha="left", va="bottom", arrowprops=leader)


def printable(token: str) -> str:
    """Leading spaces show as ␣ (so " pursuant" and "pursuant" stay distinct); other whitespace is escaped."""
    shown = repr(token)[1:-1] if any(character.isspace() and character != " " for character in token) or not token.strip() else token
    return "␣" + shown[1:] if shown.startswith(" ") else shown


def logit_lens(entry: ModelEntry, out: Path) -> None:
    if entry["logit_lens"] is None:
        return
    lens = json.loads(Path(entry["logit_lens"]).read_text())
    null = float(np.quantile(lens["random_extreme_z"]["cosine"], 0.95))
    directions = list(dict.fromkeys(item["direction"] for item in lens["lenses"]))
    layers = sorted({item["layer"] for item in lens["lenses"]})
    figure, panels = plt.subplots(len(directions), len(layers), figsize=(3.3 * len(layers), 3.6 * len(directions)), squeeze=False)
    for item in lens["lenses"]:
        panel = panels[directions.index(item["direction"])][layers.index(item["layer"])]
        ranking = item["tokens_by_ranking"]["cosine"]
        positive, negative = ranking["positive"][:TOP_TOKENS], ranking["negative"][:TOP_TOKENS][::-1]
        values = [token["z"] for token in positive] + [-abs(token["z"]) for token in negative]
        names = [printable(token["token"]) for token in positive + negative]
        rows = np.arange(len(values))
        panel.barh(rows, values, color=["#e34948"] * len(positive) + ["#2a78d6"] * len(negative), height=0.7)
        for side in (null, -null):
            panel.axvline(side, color=INK, linewidth=0.8, linestyle="--")
        panel.set_yticks(rows, names, fontsize=7)
        panel.invert_yaxis()
        panel.axvline(0, color=MUTED_INK, linewidth=0.5)
        style(panel)
        panel.set_title(f"{label(item['direction'])}, layer {item['layer']}", fontsize=8, color=INK, loc="left")
    for panel in panels[-1]:
        panel.set_xlabel("z vs vocabulary (dashed: random-direction 95th pct |z|)", fontsize=7, color=SECONDARY_INK)
    figure.suptitle(f"{entry['label']}: logit lens, top and bottom {TOP_TOKENS} tokens by cosine with the unembedding (␣ = leading space)",
                    fontsize=10, color=INK, x=0.01, ha="left")
    figure.tight_layout(rect=(0, 0, 1, 0.985))
    save(figure, out, f"logit_lens_{entry['label']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    entries: list[ModelEntry] = json.loads(args.manifest.read_text())["models"]
    args.out.mkdir(parents=True, exist_ok=True)
    directions_by_model = {entry["label"]: load_directions(entry["directions"]) for entry in entries}
    geometries = {entry["label"]: json.loads(Path(entry["geometry"]).read_text()) for entry in entries}
    pairs_by_model = {entry["label"]: load_pairs(entry) for entry in entries}
    for entry in entries:
        cosine_heatmap(entry, directions_by_model[entry["label"]], geometries[entry["label"]], args.out)
        logit_lens(entry, args.out)
    for reference_of in ("gi_v2", "chris"):
        cosine_by_layer(entries, directions_by_model, reference_of, args.out)
    pair_projections(entries, directions_by_model, pairs_by_model, args.out)
    reply_auroc(entries, directions_by_model, pairs_by_model, args.out)
    pair_gi_vs_accuracy(entries, geometries, args.out)


if __name__ == "__main__":
    main()
