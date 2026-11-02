"""Geometry of a reward-hacker persona axis against the steered deceiver axis (gi_v2).

    python scripts/rh_axis_geometry.py --vectors-dir <root>/vectors_response_only --gi-axis <gi_v2 persona_contrast.pt>
        --groups data/persona_groups_paper_v2.json --layer 17 --out <dir> [--context-axis <17-role persona_contrast.pt>]

rh_contrast            = mean(deceptive_rh) - mean(honest_rh)
rh_raw                 = mean(deceptive_rh) - default
nonrh_holdout_contrast = mean(deceptive_holdout) - mean(honest_holdout)   (reference: unseen non-RH deception)
nonrh_holdout_raw      = mean(deceptive_holdout) - default
rh_residual            = rh_contrast with the gi_v2 component removed per layer

Writes rh_geometry.json (cosines per layer, per-role and leave-one-out cosines at --layer), the rh_* artifacts in the
build_vectors.py format, and rh_geometry.{png,pdf}.
"""
import argparse
import json
from pathlib import Path
from typing import Literal, TypedDict

import matplotlib
matplotlib.use("Agg")
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from jaxtyping import Float
import torch

from build_vectors import load_axis, load_role, project_out, sha256, unit

GroupName = Literal["deceptive_main", "honest_main", "deceptive_rh", "honest_rh", "deceptive_holdout", "honest_holdout"]
Activations = Float[torch.Tensor, "layer hidden"]
PerLayer = Float[torch.Tensor, "layer"]

COLOR_BY_AXIS = {"rh_contrast": "#eb6834", "rh_raw": "#1baf7a", "nonrh_holdout_contrast": "#2a78d6"}
COLOR_BY_GROUP: dict[GroupName, str] = {"deceptive_rh": "#eb6834", "honest_rh": "#eb6834",
                                        "deceptive_holdout": "#2a78d6", "honest_holdout": "#2a78d6"}
TEXT_COLOR, MUTED_COLOR, SURFACE_COLOR = "#0b0b0b", "#52514e", "#fcfcfb"


class RoleCosine(TypedDict):
    role: str
    group: GroupName
    minus_default: float
    minus_honest_rh: float


class Geometry(TypedDict):
    layer: int
    gi_axis: str
    gi_axis_sha256: str
    vectors_dir: str
    groups: dict[str, list[str]]
    sanity_cosine_recomputed_gi_vs_artifact: float
    sanity_cosine_gi_vs_persona_raw: float
    cosine_with_gi_by_axis: dict[str, list[float]]
    cosine_rh_raw_vs_persona_raw: list[float]
    cosine_rh_contrast_vs_nonrh_holdout_contrast: list[float]
    cosine_with_context_by_axis: dict[str, list[float]] | None
    share_of_rh_contrast_along_gi: float
    role_cosines: list[RoleCosine]
    leave_one_out_rh_contrast_cosine_by_dropped_role: dict[str, float]


def cosine_by_layer(first: Activations, second: Activations) -> PerLayer:
    return (unit(first) * unit(second)).sum(-1)


def load_roles(vectors_dir: Path, roles: list[str]) -> dict[str, Activations]:
    vector_by_role = {role: load_role(vectors_dir, role) for role in roles}
    missing = sorted(role for role, vector in vector_by_role.items() if vector is None)
    if missing:
        raise SystemExit(f"No vector in {vectors_dir} for {missing}")
    return {role: vector for role, vector in vector_by_role.items() if vector is not None}


def mean_of(vector_by_role: dict[str, Activations], roles: list[str]) -> Activations:
    return torch.stack([vector_by_role[role] for role in roles]).mean(0)


def rounded(values: PerLayer) -> list[float]:
    return [round(value, 4) for value in values.tolist()]


def compute_geometry(vector_by_role: dict[str, Activations], roles_by_group: dict[GroupName, list[str]], gi_axis: Activations,
                     context_axis: Activations | None, layer: int) -> tuple[Geometry, dict[str, Activations]]:
    def mean_group(group: GroupName) -> Activations:
        return mean_of(vector_by_role, roles_by_group[group])

    default = vector_by_role["default"]
    honest_rh_mean = mean_group("honest_rh")
    rh_contrast = mean_group("deceptive_rh") - honest_rh_mean
    axis_by_name = {"rh_contrast": rh_contrast,
                    "rh_raw": mean_group("deceptive_rh") - default,
                    "nonrh_holdout_contrast": mean_group("deceptive_holdout") - mean_group("honest_holdout"),
                    "nonrh_holdout_raw": mean_group("deceptive_holdout") - default}
    persona_raw = mean_group("deceptive_main") - default
    recomputed_gi = mean_group("deceptive_main") - mean_group("honest_main")
    role_cosines: list[RoleCosine] = [
        {"role": role, "group": group,
         "minus_default": round(cosine_by_layer(vector_by_role[role] - default, gi_axis)[layer].item(), 4),
         "minus_honest_rh": round(cosine_by_layer(vector_by_role[role] - honest_rh_mean, gi_axis)[layer].item(), 4)}
        for group in ("deceptive_rh", "honest_rh", "deceptive_holdout", "honest_holdout") for role in roles_by_group[group]]
    leave_one_out = {}
    for dropped in roles_by_group["deceptive_rh"] + roles_by_group["honest_rh"]:
        deceptive = [role for role in roles_by_group["deceptive_rh"] if role != dropped]
        honest = [role for role in roles_by_group["honest_rh"] if role != dropped]
        if not deceptive or not honest:
            continue
        contrast = mean_of(vector_by_role, deceptive) - mean_of(vector_by_role, honest)
        leave_one_out[dropped] = round(cosine_by_layer(contrast, gi_axis)[layer].item(), 4)
    geometry: Geometry = {
        "layer": layer, "gi_axis": "", "gi_axis_sha256": "", "vectors_dir": "",
        "groups": {group: roles for group, roles in roles_by_group.items()},
        "sanity_cosine_recomputed_gi_vs_artifact": round(cosine_by_layer(recomputed_gi, gi_axis)[layer].item(), 4),
        "sanity_cosine_gi_vs_persona_raw": round(cosine_by_layer(persona_raw, gi_axis)[layer].item(), 4),
        "cosine_with_gi_by_axis": {name: rounded(cosine_by_layer(axis, gi_axis)) for name, axis in axis_by_name.items()},
        "cosine_rh_raw_vs_persona_raw": rounded(cosine_by_layer(axis_by_name["rh_raw"], persona_raw)),
        "cosine_rh_contrast_vs_nonrh_holdout_contrast": rounded(cosine_by_layer(rh_contrast, axis_by_name["nonrh_holdout_contrast"])),
        "cosine_with_context_by_axis": None if context_axis is None else
            {name: rounded(cosine_by_layer(axis, context_axis)) for name, axis in axis_by_name.items()},
        "share_of_rh_contrast_along_gi": round(cosine_by_layer(rh_contrast, gi_axis)[layer].item() ** 2, 4),
        "role_cosines": role_cosines,
        "leave_one_out_rh_contrast_cosine_by_dropped_role": leave_one_out,
    }
    artifact_by_name = {"rh_contrast": rh_contrast, "rh_raw": axis_by_name["rh_raw"], "rh_residual": project_out(rh_contrast, gi_axis)}
    return geometry, artifact_by_name


def style(axes: Axes) -> None:
    axes.set_facecolor(SURFACE_COLOR)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(MUTED_COLOR)
    axes.tick_params(colors=MUTED_COLOR, labelcolor=TEXT_COLOR)


def spread_end_labels(end_value_by_name: dict[str, float], minimum_gap: float = 0.04) -> dict[str, float]:
    label_y_by_name: dict[str, float] = {}
    previous_y = float("-inf")
    for name, end_value in sorted(end_value_by_name.items(), key=lambda item: item[1]):
        previous_y = max(end_value, previous_y + minimum_gap)
        label_y_by_name[name] = previous_y
    return label_y_by_name


def plot_geometry(geometry: Geometry, model_label: str, path_stem: Path) -> None:
    figure, (layer_axes, role_axes) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.4, 1]}, facecolor=SURFACE_COLOR)
    layer = geometry["layer"]
    for name, color in COLOR_BY_AXIS.items():
        cosines = geometry["cosine_with_gi_by_axis"][name]
        layer_axes.plot(range(len(cosines)), cosines, color=color, linewidth=2, label=name.replace("_", " "))
        layer_axes.scatter([layer], [cosines[layer]], s=40, color=color, edgecolors=SURFACE_COLOR, linewidths=1.5, zorder=3)
    for name, label_y in spread_end_labels({name: geometry["cosine_with_gi_by_axis"][name][-1] for name in COLOR_BY_AXIS}).items():
        layer_axes.text(len(geometry["cosine_with_gi_by_axis"][name]) - 0.6, label_y, name.replace("_", " "), va="center", ha="left",
                        fontsize=8, color=TEXT_COLOR)
    layer_axes.axvline(layer, color=MUTED_COLOR, linewidth=0.8, linestyle=":")
    layer_axes.text(layer, 1.0, f" steer layer {layer}", va="top", ha="left", fontsize=8, color=MUTED_COLOR)
    layer_axes.axhline(0, color=MUTED_COLOR, linewidth=0.8)
    layer_axes.set_ylim(min(0.0, *[min(c) for c in geometry["cosine_with_gi_by_axis"].values()]) - 0.05, 1.0)
    layer_axes.set_xlabel("layer", color=TEXT_COLOR)
    layer_axes.set_ylabel("cosine with gi_v2 deceiver axis", color=TEXT_COLOR)
    layer_axes.legend(frameon=False, fontsize=8, loc="upper left")
    style(layer_axes)

    role_cosines = geometry["role_cosines"]
    for row, role_cosine in enumerate(role_cosines):
        is_honest = role_cosine["group"].startswith("honest")
        color = COLOR_BY_GROUP[role_cosine["group"]]
        role_axes.scatter(role_cosine["minus_default"], row, s=48, color=SURFACE_COLOR if is_honest else color, edgecolors=color,
                          linewidths=1.5, zorder=3)
    role_axes.set_yticks(range(len(role_cosines)), [r["role"].removesuffix("_fic").replace("_", " ") for r in role_cosines], fontsize=8,
                         color=TEXT_COLOR)
    role_axes.invert_yaxis()
    role_axes.axvline(0, color=MUTED_COLOR, linewidth=0.8)
    role_axes.set_xlabel(f"cosine of (persona − default) with gi_v2, layer {layer}", color=TEXT_COLOR)
    handles = [Line2D([], [], marker="o", linestyle="", color=COLOR_BY_GROUP["deceptive_rh"], label="RH-adjacent"),
               Line2D([], [], marker="o", linestyle="", color=COLOR_BY_GROUP["deceptive_holdout"], label="non-RH holdout"),
               Line2D([], [], marker="o", linestyle="", color=MUTED_COLOR, label="filled = deceptive"),
               Line2D([], [], marker="o", linestyle="", markerfacecolor=SURFACE_COLOR, markeredgecolor=MUTED_COLOR, label="open = honest")]
    role_axes.legend(handles=handles, frameon=False, fontsize=8, loc="lower right")
    style(role_axes)
    figure.suptitle(f"{model_label}: reward-hacker persona axes vs the gi_v2 deceiver axis", color=TEXT_COLOR, fontsize=11)
    figure.tight_layout()
    for suffix in (".png", ".pdf"):
        figure.savefig(path_stem.with_suffix(suffix), dpi=200, bbox_inches="tight", facecolor=SURFACE_COLOR)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vectors-dir", type=Path, required=True)
    parser.add_argument("--gi-axis", type=Path, required=True)
    parser.add_argument("--context-axis", type=Path, help="Another axis artifact to report cosines with (e.g. the superseded 17-role gi vector)")
    parser.add_argument("--groups", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--model-label", default="gpt-oss-120b")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    group_names: list[GroupName] = ["deceptive_main", "honest_main", "deceptive_rh", "honest_rh", "deceptive_holdout", "honest_holdout"]
    all_groups = json.loads(args.groups.read_text())
    roles_by_group: dict[GroupName, list[str]] = {group: all_groups[group] for group in group_names}
    vector_by_role = load_roles(args.vectors_dir, ["default", *[role for roles in roles_by_group.values() for role in roles]])
    gi_axis = load_axis(args.gi_axis)
    context_axis = load_axis(args.context_axis) if args.context_axis is not None else None
    geometry, artifact_by_name = compute_geometry(vector_by_role, roles_by_group, gi_axis, context_axis, args.layer)
    geometry["gi_axis"], geometry["gi_axis_sha256"], geometry["vectors_dir"] = str(args.gi_axis), sha256(args.gi_axis), str(args.vectors_dir)

    args.out.mkdir(parents=True, exist_ok=True)
    model = torch.load(args.gi_axis, map_location="cpu", weights_only=False)["model"]
    for name, axis in artifact_by_name.items():
        torch.save({"axis": axis, "model": model, "name": name, "vectors_dir": str(args.vectors_dir), "gi_axis": str(args.gi_axis),
                    "groups": {group: roles_by_group[group] for group in ("deceptive_rh", "honest_rh")}}, args.out / f"{name}.pt")
    (args.out / "rh_geometry.json").write_text(json.dumps(geometry, indent=1))
    plot_geometry(geometry, args.model_label, args.out / "rh_geometry")


if __name__ == "__main__":
    main()
