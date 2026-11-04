"""Geometry of a reward-hacker persona axis against the steered deceiver axis (gi_v2).

    python scripts/rh_axis_geometry.py --vectors-dir <root>/vectors_response_only --gi-axis <gi_v2 persona_contrast.pt>
        --groups data/persona_groups.json --layer 17 --out <dir> [--context-axis <17-role persona_contrast.pt>]
        [--rh-groups data/persona_groups_letter_spirit.json --rh-positive letter --rh-negative spirit]

The RH poles default to deceptive_rh / honest_rh of --groups; --rh-groups adds groups (poles and controls) from another file.
rh_contrast            = mean(rh_positive) - mean(rh_negative)
rh_raw                 = mean(rh_positive) - default
nonrh_holdout_contrast = mean(deceptive_holdout) - mean(honest_holdout)   (reference: unseen non-RH deception)
nonrh_holdout_raw      = mean(deceptive_holdout) - default
rh_residual            = rh_contrast with the gi_v2 component removed per layer

Writes rh_geometry.json (cosines per layer; per-role cosines with gi_v2 and with rh_contrast and leave-one-out cosines at --layer), the rh_* artifacts in the
build_vectors.py format, and rh_geometry.{png,pdf}.
"""
import argparse
import json
from pathlib import Path
from typing import TypedDict

import matplotlib
matplotlib.use("Agg")
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from jaxtyping import Float
import torch

from build_vectors import load_axis, load_role, project_out, sha256, unit

Activations = Float[torch.Tensor, "layer hidden"]
PerLayer = Float[torch.Tensor, "layer"]

COLOR_BY_AXIS = {"rh_contrast": "#eb6834", "rh_raw": "#1baf7a", "nonrh_holdout_contrast": "#2a78d6"}
AXIS_POLE_COLOR, HELD_OUT_RH_COLOR, NON_RH_COLOR = "#eb6834", "#1baf7a", "#2a78d6"
COMPARISON_GROUPS = ["deceptive_main", "honest_main", "deceptive_rh", "honest_rh", "deceptive_holdout", "honest_holdout"]
TEXT_COLOR, MUTED_COLOR, SURFACE_COLOR = "#0b0b0b", "#52514e", "#fcfcfb"


class RoleCosine(TypedDict):
    role: str
    group: str
    is_positive_side: bool
    minus_default: float
    minus_negative_pole: float
    minus_default_with_rh_contrast: float


class Geometry(TypedDict):
    layer: int
    gi_axis: str
    gi_axis_sha256: str
    vectors_dir: str
    groups: dict[str, list[str]]
    rh_positive: str
    rh_negative: str
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


def is_positive_side(group: str, rh_positive: str) -> bool:
    return group == rh_positive or group.startswith(("deceptive", "letter"))


def compute_geometry(vector_by_role: dict[str, Activations], roles_by_group: dict[str, list[str]], rh_positive: str, rh_negative: str,
                     gi_axis: Activations, context_axis: Activations | None, layer: int) -> tuple[Geometry, dict[str, Activations]]:
    def mean_group(group: str) -> Activations:
        return mean_of(vector_by_role, roles_by_group[group])

    def cosine_at_layer(first: Activations, second: Activations) -> float:
        return round(cosine_by_layer(first, second)[layer].item(), 4)

    default = vector_by_role["default"]
    negative_mean = mean_group(rh_negative)
    rh_contrast = mean_group(rh_positive) - negative_mean
    axis_by_name = {"rh_contrast": rh_contrast,
                    "rh_raw": mean_group(rh_positive) - default,
                    "nonrh_holdout_contrast": mean_group("deceptive_holdout") - mean_group("honest_holdout"),
                    "nonrh_holdout_raw": mean_group("deceptive_holdout") - default}
    persona_raw = mean_group("deceptive_main") - default
    recomputed_gi = mean_group("deceptive_main") - mean_group("honest_main")
    listed_groups = list(dict.fromkeys([rh_positive, rh_negative, *[g for g in roles_by_group if g not in COMPARISON_GROUPS],
                                        "deceptive_rh", "honest_rh", "deceptive_holdout", "honest_holdout"]))
    role_cosines: list[RoleCosine] = [
        {"role": role, "group": group, "is_positive_side": is_positive_side(group, rh_positive),
         "minus_default": cosine_at_layer(vector_by_role[role] - default, gi_axis),
         "minus_negative_pole": cosine_at_layer(vector_by_role[role] - negative_mean, gi_axis),
         "minus_default_with_rh_contrast": cosine_at_layer(vector_by_role[role] - default, rh_contrast)}
        for group in listed_groups for role in roles_by_group[group]]
    leave_one_out = {}
    for dropped in roles_by_group[rh_positive] + roles_by_group[rh_negative]:
        positive = [role for role in roles_by_group[rh_positive] if role != dropped]
        negative = [role for role in roles_by_group[rh_negative] if role != dropped]
        if not positive or not negative:
            continue
        leave_one_out[dropped] = cosine_at_layer(mean_of(vector_by_role, positive) - mean_of(vector_by_role, negative), gi_axis)
    geometry: Geometry = {
        "layer": layer, "gi_axis": "", "gi_axis_sha256": "", "vectors_dir": "",
        "groups": roles_by_group, "rh_positive": rh_positive, "rh_negative": rh_negative,
        "sanity_cosine_recomputed_gi_vs_artifact": cosine_at_layer(recomputed_gi, gi_axis),
        "sanity_cosine_gi_vs_persona_raw": cosine_at_layer(persona_raw, gi_axis),
        "cosine_with_gi_by_axis": {name: rounded(cosine_by_layer(axis, gi_axis)) for name, axis in axis_by_name.items()},
        "cosine_rh_raw_vs_persona_raw": rounded(cosine_by_layer(axis_by_name["rh_raw"], persona_raw)),
        "cosine_rh_contrast_vs_nonrh_holdout_contrast": rounded(cosine_by_layer(rh_contrast, axis_by_name["nonrh_holdout_contrast"])),
        "cosine_with_context_by_axis": None if context_axis is None else
            {name: rounded(cosine_by_layer(axis, context_axis)) for name, axis in axis_by_name.items()},
        "share_of_rh_contrast_along_gi": round(cosine_at_layer(rh_contrast, gi_axis) ** 2, 4),
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


def role_color(group: str, geometry: Geometry) -> str:
    if group in (geometry["rh_positive"], geometry["rh_negative"]):
        return AXIS_POLE_COLOR
    if group in ("deceptive_rh", "honest_rh"):
        return HELD_OUT_RH_COLOR
    if group in ("deceptive_holdout", "honest_holdout"):
        return NON_RH_COLOR
    return MUTED_COLOR


def plot_role_cosines(axes: Axes, geometry: Geometry, field: str, xlabel: str, show_labels: bool) -> None:
    role_cosines = geometry["role_cosines"]
    for row, role_cosine in enumerate(role_cosines):
        color = role_color(role_cosine["group"], geometry)
        value = role_cosine["minus_default"] if field == "minus_default" else role_cosine["minus_default_with_rh_contrast"]
        axes.scatter(value, row, s=48, marker="s" if color == MUTED_COLOR else "o", color=color if role_cosine["is_positive_side"] else SURFACE_COLOR,
                     edgecolors=color, linewidths=1.5, zorder=3)
    axes.set_yticks(range(len(role_cosines)), [r["role"].removesuffix("_fic").replace("_", " ") for r in role_cosines] if show_labels else [],
                    fontsize=8, color=TEXT_COLOR)
    axes.set_ylim(len(role_cosines) - 0.5, -0.5)
    axes.axvline(0, color=MUTED_COLOR, linewidth=0.8)
    axes.set_xlabel(xlabel, color=TEXT_COLOR)
    style(axes)


def plot_geometry(geometry: Geometry, model_label: str, path_stem: Path) -> None:
    role_count = len(geometry["role_cosines"])
    figure, (layer_axes, gi_role_axes, rh_role_axes) = plt.subplots(1, 3, figsize=(17, max(4.6, 0.22 * role_count + 1)),
                                                                   gridspec_kw={"width_ratios": [1.4, 1, 1]}, facecolor=SURFACE_COLOR)
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

    plot_role_cosines(gi_role_axes, geometry, "minus_default", f"cosine of (persona − default) with gi_v2, layer {layer}", show_labels=True)
    plot_role_cosines(rh_role_axes, geometry, "minus_default_with_rh_contrast", f"cosine of (persona − default) with rh_contrast, layer {layer}",
                      show_labels=False)
    handles = [Line2D([], [], marker="o", linestyle="", color=AXIS_POLE_COLOR, label=f"RH axis poles ({geometry['rh_positive']} / {geometry['rh_negative']})"),
               Line2D([], [], marker="o", linestyle="", color=HELD_OUT_RH_COLOR, label="earlier RH-adjacent roles"),
               Line2D([], [], marker="o", linestyle="", color=NON_RH_COLOR, label="non-RH holdout"),
               Line2D([], [], marker="s", linestyle="", color=MUTED_COLOR, label="controls"),
               Line2D([], [], marker="o", linestyle="", color=TEXT_COLOR, label="filled = letter / deceptive side"),
               Line2D([], [], marker="o", linestyle="", markerfacecolor=SURFACE_COLOR, markeredgecolor=TEXT_COLOR, label="open = spirit / honest side")]
    figure.legend(handles=handles, frameon=False, fontsize=8, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.04))
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
    parser.add_argument("--rh-groups", type=Path, help="Extra groups file (RH poles and controls), merged into --groups")
    parser.add_argument("--rh-positive", default="deceptive_rh")
    parser.add_argument("--rh-negative", default="honest_rh")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--model-label", default="gpt-oss-120b")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    comparison_groups = json.loads(args.groups.read_text())
    rh_groups = json.loads(args.rh_groups.read_text()) if args.rh_groups is not None else {}
    roles_by_group: dict[str, list[str]] = {**{group: comparison_groups[group] for group in COMPARISON_GROUPS}, **rh_groups}
    vector_by_role = load_roles(args.vectors_dir, ["default", *[role for roles in roles_by_group.values() for role in roles]])
    gi_axis = load_axis(args.gi_axis)
    context_axis = load_axis(args.context_axis) if args.context_axis is not None else None
    geometry, artifact_by_name = compute_geometry(vector_by_role, roles_by_group, args.rh_positive, args.rh_negative, gi_axis, context_axis, args.layer)
    geometry["gi_axis"], geometry["gi_axis_sha256"], geometry["vectors_dir"] = str(args.gi_axis), sha256(args.gi_axis), str(args.vectors_dir)

    args.out.mkdir(parents=True, exist_ok=True)
    model = torch.load(args.gi_axis, map_location="cpu", weights_only=False)["model"]
    for name, axis in artifact_by_name.items():
        torch.save({"axis": axis, "model": model, "name": name, "vectors_dir": str(args.vectors_dir), "gi_axis": str(args.gi_axis),
                    "groups": {group: roles_by_group[group] for group in (args.rh_positive, args.rh_negative)}}, args.out / f"{name}.pt")
    (args.out / "rh_geometry.json").write_text(json.dumps(geometry, indent=1))
    plot_geometry(geometry, args.model_label, args.out / "rh_geometry")


if __name__ == "__main__":
    main()
