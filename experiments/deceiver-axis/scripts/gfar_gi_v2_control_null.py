"""gfar against gi_v2 at one layer, read against a chosen subset of the antonym control directions. (CPU)

Cosine: |cos(gfar, gi_v2)| as a percentile among |cos(gi_v2, control)| and |cos(gfar, control)|.
Activation correlation (episode level: one token-mean residual per episode and segment, from `axis_acts_extract.py`):
|r| of the two projections as a percentile among |r(gi_v2, control)| and |r(gfar, control)|, raw and within problem.
Percentile = share of controls with a smaller absolute value, as in axis_acts_analysis.py; also the percentile among the
|value| of every control-to-control pair.

    python experiments/deceiver-axis/scripts/gfar_gi_v2_control_null.py --acts <gfar_vs_gi_v2>/acts/muse --layer 26 \
        --gi-v2 <persona_contrast.pt> --gfar <directions_headline.pt>:goodfire_all_response_pl_all_families \
        --control-dirs <antonym_controls/dirs_isolated> --control-glob 'antonym_p*.pt' --out <dir>/muse.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TypedDict

import numpy as np
import torch
from jaxtyping import Float

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hack_separation import SEGMENTS, Segment, load_episodes, load_unit_direction  # noqa: E402


class AbsoluteValueAgainstControls(TypedDict):
    value: float
    percentile_among_gi_v2_controls: float
    percentile_among_gfar_controls: float
    gi_v2_controls_abs_median: float
    gfar_controls_abs_median: float
    percentile_among_control_pairs: float
    control_pairs_abs_median: float


class SegmentCorrelation(TypedDict):
    episode: AbsoluteValueAgainstControls
    within_problem: AbsoluteValueAgainstControls


class ControlNullResult(TypedDict):
    layer: int
    control_files: list[str]
    episodes: int
    cosine: AbsoluteValueAgainstControls
    correlation_by_segment: dict[Segment, SegmentCorrelation]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--acts", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--gi-v2", required=True, help="path[:key] of the gi_v2 direction")
    parser.add_argument("--gfar", required=True, help="path[:key] of the gfar direction")
    parser.add_argument("--control-dirs", type=Path, required=True)
    parser.add_argument("--control-glob", default="*.pt")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def load_control_units(control_files: list[Path], layer: int) -> Float[np.ndarray, "controls hidden"]:
    axes = np.stack([torch.load(path, weights_only=False, map_location="cpu")["axis"][layer].float().numpy() for path in control_files])
    return axes / np.linalg.norm(axes, axis=1, keepdims=True)


def pearson_with_columns(values: Float[np.ndarray, "episodes"], columns: Float[np.ndarray, "episodes columns"]) -> Float[np.ndarray, "columns"]:
    centered_values = values - values.mean()
    centered_columns = columns - columns.mean(0)
    return (centered_values @ centered_columns) / (np.linalg.norm(centered_values) * np.linalg.norm(centered_columns, axis=0))


def demean_within_problem(values: Float[np.ndarray, "episodes ..."], problem_ids: list[str]) -> Float[np.ndarray, "episodes ..."]:
    problem_index_by_id = {problem_id: index for index, problem_id in enumerate(sorted(set(problem_ids)))}
    onehot = np.zeros((len(problem_ids), len(problem_index_by_id)))
    onehot[np.arange(len(problem_ids)), [problem_index_by_id[problem_id] for problem_id in problem_ids]] = 1.0
    problem_means = (onehot.T @ values) / onehot.sum(0).reshape(-1, *([1] * (values.ndim - 1)))
    return values - onehot @ problem_means


def upper_triangle(pairwise: Float[np.ndarray, "controls controls"]) -> Float[np.ndarray, "pairs"]:
    return pairwise[np.triu_indices(pairwise.shape[0], 1)]


def against_controls(value: float, gi_v2_controls: Float[np.ndarray, "controls"], gfar_controls: Float[np.ndarray, "controls"],
                     control_pairs: Float[np.ndarray, "pairs"]) -> AbsoluteValueAgainstControls:
    return {
        "value": value,
        "percentile_among_gi_v2_controls": float((np.abs(gi_v2_controls) < abs(value)).mean()),
        "percentile_among_gfar_controls": float((np.abs(gfar_controls) < abs(value)).mean()),
        "gi_v2_controls_abs_median": float(np.median(np.abs(gi_v2_controls))),
        "gfar_controls_abs_median": float(np.median(np.abs(gfar_controls))),
        "percentile_among_control_pairs": float((np.abs(control_pairs) < abs(value)).mean()),
        "control_pairs_abs_median": float(np.median(np.abs(control_pairs))),
    }


def correlation_against_controls(projections_by_name: dict[str, Float[np.ndarray, "episodes"]],
                                 control_projections: Float[np.ndarray, "episodes controls"]) -> AbsoluteValueAgainstControls:
    gi_v2, gfar = projections_by_name["gi_v2"], projections_by_name["gfar"]
    return against_controls(float(pearson_with_columns(gi_v2, gfar[:, None])[0]),
                            pearson_with_columns(gi_v2, control_projections), pearson_with_columns(gfar, control_projections),
                            upper_triangle(np.corrcoef(control_projections.T)))


def main() -> None:
    args = parse_args()
    control_files = sorted(args.control_dirs.glob(args.control_glob))
    control_units = load_control_units(control_files, args.layer)
    _, gi_v2_unit = load_unit_direction(f"gi_v2={args.gi_v2}", args.layer)
    _, gfar_unit = load_unit_direction(f"gfar={args.gfar}", args.layer)
    episodes = load_episodes(args.acts, args.layer)

    correlation_by_segment: dict[Segment, SegmentCorrelation] = {}
    for segment in SEGMENTS:
        means = episodes["means_by_segment"][segment]
        has_tokens = np.abs(means).sum(1) > 0
        means = means[has_tokens]
        problem_ids = [problem_id for problem_id, keep in zip(episodes["problem_ids"], has_tokens) if keep]
        projections_by_name = {"gi_v2": means @ gi_v2_unit, "gfar": means @ gfar_unit}
        control_projections = means @ control_units.T
        correlation_by_segment[segment] = {
            "episode": correlation_against_controls(projections_by_name, control_projections),
            "within_problem": correlation_against_controls(
                {name: demean_within_problem(projection, problem_ids) for name, projection in projections_by_name.items()},
                demean_within_problem(control_projections, problem_ids)),
        }

    result: ControlNullResult = {
        "layer": args.layer,
        "control_files": [path.name for path in control_files],
        "episodes": len(episodes["problem_ids"]),
        "cosine": against_controls(float(gi_v2_unit @ gfar_unit), control_units @ gi_v2_unit, control_units @ gfar_unit,
                                   upper_triangle(control_units @ control_units.T)),
        "correlation_by_segment": correlation_by_segment,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
