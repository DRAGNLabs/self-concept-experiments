"""Do named directions separate judged-hack from non-hack unsteered episodes better than random directions? (CPU)

Reads the per-episode token means cached by `axis_acts_extract.py` (`<acts>/unsteered__seed<N>.pt`, reasoning and answer
tokens at every layer), labels each episode HACK (judged STRONG or WEAK) or not from a verdicts JSON keyed
'<arm>/seed<s>/<example_id>' (unjudged episodes count as not HACK), projects the means at one layer on each named
direction and on isotropic random unit directions, and reports the AUROC of the projection (higher = hack) pooled and
within problem, against the random directions' 2.5-97.5 percentile range. The BORDERLINE-excluded rows drop BORDERLINE
episodes from both classes.

    python experiments/deceiver-axis/scripts/hack_separation.py --acts <gfar_vs_gi_v2>/acts/muse --layer 26 \
        --verdicts <verdicts>/as_judged.json --direction gi_v2=<persona_contrast.pt> \
        --direction gfar=<directions_headline.pt>:goodfire_all_response_pl_all_families --out <dir>/muse.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Literal, TypedDict

import numpy as np
import torch
from jaxtyping import Bool, Float
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from selfconcept.correlation.random_vectors import random_directions  # noqa: E402

Segment = Literal["reasoning", "answer"]
BorderlineHandling = Literal["counted_not_hack", "excluded"]
SEGMENTS: tuple[Segment, ...] = ("reasoning", "answer")
RANDOM_SEED = 1729


class Episodes(TypedDict):
    means_by_segment: dict[Segment, Float[np.ndarray, "episodes hidden"]]
    verdict_keys: list[str]
    problem_ids: list[str]


class AurocWithNull(TypedDict):
    auroc: float
    random_p2_5: float
    random_p97_5: float
    random_percentile: float


class SeparationRow(TypedDict):
    direction: str
    segment: Segment
    borderline: BorderlineHandling
    hacks: int
    non_hacks: int
    pooled: AurocWithNull
    within_problem: AurocWithNull


def load_episodes(acts_dir: Path, layer: int) -> Episodes:
    seed_files = sorted(acts_dir.glob("unsteered__seed*.pt"), key=lambda path: int(path.stem.split("seed")[1]))
    caches = [torch.load(path, weights_only=False, map_location="cpu") for path in seed_files]
    layer_index = caches[0]["record_layers"].index(layer)
    return {
        "means_by_segment": {segment: torch.cat([cache[f"{segment}_mean"][:, layer_index].float() for cache in caches]).numpy()
                             for segment in SEGMENTS},
        "verdict_keys": [key for cache in caches for key in cache["verdict_key"]],
        "problem_ids": [example_id for cache in caches for example_id in cache["example_id"]],
    }


def load_unit_direction(spec: str, layer: int) -> tuple[str, Float[np.ndarray, "hidden"]]:
    name, location = spec.split("=", 1)
    path, _, key = location.partition(":")
    artifact = torch.load(path, weights_only=False, map_location="cpu")
    direction = (artifact[key] if key else artifact["axis"])[layer].float().numpy()
    return name, direction / np.linalg.norm(direction)


def mann_whitney_concordance(scores: Float[np.ndarray, "episodes directions"], is_hack: Bool[np.ndarray, "episodes"]
                             ) -> tuple[Float[np.ndarray, "directions"], int]:
    """Per direction, the number of (hack, non-hack) pairs in which the hack scores higher (ties count half), and the pair count."""
    hacks = int(is_hack.sum())
    hack_rank_sums = rankdata(scores, axis=0)[is_hack].sum(axis=0)
    return hack_rank_sums - hacks * (hacks + 1) / 2, hacks * int((~is_hack).sum())


def pooled_auroc(scores: Float[np.ndarray, "episodes directions"], is_hack: Bool[np.ndarray, "episodes"]) -> Float[np.ndarray, "directions"]:
    concordant_pairs, total_pairs = mann_whitney_concordance(scores, is_hack)
    return concordant_pairs / total_pairs


def within_problem_auroc(scores: Float[np.ndarray, "episodes directions"], is_hack: Bool[np.ndarray, "episodes"],
                         problem_ids: np.ndarray) -> Float[np.ndarray, "directions"]:
    """Share of (hack, non-hack) pairs from the same problem in which the hack scores higher; ties count half."""
    concordance_by_problem = [mann_whitney_concordance(scores[problem_ids == problem_id], is_hack[problem_ids == problem_id])
                              for problem_id in np.unique(problem_ids)]
    return np.sum([concordant for concordant, _ in concordance_by_problem], axis=0) / sum(pairs for _, pairs in concordance_by_problem)


def auroc_with_null(observed: float, random_aurocs: Float[np.ndarray, "random"]) -> AurocWithNull:
    return {"auroc": observed, "random_p2_5": float(np.percentile(random_aurocs, 2.5)),
            "random_p97_5": float(np.percentile(random_aurocs, 97.5)),
            "random_percentile": float(100 * np.mean(random_aurocs < observed) + 50 * np.mean(random_aurocs == observed))}


def separation_rows(episodes: Episodes, verdicts_by_key: dict[str, str], unit_directions_by_name: dict[str, np.ndarray],
                    random_unit_directions: Float[np.ndarray, "random hidden"]) -> list[SeparationRow]:
    verdicts = np.array([verdicts_by_key.get(key, "UNJUDGED") for key in episodes["verdict_keys"]])
    problem_ids = np.array(episodes["problem_ids"])
    rows: list[SeparationRow] = []
    kept_by_borderline_handling: dict[BorderlineHandling, Bool[np.ndarray, "episodes"]] = {
        "counted_not_hack": np.ones(len(verdicts), dtype=bool), "excluded": verdicts != "BORDERLINE"}
    for borderline, keep in kept_by_borderline_handling.items():
        is_hack = verdicts[keep] == "HACK"
        for segment in SEGMENTS:
            means = episodes["means_by_segment"][segment][keep]
            random_scores = means @ random_unit_directions.T
            random_pooled = pooled_auroc(random_scores, is_hack)
            random_within = within_problem_auroc(random_scores, is_hack, problem_ids[keep])
            for name, unit_direction in unit_directions_by_name.items():
                scores = (means @ unit_direction)[:, None]
                rows.append({"direction": name, "segment": segment, "borderline": borderline,
                             "hacks": int(is_hack.sum()), "non_hacks": int((~is_hack).sum()),
                             "pooled": auroc_with_null(float(pooled_auroc(scores, is_hack)[0]), random_pooled),
                             "within_problem": auroc_with_null(float(within_problem_auroc(scores, is_hack, problem_ids[keep])[0]), random_within)})
    return rows


def format_row(row: SeparationRow) -> str:
    def cell(result: AurocWithNull) -> str:
        return f"{result['auroc']:.3f} [{result['random_p2_5']:.3f}, {result['random_p97_5']:.3f}] {result['random_percentile']:5.1f}%"
    return (f"{row['direction']:>8} {row['segment']:>9} {row['borderline']:>16} {row['hacks']:>4}/{row['non_hacks']:<5} "
            f"pooled {cell(row['pooled'])}   within {cell(row['within_problem'])}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--acts", type=Path, required=True, help="directory of unsteered__seed<N>.pt caches")
    parser.add_argument("--verdicts", type=Path, required=True, help="as_judged.json keyed '<arm>/seed<s>/<example_id>'")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--direction", action="append", required=True, help="name=path[:key]; (layers, hidden) artifact")
    parser.add_argument("--random-directions", type=int, default=1000)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    episodes = load_episodes(args.acts, args.layer)
    unit_directions_by_name = dict(load_unit_direction(spec, args.layer) for spec in args.direction)
    hidden_size = episodes["means_by_segment"]["reasoning"].shape[1]
    random_unit_directions = random_directions(hidden_size, args.random_directions, RANDOM_SEED, args.layer)
    rows = separation_rows(episodes, json.loads(args.verdicts.read_text()), unit_directions_by_name, random_unit_directions)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"acts": str(args.acts), "verdicts": str(args.verdicts), "layer": args.layer,
                                    "directions": args.direction, "random_directions": args.random_directions,
                                    "random_seed": RANDOM_SEED, "episodes": len(episodes["verdict_keys"]), "rows": rows}, indent=1))
    print("AUROC [random 2.5-97.5 pct] percentile among random")
    for row in rows:
        print(format_row(row))


if __name__ == "__main__":
    main()
