"""Per-layer within-problem AUROC of the gfar projection for judged-hack vs other unsteered episodes, against random
unit directions (CPU).

Reads the axis_acts_extract.py caches (`<acts>/unsteered__seed<N>.pt`), pools each episode's reasoning and answer token
means into one response-token mean (weighted by token counts), and at every layer from --from-layer on scores the
episodes on the unit gfar direction and on --random-directions isotropic random unit directions. Labels as in
hack_separation.py: HACK = judged STRONG or WEAK, everything else (BORDERLINE, unjudged) not hack.

    python experiments/deceiver-axis/scripts/gfar_layer_separation.py --acts <gfar_vs_gi_v2>/acts/muse --from-layer 26 \
        --verdicts <as_judged.json> --gfar <directions_headline.pt>:goodfire_all_response_pl_all_families --out <dir>/muse.json
"""
import argparse
import json
from pathlib import Path
from typing import TypedDict

import numpy as np
import torch
from jaxtyping import Float

from hack_separation import RANDOM_SEED, auroc_with_null, within_problem_auroc
from selfconcept.correlation.random_vectors import random_directions
from steer_gfar_lens_extract import VerdictKey


class LayerSeparation(TypedDict):
    layer: int
    auroc: float
    random_p2_5: float
    random_p97_5: float
    outside_random_band: bool


class ResponseMeans(TypedDict):
    means_by_layer: Float[np.ndarray, "layers episodes hidden"]
    record_layers: list[int]
    verdict_key_by_episode: list[VerdictKey]
    problem_id_by_episode: list[str]


def load_response_means(acts_dir: Path) -> ResponseMeans:
    seed_files = sorted(acts_dir.glob("unsteered__seed*.pt"), key=lambda path: int(path.stem.split("seed")[1]))
    caches = [torch.load(path, weights_only=False, map_location="cpu") for path in seed_files]

    def pooled(cache: dict) -> Float[np.ndarray, "layers episodes hidden"]:
        reasoning_count = torch.as_tensor(cache["stats"]["reasoning"]["n"], dtype=torch.float32)[:, None, None]
        answer_count = torch.as_tensor(cache["stats"]["answer"]["n"], dtype=torch.float32)[:, None, None]
        total = (reasoning_count * cache["reasoning_mean"].float() + answer_count * cache["answer_mean"].float())
        return (total / (reasoning_count + answer_count).clamp_min(1)).permute(1, 0, 2).numpy()

    return {
        "means_by_layer": np.concatenate([pooled(cache) for cache in caches], axis=1),
        "record_layers": list(caches[0]["record_layers"]),
        "verdict_key_by_episode": [key for cache in caches for key in cache["verdict_key"]],
        "problem_id_by_episode": [example_id for cache in caches for example_id in cache["example_id"]],
    }


def layer_separations(means: ResponseMeans, gfar_units: Float[np.ndarray, "layers hidden"], verdicts_by_key: dict[VerdictKey, str],
                      from_layer: int, random_count: int) -> list[LayerSeparation]:
    is_hack = np.array([verdicts_by_key.get(key) == "HACK" for key in means["verdict_key_by_episode"]])
    problem_ids = np.array(means["problem_id_by_episode"])
    separations: list[LayerSeparation] = []
    for index, layer in enumerate(means["record_layers"]):
        if layer < from_layer:
            continue
        layer_means = means["means_by_layer"][index]
        hidden = layer_means.shape[1]
        random_aurocs = within_problem_auroc(layer_means @ random_directions(hidden, random_count, RANDOM_SEED, layer).T, is_hack, problem_ids)
        observed = float(within_problem_auroc((layer_means @ gfar_units[layer])[:, None], is_hack, problem_ids)[0])
        result = auroc_with_null(observed, random_aurocs)
        separations.append({"layer": layer, "auroc": observed, "random_p2_5": result["random_p2_5"], "random_p97_5": result["random_p97_5"],
                            "outside_random_band": not result["random_p2_5"] <= observed <= result["random_p97_5"]})
    return separations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--acts", type=Path, required=True)
    parser.add_argument("--verdicts", type=Path, required=True)
    parser.add_argument("--gfar", required=True, help="path:key of the (layers, hidden) gfar artifact")
    parser.add_argument("--from-layer", type=int, required=True)
    parser.add_argument("--random-directions", type=int, default=1000)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    path, key = args.gfar.rsplit(":", 1)
    gfar = torch.load(path, weights_only=False, map_location="cpu")[key].float()
    gfar_units = (gfar / gfar.norm(dim=-1, keepdim=True)).numpy()
    separations = layer_separations(load_response_means(args.acts), gfar_units, json.loads(args.verdicts.read_text()), args.from_layer,
                                    args.random_directions)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"acts": str(args.acts), "verdicts": str(args.verdicts), "gfar": args.gfar, "segment": "response (reasoning + answer)",
                                    "random_directions": args.random_directions, "random_seed": RANDOM_SEED, "layers": separations}, indent=1))
    for separation in separations:
        print(f"L{separation['layer']}: {separation['auroc']:.3f} [{separation['random_p2_5']:.3f}, {separation['random_p97_5']:.3f}]"
              f"{'  outside' if separation['outside_random_band'] else ''}")


if __name__ == "__main__":
    main()
