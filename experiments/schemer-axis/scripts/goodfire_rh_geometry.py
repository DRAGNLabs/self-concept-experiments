"""Goodfire's synthetic reward-hacking directions (difference of means over matched hack/honest conversations) against
our directions.

    python scripts/goodfire_rh_geometry.py --readouts <goodfire_readouts.pt> --contrasts <contrasts.json>
        --reference-directions <chris_rh_axis/geometry/directions.pt> --out <dir>

Readouts come from goodfire_readout_vectors.py. For each readout (close, last, all_response) and each contrast in
their contrasts.json, plus `pl_all_families` (every hack cell against honest across the four released families), the
direction is the mean over train tasks of (mean hack cells - mean comparison cells) within the task. Per direction and
layer: AUROC on their validation split, split-half reliability over train tasks, and its cosine with every reference
direction against a sign-flip null (each task's difference flipped at random, 95th percentile |cosine|). With
--pair-vectors and --layers, also its AUROC on Chris's real hack / honest replies, with two nulls: the percentile
among isotropic random directions, and the share of the sign-flipped (shuffled-label) directions that do as well.

Also runs on our own minimal-harness pairs (synthetic_rh_tasks.py; --prefix ours --headline pl_all), where
pl_all_families does not apply. Writes goodfire_rh_geometry.json, directions_headline.pt (the --headline contrast per
readout) and
directions_contrasts_<readout>.pt (every contrast), each name -> (layer, hidden).
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Literal, TypedDict

import torch
from jaxtyping import Float

from build_vectors import unit

Readout = Literal["close", "last", "all_response", "tool_calls", "report"]
READOUTS: tuple[Readout, ...] = ("close", "last", "all_response", "tool_calls", "report")
CONVERSATION_READOUTS: tuple[Readout, ...] = ("all_response", "tool_calls", "report")
FAMILY_CONTRASTS = ("pl_suite", "pl_validator", "pl_scorefile", "pl_acquire")
N_SIGN_FLIPS = 1000
N_SPLIT_HALVES = 100
N_RANDOM_DIRECTIONS = 2000


class Contrast(TypedDict):
    dataset: str
    span: str
    pos: list[str]
    neg: list[str]


class Membership(TypedDict):
    index: int
    group: str
    split: str
    is_hack: bool


def label_members(rows: list[dict], contrast_name: str) -> list[Membership]:
    """close / last rows: Goodfire's own readout labels say which contrasts a readout belongs to and on which side."""
    return [{"index": index, "group": row["group_id"], "split": row["split"], "is_hack": bool(row["labels"][contrast_name])}
            for index, row in enumerate(rows) if contrast_name in row["labels"]]


def all_families_label_members(rows: list[dict]) -> list[Membership]:
    return [{"index": index, "group": row["group_id"], "split": row["split"],
             "is_hack": any(row["labels"][name] for name in FAMILY_CONTRASTS if name in row["labels"])}
            for index, row in enumerate(rows) if any(name in row["labels"] for name in FAMILY_CONTRASTS)]


def cell_members(rows: list[dict], contrasts: list[Contrast]) -> list[Membership]:
    """all_response rows (one per conversation): membership by dataset and cell."""
    return [{"index": index, "group": row["group_id"], "split": row["split"], "is_hack": row["cell"] in contrast["pos"]}
            for index, row in enumerate(rows)
            for contrast in contrasts if row["dataset"] == contrast["dataset"] and row["cell"] in contrast["pos"] + contrast["neg"]]


def group_differences(activations: torch.Tensor, members: list[Membership], split: str) -> Float[torch.Tensor, "group layer hidden"]:
    sides: dict[str, dict[bool, list[int]]] = defaultdict(lambda: {True: [], False: []})
    for member in members:
        if member["split"] == split:
            sides[member["group"]][member["is_hack"]].append(member["index"])
    complete = [group for group in sides.values() if group[True] and group[False]]
    return torch.stack([activations[group[True]].float().mean(0) - activations[group[False]].float().mean(0) for group in complete])


def auroc_by_layer(activations: torch.Tensor, members: list[Membership], direction: Float[torch.Tensor, "layer hidden"]) -> list[float]:
    validation = [member for member in members if member["split"] == "validation"]
    projections = torch.einsum("nlh,lh->nl", activations[[member["index"] for member in validation]].float(), unit(direction))
    is_hack = torch.tensor([member["is_hack"] for member in validation])
    margins = projections[is_hack][:, None] - projections[~is_hack][None, :]
    return ((margins > 0).float() + 0.5 * (margins == 0).float()).mean((0, 1)).tolist()


def cosine_by_layer(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    return (unit(first) * unit(second)).sum(-1)


class RealPairs(TypedDict):
    """Chris's real hack / honest replies at the scored layers, with the AUROC of isotropic random directions."""
    hack: Float[torch.Tensor, "pair layer hidden"]
    honest: Float[torch.Tensor, "pair layer hidden"]
    random_aurocs: Float[torch.Tensor, "layer random"]


def pair_aurocs(hack: Float[torch.Tensor, "pair hidden"], honest: Float[torch.Tensor, "pair hidden"],
                axes: Float[torch.Tensor, "axis hidden"]) -> Float[torch.Tensor, "axis"]:
    """P(a hack reply projects above an honest reply), over every hack x honest reply combination."""
    hack_projections, honest_projections = hack @ unit(axes).T, honest @ unit(axes).T
    return ((hack_projections[:, None] - honest_projections[None]) > 0).float().mean((0, 1))


def load_real_pairs(path: Path, layers: list[int], generator: torch.Generator) -> RealPairs:
    pair_vectors = torch.load(path, map_location="cpu", weights_only=False)
    rows, activations = pair_vectors["rows"], pair_vectors["activations"][:, layers].float()
    index_by_key = {(row["pair_id"], row["condition"]): i for i, row in enumerate(rows)}
    pair_ids = sorted({row["pair_id"] for row in rows})
    hack = torch.stack([activations[index_by_key[(pair_id, "hack")]] for pair_id in pair_ids])
    honest = torch.stack([activations[index_by_key[(pair_id, "honest")]] for pair_id in pair_ids])
    random_axes = torch.randn(N_RANDOM_DIRECTIONS, hack.shape[-1], generator=generator)
    return {"hack": hack, "honest": honest,
            "random_aurocs": torch.stack([pair_aurocs(hack[:, i], honest[:, i], random_axes) for i in range(len(layers))])}


def real_pair_separation(direction: torch.Tensor, null_directions: torch.Tensor, real_pairs: dict[str, RealPairs], layers: list[int]) -> dict:
    """Per pair set and layer: AUROC on the real pairs, its percentile among random directions, and the share of
    shuffled-label (sign-flipped) directions from the same synthetic tasks that separate at least as well."""
    report: dict = {}
    for name, pairs in real_pairs.items():
        report[name] = {}
        for i, layer in enumerate(layers):
            observed = pair_aurocs(pairs["hack"][:, i], pairs["honest"][:, i], direction[layer][None])[0]
            shuffled = pair_aurocs(pairs["hack"][:, i], pairs["honest"][:, i], null_directions[:, layer])
            report[name][layer] = {"auroc": observed.item(),
                                   "random_direction_percentile": (pairs["random_aurocs"][i] < observed).float().mean().item(),
                                   "shuffled_label_share_at_least": (shuffled >= observed).float().mean().item()}
    return report


def summarize(differences: Float[torch.Tensor, "group layer hidden"], activations: torch.Tensor, members: list[Membership],
              references: dict[str, torch.Tensor], real_pairs: dict[str, RealPairs], layers: list[int],
              generator: torch.Generator) -> tuple[torch.Tensor, dict]:
    direction = differences.mean(0)
    signs = torch.randint(0, 2, (N_SIGN_FLIPS, differences.shape[0]), generator=generator).float() * 2 - 1
    null_directions = torch.einsum("rg,glh->rlh", signs, differences) / differences.shape[0]
    halves = [torch.randperm(differences.shape[0], generator=generator) for _ in range(N_SPLIT_HALVES)]
    middle = differences.shape[0] // 2
    return direction, {
        "n_train_tasks": differences.shape[0],
        "validation_auroc": auroc_by_layer(activations, members, direction),
        "split_half_over_tasks_mean": torch.stack([cosine_by_layer(differences[order[:middle]].mean(0), differences[order[middle:]].mean(0))
                                                   for order in halves]).mean(0).tolist(),
        "cosine_with_reference": {name: cosine_by_layer(direction, reference).tolist() for name, reference in references.items()},
        "sign_flip_null_abs_cosine_95th": {
            name: torch.quantile(cosine_by_layer(null_directions, reference.expand_as(null_directions)).abs(), 0.95, dim=0).tolist()
            for name, reference in references.items()},
        "real_pairs": real_pair_separation(direction, null_directions, real_pairs, layers),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--readouts", type=Path, required=True)
    parser.add_argument("--contrasts", type=Path, required=True)
    parser.add_argument("--reference-directions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prefix", default="goodfire", help="direction names are <prefix>_<readout>_<contrast>")
    parser.add_argument("--pair-vectors", type=Path, nargs="*", default=[], help="Chris's pair vectors (pair_reply_vectors.py)")
    parser.add_argument("--layers", type=int, nargs="*", default=[], help="layers at which the real pairs are scored")
    parser.add_argument("--only-readouts", nargs="*", default=[], help="analyze only these readouts (default: all present)")
    parser.add_argument("--headline", default="pl_all_families", help="the contrast saved to directions_headline.pt")
    args = parser.parse_args()

    data = torch.load(args.readouts, map_location="cpu", weights_only=False)
    contrasts: dict[str, Contrast] = json.loads(args.contrasts.read_text())
    references = {name: vector.float() for name, vector in torch.load(args.reference_directions, map_location="cpu").items()}
    generator = torch.Generator().manual_seed(0)
    real_pairs = {path.stem: load_real_pairs(path, args.layers, generator) for path in args.pair_vectors}
    report: dict = {"readouts": str(args.readouts), "model": data["model"], "by_readout": {}}
    headline: dict[str, torch.Tensor] = {}
    args.out.mkdir(parents=True, exist_ok=True)
    for readout in [name for name in READOUTS if name in data and (not args.only_readouts or name in args.only_readouts)]:
        activations = data[readout]
        if readout in CONVERSATION_READOUTS:
            rows = data["conversation_rows"]
            members_by_contrast = {name: cell_members(rows, [contrast]) for name, contrast in contrasts.items()}
            if all(name in contrasts for name in FAMILY_CONTRASTS):
                members_by_contrast["pl_all_families"] = cell_members(rows, [contrasts[name] for name in FAMILY_CONTRASTS])
        else:
            rows = data["readout_rows"]
            members_by_contrast = {name: label_members(rows, name) for name in contrasts}
            if all(name in contrasts for name in FAMILY_CONTRASTS):
                members_by_contrast["pl_all_families"] = all_families_label_members(rows)
        directions: dict[str, torch.Tensor] = {}
        report["by_readout"][readout] = {}
        for name, members in members_by_contrast.items():
            direction, summary = summarize(group_differences(activations, members, "train"), activations, members, references,
                                           real_pairs, args.layers, generator)
            directions[f"{args.prefix}_{readout}_{name}"] = direction
            report["by_readout"][readout][name] = summary
            print(f"{readout:12} {name:16} tasks {summary['n_train_tasks']:4}  max validation AUROC {max(summary['validation_auroc']):.3f}")
        headline[f"{args.prefix}_{readout}_{args.headline}"] = directions[f"{args.prefix}_{readout}_{args.headline}"]
        torch.save(directions, args.out / f"directions_contrasts_{readout}.pt")
    torch.save(headline, args.out / "directions_headline.pt")
    (args.out / "goodfire_rh_geometry.json").write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
