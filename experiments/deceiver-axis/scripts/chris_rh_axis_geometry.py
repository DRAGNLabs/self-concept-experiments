"""Chris's reply-pair reward-hack axis (origin/Chris, experiments/reward-hack-axis) against our persona axes.

    python scripts/chris_rh_axis_geometry.py --pair-vectors <dir>/same_prompt_pairs.pt <dir>/stagec_gptoss_pairs.pt
        --vectors-dir <root>/vectors_response_only --gi-axis <gi_v2 persona_contrast.pt> --context-axis <17-role persona_contrast.pt>
        --groups data/persona_groups.json --letter-spirit-groups data/persona_groups_letter_spirit.json --out <dir>

Pair vectors come from pair_reply_vectors.py. His axis is mean(train hack) - mean(train honest), as in his extract_response_mean.py.
Per pair set: cosine with each persona axis by layer, a sign-flip null (each train pair's hack/honest labels flipped at random),
split-half reliability over problems, holdout pair accuracy (hack reply projects above its honest twin) and AUROC over all
replies for every axis, and each persona's cosine (role - default) with his axis. Per matched letter/spirit pair (and the
jobsworth control): cosine with gi_v2, with the letter-vs-spirit axis (all pairs and without that pair) and with his axis,
and its pair accuracy. --vectors-dir may list several roots (first hit wins; default comes from the first); persona axes
whose roles are missing are skipped. Writes chris_rh_axis_geometry.json, chris_rh_axis_geometry.{png,pdf} and
directions.pt (every direction above, name -> (layer, hidden)).
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from jaxtyping import Float
import torch

from build_vectors import load_axis, load_role, unit

Activations = Float[torch.Tensor, "layer hidden"]
PerLayer = Float[torch.Tensor, "layer"]

N_RESAMPLES = 2000
COLOR_BY_PERSONA_AXIS = {"gi_v2": "#2a78d6", "letter_vs_spirit": "#eb6834", "rh_personas_7v3": "#1baf7a", "nonrh_holdout": "#8a5cd6"}
TEXT_COLOR, MUTED_COLOR = "#0b0b0b", "#52514e"


def find_role(vectors_dirs: list[Path], role: str) -> Activations | None:
    return next((vector for vector in (load_role(vectors_dir, role) for vectors_dir in vectors_dirs) if vector is not None), None)


def mean_of(vectors_dirs: list[Path], roles: list[str]) -> Activations | None:
    vectors = [find_role(vectors_dirs, role) for role in roles]
    if any(vector is None for vector in vectors):
        return None
    return torch.stack([vector for vector in vectors if vector is not None]).mean(0)


def contrast(vectors_dirs: list[Path], positive: list[str], negative: list[str]) -> Activations | None:
    positive_mean, negative_mean = mean_of(vectors_dirs, positive), mean_of(vectors_dirs, negative)
    return None if positive_mean is None or negative_mean is None else positive_mean - negative_mean


def cosine_by_layer(first: Activations, second: Activations) -> PerLayer:
    return (unit(first) * unit(second)).sum(-1)


def pair_differences(pair_vectors: dict, split: str | None) -> tuple[Float[torch.Tensor, "pair layer hidden"], list[str]]:
    rows, activations = pair_vectors["rows"], pair_vectors["activations"]
    index_by_key = {(row["pair_id"], row["condition"]): i for i, row in enumerate(rows)}
    pair_ids = sorted({row["pair_id"] for row in rows if split is None or row["split"] == split})
    differences = torch.stack([activations[index_by_key[(pair_id, "hack")]] - activations[index_by_key[(pair_id, "honest")]] for pair_id in pair_ids])
    task_by_pair = {row["pair_id"]: row["task_id"] for row in rows}
    return differences, [task_by_pair[pair_id] for pair_id in pair_ids]


def pair_accuracy(differences: Float[torch.Tensor, "pair layer hidden"], axis: Activations, layer: int) -> float:
    return ((differences[:, layer] @ axis[layer]) > 0).float().mean().item()


def reply_auroc(pair_vectors: dict, axis: Activations, layer: int) -> float:
    """P(a hack reply projects above an honest reply), over every hack x honest reply combination."""
    projections = pair_vectors["activations"][:, layer] @ axis[layer]
    is_hack = torch.tensor([row["condition"] == "hack" for row in pair_vectors["rows"]])
    margins = projections[is_hack][:, None] - projections[~is_hack][None, :]
    return ((margins > 0).float() + 0.5 * (margins == 0).float()).mean().item()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pair-vectors", type=Path, nargs="+", required=True)
    parser.add_argument("--vectors-dir", type=Path, nargs="+", required=True, help="Role vector roots, searched in order")
    parser.add_argument("--gi-axis", type=Path, required=True)
    parser.add_argument("--context-axis", type=Path, required=True)
    parser.add_argument("--groups", type=Path, required=True)
    parser.add_argument("--letter-spirit-groups", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=[17, 18])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    groups = json.loads(args.groups.read_text()) | json.loads(args.letter_spirit_groups.read_text())
    default = load_role(args.vectors_dir[0], "default")
    if default is None:
        raise SystemExit(f"No default vector in {args.vectors_dir[0]}")
    contrasts = {
        "letter_vs_spirit": contrast(args.vectors_dir, groups["letter"], groups["spirit"]),
        "rh_personas_7v3": contrast(args.vectors_dir, groups["deceptive_rh"], groups["honest_rh"]),
        "rh_narrow": contrast(args.vectors_dir, ["metric_gamer_fic", "rules_lawyer_fic"], ["spirit_keeper_fic"]),
        "nonrh_holdout": contrast(args.vectors_dir, groups["deceptive_holdout"], groups["honest_holdout"]),
        "jobsworth_vs_discretionary": contrast(args.vectors_dir, groups["letter_control"], groups["spirit_control"]),
    }
    persona_axes: dict[str, Activations] = {"gi_v2": load_axis(args.gi_axis)} | {key: axis for key, axis in contrasts.items() if axis is not None}
    persona_axes["gi_17role"] = load_axis(args.context_axis)
    group_by_role = {role: name for name, members in groups.items() if not name.endswith("_v1") for role in members}
    vector_by_role = {role: find_role(args.vectors_dir, role) for role in sorted(group_by_role)}
    role_offsets = {role: vector - default for role, vector in vector_by_role.items() if vector is not None}
    axis_pairs = list(zip(groups["letter"], groups["spirit"]))
    letter_spirit_pairs = axis_pairs + list(zip(groups["letter_control"], groups["spirit_control"]))
    pair_axes = {f"{letter}-{spirit}": axis for letter, spirit in letter_spirit_pairs if (axis := contrast(args.vectors_dir, [letter], [spirit])) is not None}
    without_pair = {f"{letter}-{spirit}": contrast(args.vectors_dir, [other for other, _ in axis_pairs if other != letter], [other for _, other in axis_pairs if other != spirit])
                    for letter, spirit in letter_spirit_pairs}
    generator = torch.Generator().manual_seed(0)

    loaded = {path.stem: torch.load(path, map_location="cpu", weights_only=False) for path in args.pair_vectors}
    chris_axes = {name: pair_differences(pair_vectors, "train")[0].mean(0) for name, pair_vectors in loaded.items()}
    report: dict = {
        "layers": args.layers, "vectors_dirs": [str(path) for path in args.vectors_dir],
        "skipped_persona_axes": sorted(key for key, axis in contrasts.items() if axis is None),
        "missing_roles": sorted(role for role, vector in vector_by_role.items() if vector is None),
        "persona_axis_cosine_with_gi_v2": {key: cosine_by_layer(axis, persona_axes["gi_v2"]).tolist() for key, axis in persona_axes.items()},
        "letter_spirit_pairs": {name: {
            "cosine_with_gi_v2": cosine_by_layer(axis, persona_axes["gi_v2"]).tolist(),
            "cosine_with_letter_vs_spirit": cosine_by_layer(axis, persona_axes["letter_vs_spirit"]).tolist() if "letter_vs_spirit" in persona_axes else None,
            "cosine_with_letter_vs_spirit_without_this_pair": cosine_by_layer(axis, rest).tolist() if (rest := without_pair[name]) is not None else None,
        } for name, axis in pair_axes.items()},
        "cosine_between_chris_axes": {}, "pair_sets": {},
    }
    names = list(chris_axes)
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            report["cosine_between_chris_axes"][f"{first}|{second}"] = cosine_by_layer(chris_axes[first], chris_axes[second]).tolist()

    for name, pair_vectors in loaded.items():
        train, train_tasks = pair_differences(pair_vectors, "train")
        holdout, _ = pair_differences(pair_vectors, "holdout")
        everything, _ = pair_differences(pair_vectors, None)
        axis = chris_axes[name]
        signs = torch.randint(0, 2, (N_RESAMPLES, train.shape[0]), generator=generator).float() * 2 - 1
        null_axes = torch.einsum("rp,plh->rlh", signs, train) / train.shape[0]
        tasks = sorted(set(train_tasks))
        split_half: list[PerLayer] = []
        for _ in range(200):
            chosen = {tasks[j] for j in torch.randperm(len(tasks), generator=generator)[: len(tasks) // 2].tolist()}
            in_half = torch.tensor([task in chosen for task in train_tasks])
            split_half.append(cosine_by_layer(train[in_half].mean(0), train[~in_half].mean(0)))
        units = unit(train)
        mutual = torch.einsum("plh,qlh->lpq", units, units)
        off_diagonal = ~torch.eye(train.shape[0], dtype=torch.bool)
        report["pair_sets"][name] = {
            "n_train_pairs": train.shape[0], "n_holdout_pairs": holdout.shape[0], "n_train_problems": len(tasks),
            "cosine_with_persona_axis": {key: cosine_by_layer(axis, persona).tolist() for key, persona in persona_axes.items()},
            "sign_flip_null_abs_cosine_95th_with_persona_axis": {
                key: torch.quantile(cosine_by_layer(null_axes, persona.expand_as(null_axes)).abs(), 0.95, dim=0).tolist() for key, persona in persona_axes.items()},
            "split_half_over_problems_mean": torch.stack(split_half).mean(0).tolist(),
            "pair_difference_mutual_cosine_mean": mutual[:, off_diagonal].mean(-1).tolist(),
            "holdout_pair_accuracy": {key: {layer: pair_accuracy(holdout, persona, layer) for layer in args.layers}
                                      for key, persona in persona_axes.items() | {"chris_axis": axis}.items()},
            "all_pair_accuracy_persona_axes": {key: {layer: pair_accuracy(everything, persona, layer) for layer in args.layers} for key, persona in persona_axes.items()},
            "all_reply_auroc": {key: {layer: reply_auroc(pair_vectors, persona, layer) for layer in args.layers}
                                for key, persona in persona_axes.items() | {"chris_axis_in_sample": axis}.items()},
            "letter_spirit_pairs": {pair_name: {"cosine_with_chris_axis": cosine_by_layer(pair_axis, axis).tolist(),
                                                "all_pair_accuracy": {layer: pair_accuracy(everything, pair_axis, layer) for layer in args.layers}}
                                    for pair_name, pair_axis in pair_axes.items()},
            "role_cosine_with_chris_axis": {layer: sorted(
                ({"role": role, "group": group_by_role.get(role, "?"), "cosine": cosine_by_layer(offset, axis)[layer].item()} for role, offset in role_offsets.items()),
                key=lambda entry: -entry["cosine"]) for layer in args.layers},
        }

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "chris_rh_axis_geometry.json").write_text(json.dumps(report, indent=1) + "\n")
    torch.save(persona_axes | pair_axes | {f"chris_{name}": axis for name, axis in chris_axes.items()}, args.out / "directions.pt")
    figure, panels = plt.subplots(1, len(loaded), figsize=(6 * len(loaded), 4), sharey=True, squeeze=False)
    for panel, (name, entry) in zip(panels[0], report["pair_sets"].items()):
        for key, color in COLOR_BY_PERSONA_AXIS.items():
            if key not in entry["cosine_with_persona_axis"]:
                continue
            panel.plot(entry["cosine_with_persona_axis"][key], color=color, linewidth=2, label=key)
        panel.plot(entry["sign_flip_null_abs_cosine_95th_with_persona_axis"]["gi_v2"], color=MUTED_COLOR, linewidth=1, linestyle="--", label="sign-flip null 95th (gi_v2)")
        for layer in args.layers:
            panel.axvline(layer, color=MUTED_COLOR, linewidth=0.5)
        panel.axhline(0, color=MUTED_COLOR, linewidth=0.5)
        panel.set_title(f"{name}: {entry['n_train_pairs']} train pairs", color=TEXT_COLOR)
        panel.set_xlabel("layer")
        for side in ("top", "right"):
            panel.spines[side].set_visible(False)
    panels[0][0].set_ylabel("cosine with Chris's hack - honest axis")
    handles, labels = panels[0][0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=5, frameon=False)
    figure.tight_layout(rect=(0, 0.1, 1, 1))
    for suffix in ("png", "pdf"):
        figure.savefig(args.out / f"chris_rh_axis_geometry.{suffix}", dpi=200)


if __name__ == "__main__":
    main()
