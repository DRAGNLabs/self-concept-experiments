"""Chris's reply-pair reward-hack axis (origin/Chris, experiments/reward-hack-axis) against our persona axes.

    python scripts/chris_rh_axis_geometry.py --pair-vectors <dir>/same_prompt_pairs.pt <dir>/stagec_gptoss_pairs.pt
        --vectors-dir <root>/vectors_response_only --gi-axis <gi_v2 persona_contrast.pt> --context-axis <17-role persona_contrast.pt>
        --groups data/persona_groups.json --letter-spirit-groups data/persona_groups_letter_spirit.json --out <dir>

Pair vectors come from pair_reply_vectors.py. His axis is mean(train hack) - mean(train honest), as in his extract_response_mean.py.
Per pair set: cosine with each persona axis by layer, a sign-flip null (each train pair's hack/honest labels flipped at random),
split-half reliability over problems, holdout pair accuracy (hack reply projects above its honest twin) for every axis, and
each persona's cosine (role - default) with his axis. Writes chris_rh_axis_geometry.json and chris_rh_axis_geometry.{png,pdf}.
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


def mean_of(vectors_dir: Path, roles: list[str]) -> Activations:
    vectors = [load_role(vectors_dir, role) for role in roles]
    missing = [role for role, vector in zip(roles, vectors) if vector is None]
    if missing:
        raise SystemExit(f"No vector in {vectors_dir} for {missing}")
    return torch.stack([vector for vector in vectors if vector is not None]).mean(0)


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pair-vectors", type=Path, nargs="+", required=True)
    parser.add_argument("--vectors-dir", type=Path, required=True)
    parser.add_argument("--gi-axis", type=Path, required=True)
    parser.add_argument("--context-axis", type=Path, required=True)
    parser.add_argument("--groups", type=Path, required=True)
    parser.add_argument("--letter-spirit-groups", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=[17, 18])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    groups = json.loads(args.groups.read_text()) | json.loads(args.letter_spirit_groups.read_text())
    default = mean_of(args.vectors_dir, ["default"])
    persona_axes: dict[str, Activations] = {
        "gi_v2": load_axis(args.gi_axis),
        "letter_vs_spirit": mean_of(args.vectors_dir, groups["letter"]) - mean_of(args.vectors_dir, groups["spirit"]),
        "rh_personas_7v3": mean_of(args.vectors_dir, groups["deceptive_rh"]) - mean_of(args.vectors_dir, groups["honest_rh"]),
        "rh_narrow": mean_of(args.vectors_dir, ["metric_gamer_fic", "rules_lawyer_fic"]) - mean_of(args.vectors_dir, ["spirit_keeper_fic"]),
        "nonrh_holdout": mean_of(args.vectors_dir, groups["deceptive_holdout"]) - mean_of(args.vectors_dir, groups["honest_holdout"]),
        "jobsworth_vs_discretionary": mean_of(args.vectors_dir, groups["letter_control"]) - mean_of(args.vectors_dir, groups["spirit_control"]),
        "gi_17role": load_axis(args.context_axis),
    }
    group_by_role = {role: name for name, members in groups.items() if not name.endswith("_v1") for role in members}
    role_offsets = {role: mean_of(args.vectors_dir, [role]) - default for role in sorted(group_by_role)}
    generator = torch.Generator().manual_seed(0)

    loaded = {path.stem: torch.load(path, map_location="cpu", weights_only=False) for path in args.pair_vectors}
    chris_axes = {name: pair_differences(pair_vectors, "train")[0].mean(0) for name, pair_vectors in loaded.items()}
    report: dict = {"layers": args.layers, "cosine_between_chris_axes": {}, "pair_sets": {}}
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
            "role_cosine_with_chris_axis": {layer: sorted(
                ({"role": role, "group": group_by_role.get(role, "?"), "cosine": cosine_by_layer(offset, axis)[layer].item()} for role, offset in role_offsets.items()),
                key=lambda entry: -entry["cosine"]) for layer in args.layers},
        }

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "chris_rh_axis_geometry.json").write_text(json.dumps(report, indent=1) + "\n")
    figure, panels = plt.subplots(1, len(loaded), figsize=(6 * len(loaded), 4), sharey=True, squeeze=False)
    for panel, (name, entry) in zip(panels[0], report["pair_sets"].items()):
        for key, color in COLOR_BY_PERSONA_AXIS.items():
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
