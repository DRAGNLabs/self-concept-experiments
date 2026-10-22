"""Figures for the write-up "Can the Assistant Axis or Schemer Axis predict reward hacking?".

Recomputes the per-random-direction AUROCs on the layer-17 region-mean cache (the same 256 isotropic directions,
seed 1729, problem/family/turn strata as direction_analysis) so the named directions can be drawn against the
full random-direction distribution, and reads the two Phase 2 analysis.json files for the CI forest plot.
The role-compliance figure uses the judge counts recorded in FINDINGS.md.

    python experiments/schemer-axis/scripts/post_figures.py --out experiments/schemer-axis/post/figures
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from selfconcept.assistant_axis.projection import load_unit_axes
from selfconcept.correlation.analyze import binary_outcome
from selfconcept.correlation.direction_analysis import orthogonalize
from selfconcept.correlation.random_vector_analysis import group_indices, load_cache, pair_totals, project
from selfconcept.correlation.random_vectors import REGIONS, random_directions

HOME = Path.home()
SCRATCH = HOME / "nobackup/autodelete/self-concept-experiments/schemer-axis/roles_fic"
CACHE = HOME / "nobackup/autodelete/rh-random-control-v3/full"
AA = Path("/grphome/grp_mirror/nobackup/autodelete/self-concept-experiments/assistant-axis/gptoss12b-thinking/axis_response_only.py")
LAYER = 17
COLORS = {"aa": "#b5473a", "cot_contrast": "#2b6ca3", "fin_contrast_orth": "#3a9a5b", "ceiling": "#666666"}
LABELS = {"aa": "Assistant Axis", "cot_contrast": "Schemer contrast (reasoning tokens)",
          "fin_contrast_orth": "Schemer contrast (answer tokens) ⊥ AA", "ceiling": "held-out diff-of-means ceiling"}


def stratified_auroc(scores, labels, strata, problems):
    informative = [i for i in group_indices(strata).values() if len(np.unique(labels[i])) == 2]
    keep = np.sort(np.concatenate(informative))
    u, pairs, _ = pair_totals(scores[keep], labels[keep], [strata[i] for i in keep], [problems[i] for i in keep])
    return u.sum(axis=0) / pairs.sum()


def random_control_figure(out: Path, random_count: int, seed: int) -> dict:
    records, manifest = load_cache([d for d in sorted(CACHE.glob("shard_*")) if d.is_dir()])
    identity = manifest["identity"]
    hidden, num_layers = manifest["hidden_size"], manifest["num_layers"]
    named = {
        "aa": AA,
        "cot_contrast": SCRATCH / "candidates_with_cot_vs_default/persona_contrast.pt",
        "fin_contrast_orth": SCRATCH / "candidates_response_only_vs_default/persona_contrast_orth.pt",
    }
    axes = {k: load_unit_axes(p, [LAYER], num_layers, hidden, identity["model"])[LAYER]["direction"] for k, p in named.items()}
    # fin_contrast_orth was orthogonalized by build_vectors; re-orthogonalize against the AA for safety (no-op if already).
    axes["fin_contrast_orth"] = orthogonalize(axes["fin_contrast_orth"], axes["aa"])
    completed = [(d, r) for d, r in records if r["status"] == "complete" and binary_outcome(r["outcome"])[0] is not None]
    rows = [r for _, r in completed]
    labels = np.asarray([binary_outcome(r["outcome"])[0] for r in rows])
    problems = [json.dumps([r["scenario"], r["outcome"]["problem"]]) for r in rows]
    strata = [r["outcome"]["stratum"] for r in rows]
    randoms = random_directions(hidden, random_count, seed, LAYER)
    columns = np.vstack([*(axes[k] for k in named), randoms])
    matrices = np.stack([np.load(d / r["residuals"])[f"layer_{LAYER}"] for d, r in completed])
    results = {}
    fig, panels = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for panel, (region_index, region) in zip(panels, enumerate(REGIONS)):
        residuals = matrices[:, region_index, :]
        valid = np.isfinite(residuals).all(axis=1)
        idx = np.flatnonzero(valid)
        scores = project(residuals[idx], columns)
        aucs = stratified_auroc(scores, labels[idx], [strata[i] for i in idx], [problems[i] for i in idx])
        named_aucs = dict(zip(named, aucs[: len(named)]))
        random_aucs = aucs[len(named):]
        results[region] = {"named": {k: float(v) for k, v in named_aucs.items()},
                           "random_quantiles": np.quantile(random_aucs, [.025, .5, .975]).tolist(),
                           "random": random_aucs.tolist()}
        panel.hist(random_aucs, bins=np.linspace(0.1, 0.9, 33), color="#c9d3df", edgecolor="white", label=f"{random_count} random directions")
        for key in named:
            panel.axvline(named_aucs[key], color=COLORS[key], lw=2, label=LABELS[key])
        panel.axvline(0.5, color="black", lw=0.8, ls=":")
        panel.set_title({"prompt": "Prompt tokens", "cot": "Reasoning tokens", "final": "Answer tokens"}[region])
        panel.set_xlabel("stratified AUROC, hack vs non-hack")
    panels[0].set_ylabel("random directions")
    handles, labels_ = panels[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("gpt-oss-120b, layer 17, 824 episodes (189 hacks), hacks compared with non-hacks of the same problem, family and turn", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "fig_random_controls.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return results


def forest_figure(out: Path) -> None:
    cells = {}
    for name, path in (("cot", SCRATCH / "phase2_with_cot/analysis.json"), ("fin", SCRATCH / "phase2_response_only/analysis.json")):
        data = json.load(open(path))
        cells[name] = next(c for c in data["cells"] if c["region"] == "cot" and c["scenario"] == "all" and c["grouping"] == "problem_family_turn")
    rows = [
        ("Assistant Axis", cells["cot"]["directions"]["aa"], COLORS["aa"]),
        ("Schemer contrast, reasoning tokens (primary)", cells["cot"]["directions"]["persona_contrast"], COLORS["cot_contrast"]),
        ("  same, ⊥ AA", cells["cot"]["directions"]["persona_contrast|orth_aa"], COLORS["cot_contrast"]),
        ("Schemer minus default, reasoning tokens", cells["cot"]["directions"]["persona_raw"], "#7fa3c7"),
        ("Schemer contrast, answer tokens", cells["fin"]["directions"]["persona_contrast"], "#7fc49a"),
        ("  same, ⊥ AA (secondary)", cells["fin"]["directions"]["persona_contrast|orth_aa"], COLORS["fin_contrast_orth"]),
        ("Schemer minus default, answer tokens", cells["fin"]["directions"]["persona_raw"], "#a9d9bb"),
    ]
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    for i, (label, d, color) in enumerate(rows):
        lo, hi = d["problem_bootstrap_ci"]
        v = d["auroc"]["value"]
        ax.plot([lo, hi], [i, i], color=color, lw=2)
        ax.plot(v, i, "o", color=color, ms=7)
        ax.text(0.98, i, f"tail {d['auroc']['random_direction_tail_fraction']:.2f}", va="center", ha="right", fontsize=8, color="#444")
    q = cells["cot"]["directions"]["aa"]["auroc"]["random_quantiles"]
    ax.axvspan(q[0], q[2], color="#eeeeee", zorder=0, label="random directions, 2.5–97.5%")
    ax.axvline(0.5, color="black", lw=0.8, ls=":")
    ax.axvline(cells["cot"]["held_out_diff_means_auroc"], color=COLORS["ceiling"], lw=1, ls="--", label="held-out diff-of-means ceiling")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0.2, 1.0)
    ax.set_xlabel("stratified AUROC on reasoning tokens, 95% whole-problem bootstrap CI")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "fig_auroc_forest.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def compliance_figure(out: Path) -> None:
    # Judge score-3 ("fully in role") counts. Smoke: 15 responses per role (3 questions x 5 prompts). Full: 200.
    smoke = [("cheater", "bare persona", 0), ("deceiver", "bare persona", 0), ("schemer", "bare persona", 1), ("strategist", "bare persona", 0),
             ("cheater", "named character", 11), ("deceiver", "named character", 12), ("schemer", "named character", 15)]
    full = [("schemer", 193), ("deceiver", 151), ("cheater", 147), ("straight shooter", 178), ("rule follower", 195), ("neutral", 198)]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.4), gridspec_kw={"width_ratios": [4, 3]})
    names = ["cheater", "deceiver", "schemer", "strategist"]
    x = np.arange(len(names))
    bare = [next((s[2] for s in smoke if s[0] == n and s[1] == "bare persona"), 0) for n in names]
    fic = [next((s[2] for s in smoke if s[0] == n and s[1] == "named character"), np.nan) for n in names]
    a.bar(x - 0.2, bare, 0.4, color="#b5473a", label='bare "You are a …" system prompt')
    a.bar(x + 0.2, fic, 0.4, color="#2b6ca3", label="named fictional character")
    for xi, v in zip(x, bare):
        a.text(xi - 0.2, v + 0.2, str(v), ha="center", fontsize=8, color="#b5473a")
    a.set_xticks(x)
    a.set_xticklabels(names)
    a.set_ylim(0, 15)
    a.set_ylabel("responses judged fully in role (of 15)")
    a.set_title("Framing smoke: the model refuses bare personas", fontsize=10)
    a.legend(fontsize=8, frameon=False, loc="upper left")
    b.barh([f[0] for f in full][::-1], [f[1] for f in full][::-1], color=["#9fb7cc"] * 3 + ["#2b6ca3"] * 3)
    b.axvline(50, color="black", ls=":", lw=0.8)
    b.set_xlim(0, 200)
    b.set_xlabel("responses judged fully in role (of 200)")
    b.set_title("Full extraction, fiction-framed roles (gate: 50)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "fig_role_compliance.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--random-count", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    compliance_figure(args.out)
    forest_figure(args.out)
    results = random_control_figure(args.out, args.random_count, args.seed)
    (args.out / "random_controls.json").write_text(json.dumps(results, indent=1))
    for region, r in results.items():
        print(region, {k: round(v, 3) for k, v in r["named"].items()}, "random 2.5/50/97.5:", [round(q, 3) for q in r["random_quantiles"]])


if __name__ == "__main__":
    main()
