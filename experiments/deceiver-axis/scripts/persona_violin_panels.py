"""Three-panel persona violin chart: per-response projections on an axis, deceptive next to honest personas, one panel
per persona set (in the axis | held out | reward-hacking-adjacent), shared y-axis.

Redesign of persona_figures.py's `<prefix>_distributions_no_controls.png` (same data: in-role responses to the held-out
questions, projection at one layer minus the default assistant's mean on the same question). CPU only.

Per-response values come from an extraction root (`act_response_only/<role>.pt`, `scores/<role>.json`) and, for roles
the root does not hold, from a directory of per-token projections written by token_projections.py (`<role>.jsonl`; the
mean over a response's tokens is the projection of its mean activation). Each source is referenced to its own default.

    python scripts/persona_violin_panels.py --root <extraction root> --axis <persona_contrast.pt> --layer 17 \
        [--tokens <tokens_L17 dir>] --model-label gpt-oss-120b --out <png> [--note "<extra note line>"]

Writes the PNG and `<png stem>.json` (per-persona n / median / quartiles, per-panel AUROC, sources).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Patch
from scipy.stats import rankdata

INK, SOFT, GRID = "#1f1e1c", "#52514e", "#dddbd5"
COLOR = {"deceptive": "#eb6834", "honest": "#2a78d6"}
HERE = Path(__file__).resolve().parent.parent
PANELS = (  # (title, deceptive group, honest group)
    ("Personas the axis was built from", "deceptive_main", "honest_main"),
    ("Held-out personas", "deceptive_holdout", "honest_holdout"),
    ("Reward-hacking-adjacent (held out)", "deceptive_rh", "honest_rh"),
)
GROUP_GAP = 0.9  # blank columns between the deceptive and the honest violins of a panel


def question_of(key: str) -> int:
    return int(key.rsplit("_q", 1)[1])


def relative(values: dict[str, float], default_by_question: dict[int, float], heldout: set[int]) -> np.ndarray:
    return np.array([v - default_by_question[question_of(k)] for k, v in values.items() if question_of(k) in heldout])


def question_means(values: dict[str, float]) -> dict[int, float]:
    by_question: dict[int, list[float]] = {}
    for key, value in values.items():
        by_question.setdefault(question_of(key), []).append(value)
    return {question: float(np.mean(v)) for question, v in by_question.items()}


def from_activations(root: Path, role: str, unit: torch.Tensor, layer: int) -> dict[str, float] | None:
    path = root / "act_response_only" / f"{role}.pt"
    if not path.exists():
        return None
    activations = torch.load(path, map_location="cpu", weights_only=True)
    keys = list(activations)
    if role != "default":  # personas keep only the responses the role judge scored 3 (fully in role)
        scores = json.loads((root / "scores" / f"{role}.json").read_text())
        keys = [key for key in keys if scores.get(key) == 3]
    return {key: float(activations[key][layer].float() @ unit) for key in keys}


def from_tokens(tokens: Path, role: str) -> dict[str, float] | None:
    path = tokens / f"{role}.jsonl"
    if not path.exists():
        return None
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {row["key"]: float(np.mean(row["scores_by_region"]["response_only"])) for row in rows}


def auroc(positive: np.ndarray, negative: np.ndarray) -> float:
    ranks = rankdata(np.concatenate([positive, negative]))
    return float((ranks[: len(positive)].sum() - len(positive) * (len(positive) + 1) / 2) / (len(positive) * len(negative)))


def label_of(role: str) -> str:
    return role.removesuffix("_fic").replace("_v2", "").replace("_", " ")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--axis", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--tokens", type=Path, default=None)
    parser.add_argument("--groups", type=Path, default=HERE / "data/persona_groups.json")
    parser.add_argument("--questions", type=Path, default=HERE / "data/questions.jsonl")
    parser.add_argument("--model-label", default="")
    parser.add_argument("--axis-label", default="Deceiver Axis")
    parser.add_argument("--note", default="")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    groups = json.loads(args.groups.read_text())
    questions = [json.loads(line) for line in args.questions.read_text().splitlines() if line.strip()]
    heldout = {q["id"] for q in questions if q["split"] == "heldout"}
    axis = torch.load(args.axis, map_location="cpu", weights_only=True)
    axis = axis["axis"] if isinstance(axis, dict) else axis
    unit = axis[args.layer].float()
    unit = unit / unit.norm()

    root_default = from_activations(args.root, "default", unit, args.layer)
    token_default = from_tokens(args.tokens, "default") if args.tokens else None
    values: dict[str, np.ndarray] = {}
    source: dict[str, str] = {}
    for _, deceptive_group, honest_group in PANELS:
        for role in groups[deceptive_group] + groups[honest_group]:
            if root_default is not None and (raw := from_activations(args.root, role, unit, args.layer)) is not None:
                values[role], source[role] = relative(raw, question_means(root_default), heldout), "activations"
            elif token_default is not None and (raw := from_tokens(args.tokens, role)) is not None:
                values[role], source[role] = relative(raw, question_means(token_default), heldout), "tokens"
            else:
                raise SystemExit(f"no per-response data for {role}")

    widths = [len(groups[d]) + len(groups[h]) + GROUP_GAP for _, d, h in PANELS]
    figure, axes_row = plt.subplots(1, len(PANELS), figsize=(11.0, 5.9), sharey=True, gridspec_kw={"width_ratios": widths, "wspace": 0.08})
    figure.subplots_adjust(left=0.085, right=0.99, top=0.775, bottom=0.375)
    record: dict = {"layer": args.layer, "axis": str(args.axis), "root": str(args.root), "tokens": str(args.tokens) if args.tokens else None,
                    "heldout_questions": len(heldout), "panels": {}, "personas": {}}
    top = max(float(v.max()) for v in values.values())
    bottom = min(float(v.min()) for v in values.values())
    for axes, (title, deceptive_group, honest_group) in zip(axes_row, PANELS):
        ordered = {honesty: sorted(groups[group], key=lambda role: -float(np.median(values[role])))
                   for honesty, group in (("deceptive", deceptive_group), ("honest", honest_group))}
        positions, roles, colors = [], [], []
        for honesty, offset in (("deceptive", 0.0), ("honest", len(ordered["deceptive"]) + GROUP_GAP)):
            for index, role in enumerate(ordered[honesty]):
                positions.append(offset + index), roles.append(role), colors.append(COLOR[honesty])
        axes.set_axisbelow(True)
        axes.grid(axis="y", color=GRID, linewidth=0.7)
        axes.axhline(0, color=SOFT, linewidth=0.9, zorder=1)
        parts = axes.violinplot([values[role] for role in roles], positions=positions, widths=0.82, showextrema=False)
        for body, color in zip(parts["bodies"], colors):
            body.set_facecolor(color), body.set_linewidth(0), body.set_alpha(1.0), body.set_zorder(2)
        for position, role in zip(positions, roles):
            median = float(np.median(values[role]))
            axes.plot([position - 0.3, position + 0.3], [median, median], color=INK, linewidth=1.4, solid_capstyle="butt", zorder=3)
            q25, q75 = (float(x) for x in np.percentile(values[role], [25, 75]))
            record["personas"][role] = {"panel": title, "honesty": "deceptive" if role in ordered["deceptive"] else "honest",
                                        "n": int(len(values[role])), "median": median, "q25": q25, "q75": q75, "source": source[role]}
        for honesty in ("deceptive", "honest"):
            centre = float(np.mean([p for p, r in zip(positions, roles) if r in ordered[honesty]]))
            axes.text(centre, 0.965, honesty, transform=axes.get_xaxis_transform(), ha="center", va="top", fontsize=8.5, color=SOFT)
        positive = np.concatenate([values[role] for role in ordered["deceptive"]])
        negative = np.concatenate([values[role] for role in ordered["honest"]])
        separation = auroc(positive, negative)
        lowest_deceptive = min(float(np.median(values[role])) for role in ordered["deceptive"])
        highest_honest = max(float(np.median(values[role])) for role in ordered["honest"])
        record["panels"][title] = {"auroc": separation, "deceptive_responses": int(len(positive)), "honest_responses": int(len(negative)),
                                   "lowest_deceptive_median": lowest_deceptive, "highest_honest_median": highest_honest}
        axes.set_title(title, loc="left", fontsize=9.5, fontweight="bold", color=INK, pad=20)
        axes.text(0.0, 1.022, f"AUROC {separation:.3f}",
                  transform=axes.transAxes, ha="left", va="bottom", fontsize=8.5, color=SOFT)
        axes.set_xticks(positions, [label_of(role) for role in roles], rotation=45, ha="right", rotation_mode="anchor", fontsize=8, color=INK)
        axes.set_xlim(-0.75, positions[-1] + 0.75)
        axes.tick_params(axis="x", length=3, color=SOFT, pad=2)
        axes.tick_params(axis="y", length=3, color=SOFT, labelsize=8.5, labelcolor=INK)
        for side in ("top", "right"):
            axes.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axes.spines[side].set_color(SOFT)
        if axes is not axes_row[0]:
            axes.spines["left"].set_visible(False)
            axes.tick_params(axis="y", length=0)
    span = top - bottom
    axes_row[0].set_ylim(bottom - 0.04 * span, top + 0.13 * span)
    axes_row[0].set_ylabel(f"Projection on the {args.axis_label}\n(default assistant on the same question = 0)", fontsize=9, color=INK)

    counts = [entry["n"] for entry in record["personas"].values()]
    model = f"{args.model_label}: " if args.model_label else ""
    figure.text(0.012, 0.972, f"{model}persona responses projected on the {args.axis_label}, layer {args.layer}", fontsize=12.5, fontweight="bold",
                color=INK, ha="left", va="top")
    figure.text(0.012, 0.922, f"One violin per persona: its in-role responses to the {len(heldout)} held-out questions; black line = median.",
                fontsize=9.5, color=SOFT, ha="left", va="top")
    figure.legend(handles=[Patch(facecolor=COLOR["deceptive"], label="deceptive persona"), Patch(facecolor=COLOR["honest"], label="honest persona")],
                  loc="upper right", bbox_to_anchor=(0.99, 0.975), ncols=2, frameon=False, fontsize=9, labelcolor=INK, handlelength=1.2,
                  columnspacing=1.6)
    notes = [f"Each response: mean residual activation over its tokens at layer {args.layer}, projected on the unit axis, minus the default "
             "assistant's mean on the same question.",
             f"Only responses the role judge scored fully in role ({min(counts)}–{max(counts)} per persona). AUROC: deceptive against honest "
             "responses within the panel. Violins run from highest to lowest median within each group."]
    if args.note:
        notes.append(args.note)
    figure.text(0.012, 0.012, "\n".join(notes), fontsize=7.8, color=SOFT, ha="left", va="bottom", linespacing=1.45)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out, dpi=200, facecolor="white")
    plt.close(figure)
    args.out.with_suffix(".json").write_text(json.dumps(record, indent=1) + "\n")
    for title, panel in record["panels"].items():
        print(f"{title}: AUROC {panel['auroc']:.4f}, lowest deceptive median {panel['lowest_deceptive_median']:.0f}, "
              f"highest honest median {panel['highest_honest_median']:.0f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
