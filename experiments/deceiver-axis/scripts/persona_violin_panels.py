"""Three-panel persona violin chart: per-response projections on an axis, deceptive next to honest personas, one panel
per persona set (in the axis | held out | reward-hacking-adjacent), shared y-axis.

Redesign of persona_figures.py's `<prefix>_distributions_no_controls.png` (same data: in-role responses to the held-out
questions, projection at one layer minus the default assistant's mean on the same question). CPU only.

With --controls it draws the same style for the test personas instead (redesign of persona_figures.py's
`<prefix>_distributions_controls.png`): one panel, the axis's deceptive personas pooled, each test persona by median, the
axis's honest personas pooled.

Per-response values come from an extraction root (`act_response_only/<role>.pt`, `scores/<role>.json`) and, for roles
the root does not hold, from a directory of per-token projections written by token_projections.py (`<role>.jsonl`; the
mean over a response's tokens is the projection of its mean activation). Each source is referenced to its own default.

    python scripts/persona_violin_panels.py --root <extraction root> --axis <persona_contrast.pt> --layer 17 \
        [--tokens <tokens_L17 dir>] [--cache <npz>] [--controls] --model-label gpt-oss-120b --out <png> [--note "<extra note line>"]

Writes the PNG and `<png stem>.json` (per-persona n / median / quartiles, per-panel AUROC, sources). With --cache the
per-response values are stored on the first run and read back on later ones (the root is then not opened again).
"""
from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path
from typing import cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.collections import PolyCollection
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from scipy.stats import rankdata

INK, SOFT, GRID = "#1f1e1c", "#52514e", "#dddbd5"
COLOR = {"deceptive": "#eb6834", "test": "#27a06f", "honest": "#2a78d6"}
HERE = Path(__file__).resolve().parent.parent
PANELS = (  # (title, deceptive group, honest group)
    ("Personas the axis was built from", "deceptive_main", "honest_main"),
    ("Held-out personas", "deceptive_holdout", "honest_holdout"),
    ("Reward-hacking-adjacent (held out)", "deceptive_rh", "honest_rh"),
)
GROUP_GAP = 0.9  # blank columns between the deceptive and the honest violins of a panel
CONTROLS_LEFT_OUT = ("neutral_fic",)  # as in persona_figures.py's controls violins


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


def summary(values: np.ndarray) -> dict[str, float]:
    q25, median, q75 = (float(x) for x in np.percentile(values, [25, 50, 75]))
    return {"n": int(len(values)), "median": median, "q25": q25, "q75": q75}


def draw_violins(axes: Axes, positions: list[float], values_by_column: list[np.ndarray], colors: list[str]) -> None:
    axes.set_axisbelow(True)
    axes.grid(axis="y", color=GRID, linewidth=0.6)
    axes.axhline(0, color=SOFT, linewidth=0.9, zorder=1)
    parts = axes.violinplot(values_by_column, positions=positions, widths=0.82, showextrema=False)
    bodies = cast(list[PolyCollection], parts["bodies"])
    for body, color in zip(bodies, colors):
        body.set_facecolor(color), body.set_linewidth(0), body.set_alpha(1.0), body.set_zorder(2)
    for position, values in zip(positions, values_by_column):
        median = float(np.median(values))
        axes.plot([position - 0.3, position + 0.3], [median, median], color=INK, linewidth=1.4, solid_capstyle="butt", zorder=3)


def style_axes(axes: Axes, positions: list[float], labels: list[str], shows_y_axis: bool) -> None:
    axes.set_xticks(positions, labels, rotation=45, ha="right", rotation_mode="anchor", fontsize=8, color=INK)
    axes.set_xlim(-0.75, positions[-1] + 0.75)
    axes.tick_params(axis="x", length=3, color=SOFT, pad=2)
    axes.tick_params(axis="y", length=3, color=SOFT, labelsize=8.5, labelcolor=INK)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(SOFT)
    if not shows_y_axis:
        axes.spines["left"].set_visible(False)
        axes.tick_params(axis="y", length=0)


def label_y_axis(axes: Axes, values_by_column: list[np.ndarray], axis_label: str) -> None:
    top = max(float(v.max()) for v in values_by_column)
    bottom = min(float(v.min()) for v in values_by_column)
    span = top - bottom
    axes.set_ylim(bottom - 0.04 * span, top + 0.04 * span)
    axes.set_ylabel(f"Projection on the {axis_label}", fontsize=9.5, color=INK)
    axes.annotate("default assistant", xy=(-0.6, 0), xytext=(0, -4), textcoords="offset points", ha="left", va="top", fontsize=8, color=SOFT)


def add_header(figure: Figure, title: str, legend_labels_by_color: dict[str, str], legend_below_title: bool) -> None:
    inch = 1 / figure.get_figheight()
    figure.text(0.012, 1 - 0.13 * inch, title, fontsize=12.5, fontweight="bold", color=INK, ha="left", va="top")
    handles = [Patch(facecolor=color, label=label) for color, label in legend_labels_by_color.items()]
    figure.legend(handles=handles, loc="upper left" if legend_below_title else "upper right",
                  bbox_to_anchor=(0.004, 1 - 0.42 * inch) if legend_below_title else (0.992, 1 - 0.08 * inch), ncols=len(handles), frameon=False, fontsize=9.5, labelcolor=INK, handlelength=1.1,
                  handleheight=0.9, columnspacing=1.6)


def draw_panels(values_by_role: dict[str, np.ndarray], source_by_role: dict[str, str], groups: dict[str, list[str]], args: argparse.Namespace,
                heldout_question_count: int) -> tuple[Figure, dict]:
    widths = [len(groups[d]) + len(groups[h]) + GROUP_GAP for _, d, h in PANELS]
    figure, axes_row = plt.subplots(1, len(PANELS), figsize=(11.0, 4.62 if args.no_notes else 5.02), sharey=True,
                                    gridspec_kw={"width_ratios": widths, "wspace": 0.08})
    figure.subplots_adjust(left=0.068, right=0.99, top=1 - 0.86 / figure.get_figheight(), bottom=(1.46 if args.no_notes else 1.855) / figure.get_figheight())
    record: dict = {"layer": args.layer, "axis": str(args.axis), "root": str(args.root), "tokens": str(args.tokens) if args.tokens else None,
                    "heldout_questions": heldout_question_count, "panels": {}, "personas": {}}
    for axes, (title, deceptive_group, honest_group) in zip(axes_row, PANELS):
        ordered = {honesty: sorted(groups[group], key=lambda role: -float(np.median(values_by_role[role])))
                   for honesty, group in (("deceptive", deceptive_group), ("honest", honest_group))}
        positions, roles, colors = [], [], []
        for honesty, offset in (("deceptive", 0.0), ("honest", len(ordered["deceptive"]) + GROUP_GAP)):
            for index, role in enumerate(ordered[honesty]):
                positions.append(offset + index), roles.append(role), colors.append(COLOR[honesty])
        draw_violins(axes, positions, [values_by_role[role] for role in roles], colors)
        for role in roles:
            record["personas"][role] = {"panel": title, "honesty": "deceptive" if role in ordered["deceptive"] else "honest",
                                        **summary(values_by_role[role]), "source": source_by_role[role]}
        positive = np.concatenate([values_by_role[role] for role in ordered["deceptive"]])
        negative = np.concatenate([values_by_role[role] for role in ordered["honest"]])
        separation = auroc(positive, negative)
        lowest_deceptive = min(float(np.median(values_by_role[role])) for role in ordered["deceptive"])
        highest_honest = max(float(np.median(values_by_role[role])) for role in ordered["honest"])
        record["panels"][title] = {"auroc": separation, "deceptive_responses": int(len(positive)), "honest_responses": int(len(negative)),
                                   "lowest_deceptive_median": lowest_deceptive, "highest_honest_median": highest_honest}
        axes.set_title(title, loc="left", fontsize=9.5, fontweight="bold", color=INK, pad=19)
        axes.text(0.0, 1.02, f"AUROC {separation:.3f}", transform=axes.transAxes, ha="left", va="bottom", fontsize=8.5, color=SOFT)
        style_axes(axes, positions, [label_of(role) for role in roles], shows_y_axis=axes is axes_row[0])
    label_y_axis(axes_row[0], [values_by_role[role] for role in record["personas"]], args.axis_label)
    model = f"{args.model_label}: " if args.model_label else ""
    add_header(figure, f"{model}persona responses on the {args.axis_label}",
               {COLOR["deceptive"]: "deceptive personas", COLOR["honest"]: "honest personas"}, legend_below_title=False)
    counts = [entry["n"] for entry in record["personas"].values()]
    record["notes"] = ["Each response: mean residual activation over its response tokens, projected on the unit axis, relative to the default "
                       "assistant's mean on the same question.",
                       f"Only responses judged fully in role ({min(counts)}–{max(counts)} per persona). AUROC: deceptive vs honest responses within the panel."]
    return figure, record


def draw_controls(values_by_role: dict[str, np.ndarray], source_by_role: dict[str, str], groups: dict[str, list[str]], args: argparse.Namespace,
                  heldout_question_count: int) -> tuple[Figure, dict]:
    test_roles = sorted((role for role in groups["controls"] if role not in CONTROLS_LEFT_OUT), key=lambda role: -float(np.median(values_by_role[role])))
    columns = [("deceptive (pooled)", "deceptive", np.concatenate([values_by_role[role] for role in groups["deceptive_main"]])),
               *[(label_of(role), "test", values_by_role[role]) for role in test_roles],
               ("honest (pooled)", "honest", np.concatenate([values_by_role[role] for role in groups["honest_main"]]))]
    positions = [0.0, *[GROUP_GAP + index for index in range(1, len(test_roles) + 1)], 2 * GROUP_GAP + len(test_roles) + 1]
    figure, axes = plt.subplots(figsize=(6.0, 4.1 if args.no_notes else 4.6))
    figure.subplots_adjust(left=0.125, right=0.985, top=1 - 0.78 / figure.get_figheight(), bottom=(1.26 if args.no_notes else 1.76) / figure.get_figheight())
    draw_violins(axes, positions, [values for _, _, values in columns], [COLOR[kind] for _, kind, _ in columns])
    style_axes(axes, positions, [label for label, _, _ in columns], shows_y_axis=True)
    label_y_axis(axes, [values for _, _, values in columns], args.axis_label)
    model = f"{args.model_label}: " if args.model_label else ""
    add_header(figure, f"{model}test personas on the {args.axis_label}",
               {COLOR["deceptive"]: "deceptive axis personas", COLOR["test"]: "test personas", COLOR["honest"]: "honest axis personas"},
               legend_below_title=True)
    record: dict = {"layer": args.layer, "axis": str(args.axis), "root": str(args.root), "tokens": str(args.tokens) if args.tokens else None,
                    "heldout_questions": heldout_question_count, "left_out": list(CONTROLS_LEFT_OUT),
                    "columns": {label: {"kind": kind, **summary(values)} for label, kind, values in columns},
                    "sources": {role: source_by_role[role] for role in [*groups["deceptive_main"], *test_roles, *groups["honest_main"]]}}
    record["notes"] = ["Each response: mean residual activation over its response tokens, projected on the unit axis, relative to the default "
                       "assistant's mean on the same question.",
                       "Only responses judged fully in role. Pooled: every response of the axis's nine deceptive or nine honest personas."]
    return figure, record


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
    parser.add_argument("--no-notes", action="store_true", help="leave the note lines under the panels out (caption goes in the paper)")
    parser.add_argument("--controls", action="store_true", help="draw the test-persona chart instead of the three persona-set panels")
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    groups = json.loads(args.groups.read_text())
    questions = [json.loads(line) for line in args.questions.read_text().splitlines() if line.strip()]
    heldout = {q["id"] for q in questions if q["split"] == "heldout"}
    axis = torch.load(args.axis, map_location="cpu", weights_only=True)
    axis = axis["axis"] if isinstance(axis, dict) else axis
    unit = axis[args.layer].float()
    unit = unit / unit.norm()

    panel_roles = [role for _, deceptive_group, honest_group in PANELS for role in groups[deceptive_group] + groups[honest_group]]
    test_roles = [role for role in groups["controls"] if role not in CONTROLS_LEFT_OUT]
    roles = [*groups["deceptive_main"], *test_roles, *groups["honest_main"]] if args.controls else panel_roles
    values: dict[str, np.ndarray] = {}
    source: dict[str, str] = {}
    cached = args.cache is not None and args.cache.exists()
    if cached:
        with np.load(args.cache) as stored:
            values = {role: stored[role] for role in stored.files}
        source = {role: "cache" for role in values}
    root_default = None if cached else from_activations(args.root, "default", unit, args.layer)
    token_default = from_tokens(args.tokens, "default") if args.tokens and not cached else None
    for role in roles:
        if role in values:
            continue
        if root_default is not None and (raw := from_activations(args.root, role, unit, args.layer)) is not None:
            values[role], source[role] = relative(raw, question_means(root_default), heldout), "activations"
        elif token_default is not None and (raw := from_tokens(args.tokens, role)) is not None:
            values[role], source[role] = relative(raw, question_means(token_default), heldout), "tokens"
        else:
            raise SystemExit(f"no per-response data for {role}")

    if args.cache is not None and not cached:
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.cache, **values)

    draw = draw_controls if args.controls else draw_panels
    figure, record = draw(values, source, groups, args, len(heldout))
    notes = record["notes"] + ([args.note] if args.note else [])
    record["notes"] = notes
    if not args.no_notes:
        characters_per_line = int(17 * figure.get_figwidth())
        wrapped_notes = "\n".join(textwrap.fill(note, characters_per_line) for note in notes)
        figure.text(0.012, 0.014, wrapped_notes, fontsize=8, color=SOFT, ha="left", va="bottom", linespacing=1.45)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out, dpi=200, facecolor="white")
    plt.close(figure)
    args.out.with_suffix(".json").write_text(json.dumps(record, indent=1) + "\n")
    for title, panel in record.get("panels", {}).items():
        print(f"{title}: AUROC {panel['auroc']:.4f}, lowest deceptive median {panel['lowest_deceptive_median']:.0f}, "
              f"highest honest median {panel['highest_honest_median']:.0f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
