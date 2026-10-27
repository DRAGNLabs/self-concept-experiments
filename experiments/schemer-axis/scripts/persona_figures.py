"""Figures for the expanded persona set (PLAN "Expanded persona set"): projections of every persona's in-role
responses onto a schemer axis, compared across persona groups and question splits.

Every projection is taken relative to the default role: the default's mean projection on the same question is
subtracted. Persona means average per-question means, so each question counts once; 95% intervals bootstrap over
questions. The grey bands are mean ± 2 sd of the same quantity on isotropic random directions.

    python scripts/persona_figures.py --run.root <extraction root> --run.region with_cot --run.layer 17 \
        --run.axis <candidates>/persona_contrast.pt --run.out <dir> [--run.groups data/persona_groups.json] [--run.questions data/questions.jsonl]
"""
from __future__ import annotations

import json
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypedDict, cast

import matplotlib
matplotlib.use("Agg")
from matplotlib.axes import Axes
from matplotlib.collections import PolyCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import matplotlib.pyplot as plt
from jaxtyping import Float, Int
import numpy as np
from sklearn.metrics import roc_auc_score
import torch

from selfconcept.correlation.random_vectors import random_directions
from selfconcept.measurement.unit_axes import load_unit_axes

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

GroupName = Literal["deceptive_main", "honest_main", "deceptive_rh", "honest_rh", "deceptive_holdout", "honest_holdout", "controls"]
Honesty = Literal["deceptive", "honest", "control"]
QuestionSet = Literal["train", "heldout", "heldout_task", "heldout_other"]
QuestionRecipe = Literal["split", "generic_fit"]  # generic_fit: train = generic questions, held out = every intent and task question
Region = Literal["with_cot", "response_only"]

HONESTY_BY_GROUP: dict[GroupName, Honesty] = {
    "deceptive_main": "deceptive", "deceptive_rh": "deceptive", "deceptive_holdout": "deceptive",
    "honest_main": "honest", "honest_rh": "honest", "honest_holdout": "honest", "controls": "control",
}
COLOR_BY_HONESTY: dict[Honesty, str] = {"honest": "#2a78d6", "deceptive": "#eb6834", "control": "#1baf7a"}
TEXT_COLOR, MUTED_COLOR, SURFACE_COLOR, RANDOM_BAND_COLOR = "#0b0b0b", "#52514e", "#fcfcfb", "#dddcd7"
DIVERGING = LinearSegmentedColormap.from_list("honest_to_deceptive", ["#2a78d6", "#f0efec", "#e34948"])
DIVERGING_CVD = LinearSegmentedColormap.from_list("blue_to_orange", ["#2a78d6", "#f0efec", "#eb6834"])  # poles pass the dataviz CVD check
BOOTSTRAP_RESAMPLES = 1000
HONESTIES: tuple[Honesty, ...] = ("deceptive", "honest", "control")
HELDOUT_MARKERS: tuple[tuple[QuestionSet, str], ...] = (("heldout_task", "^"), ("heldout_other", "o"))

type Subsection = list[str]
type Section = tuple[str, list[Subsection]]


class Projections(TypedDict):
    question_ids: Int[np.ndarray, "response"]
    axis: Float[np.ndarray, "response layer"]
    random: Float[np.ndarray, "response layer direction"]


class Summary(TypedDict):
    mean: float
    ci_low: float
    ci_high: float
    random_mean: float
    random_sd: float
    questions: int


class AurocVersusRandom(TypedDict):
    axis: float
    random_2_5: float
    random_50: float
    random_97_5: float
    random_fraction_at_least_as_far_from_half: float
    positive_responses: int
    negative_responses: int


def question_index_of(response_key: str) -> int:
    return int(response_key.rsplit("_q", 1)[1])


def load_projections(root: Path, region: Region, role: str, unit_axes: Float[torch.Tensor, "layer hidden"],
                     random_axes: Float[torch.Tensor, "layer direction hidden"]) -> Projections | None:
    """Projections of the role's responses; personas keep only responses the judge scored 3 (fully in role)."""
    activations_path = root / f"act_{region}" / f"{role}.pt"
    if not activations_path.exists():
        logger.warning(f"no activations for {role}")
        return None
    activations_by_key = torch.load(activations_path, map_location="cpu", weights_only=False)
    if role == "default":
        keys = list(activations_by_key)
    else:
        scores_by_key = json.loads((root / "scores" / f"{role}.json").read_text())
        keys = [key for key in activations_by_key if scores_by_key.get(key) == 3]
    if not keys:
        logger.warning(f"no in-role responses for {role}")
        return None
    stacked = torch.stack([activations_by_key[key] for key in keys]).float()
    return {"question_ids": np.array([question_index_of(key) for key in keys]),
            "axis": torch.einsum("nlh,lh->nl", stacked, unit_axes).numpy(),
            "random": torch.einsum("nlh,ldh->nld", stacked, random_axes).numpy()}


def per_question_means(values: Float[np.ndarray, "response *rest"], question_ids: Int[np.ndarray, "response"],
                       question_count: int) -> Float[np.ndarray, "question *rest"]:
    """Mean over each question's responses; NaN for questions without responses."""
    sums = np.zeros((question_count, *values.shape[1:]))
    np.add.at(sums, question_ids, values)
    counts = np.bincount(question_ids, minlength=question_count).reshape(-1, *[1] * (values.ndim - 1))
    with np.errstate(invalid="ignore"):
        return sums / counts


def relative_to_default(projections: Projections, default: Projections, question_count: int) -> Projections:
    default_axis = per_question_means(default["axis"], default["question_ids"], question_count)
    default_random = per_question_means(default["random"], default["question_ids"], question_count)
    question_ids = projections["question_ids"]
    return {"question_ids": question_ids, "axis": projections["axis"] - default_axis[question_ids],
            "random": projections["random"] - default_random[question_ids]}


def restrict_to_questions(projections: Projections, question_ids: Int[np.ndarray, "subset"]) -> Projections:
    keep = np.isin(projections["question_ids"], question_ids)
    return {"question_ids": projections["question_ids"][keep], "axis": projections["axis"][keep], "random": projections["random"][keep]}


def bootstrap_ci(per_question: Float[np.ndarray, "question"], rng: np.random.Generator) -> tuple[float, float]:
    resampled = rng.integers(0, len(per_question), (BOOTSTRAP_RESAMPLES, len(per_question)))
    low, high = np.percentile(per_question[resampled].mean(axis=1), [2.5, 97.5])
    return float(low), float(high)


def summarize(per_question_axis: Float[np.ndarray, "question"], per_question_random: Float[np.ndarray, "question direction"],
              rng: np.random.Generator) -> Summary | None:
    """Mean over the questions with data, its bootstrap CI, and the random-direction spread of the same mean."""
    has_data = ~np.isnan(per_question_axis)
    if not has_data.any():
        return None
    low, high = bootstrap_ci(per_question_axis[has_data], rng)
    random_means = per_question_random[has_data].mean(axis=0)
    return {"mean": float(per_question_axis[has_data].mean()), "ci_low": low, "ci_high": high,
            "random_mean": float(random_means.mean()), "random_sd": float(random_means.std()), "questions": int(has_data.sum())}


def per_question_at_layer(relative: Projections, layer: int, question_ids: Int[np.ndarray, "subset"], question_count: int
                          ) -> tuple[Float[np.ndarray, "subset"], Float[np.ndarray, "subset direction"]]:
    axis = per_question_means(relative["axis"][:, layer], relative["question_ids"], question_count)
    random = per_question_means(relative["random"][:, layer], relative["question_ids"], question_count)
    return axis[question_ids], random[question_ids]


def group_summary(relative_by_role: dict[str, Projections], roles: list[str], layer: int, question_ids: Int[np.ndarray, "subset"],
                  question_count: int, rng: np.random.Generator) -> Summary | None:
    """Per question, the mean over the group's personas; then the same summary as for one persona."""
    pairs = [per_question_at_layer(relative_by_role[role], layer, question_ids, question_count) for role in roles if role in relative_by_role]
    if not pairs:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN questions stay NaN
        axis = np.nanmean(np.stack([axis for axis, _ in pairs]), axis=0)
        random = np.nanmean(np.stack([random for _, random in pairs]), axis=0)
    return summarize(axis, random, rng)


def selected_projections(relative_by_role: dict[str, Projections], roles: list[str], question_ids: Int[np.ndarray, "subset"]) -> list[Projections]:
    return [restrict_to_questions(relative_by_role[role], question_ids) for role in roles if role in relative_by_role]


def auroc_by_layer(relative_by_role: dict[str, Projections], positive: list[str], negative: list[str],
                   question_ids: Int[np.ndarray, "subset"]) -> tuple[Float[np.ndarray, "layer"], Float[np.ndarray, "layer direction"]] | None:
    """Response-level AUROC of the positive vs the negative roles along the axis and along each random direction."""
    positive_sets, negative_sets = selected_projections(relative_by_role, positive, question_ids), selected_projections(relative_by_role, negative, question_ids)
    if not positive_sets or not negative_sets:
        return None
    axis = np.concatenate([p["axis"] for p in positive_sets + negative_sets])
    random = np.concatenate([p["random"] for p in positive_sets + negative_sets])
    labels = np.concatenate([np.ones(sum(len(p["axis"]) for p in positive_sets)), np.zeros(sum(len(p["axis"]) for p in negative_sets))])
    axis_auroc = np.array([roc_auc_score(labels, axis[:, layer]) for layer in range(axis.shape[1])])
    random_auroc = np.array([[roc_auc_score(labels, random[:, layer, direction]) for direction in range(random.shape[2])]
                             for layer in range(random.shape[1])])
    return axis_auroc, random_auroc


def auroc_versus_random(relative_by_role: dict[str, Projections], positive: list[str], negative: list[str], layer: int,
                        question_ids: Int[np.ndarray, "subset"]) -> AurocVersusRandom | None:
    """The axis AUROC at one layer against the 95% range of the random-direction AUROCs."""
    at_layer = {role: {"question_ids": p["question_ids"], "axis": p["axis"][:, layer:layer + 1], "random": p["random"][:, layer:layer + 1]}
                for role, p in relative_by_role.items()}
    result = auroc_by_layer(cast(dict[str, Projections], at_layer), positive, negative, question_ids)
    if result is None:
        return None
    [axis_auroc], [random_auroc] = result
    low, median, high = np.quantile(random_auroc, [0.025, 0.5, 0.975])
    return {"axis": float(axis_auroc), "random_2_5": float(low), "random_50": float(median), "random_97_5": float(high),
            "random_fraction_at_least_as_far_from_half": float((np.abs(random_auroc - 0.5) >= abs(axis_auroc - 0.5)).mean()),
            "positive_responses": sum(len(p["axis"]) for p in selected_projections(relative_by_role, positive, question_ids)),
            "negative_responses": sum(len(p["axis"]) for p in selected_projections(relative_by_role, negative, question_ids))}


def mean_or_nan(summary: Summary | None) -> float:
    return np.nan if summary is None else summary["mean"]


def persona_label(role: str, groups: dict[GroupName, list[str]]) -> str:
    in_axis = role in groups["deceptive_main"] or role in groups["honest_main"]
    return role.removesuffix("_fic").replace("_", " ") + (" †" if in_axis else "")


def group_of(role: str, groups: dict[GroupName, list[str]]) -> GroupName:
    return next(group for group, roles in groups.items() if role in roles)


def honesty_of(role: str, groups: dict[GroupName, list[str]]) -> Honesty:
    return HONESTY_BY_GROUP[group_of(role, groups)]


def is_rh_adjacent(role: str, groups: dict[GroupName, list[str]]) -> bool:
    return role in groups["deceptive_rh"] or role in groups["honest_rh"]


def style_axes(axes: Axes) -> None:
    axes.set_facecolor(SURFACE_COLOR)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(MUTED_COLOR)
    axes.tick_params(colors=MUTED_COLOR, labelcolor=TEXT_COLOR)
    axes.grid(axis="y", color="#e6e5e0", linewidth=0.6)
    axes.set_axisbelow(True)


def section_columns(sections: list[Section], present: set[str]) -> tuple[list[str], list[tuple[str, int, int]]]:
    """Column order of the present personas and each section's (title, first column, last column + 1)."""
    columns: list[str] = []
    spans = []
    for title, subsections in sections:
        start = len(columns)
        columns += [role for subsection in subsections for role in subsection if role in present]
        if len(columns) > start:
            spans.append((title, start, len(columns)))
    return columns, spans


def with_model(model_label: str, title: str) -> str:
    return f"{model_label}: {title}" if model_label else title


def draw_section_titles(axes: Axes, spans: list[tuple[str, int, int]]) -> None:
    for index, (title, start, end) in enumerate(spans):
        axes.text((start + end - 1) / 2, 1.02, title, transform=axes.get_xaxis_transform(), ha="center", va="bottom",
                  color=TEXT_COLOR, fontsize=9)
        if index:
            axes.axvline(start - 0.5, color=MUTED_COLOR, linewidth=0.8)


def heatmap_figure(summaries: dict[str, dict[QuestionSet, Summary | None]], sections: list[Section],
                   groups: dict[GroupName, list[str]], path: Path, model_label: str) -> None:
    columns, spans = section_columns(sections, set(summaries))
    rows: list[tuple[QuestionSet, str]] = [("heldout_task", "held-out task questions"), ("heldout_other", "other held-out questions")]
    matrix = np.array([[mean_or_nan(summaries[role][row]) for role in columns] for row, _ in rows])
    limit = float(np.nanmax(np.abs(matrix))) if np.isfinite(matrix).any() else 1.0
    figure, axes = plt.subplots(figsize=(max(6.0, 0.32 * len(columns) + 2.5), 3.4))
    image = axes.imshow(np.ma.masked_invalid(matrix), cmap=DIVERGING, vmin=-limit, vmax=limit, aspect="auto")
    axes.set_yticks(range(len(rows)), [label for _, label in rows])
    axes.set_xticks(range(len(columns)), [persona_label(role, groups) for role in columns], rotation=60, ha="right", fontsize=8)
    for side in axes.spines.values():
        side.set_visible(False)
    for index, (_, start, _) in enumerate(spans):
        if index:
            axes.axvline(start - 0.5, color=SURFACE_COLOR, linewidth=3)
    for title, start, end in spans:
        axes.text((start + end - 1) / 2, -0.6, title, ha="center", va="bottom", color=TEXT_COLOR, fontsize=9)
    colorbar = figure.colorbar(image, ax=axes, fraction=0.03, pad=0.02)
    colorbar.set_label("projection − default (same question)", color=TEXT_COLOR)
    axes.set_title(with_model(model_label, "Mean schemer-axis projection on held-out questions († = persona used to build the axis)"),
                   color=TEXT_COLOR, fontsize=10, pad=34)
    figure.savefig(path, dpi=200, bbox_inches="tight", facecolor=SURFACE_COLOR)
    plt.close(figure)


def bar_figure(summaries: dict[str, dict[QuestionSet, Summary | None]], group_summaries: dict[str, Summary | None],
               sections: list[Section], groups: dict[GroupName, list[str]], title: str, path: Path,
               model_label: str, references: tuple[tuple[str, Summary | None], ...] = ()) -> None:
    """Per-persona held-out means with bootstrap CIs over grey random-direction bands, and a mean line per subsection."""
    present = {role for role, by_set in summaries.items() if by_set["heldout"] is not None}
    columns, spans = section_columns(sections, present)
    if not columns:
        logger.warning(f"no personas with held-out data for {path.name}")
        return
    figure, axes = plt.subplots(figsize=(max(5.0, 0.38 * len(columns) + 2.5), 4.2))
    style_axes(axes)
    for column, role in enumerate(columns):
        summary = summaries[role]["heldout"]
        assert summary is not None
        axes.add_patch(Rectangle((column - 0.42, summary["random_mean"] - 2 * summary["random_sd"]), 0.84, 4 * summary["random_sd"],
                                     facecolor=RANDOM_BAND_COLOR, linewidth=0, zorder=1))
        axes.bar(column, summary["mean"], width=0.62, color=COLOR_BY_HONESTY[honesty_of(role, groups)], zorder=2,
                 hatch="////" if is_rh_adjacent(role, groups) else None, edgecolor=SURFACE_COLOR, linewidth=0)
        axes.errorbar(column, summary["mean"], yerr=[[summary["mean"] - summary["ci_low"]], [summary["ci_high"] - summary["mean"]]],
                      color=TEXT_COLOR, linewidth=1, capsize=2, zorder=3)
    for _, subsections in sections:
        for subsection in subsections:
            subsection_columns = [columns.index(role) for role in subsection if role in present]
            group = group_summaries.get(" ".join(subsection))
            if not subsection_columns or group is None:
                continue
            left, right = min(subsection_columns) - 0.45, max(subsection_columns) + 0.45
            axes.plot([left, right], [group["mean"]] * 2, color=MUTED_COLOR, linewidth=1.5, zorder=5)
    for label, reference in references:
        if reference is not None:
            axes.axhline(reference["mean"], color=MUTED_COLOR, linewidth=1, zorder=1)
            axes.text(-0.55, reference["mean"], label, va="bottom", ha="left", color=MUTED_COLOR, fontsize=8)
    axes.axhline(0, color=MUTED_COLOR, linewidth=0.8, zorder=1)
    draw_section_titles(axes, spans)
    axes.set_xlim(-0.6, len(columns) - 0.4)
    axes.set_xticks(range(len(columns)), [persona_label(role, groups) for role in columns], rotation=60, ha="right", fontsize=8)
    axes.set_ylabel("projection − default, held-out questions", color=TEXT_COLOR)
    axes.set_title(f"{with_model(model_label, title)}\n(error bars: 95% bootstrap CI over questions; † = persona used to build the axis)", color=TEXT_COLOR,
                   fontsize=10, pad=20)
    honesties: list[Honesty] = [honesty for honesty in HONESTIES if any(honesty_of(role, groups) == honesty for role in columns)]
    legend = [Patch(color=COLOR_BY_HONESTY[honesty], label=honesty) for honesty in honesties]
    legend += [Patch(facecolor=MUTED_COLOR, hatch="////", edgecolor=SURFACE_COLOR, label="RH-adjacent")] if any(is_rh_adjacent(r, groups) for r in columns) else []
    legend += [Patch(color=RANDOM_BAND_COLOR, label="random directions ± 2 sd"), Line2D([], [], color=MUTED_COLOR, linewidth=1.5, label="group mean")]
    axes.legend(handles=legend, frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    figure.savefig(path, dpi=200, bbox_inches="tight", facecolor=SURFACE_COLOR)
    plt.close(figure)


def generalization_figure(summaries: dict[str, dict[QuestionSet, Summary | None]], groups: dict[GroupName, list[str]], path: Path, model_label: str) -> None:
    figure, axes = plt.subplots(figsize=(5.4, 5.0))
    style_axes(axes)
    axes.grid(axis="both", color="#e6e5e0", linewidth=0.6)
    values = []
    for role, by_set in summaries.items():
        train = by_set["train"]
        if train is None:
            continue
        for question_set, marker in HELDOUT_MARKERS:
            heldout = by_set[question_set]
            if heldout is not None:
                axes.scatter(train["mean"], heldout["mean"], marker=marker, s=42, color=COLOR_BY_HONESTY[honesty_of(role, groups)],
                             edgecolors=SURFACE_COLOR, linewidths=1.5, zorder=3)
                values += [train["mean"], heldout["mean"]]
    if values:
        low, high = min(values), max(values)
        margin = 0.05 * (high - low or 1)
        axes.plot([low - margin, high + margin], [low - margin, high + margin], color=MUTED_COLOR, linewidth=1, zorder=2)
    axes.set_xlabel("persona mean, train questions", color=TEXT_COLOR)
    axes.set_ylabel("persona mean, held-out questions", color=TEXT_COLOR)
    axes.set_title(with_model(model_label, "Train vs held-out questions (diagonal = equal)"), color=TEXT_COLOR, fontsize=10)
    legend = [Patch(color=COLOR_BY_HONESTY[h], label=h) for h in HONESTIES]
    legend += [Line2D([], [], marker="^", linestyle="", color=MUTED_COLOR, label="held-out task"),
               Line2D([], [], marker="o", linestyle="", color=MUTED_COLOR, label="other held-out")]
    axes.legend(handles=legend, frameon=False, fontsize=8, loc="upper left")
    figure.savefig(path, dpi=200, bbox_inches="tight", facecolor=SURFACE_COLOR)
    plt.close(figure)


def layer_sweep_figure(aurocs: dict[str, tuple[Float[np.ndarray, "layer"], Float[np.ndarray, "layer direction"]]], layer: int,
                       region: Region, path: Path, model_label: str) -> None:
    figure, axes = plt.subplots(figsize=(6.4, 3.8))
    style_axes(axes)
    colors = ["#2a78d6", "#eb6834", "#1baf7a"]
    for (label, (axis_auroc, random_auroc)), color in zip(aurocs.items(), colors):
        layers = np.arange(len(axis_auroc))
        if label == next(iter(aurocs)):
            center, spread = random_auroc.mean(axis=1), random_auroc.std(axis=1)
            axes.fill_between(layers, center - 2 * spread, center + 2 * spread, color=RANDOM_BAND_COLOR, linewidth=0,
                              label=f"random directions ± 2 sd ({label})")
        axes.plot(layers, axis_auroc, color=color, linewidth=2, label=label)
    axes.axvline(layer, color=MUTED_COLOR, linewidth=0.8)
    axes.axhline(0.5, color=MUTED_COLOR, linewidth=0.8)
    axes.set_ylim(0, 1.02)
    axes.set_xlabel("layer", color=TEXT_COLOR)
    axes.set_ylabel("AUROC, deceptive vs honest", color=TEXT_COLOR)
    axes.set_title(with_model(model_label, f"Held-out questions, {region} activations"), color=TEXT_COLOR, fontsize=10)
    axes.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    figure.savefig(path, dpi=200, bbox_inches="tight", facecolor=SURFACE_COLOR)
    plt.close(figure)


def random_band_key(axes: Axes, center_column: float) -> None:
    """Key for the per-response random bands, inside the plot at the top, centred over a column position."""
    axes.legend(handles=[Patch(color=RANDOM_BAND_COLOR, label="random directions ± 2 sd, per response")], frameon=False, fontsize=8,
                loc="upper center", bbox_to_anchor=(center_column, 1.0), bbox_transform=axes.get_xaxis_transform())


def distribution_figure(relative_by_role: dict[str, Projections], sections: list[Section], groups: dict[GroupName, list[str]],
                        layer: int, heldout_ids: Int[np.ndarray, "subset"], path: Path, model_label: str, random_bands: bool = False,
                        sort_by_median: bool = False) -> None:
    """Per-response violins; with random_bands, a grey band per persona spans mean ± 2 sd of the same per-response
    values projected onto the random directions instead of the axis; with sort_by_median, each section runs from
    highest to lowest median."""
    values_by_role = {role: projections["axis"][np.isin(projections["question_ids"], heldout_ids), layer]
                      for role, projections in relative_by_role.items() if role != "default"}
    columns, spans = section_columns(sections, {role for role, values in values_by_role.items() if len(values) > 1})
    if sort_by_median:
        columns = [role for _, start, end in spans for role in sorted(columns[start:end], key=lambda role: -float(np.median(values_by_role[role])))]
    figure, axes = plt.subplots(figsize=(max(6.0, 0.36 * len(columns) + 2.0), 4.2))
    style_axes(axes)
    for column, role in enumerate(columns if random_bands else []):
        projections = relative_by_role[role]
        random_values = projections["random"][np.isin(projections["question_ids"], heldout_ids), layer]
        center, spread = float(random_values.mean()), float(random_values.std())
        axes.add_patch(Rectangle((column - 0.42, center - 2 * spread), 0.84, 4 * spread, facecolor=RANDOM_BAND_COLOR, linewidth=0, zorder=0))
    if columns:
        parts = axes.violinplot([values_by_role[role] for role in columns], positions=range(len(columns)), widths=0.8,
                                showextrema=False, showmedians=True)
        for body, role in zip(cast(list[PolyCollection], parts["bodies"]), columns):
            body.set_facecolor(COLOR_BY_HONESTY[honesty_of(role, groups)])
            body.set_alpha(0.85)
            body.set_linewidth(0)
        parts["cmedians"].set_color(TEXT_COLOR)
        parts["cmedians"].set_linewidth(1)
    axes.axhline(0, color=MUTED_COLOR, linewidth=0.8)
    draw_section_titles(axes, spans)
    axes.set_xticks(range(len(columns)), [persona_label(role, groups) for role in columns], rotation=60, ha="right", fontsize=8)
    axes.set_ylabel("projection − default, per response", color=TEXT_COLOR)
    axes.set_title(with_model(model_label, "Held-out in-role responses (line = median; † = persona used to build the axis)"), color=TEXT_COLOR,
                   fontsize=10, pad=32)
    if random_bands:
        _, widest_start, widest_end = max(spans, key=lambda span: span[2] - span[1])
        random_band_key(axes, (widest_start + widest_end - 1) / 2)
    figure.savefig(path, dpi=200, bbox_inches="tight", facecolor=SURFACE_COLOR)
    plt.close(figure)


def controls_distribution_figure(relative_by_role: dict[str, Projections], groups: dict[GroupName, list[str]], layer: int,
                                 heldout_ids: Int[np.ndarray, "subset"], path: Path, model_label: str, skipped_controls: frozenset[str] = frozenset()) -> None:
    """Per-response violins for each control, highest median first, between the pooled deceptive and pooled honest axis
    personas, each over its per-response random-direction band (mean ± 2 sd)."""
    def heldout(role: str, kind: Literal["axis", "random"]) -> np.ndarray:
        projections = relative_by_role[role]
        return projections[kind][np.isin(projections["question_ids"], heldout_ids), layer]
    def pooled(roles: list[str], kind: Literal["axis", "random"]) -> np.ndarray:
        return np.concatenate([heldout(role, kind) for role in roles if role in relative_by_role])
    controls = [role for role in groups["controls"] if role not in skipped_controls and role in relative_by_role and len(heldout(role, "axis")) > 1]
    controls = sorted(controls, key=lambda role: -float(np.median(heldout(role, "axis"))))
    columns: list[tuple[str, Honesty, np.ndarray, np.ndarray]] = [
        ("deceptive axis\npersonas (pooled)", "deceptive", pooled(groups["deceptive_main"], "axis"), pooled(groups["deceptive_main"], "random")),
        *[(persona_label(role, groups), "control", heldout(role, "axis"), heldout(role, "random")) for role in controls],
        ("honest axis\npersonas (pooled)", "honest", pooled(groups["honest_main"], "axis"), pooled(groups["honest_main"], "random"))]
    figure, axes = plt.subplots(figsize=(0.75 * len(columns) + 2.5, 4.2))
    style_axes(axes)
    for column, (_, _, _, random_values) in enumerate(columns):
        center, spread = float(random_values.mean()), float(random_values.std())
        axes.add_patch(Rectangle((column - 0.42, center - 2 * spread), 0.84, 4 * spread, facecolor=RANDOM_BAND_COLOR, linewidth=0, zorder=0))
    parts = axes.violinplot([values for _, _, values, _ in columns], positions=range(len(columns)), widths=0.8, showextrema=False, showmedians=True)
    for body, (_, honesty, _, _) in zip(cast(list[PolyCollection], parts["bodies"]), columns):
        body.set_facecolor(COLOR_BY_HONESTY[honesty])
        body.set_alpha(0.85)
        body.set_linewidth(0)
    parts["cmedians"].set_color(TEXT_COLOR)
    parts["cmedians"].set_linewidth(1)
    for boundary in (0.5, len(columns) - 1.5):
        axes.axvline(boundary, color=MUTED_COLOR, linewidth=0.8)
    axes.axhline(0, color=MUTED_COLOR, linewidth=0.8)
    axes.set_xticks(range(len(columns)), [label for label, _, _, _ in columns], rotation=60, ha="right", fontsize=8)
    axes.set_ylabel("projection − default, per response", color=TEXT_COLOR)
    axes.set_title(with_model(model_label, "Controls: held-out in-role responses (line = median)"), color=TEXT_COLOR, fontsize=10)
    random_band_key(axes, (len(columns) - 1) / 2)
    figure.savefig(path, dpi=200, bbox_inches="tight", facecolor=SURFACE_COLOR)
    plt.close(figure)


def geometry_figures(root: Path, region: Region, layer: int, sections: list[Section], non_control_sections: list[Section],
                     groups: dict[GroupName, list[str]], pca_path: Path, pc1_path: Path, cosine_path: Path, centered_cosine_path: Path,
                     non_control_centered_cosine_path: Path, model_label: str) -> None:
    """2-D PCA, PC1 of the non-control personas alone, and pairwise cosine of the per-persona vectors (train questions,
    in-role), each minus the default vector; the second cosine plot also subtracts the mean over the plotted personas."""
    vectors_dir = root / f"vectors_{region}"
    def vector(role: str) -> Float[np.ndarray, "hidden"]:
        return torch.load(vectors_dir / f"{role}.pt", map_location="cpu", weights_only=False)["vector"][layer].float().numpy()
    default = vector("default")
    columns, spans = section_columns(sections, {path.stem for path in vectors_dir.glob("*.pt")})
    offsets = np.stack([vector(role) - default for role in columns])
    centered = offsets - offsets.mean(axis=0)
    _, singular_values, components = np.linalg.svd(centered, full_matrices=False)
    coordinates = centered @ components[:2].T
    explained = singular_values[:2] ** 2 / (singular_values ** 2).sum()
    marker_by_group: dict[GroupName, str] = {"deceptive_main": "o", "honest_main": "o", "deceptive_holdout": "s", "honest_holdout": "s",
                                             "deceptive_rh": "^", "honest_rh": "^", "controls": "D"}
    figure, axes = plt.subplots(figsize=(7.0, 6.0))
    style_axes(axes)
    axes.grid(axis="both", color="#e6e5e0", linewidth=0.6)
    for role, (x, y) in zip(columns, coordinates):
        persona_group = group_of(role, groups)
        axes.scatter(x, y, marker=marker_by_group[persona_group], s=46, color=COLOR_BY_HONESTY[HONESTY_BY_GROUP[persona_group]],
                     edgecolors=SURFACE_COLOR, linewidths=1.5, zorder=3)
    axes.set_aspect("equal", adjustable="datalim")
    label_without_overlap(axes, coordinates, [persona_label(role, groups) for role in columns])
    axes.set_xlabel(f"PC1 ({explained[0]:.0%})", color=TEXT_COLOR)
    axes.set_ylabel(f"PC2 ({explained[1]:.0%})", color=TEXT_COLOR)
    axes.set_title(with_model(model_label, f"Persona vectors − default, layer {layer} (labels thinned where they would overlap)"), color=TEXT_COLOR, fontsize=10)
    legend = [Patch(color=COLOR_BY_HONESTY[h], label=h) for h in HONESTIES]
    legend += [Line2D([], [], marker=marker, linestyle="", color=MUTED_COLOR, label=label)
               for marker, label in (("o", "in axis"), ("s", "held out"), ("^", "held out, RH-adjacent"), ("D", "control"))]
    axes.legend(handles=legend, frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    figure.savefig(pca_path, dpi=200, bbox_inches="tight", facecolor=SURFACE_COLOR)
    plt.close(figure)

    non_control = [index for index, role in enumerate(columns) if group_of(role, groups) != "controls"]
    pc1_figure(offsets[non_control], [columns[index] for index in non_control], groups, layer, pc1_path, model_label)

    labels = [persona_label(role, groups) for role in columns]
    cosine_figure(offsets, labels, spans, with_model(model_label, f"Persona vectors − default, layer {layer}"), cosine_path)
    cosine_figure(centered, labels, spans, with_model(model_label, f"Persona vectors − mean persona vector, layer {layer}"), centered_cosine_path)
    non_control_columns, non_control_spans = section_columns(non_control_sections, set(columns))
    non_control_offsets = np.stack([offsets[columns.index(role)] for role in non_control_columns])
    cosine_figure(non_control_offsets - non_control_offsets.mean(axis=0), [persona_label(role, groups) for role in non_control_columns], non_control_spans,
                  with_model(model_label, f"Persona vectors − mean persona vector, layer {layer}"), non_control_centered_cosine_path)


def label_without_overlap(axes: Axes, points: Float[np.ndarray, "point 2"], labels: list[str]) -> None:
    """Label the most isolated points first, skipping any label whose box would overlap one already placed."""
    axes.apply_aspect()
    distances = np.linalg.norm(points[:, None] - points[None], axis=-1) + np.diag(np.full(len(points), np.inf))
    placed = []
    for index in np.argsort(-distances.min(axis=1)):
        text = axes.annotate(labels[index], tuple(points[index]), xytext=(4, 3), textcoords="offset points", fontsize=7, color=TEXT_COLOR)
        box = text.get_window_extent().expanded(1.05, 1.15)
        if any(box.overlaps(other) for other in placed):
            text.remove()
        else:
            placed.append(box)


def spread_apart(targets: Float[np.ndarray, "point"], gap: float) -> Float[np.ndarray, "point"]:
    """Positions near the targets with neighbours at least gap apart (pairwise pushes until nothing overlaps)."""
    order = np.argsort(targets)
    positions = targets[order].astype(float)
    for _ in range(500):
        overlaps = gap - np.diff(positions)
        if (overlaps <= 1e-9).all():
            break
        push = np.where(overlaps > 0, overlaps / 2, 0.0)
        positions[:-1] -= push
        positions[1:] += push
    spread = np.empty_like(positions)
    spread[order] = positions
    return spread


def pc1_figure(offsets: Float[np.ndarray, "persona hidden"], roles: list[str], groups: dict[GroupName, list[str]], layer: int, path: Path,
               model_label: str) -> None:
    """Each persona's PC1 coordinate (PCA over these personas only), one column per persona group, deceptive side up.
    Columns sit close together; labels are gathered into lanes (one column of text shared by neighbouring columns,
    spread vertically together) joined to their points by leader lines. The deceptive and honest groups sit far apart
    on PC1, so a lane can run under or over the other honesty's columns."""
    centroid = offsets.mean(axis=0)
    _, singular_values, components = np.linalg.svd(offsets - centroid, full_matrices=False)
    pc1 = components[0]
    deceptive = [index for index, role in enumerate(roles) if HONESTY_BY_GROUP[group_of(role, groups)] == "deceptive"]
    pc1 = pc1 if ((offsets[deceptive] - centroid) @ pc1).mean() > 0 else -pc1
    coordinates = (offsets - centroid) @ pc1
    default_coordinate = float(-centroid @ pc1)
    column_spacing = 0.3  # x units; the figure is sized so one x unit is about an inch
    column_groups: list[tuple[GroupName, str, str]] = [  # (group, tick label, label lane)
        ("deceptive_main", "deceptive, in axis", "deceptive left"), ("deceptive_holdout", "deceptive, held out", "deceptive right"),
        ("deceptive_rh", "deceptive, RH-adjacent", "deceptive right"), ("honest_main", "honest, in axis", "honest left"),
        ("honest_holdout", "honest, held out", "honest right"), ("honest_rh", "honest, RH-adjacent", "honest right")]
    x_by_group = {group_name: column * column_spacing for column, (group_name, _, _) in enumerate(column_groups)}
    lane_columns: dict[str, list[GroupName]] = {}
    for group_name, _, lane in column_groups:
        lane_columns.setdefault(lane, []).append(group_name)
    # text anchored just past the lane's outermost column, on the side the lane faces
    lane_anchor = {lane: (min(x_by_group[g] for g in members) - 0.12, "right") if lane.endswith("left") else
                   (max(x_by_group[g] for g in members) + 0.12, "left") for lane, members in lane_columns.items()}
    x_low, x_high = -0.95, x_by_group["honest_rh"] + 1.15
    figure, axes = plt.subplots(figsize=((x_high - x_low) / 0.775, 7.0))
    style_axes(axes)
    low, high = min(coordinates.min(), default_coordinate), max(coordinates.max(), default_coordinate)
    margin = 0.06 * (high - low)
    axes.set_ylim(low - margin, high + margin)
    axes.set_xlim(x_low, x_high)
    axes_height_points = axes.get_position().height * figure.get_figheight() * 72
    gap = 9.0 / axes_height_points * (high - low + 2 * margin)
    axes.axhline(default_coordinate, color=MUTED_COLOR, linewidth=0.8, linestyle="--", zorder=1)
    axes.text(x_high - 0.05, default_coordinate, "default", ha="right", va="bottom", fontsize=8, color=MUTED_COLOR)
    for lane, members in lane_columns.items():
        points = [(index, x_by_group[group_name]) for group_name in members for index, role in enumerate(roles) if group_of(role, groups) == group_name]
        if not points:
            continue
        values = coordinates[[index for index, _ in points]]
        anchor_x, alignment = lane_anchor[lane]
        direction = 1 if alignment == "left" else -1
        for (index, x), value, label_y in zip(points, values, spread_apart(values, gap)):
            color = COLOR_BY_HONESTY[HONESTY_BY_GROUP[group_of(roles[index], groups)]]
            axes.scatter(x, value, s=40, color=color, edgecolors=SURFACE_COLOR, linewidths=1.5, zorder=3)
            axes.plot([x + direction * 0.03, anchor_x - direction * 0.02], [value, label_y], color="#c9c8c3", linewidth=0.6, zorder=2)
            axes.text(anchor_x, label_y, persona_label(roles[index], groups), ha=alignment, va="center", fontsize=7, color=TEXT_COLOR)
    axes.set_xticks(list(x_by_group.values()), [label for _, label, _ in column_groups], color=TEXT_COLOR, fontsize=8, rotation=45, ha="right",
                    rotation_mode="anchor")
    explained = singular_values[0] ** 2 / (singular_values ** 2).sum()
    axes.set_ylabel(f"PC1 ({explained:.0%} of variance), deceptive side up", color=TEXT_COLOR)
    axes.set_title(with_model(model_label, f"Persona vectors − default, layer {layer}:\nfirst principal component"), color=TEXT_COLOR, fontsize=10)
    figure.savefig(path, dpi=200, bbox_inches="tight", facecolor=SURFACE_COLOR)
    plt.close(figure)


def cosine_figure(vectors: Float[np.ndarray, "persona hidden"], labels: list[str], spans: list[tuple[str, int, int]], title: str, path: Path,
                  colormap: LinearSegmentedColormap = DIVERGING_CVD) -> None:
    unit_vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    cosine = unit_vectors @ unit_vectors.T
    figure, axes = plt.subplots(figsize=(0.26 * len(labels) + 3.0, 0.26 * len(labels) + 2.0))
    image = axes.imshow(cosine, cmap=colormap, vmin=-1, vmax=1)
    axes.set_xticks(range(len(labels)), labels, rotation=90, fontsize=7)
    axes.set_yticks(range(len(labels)), labels, fontsize=7)
    for _, start, _ in spans[1:]:
        axes.axvline(start - 0.5, color=SURFACE_COLOR, linewidth=2)
        axes.axhline(start - 0.5, color=SURFACE_COLOR, linewidth=2)
    for side in axes.spines.values():
        side.set_visible(False)
    figure.colorbar(image, ax=axes, fraction=0.04, pad=0.02).set_label("cosine similarity", color=TEXT_COLOR)
    axes.set_title(title, color=TEXT_COLOR, fontsize=10)
    figure.savefig(path, dpi=200, bbox_inches="tight", facecolor=SURFACE_COLOR)
    plt.close(figure)


def question_ids_by_set(questions_path: Path, recipe: QuestionRecipe) -> dict[QuestionSet, Int[np.ndarray, "subset"]]:
    questions = [json.loads(line) for line in questions_path.read_text().splitlines() if line.strip()]
    def ids(keep) -> Int[np.ndarray, "subset"]:
        return np.array([q["id"] for q in questions if keep(q)])
    if recipe == "generic_fit":
        return {"train": ids(lambda q: q["group"] == "generic"), "heldout": ids(lambda q: q["group"] != "generic"),
                "heldout_task": ids(lambda q: q["group"] == "task"), "heldout_other": ids(lambda q: q["group"] == "intent")}
    return {"train": ids(lambda q: q["split"] == "train"), "heldout": ids(lambda q: q["split"] == "heldout"),
            "heldout_task": ids(lambda q: q["split"] == "heldout" and q["group"] == "task"),
            "heldout_other": ids(lambda q: q["split"] == "heldout" and q["group"] != "task")}


_SCRIPT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class RunConfig:
    """Extraction root and axis in, figures and summary JSON out."""

    root: Path  # extraction root with act_<region>/, scores/, vectors_<region>/
    region: Literal["with_cot", "response_only"]
    axis: Path  # (layers, hidden) axis artifact, e.g. build_vectors.py's persona_contrast.pt
    layer: int
    out: Path
    groups: Path = _SCRIPT_ROOT / "data/persona_groups.json"
    questions: Path = _SCRIPT_ROOT / "data/questions.jsonl"
    question_recipe: QuestionRecipe = "split"
    random_count: int = 256
    seed: int = 1729
    controls_violins_skip: tuple[str, ...] = ("neutral_fic",)  # controls left out of the controls violin plot only
    model_label: str = ""  # model name for figure titles, e.g. 'GPT-OSS 120B'


def main(run: RunConfig) -> None:
    groups: dict[GroupName, list[str]] = json.loads(run.groups.read_text())
    question_ids = question_ids_by_set(run.questions, run.question_recipe)
    question_count = max(int(ids.max()) for ids in question_ids.values() if len(ids)) + 1
    default_vector = torch.load(run.root / f"vectors_{run.region}" / "default.pt", map_location="cpu", weights_only=False)["vector"]
    num_layers, hidden = default_vector.shape
    unit_by_layer = load_unit_axes(run.axis, list(range(num_layers)), num_layers, hidden)
    unit_axes = torch.as_tensor(np.stack([unit_by_layer[layer]["direction"] for layer in range(num_layers)]))
    random_axes = torch.as_tensor(np.stack([random_directions(hidden, run.random_count, run.seed, layer) for layer in range(num_layers)]))

    roles = ["default", *[role for roles in groups.values() for role in roles]]
    projections_by_role = {role: p for role in roles if (p := load_projections(run.root, run.region, role, unit_axes, random_axes)) is not None}
    default = projections_by_role.pop("default")
    relative_by_role = {role: relative_to_default(p, default, question_count) for role, p in projections_by_role.items()}

    rng = np.random.default_rng(run.seed)
    summaries: dict[str, dict[QuestionSet, Summary | None]] = {role: {question_set: summarize(*per_question_at_layer(relative, run.layer, ids, question_count), rng)
                        for question_set, ids in question_ids.items()}
                 for role, relative in relative_by_role.items()}

    rh_section: Section = ("RH-adjacent", [groups["deceptive_rh"], groups["honest_rh"]])
    not_rh_section: Section = ("not RH-adjacent", [groups["deceptive_main"] + groups["deceptive_holdout"], groups["honest_main"] + groups["honest_holdout"]])
    deceptive_section: Section = ("deceptive", [groups["deceptive_rh"], groups["deceptive_main"] + groups["deceptive_holdout"]])
    honest_section: Section = ("honest", [groups["honest_rh"], groups["honest_main"] + groups["honest_holdout"]])
    controls_section: Section = ("controls", [groups["controls"]])
    rh_deceptive_only: Section = ("RH-adjacent\ndeceptive", [groups["deceptive_rh"]])
    rh_honest_only: Section = ("RH-adj.\nhonest", [groups["honest_rh"]])
    deceptive_only: Section = ("deceptive", [groups["deceptive_main"] + groups["deceptive_holdout"]])
    honest_only: Section = ("honest", [groups["honest_main"] + groups["honest_holdout"]])
    heatmap_sections = [rh_deceptive_only, rh_honest_only, deceptive_only, honest_only, controls_section]
    subsections = [subsection for _, subs in (rh_section, not_rh_section, deceptive_section, honest_section, controls_section) for subsection in subs]
    subsections += [groups["deceptive_main"], groups["honest_main"]]
    group_summaries = {" ".join(subsection): group_summary(relative_by_role, subsection, run.layer, question_ids["heldout"], question_count, rng)
                       for subsection in subsections}

    run.out.mkdir(parents=True, exist_ok=True)
    prefix = f"{run.region}_L{run.layer}"
    model_label = run.model_label
    heatmap_figure(summaries, heatmap_sections, groups, run.out / f"{prefix}_heatmap.png", model_label)
    bar_figure(summaries, group_summaries, [rh_section, not_rh_section], groups, "RH-adjacent vs not", run.out / f"{prefix}_bars_rh.png", model_label)
    bar_figure(summaries, group_summaries, [deceptive_section, honest_section], groups, "Deceptive vs honest", run.out / f"{prefix}_bars_honesty.png", model_label)
    bar_figure(summaries, group_summaries, [controls_section], groups, "Controls", run.out / f"{prefix}_bars_controls.png", model_label,
               references=(("deceptive axis personas", group_summaries[" ".join(groups["deceptive_main"])]),
                           ("honest axis personas", group_summaries[" ".join(groups["honest_main"])])))
    generalization_figure(summaries, groups, run.out / f"{prefix}_train_vs_heldout.png", model_label)
    distribution_figure(relative_by_role, heatmap_sections, groups, run.layer, question_ids["heldout"], run.out / f"{prefix}_distributions.png", model_label)
    split_sections: list[Section] = [("deceptive\nin axis", [groups["deceptive_main"]]), ("deceptive\nheld out", [groups["deceptive_holdout"]]),
                                     ("deceptive\nRH-adjacent", [groups["deceptive_rh"]]), ("honest\nin axis", [groups["honest_main"]]),
                                     ("honest\nheld out", [groups["honest_holdout"]]), ("honest\nRH-adjacent", [groups["honest_rh"]])]
    distribution_figure(relative_by_role, split_sections, groups, run.layer, question_ids["heldout"], run.out / f"{prefix}_distributions_no_controls.png",
                        model_label, random_bands=True, sort_by_median=True)
    controls_distribution_figure(relative_by_role, groups, run.layer, question_ids["heldout"], run.out / f"{prefix}_distributions_controls.png",
                                 model_label, frozenset(run.controls_violins_skip))
    persona_sets = {"axis personas": (groups["deceptive_main"], groups["honest_main"]),
                    "held-out personas": (groups["deceptive_holdout"], groups["honest_holdout"]),
                    "RH-adjacent personas": (groups["deceptive_rh"], groups["honest_rh"])}
    aurocs = {label: result for label, (deceptive, honest) in persona_sets.items()
              if (result := auroc_by_layer(relative_by_role, deceptive, honest, question_ids["heldout"])) is not None}
    if aurocs:
        layer_sweep_figure(aurocs, run.layer, run.region, run.out / f"{run.region}_layer_sweep.png", model_label)
    relative_with_default = relative_by_role | {"default": relative_to_default(default, default, question_count)}
    honest_sets = {"axis personas": groups["honest_main"], "held-out personas": groups["honest_holdout"], "RH-adjacent personas": groups["honest_rh"]}
    honest_sets |= {role: [role] for roles in honest_sets.values() for role in roles}
    honest_vs_default_by_question_set = {question_set: {label: auroc_versus_random(relative_with_default, roles, ["default"], run.layer, question_ids[question_set])
                                                        for label, roles in honest_sets.items()}
                                         for question_set in ("heldout", "train")}
    geometry_figures(run.root, run.region, run.layer, [*split_sections, controls_section], split_sections, groups,
                     run.out / f"{prefix}_pca.png", run.out / f"{prefix}_pc1.png", run.out / f"{prefix}_cosine.png", run.out / f"{prefix}_cosine_centered.png",
                     run.out / f"{prefix}_cosine_centered_no_controls.png", model_label)
    (run.out / f"{prefix}_summary.json").write_text(json.dumps(
        {"axis": str(run.axis), "root": str(run.root), "persona": summaries, "group": group_summaries,
         "honest_vs_default_auroc": honest_vs_default_by_question_set,
         "auroc_by_layer": {label: {"axis": axis.tolist(), "random_mean": random.mean(axis=1).tolist(), "random_sd": random.std(axis=1).tolist()}
                            for label, (axis, random) in aurocs.items()}}, indent=2))
    logger.info(f"figures in {run.out}")


if __name__ == "__main__":
    from jsonargparse import auto_cli

    auto_cli(main)
