from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from jaxtyping import Float, Int8
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, LogNorm, to_rgb
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from selfconcept.assistant_axis_cot.ablation_statistics import RANDOM_MEAN_NAME, AblationAnalysis, KLGrid
from selfconcept.assistant_axis_cot.records import (
    COMPLETION_REGIONS,
    LABELLED_SEQUENCE_REGIONS,
    AxisName,
    CompletionRegion,
    Condition,
    SequenceProjectionRecord,
    SequenceRegion,
    TokenProjectionRecord,
)
from selfconcept.assistant_axis_cot.statistics import (
    CONTENT_REGIONS,
    ContentRegion,
    ConversationSummary,
    Measure,
    token_values,
)

SURFACE = "#fcfcfb"
PRIMARY_INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
COLOR_BY_CONDITION: dict[Condition, str] = {"unprompted": "#2a78d6", "persona": "#eb6834"}
COLOR_BY_REGION: dict[CompletionRegion, str] = {"cot": "#2a78d6", "final": "#eb6834", "delimiter": "#1baf7a"}
COLOR_BY_SEQUENCE_REGION: dict[SequenceRegion, str] = {
    "prompt": "#1baf7a",
    "cot": COLOR_BY_REGION["cot"],
    "final": COLOR_BY_REGION["final"],
}
SEQUENTIAL_BLUE = LinearSegmentedColormap.from_list(
    "sequential_blue",
    ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab",
     "#184f95", "#104281", "#0d366b"],
).with_extremes(under="#cde2fb", bad="#cde2fb")
NEUTRAL_CELL = "#f0efec"
MEASURE_LABEL: dict[Measure, str] = {"projection": "projection onto axis", "cosine": "cosine with axis"}


def style_axes(ax: Axes) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRIDLINE, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(colors=SECONDARY_INK, labelsize=8)


def percent_of_response_bin_means(
    record: TokenProjectionRecord, axis: AxisName, measure: Measure, bins_per_region: int
) -> Float[np.ndarray, " bin"]:
    """Mean value per bin, CoT rescaled onto the first bins_per_region bins and the final response onto the rest;
    NaN for a bin no token lands in. Delimiter tokens are left out."""
    values = token_values(record, axis, measure)
    bin_means_by_region = []
    for region in ("cot", "final"):
        region_values = values[record["region_codes"] == COMPLETION_REGIONS.index(region)]
        bin_indices = np.arange(len(region_values)) * bins_per_region // len(region_values)
        bin_sums = np.bincount(bin_indices, weights=region_values, minlength=bins_per_region)
        bin_counts = np.bincount(bin_indices, minlength=bins_per_region)
        with np.errstate(invalid="ignore"):
            bin_means_by_region.append(bin_sums / bin_counts)
    return np.concatenate(bin_means_by_region)


def plot_percent_of_response(
    records: Sequence[TokenProjectionRecord],
    axis: AxisName,
    measure: Measure,
    title: str,
    bins_per_region: int = 50,
) -> Figure:
    """Mean and 10-90th percentile band across conversations, per condition, with each conversation's CoT
    stretched over 0-50% and its final response over 50-100%. Records must have tokens in both regions."""
    figure, ax = plt.subplots(figsize=(9, 4.5), facecolor=SURFACE)
    style_axes(ax)
    bin_centers = (np.arange(2 * bins_per_region) + 0.5) * 100 / (2 * bins_per_region)
    for condition, color in COLOR_BY_CONDITION.items():
        condition_records = [record for record in records if record["condition"] == condition]
        if not condition_records:
            continue
        bin_means = np.stack([
            percent_of_response_bin_means(record, axis, measure, bins_per_region) for record in condition_records
        ])
        low, high = np.nanpercentile(bin_means, [10, 90], axis=0)
        ax.fill_between(bin_centers, low, high, color=color, alpha=0.15, linewidth=0)
        ax.plot(
            bin_centers, np.nanmean(bin_means, axis=0), color=color, linewidth=2,
            label=f"{condition} (n={len(condition_records)})",
        )
    ax.axvline(50, color=SECONDARY_INK, linewidth=1, linestyle="--")
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100], ["CoT start", "CoT 50%", "final start", "final 50%", "end"])
    ax.set_ylabel(MEASURE_LABEL[measure], color=SECONDARY_INK)
    ax.set_title(title, color=PRIMARY_INK, fontsize=11, loc="left")
    ax.legend(frameon=False, labelcolor=SECONDARY_INK, fontsize=8, title="mean, 10–90% band", title_fontsize=8)
    figure.tight_layout()
    return figure


def centered_rolling_mean(values: Float[np.ndarray, " n_tokens"], window: int) -> Float[np.ndarray, " n_tokens"]:
    """Shrinks the window at the ends rather than padding."""
    kernel = np.ones(window)
    return np.convolve(values, kernel, mode="same") / np.convolve(np.ones_like(values), kernel, mode="same")


def draw_token_trace(ax: Axes, record: TokenProjectionRecord, axis: AxisName, measure: Measure, window: int) -> None:
    """Raw per-token values as faint dots and a rolling mean of the content tokens as a line, both colored by
    region; delimiter tokens are drawn as larger diamonds, outside the rolling mean, so a spike at the CoT/final
    transition stands out rather than being smeared over the window."""
    values = token_values(record, axis, measure)
    positions = np.arange(len(values))
    region_codes = record["region_codes"]
    content = region_codes != COMPLETION_REGIONS.index("delimiter")
    region_colors = np.array([COLOR_BY_REGION[COMPLETION_REGIONS[code]] for code in region_codes])

    content_positions = positions[content]
    content_colors = region_colors[content]
    ax.scatter(content_positions, values[content], s=2, c=content_colors, alpha=0.25, linewidths=0)
    rolling = centered_rolling_mean(values[content], window)
    segment_starts = np.column_stack([content_positions[:-1], rolling[:-1]])
    segment_ends = np.column_stack([content_positions[1:], rolling[1:]])
    ax.add_collection(
        LineCollection(list(np.stack([segment_starts, segment_ends], axis=1)), colors=list(content_colors[1:]), linewidths=1.5)
    )
    ax.scatter(
        positions[~content], values[~content], s=36, marker="D", c=COLOR_BY_REGION["delimiter"],
        edgecolors=SURFACE, linewidths=1, zorder=3,
    )


def region_legend_handles() -> list[Line2D]:
    return [
        Line2D([], [], color=COLOR_BY_REGION["cot"], linewidth=2, label="CoT"),
        Line2D([], [], color=COLOR_BY_REGION["final"], linewidth=2, label="final"),
        Line2D([], [], color=COLOR_BY_REGION["delimiter"], marker="D", linestyle="", label="delimiter token"),
    ]


def plot_absolute_position(
    records_by_condition: dict[Condition, Sequence[TokenProjectionRecord]],
    axis: AxisName,
    measure: Measure,
    title: str,
    rolling_window: int = 32,
) -> Figure:
    """One panel per conversation (rows = condition), token position on x."""
    column_count = max(len(records) for records in records_by_condition.values())
    figure, axes = plt.subplots(
        len(records_by_condition), column_count, figsize=(3.2 * column_count, 2.6 * len(records_by_condition)),
        sharey=True, squeeze=False, facecolor=SURFACE,
    )
    for row, (condition, records) in enumerate(records_by_condition.items()):
        for column in range(column_count):
            ax = axes[row, column]
            if column >= len(records):
                ax.set_visible(False)
                continue
            style_axes(ax)
            record = records[column]
            draw_token_trace(ax, record, axis, measure, rolling_window)
            ax.set_title(f"{condition}: {record['role']}, q{record['question_index']}", color=PRIMARY_INK, fontsize=8, loc="left")
        axes[row, 0].set_ylabel(MEASURE_LABEL[measure], color=SECONDARY_INK, fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("token position in completion", color=SECONDARY_INK, fontsize=8)
    figure.legend(handles=region_legend_handles(), loc="upper right", ncols=3, frameon=False, fontsize=8, labelcolor=SECONDARY_INK)
    figure.suptitle(f"{title} (line: {rolling_window}-token rolling mean)", color=PRIMARY_INK, fontsize=11, x=0.01, ha="left")
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    return figure


def boundary_aligned_matrices(
    records: Sequence[TokenProjectionRecord], axis: AxisName, measure: Measure, half_width: int
) -> tuple[Float[np.ndarray, "conversation offset"], Int8[np.ndarray, "conversation offset"]]:
    """Values and region codes at offsets -half_width..half_width-1 from each conversation's first final token;
    NaN values (and code -1) where the conversation has no token at that offset."""
    values_matrix = np.full((len(records), 2 * half_width), np.nan)
    region_code_matrix = np.full((len(records), 2 * half_width), -1, dtype=np.int8)
    for row, record in enumerate(records):
        values = token_values(record, axis, measure)
        first_final = int(np.argmax(record["region_codes"] == COMPLETION_REGIONS.index("final")))
        start, end = max(first_final - half_width, 0), min(first_final + half_width, len(values))
        columns = slice(start - first_final + half_width, end - first_final + half_width)
        values_matrix[row, columns] = values[start:end]
        region_code_matrix[row, columns] = record["region_codes"][start:end]
    return values_matrix, region_code_matrix


def draw_boundary_aligned(
    ax: Axes, records: Sequence[TokenProjectionRecord], axis: AxisName, measure: Measure, half_width: int
) -> None:
    """Per region, the mean and 10-90th percentile band across conversations at each offset, wherever at least
    a tenth of the conversations have a token of that region."""
    offsets = np.arange(-half_width, half_width)
    values_matrix, region_code_matrix = boundary_aligned_matrices(records, axis, measure, half_width)
    min_conversations = max(3, len(records) // 10)
    for region in COMPLETION_REGIONS:
        region_values_matrix = np.where(region_code_matrix == COMPLETION_REGIONS.index(region), values_matrix, np.nan)
        enough = np.count_nonzero(~np.isnan(region_values_matrix), axis=0) >= min_conversations
        if not enough.any():
            continue
        mean, low, high = np.full((3, len(offsets)), np.nan)
        mean[enough] = np.nanmean(region_values_matrix[:, enough], axis=0)
        low[enough], high[enough] = np.nanpercentile(region_values_matrix[:, enough], [10, 90], axis=0)
        color = COLOR_BY_REGION[region]
        if region == "delimiter":
            ax.errorbar(
                offsets[enough], mean[enough], yerr=[mean[enough] - low[enough], high[enough] - mean[enough]],
                fmt="D", color=color, markersize=5, markeredgecolor=SURFACE, elinewidth=1, zorder=3,
            )
        else:
            ax.fill_between(offsets, low, high, color=color, alpha=0.15, linewidth=0)
            ax.plot(offsets, mean, color=color, linewidth=2)
    ax.axvline(0, color=SECONDARY_INK, linewidth=1, linestyle="--")


def plot_boundary_aligned(
    records: Sequence[TokenProjectionRecord],
    axis: AxisName,
    measure: Measure,
    title: str,
    half_width: int = 200,
) -> Figure:
    """One panel per condition; x is the token offset from the first final-response token."""
    figure, axes = plt.subplots(1, len(COLOR_BY_CONDITION), figsize=(11, 4.2), sharey=True, facecolor=SURFACE)
    for ax, condition in zip(axes, COLOR_BY_CONDITION, strict=True):
        style_axes(ax)
        condition_records = [record for record in records if record["condition"] == condition]
        if condition_records:
            draw_boundary_aligned(ax, condition_records, axis, measure, half_width)
        ax.set_title(f"{condition} (n={len(condition_records)})", color=PRIMARY_INK, fontsize=9, loc="left")
        ax.set_xlabel("tokens from first final-response token", color=SECONDARY_INK, fontsize=8)
    axes[0].set_ylabel(MEASURE_LABEL[measure], color=SECONDARY_INK, fontsize=8)
    figure.legend(handles=region_legend_handles(), loc="upper right", ncols=3, frameon=False, fontsize=8, labelcolor=SECONDARY_INK)
    figure.suptitle(f"{title} (mean, 10–90% band)", color=PRIMARY_INK, fontsize=11, x=0.01, ha="left")
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    return figure


def plot_region_mean_scatter(summaries: Sequence[ConversationSummary], measure: Measure, title: str) -> Figure:
    """One dot per conversation, mean CoT value against mean final value; below the diagonal means the CoT
    projects lower than the final response."""
    figure, ax = plt.subplots(figsize=(5.5, 5.5), facecolor=SURFACE)
    style_axes(ax)
    for condition in ("persona", "unprompted"):
        condition_summaries = [summary for summary in summaries if summary["condition"] == condition]
        ax.scatter(
            [summary["moments_by_region"]["final"]["mean"] for summary in condition_summaries],
            [summary["moments_by_region"]["cot"]["mean"] for summary in condition_summaries],
            s=16, color=COLOR_BY_CONDITION[condition], alpha=0.5, edgecolors=SURFACE, linewidths=0.5,
            label=f"{condition} (n={len(condition_summaries)})",
        )
    low, high = (min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1]))
    ax.plot([low, high], [low, high], color=SECONDARY_INK, linewidth=1, linestyle="--", label="CoT = final")
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_xlabel(f"final-response mean {MEASURE_LABEL[measure]}", color=SECONDARY_INK, fontsize=8)
    ax.set_ylabel(f"CoT mean {MEASURE_LABEL[measure]}", color=SECONDARY_INK, fontsize=8)
    ax.set_title(title, color=PRIMARY_INK, fontsize=10, loc="left")
    ax.legend(frameon=False, labelcolor=SECONDARY_INK, fontsize=8)
    figure.tight_layout()
    return figure


def plot_persona_drop_by_role(
    drop_by_region_by_role: dict[str, dict[ContentRegion, float]], measure: Measure, title: str
) -> Figure:
    """Dumbbell per persona role: the unprompted minus persona mean in CoT and in the final response, sorted by
    the final-response drop."""
    roles = sorted(drop_by_region_by_role, key=lambda role: drop_by_region_by_role[role]["final"])
    rows = np.arange(len(roles))
    figure, ax = plt.subplots(figsize=(6.5, 0.28 * len(roles) + 1.4), facecolor=SURFACE)
    style_axes(ax)
    ax.grid(axis="y", visible=False)
    drops_by_region = {
        region: np.array([drop_by_region_by_role[role][region] for role in roles]) for region in CONTENT_REGIONS
    }
    ax.hlines(rows, drops_by_region["cot"], drops_by_region["final"], color=BASELINE, linewidth=2)
    for region in CONTENT_REGIONS:
        ax.scatter(
            drops_by_region[region], rows, s=40, color=COLOR_BY_REGION[region], edgecolors=SURFACE, linewidths=1,
            zorder=3, label="CoT" if region == "cot" else region,
        )
    ax.axvline(0, color=SECONDARY_INK, linewidth=1)
    ax.set_yticks(rows, roles)
    ax.set_xlabel(f"unprompted − persona mean {MEASURE_LABEL[measure]}", color=SECONDARY_INK, fontsize=8)
    ax.set_title(title, color=PRIMARY_INK, fontsize=10, loc="left")
    ax.legend(frameon=False, labelcolor=SECONDARY_INK, fontsize=8, loc="lower right")
    figure.tight_layout()
    return figure


def plot_residual_norm_by_position(
    records: Sequence[SequenceProjectionRecord], title: str, min_transcripts: int
) -> Figure:
    """Mean, 10-90th percentile band and maximum across transcripts of the residual norm at each absolute token
    position, over the positions at least min_transcripts transcripts reach."""
    norms = np.full((len(records), max(len(record["residual_norms"]) for record in records)), np.nan)
    for row, record in enumerate(records):
        norms[row, : len(record["residual_norms"])] = record["residual_norms"]
    transcript_count_by_position = np.count_nonzero(~np.isnan(norms), axis=0)
    has_enough_transcripts = transcript_count_by_position >= min_transcripts
    positions = np.arange(norms.shape[1])[has_enough_transcripts]
    norms_at_positions = norms[:, has_enough_transcripts]
    low, high = np.nanpercentile(norms_at_positions, [10, 90], axis=0)

    figure, ax = plt.subplots(figsize=(9, 4.5), facecolor=SURFACE)
    style_axes(ax)
    ax.fill_between(positions, low, high, color=COLOR_BY_REGION["cot"], alpha=0.12, linewidth=0, label="10–90% band")
    ax.plot(positions, np.nanmean(norms_at_positions, axis=0), color=COLOR_BY_REGION["cot"], linewidth=2, label="mean")
    ax.scatter(
        positions, np.nanmax(norms_at_positions, axis=0), s=4, color=COLOR_BY_REGION["final"], linewidths=0, label="max"
    )
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlim(0, positions[-1])
    ax.set_xlabel("token position (prompt and completion)", color=SECONDARY_INK)
    ax.set_ylabel("residual norm", color=SECONDARY_INK)
    ax.set_title(
        f"{title}\nresidual norm by position, {len(records)} transcripts", color=PRIMARY_INK, fontsize=11, loc="left"
    )
    ax.legend(frameon=False, labelcolor=SECONDARY_INK, fontsize=8)
    figure.tight_layout()
    return figure


def is_dark(color: tuple[float, float, float, float] | str) -> bool:
    red, green, blue = to_rgb(color)
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue < 0.45


def draw_kl_grid(ax: Axes, grid: KLGrid, norm: LogNorm, name: str) -> None:
    """Upper-triangle cells colored by mean KL and labelled with it and its CI; the causally unaffected cells below
    the diagonal are neutral."""
    region_count = len(LABELLED_SEQUENCE_REGIONS)
    for ablated_index, ablated in enumerate(LABELLED_SEQUENCE_REGIONS):
        for measured_index, measured in enumerate(LABELLED_SEQUENCE_REGIONS):
            cell = grid[ablated][measured]
            is_unaffected = measured_index < ablated_index
            fill = NEUTRAL_CELL if is_unaffected else SEQUENTIAL_BLUE(norm(cell["mean"]))
            ax.add_patch(
                Rectangle(
                    (measured_index, ablated_index), 1, 1, facecolor=fill, edgecolor=SURFACE, linewidth=2
                )
            )
            label = (
                f"unaffected\n{cell['mean']:.2g}"
                if is_unaffected
                else f"{cell['mean']:.3g}\n[{cell['ci_low']:.2g}, {cell['ci_high']:.2g}]"
            )
            ax.text(
                measured_index + 0.5, ablated_index + 0.5, label, ha="center", va="center", fontsize=8,
                color="white" if not is_unaffected and is_dark(fill) else SECONDARY_INK,
            )
    ax.set_xlim(0, region_count)
    ax.set_ylim(region_count, 0)
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(region_count) + 0.5, LABELLED_SEQUENCE_REGIONS)
    ax.set_yticks(np.arange(region_count) + 0.5, LABELLED_SEQUENCE_REGIONS)
    ax.tick_params(colors=SECONDARY_INK, labelsize=9, length=0)
    ax.xaxis.set_ticks_position("top")
    ax.set_xlabel("measured region", color=SECONDARY_INK)
    ax.xaxis.set_label_position("top")
    ax.set_title(name, color=PRIMARY_INK, fontsize=10, loc="left", pad=8)
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_ablation_kl_grids(analysis: AblationAnalysis, title: str) -> Figure:
    """The axis's KL grid beside the random baseline's, on one shared log color scale."""
    names = [name for name in (analysis["axis_name"], RANDOM_MEAN_NAME) if name in analysis["kl_grid_by_name"]]
    affected_means = [
        analysis["kl_grid_by_name"][name][ablated][measured]["mean"]
        for name in names
        for ablated_index, ablated in enumerate(LABELLED_SEQUENCE_REGIONS)
        for measured in LABELLED_SEQUENCE_REGIONS[ablated_index:]
    ]
    positive_means = [mean for mean in affected_means if mean > 0]
    norm = LogNorm(vmin=min(positive_means), vmax=max(positive_means))

    figure, axes = plt.subplots(
        1, len(names), figsize=(4.6 * len(names) + 1.2, 5.2), facecolor=SURFACE, layout="constrained"
    )
    for ax, name in zip(np.atleast_1d(axes), names, strict=True):
        draw_kl_grid(ax, analysis["kl_grid_by_name"][name], norm, name.replace("_", " "))
    np.atleast_1d(axes)[0].set_ylabel("ablated region", color=SECONDARY_INK)
    colorbar = figure.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=SEQUENTIAL_BLUE), ax=axes, fraction=0.04)
    colorbar.set_label("mean per-token KL(clean ‖ ablated)", color=SECONDARY_INK, fontsize=9)
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(colors=SECONDARY_INK, labelsize=8)
    figure.suptitle(title, color=PRIMARY_INK, fontsize=11, x=0.02, ha="left")
    return figure


def plot_surviving_shift(analysis: AblationAnalysis, title: str) -> Figure:
    """One panel per ablated region: the RMS shift in each layer's axis projection, as a fraction of the injected
    shift, on the ablated region and the regions after it."""
    fraction_by_measured_by_ablated = analysis["surviving_shift_fraction_by_measured_by_ablated"]
    figure, axes = plt.subplots(
        1, len(fraction_by_measured_by_ablated), figsize=(12, 4), sharey=True, facecolor=SURFACE
    )
    for ax, (ablated, fraction_by_measured) in zip(axes, fraction_by_measured_by_ablated.items(), strict=True):
        style_axes(ax)
        ax.axhline(1, color=BASELINE, linewidth=1)
        for measured, fractions in fraction_by_measured.items():
            ax.plot(
                analysis["layers"], fractions, color=COLOR_BY_SEQUENCE_REGION[measured], linewidth=2, label=measured
            )
            ax.annotate(
                measured, (analysis["layers"][-1], fractions[-1]), xytext=(4, 0), textcoords="offset points",
                va="center", fontsize=8, color=SECONDARY_INK,
            )
        ax.set_xlim(analysis["layers"][0], analysis["layers"][-1])
        ax.set_title(f"{ablated} ablated", color=PRIMARY_INK, fontsize=10, loc="left")
        ax.set_xlabel("layer", color=SECONDARY_INK)
    largest_fraction = max(
        max(fractions) for fraction_by_measured in fraction_by_measured_by_ablated.values()
        for fractions in fraction_by_measured.values()
    )
    axes[0].set_ylim(0, 1.05 * max(largest_fraction, 1))
    axes[0].set_ylabel("RMS axis-projection shift / injected", color=SECONDARY_INK)
    figure.legend(
        handles=[
            Line2D([], [], color=color, linewidth=2, label=region) for region, color in COLOR_BY_SEQUENCE_REGION.items()
        ],
        loc="upper right", ncols=len(COLOR_BY_SEQUENCE_REGION), frameon=False, labelcolor=SECONDARY_INK, fontsize=8,
        title="measured region", title_fontsize=8,
    )
    figure.suptitle(title, color=PRIMARY_INK, fontsize=11, x=0.02, ha="left")
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    return figure
