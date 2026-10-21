import html
import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import cast

import numpy as np
import plotly.graph_objects as go
from plotly.colors import qualitative, sample_colorscale
from plotly.subplots import make_subplots
from jsonargparse import auto_cli

from selfconcept.arm_search.episode_tally import is_hit
from selfconcept.arm_search.interface import EpisodeResult, Tier, TierName
from selfconcept.arm_search.records import RoundRecord
from selfconcept.arm_search.state import CONFIG_FILE_NAME, load_episodes, load_rounds

logger = logging.getLogger(__name__)
TOP_GROUP_COUNT = 30
WARM_UP_ROUND_COUNT = 5
TIER_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
PANEL_HEIGHT = 420
HEAT_SCALE_TOP = 0.75  # the darkest reds hide black text
TARGET_MARK_COLOR = "#d62728"
CATEGORY_COLORS = qualitative.Pastel1 + qualitative.Pastel2 + qualitative.Set3


def _tier_color(tier_position: int) -> str:
    return TIER_COLORS[tier_position % len(TIER_COLORS)]


def _coverage_figure(rounds: list[RoundRecord], ladder: list[Tier]) -> go.Figure:
    """Groups covered per tier, against time or cumulative cost (buttons switch)."""
    results = [record["result"] for record in rounds if record["result"] is not None]
    x_values_by_axis = {
        "time": [result["completed_at"] for result in results],
        "cost": np.cumsum([result["realized_cost"] for result in results]).tolist(),
    }
    figure = go.Figure(
        [
            go.Scatter(
                x=x_values,
                y=np.cumsum([result["realized_reward_by_tier"][tier["tier_name"]] for result in results]).tolist(),
                mode="lines+markers",
                name=tier["tier_name"],
                line={"color": _tier_color(tier_position)},
                visible=x_axis == "time",
            )
            for x_axis, x_values in x_values_by_axis.items()
            for tier_position, tier in enumerate(ladder)
        ]
    )
    buttons = [
        {
            "label": x_axis,
            "method": "update",
            "args": [
                {"visible": [shown_axis == x_axis for shown_axis in x_values_by_axis for _ in ladder]},
                {"xaxis.title.text": x_axis, "xaxis.type": "date" if x_axis == "time" else "linear"},
            ],
        }
        for x_axis in x_values_by_axis
    ]
    return figure.update_layout(
        title="Groups covered per tier",
        xaxis_title="time",
        yaxis_title="groups",
        updatemenus=[{"type": "buttons", "direction": "right", "buttons": buttons, "x": 1.0, "y": 1.15}],
    )


def _tier_choice_figure(rounds: list[RoundRecord], ladder: list[Tier]) -> go.Figure:
    """Expected hits per tier (top, scaled past the warm-up rounds) and the realized hits of the tier played (bottom,
    colored by tier)."""
    figure = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.06)
    round_indices = [record["plan"]["round_index"] for record in rounds]
    for tier_position, tier in enumerate(ladder):
        figure.add_trace(
            go.Scatter(
                x=round_indices,
                y=[record["plan"]["expected_hits_by_tier"].get(tier["tier_name"]) for record in rounds],
                mode="lines",
                name=tier["tier_name"],
                line={"color": _tier_color(tier_position)},
                legendgroup=tier["tier_name"],
            ),
            row=1,
            col=1,
        )
        played_rounds = [
            record for record in rounds if record["plan"]["tier_name"] == tier["tier_name"] and record["result"]
        ]
        figure.add_trace(
            go.Scatter(
                x=[record["plan"]["round_index"] for record in played_rounds],
                y=[
                    record["result"]["realized_hits_by_tier"][tier["tier_name"]]
                    for record in played_rounds
                    if record["result"] is not None
                ],
                mode="markers",
                marker={"color": _tier_color(tier_position), "size": 5},
                name=tier["tier_name"],
                legendgroup=tier["tier_name"],
                showlegend=False,
            ),
            row=2,
            col=1,
        )
    expected_after_warm_up = [
        value
        for record in rounds[WARM_UP_ROUND_COUNT:]
        for value in record["plan"]["expected_hits_by_tier"].values()
    ]
    if expected_after_warm_up:
        figure.update_yaxes(range=[0, 1.1 * max(expected_after_warm_up)], row=1, col=1)
    figure.update_yaxes(title_text="expected hits", row=1, col=1)
    figure.update_yaxes(title_text="realized", row=2, col=1)
    figure.update_xaxes(title_text="round", row=2, col=1)
    return figure.update_layout(
        title=f"Expected hits per tier (scaled from round {WARM_UP_ROUND_COUNT}); realized hits of the tier played"
    )


def _fit_health_figure(rounds: list[RoundRecord]) -> go.Figure:
    round_indices = [record["plan"]["round_index"] for record in rounds]
    warned_round_indices = [record["plan"]["round_index"] for record in rounds if record["plan"]["fit_warnings"]]
    figure = go.Figure(
        [
            go.Scatter(
                x=round_indices,
                y=[record["plan"]["fit_diagnostic_value_by_name"].get(name) for record in rounds],
                mode="lines+markers",
                marker={"size": 4},
                name=name,
                hovertext=["; ".join(record["plan"]["fit_warnings"]) for record in rounds],
                yaxis="y2" if name == "divergence_count" else "y",
            )
            for name in ["max_r_hat", "divergence_count"]
        ]
    )
    for round_index in warned_round_indices:
        figure.add_vline(x=round_index, line={"color": "red", "width": 1}, opacity=0.3)
    return figure.update_layout(
        title=f"Fit health (red lines: fit warnings, {len(warned_round_indices)} of {len(rounds)} rounds)",
        xaxis_title="round",
        yaxis={"title": "max R-hat"},
        yaxis2={"title": "divergences", "overlaying": "y", "side": "right"},
    )


def _effect_figure(rounds: list[RoundRecord]) -> go.Figure:
    estimates = rounds[-1]["plan"]["effect_estimates"] if rounds else []
    labels = [f"{estimate['effect_name']} {estimate['level_name']}".strip() for estimate in estimates]
    figure = go.Figure(
        go.Scatter(
            x=[estimate["mean"] for estimate in estimates],
            y=labels,
            mode="markers",
            error_x={
                "type": "data",
                "symmetric": False,
                "array": [estimate["upper"] - estimate["mean"] for estimate in estimates],
                "arrayminus": [estimate["mean"] - estimate["lower"] for estimate in estimates],
            },
        )
    )
    return figure.update_layout(
        title="Effect estimates at the latest fit (mean, 90% interval)", yaxis={"tickfont": {"size": 10}}
    )


def _hits_per_group_figure(episodes: list[EpisodeResult], ladder: list[Tier]) -> go.Figure:
    """The top groups by lenient hits plus every group with a target hit (red), on a linear or log axis."""
    hit_count_by_group_by_tier: dict[TierName, Counter[str]] = {
        tier["tier_name"]: Counter(episode["arm"]["group_id"] for episode in episodes if is_hit(episode, tier))
        for tier in ladder
    }
    lenient_counts = hit_count_by_group_by_tier[ladder[0]["tier_name"]]
    target_group_ids = set(hit_count_by_group_by_tier[ladder[-1]["tier_name"]])
    top_group_ids = {group_id for group_id, _ in lenient_counts.most_common(TOP_GROUP_COUNT)}
    shown_group_ids = sorted(top_group_ids | target_group_ids, key=lambda group_id: -lenient_counts[group_id])
    figure = go.Figure(
        [
            go.Bar(
                x=shown_group_ids,
                y=[hit_counts[group_id] for group_id in shown_group_ids],
                name=tier_name,
                marker={"color": _tier_color(tier_position)},
            )
            for tier_position, (tier_name, hit_counts) in enumerate(hit_count_by_group_by_tier.items())
        ]
    )
    scale_buttons = [
        {"label": scale, "method": "relayout", "args": [{"yaxis.type": scale}]} for scale in ["linear", "log"]
    ]
    return figure.update_layout(
        title=f"Hits per group and tier (top {TOP_GROUP_COUNT} by {ladder[0]['tier_name']} hits; "
        f"<span style='color:{TARGET_MARK_COLOR}'>red</span>: {len(target_group_ids)} with a target hit)",
        barmode="group",
        yaxis_title="episodes",
        xaxis={
            "tickmode": "array",
            "tickvals": shown_group_ids,
            "ticktext": [
                f"<span style='color:{TARGET_MARK_COLOR}'><b>{group_id}</b></span>"
                if group_id in target_group_ids
                else group_id
                for group_id in shown_group_ids
            ],
        },
        updatemenus=[{"type": "buttons", "direction": "right", "buttons": scale_buttons, "x": 1.0, "y": 1.15}],
    )


def _arm_setting_values(arm_id: str, group_id: str) -> list[str]:
    """An arm id's settings after its group, split on the separators the applications use (`|` and `.`)."""
    return re.split(r"[|.]", arm_id.removeprefix(f"{group_id}|"))


def _cell(value: str, background_color: str) -> str:
    return f"<td style='background:{background_color}'>{html.escape(value)}</td>"


def _top_arms_table(rounds: list[RoundRecord]) -> str:
    """Mean hit probability as log-scaled heat, then one column per arm setting, each distinct value in a color of
    its own."""
    top_arms = rounds[-1]["plan"]["top_estimated_arms"] if rounds else []
    played_tier = rounds[-1]["plan"]["tier_name"] if rounds else ""
    columns = list(
        zip(*[[arm["group_id"], *_arm_setting_values(arm["arm_id"], arm["group_id"])] for arm in top_arms])
    )
    color_by_value = {
        value: CATEGORY_COLORS[position % len(CATEGORY_COLORS)]
        for position, value in enumerate(dict.fromkeys(value for column in columns for value in column))
    }
    log_means = np.log([arm["success_probability_mean"] for arm in top_arms])
    log_mean_range = np.ptp(log_means) if len(top_arms) else 0.0
    heat_positions = (log_means - log_means.min()) / log_mean_range if log_mean_range > 0 else np.ones_like(log_means)
    rows = [
        "<tr>"
        + _cell(
            f"{arm['success_probability_mean']:.3g}",
            str(sample_colorscale("YlOrRd", HEAT_SCALE_TOP * heat_position)[0]),
        )
        + "".join(_cell(column[row_position], color_by_value[column[row_position]]) for column in columns)
        + "</tr>"
        for row_position, (arm, heat_position) in enumerate(zip(top_arms, heat_positions.tolist(), strict=True))
    ]
    header = "".join(
        f"<th>{name}</th>"
        for name in ["mean", "group", *[f"setting {position}" for position in range(1, len(columns))]]
    )
    return (
        f"<div style='height: {PANEL_HEIGHT}px; overflow: auto; font-size: 12px'>"
        f"<b>Top estimated arms at the latest plan (tier {played_tier}, per-seed hit probability)</b>"
        f"<table style='border-collapse: collapse; white-space: nowrap' border='1' cellpadding='3'>"
        f"<tr>{header}</tr>{''.join(rows)}</table></div>"
    )


def build_dashboard(state_dir: Path, output_path: Path | None = None) -> None:
    """Write a static HTML dashboard of a search's state dir (default: dashboard.html inside it)."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    state_dir = state_dir.expanduser()
    ladder = cast(list[Tier], json.loads((state_dir / CONFIG_FILE_NAME).read_text())["ladder"])
    rounds, episodes = load_rounds(state_dir), load_episodes(state_dir)
    figures = [
        _coverage_figure(rounds, ladder),
        _tier_choice_figure(rounds, ladder),
        _fit_health_figure(rounds),
        _effect_figure(rounds),
        _hits_per_group_figure(episodes, ladder),
    ]
    body = "\n".join(
        [
            figure.update_layout(height=PANEL_HEIGHT, margin={"l": 50, "r": 30, "t": 60, "b": 40}).to_html(
                full_html=False, include_plotlyjs="cdn" if position == 0 else False
            )
            for position, figure in enumerate(figures)
        ]
        + [_top_arms_table(rounds)]
    )
    resolved_output_path = (output_path or state_dir / "dashboard.html").expanduser()
    resolved_output_path.write_text(
        f"<html><head><meta charset='utf-8'><title>arm search: {state_dir.name}</title></head>"
        f"<body style='margin: 8px; font-family: sans-serif'><h3>Arm search: {state_dir} ({len(rounds)} rounds,"
        f" {len(episodes)} episodes)</h3><div style='display: grid; grid-template-columns: 1fr 1fr; gap: 4px'>"
        f"{body}</div></body></html>"
    )
    logger.info("wrote %s", resolved_output_path)


if __name__ == "__main__":
    auto_cli(build_dashboard)
