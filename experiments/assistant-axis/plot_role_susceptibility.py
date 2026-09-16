# Derived from safety-research/assistant-axis (https://github.com/safety-research/assistant-axis),
# MIT licensed. See the NOTICE file at the repository root for the full license text.
"""Plot role-adoption fractions vs. steering strength (paper §3.2.1, Figure 4).

Reads the judged JSONL produced by ``judge_role_susceptibility.py`` (each line carries a
``label`` and a ``steering_coefficient``) and draws one line per role-adoption category: the
percentage of responses that received that label at each steering coefficient.

Fractions at a given coefficient are taken over responses with a *parseable* label; records
whose ``label`` is null (judge output couldn't be parsed) are dropped and reported, so the
category lines sum to ~100% at every coefficient.

Usage (from experiments/assistant-axis):
    python plot_role_susceptibility.py --judgements_path judgements.jsonl --output_path results/role_susceptibility.png
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import jsonlines
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from selfconcept.assistant_axis.role_susceptibility_judge import RoleLabel
from selfconcept.common.paths import scratch_dir

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Fixed category order and colourblind-safe colours (Okabe-Ito), so lines are consistent
# across runs. Keys are the judge labels from ``role_susceptibility_judge.RoleLabel``.
LABEL_ORDER: list[RoleLabel] = [
    "assistant",
    "nonhuman_role",
    "human_role",
    "weird_role",
    "ambiguous",
    "other",
    "nonsensical",
]
LABEL_COLORS: dict[str, str] = {
    "assistant": "#0072B2",
    "nonhuman_role": "#009E73",
    "human_role": "#D55E00",
    "weird_role": "#CC79A7",
    "ambiguous": "#E69F00",
    "other": "#56B4E9",
    "nonsensical": "#999999",
}


@dataclass(frozen=True)
class RunConfig:
    """Input judgements, output image, and figure title for one plot."""

    judgements_path: Path = scratch_dir("assistant-axis") / "role_susceptibility" / "judgements.jsonl"
    output_path: Path = Path("results/role_susceptibility.png")
    title: str = "Role adoption vs. steering strength (§3.2.1)"


def load_judgements(path: Path) -> list[dict]:
    """Load judged records that carry both a ``steering_coefficient`` and a ``label`` field."""
    with jsonlines.open(path) as reader:
        return [r for r in reader if "steering_coefficient" in r and "label" in r]


def label_fractions(
    records: list[dict],
) -> tuple[list[float], dict[str, dict[float, float]], int]:
    """Percentage of each label at each coefficient.

    Returns the sorted coefficients, a ``{label: {coefficient: percent}}`` table (over
    parseable labels only), and the number of dropped null-label records.
    """
    counts: dict[float, Counter[str]] = defaultdict(Counter)
    totals: Counter[float] = Counter()
    dropped = 0
    for record in records:
        label = record["label"]
        if label is None:
            dropped += 1
            continue
        coefficient = float(record["steering_coefficient"])
        counts[coefficient][label] += 1
        totals[coefficient] += 1

    coefficients = sorted(totals)
    fractions: dict[str, dict[float, float]] = {}
    for label in LABEL_ORDER:
        fractions[label] = {
            coefficient: 100.0 * counts[coefficient][label] / totals[coefficient] for coefficient in coefficients
        }
    return coefficients, fractions, dropped


def plot(coefficients: list[float], fractions: dict[str, dict[float, float]], title: str) -> Figure:
    """One line per category: percentage vs. steering coefficient."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for label in LABEL_ORDER:
        ys = [fractions[label][coefficient] for coefficient in coefficients]
        if not any(ys):
            continue  # category never occurs; omit from the legend
        ax.plot(coefficients, ys, marker="o", label=label, color=LABEL_COLORS[label])
    ax.set_xlabel("Steering coefficient (fraction of layer norm)")
    ax.set_ylabel("% of responses")
    ax.set_title(title)
    ax.set_ylim(0, 100)
    ax.axvline(0.0, color="0.8", linestyle="--", linewidth=1, zorder=0)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(title="category", loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    return fig


def main(run: RunConfig = RunConfig()) -> None:
    """Load judged traces and save the §3.2.1 role-adoption-vs-steering figure."""
    records = load_judgements(run.judgements_path)
    logger.info("Loaded %d judged records from %s", len(records), run.judgements_path)
    if not records:
        logger.warning("No records with both steering_coefficient and label; nothing to plot.")
        return

    target_models = {r.get("target_model") for r in records if r.get("target_model")}
    if len(target_models) > 1:
        logger.warning("Records mix %d target models %s; plotting them together.", len(target_models), target_models)

    coefficients, fractions, dropped = label_fractions(records)
    logger.info("Coefficients: %s; dropped %d unparseable (null-label) records", coefficients, dropped)

    fig = plot(coefficients, fractions, run.title)
    run.output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(run.output_path, dpi=150, bbox_inches="tight")
    logger.info("Wrote figure to %s", run.output_path)


if __name__ == "__main__":
    from jsonargparse import auto_cli

    auto_cli(main)
