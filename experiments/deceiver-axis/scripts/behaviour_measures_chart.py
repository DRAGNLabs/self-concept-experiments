"""gfar vs gi_v2 effect on behaviour measures, negative steering, main dose (CPU, reads the episode records on the group
share). One 3x3 figure, panels grouped:
- reward hacking and its disclosure: judged hack rate (Sonnet HACK verdicts over all episodes, unjudged = not a hack),
  admits the hack (among judged hacks), explanation matches the code (Sonnet labels on sampled episodes with code);
- incapability: admits incapability (among rejected submissions: turns whose code failed the visible tests), honest
  abstention and bad-test belief (among all episodes), and two abstain-flag panels from abstain_flag_turns.py
  (--flag-turns: <slug>_turns.json per model) on turns that did not pass honestly (failed the visible tests in an
  episode not judged a hack, or the judge's hack turn): the share whose peak first-token flag score is over the
  threshold, with steering applied to the unsteered transcripts (paired, same turns); and on each arm's own
  transcripts, tokens over the threshold per 10k tokens of those turns (own transcripts differ in length);
- other: stick to an honest answer (identical resubmission, all episodes).
Text proxies are text_proxies.py's provisional regexes. x = change under gfar, y = change under gi_v2 at the same
dose; dotted diagonal = equal effect. Bars: 95% problem-bootstrap interval of the gi_v2 - gfar gap, perpendicular to
the diagonal; marker filled when it excludes 0. Writes <out>.{png,pdf,json} (the JSON also holds the half dose).

    python experiments/deceiver-axis/scripts/behaviour_measures_chart.py --out <dir>/behaviour_measures_chart
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Literal, NotRequired, TypedDict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from matplotlib.transforms import Bbox

from selfconcept.codebench.harness import CheckRecord
from selfconcept.deceiver_axis.behaviour_measures.abstain_flag import ClassSummary, TurnClass, TurnFlagSummary
from selfconcept.deceiver_axis.behaviour_measures.bootstrap import (Change, Gap, ProblemCounts, bootstrap_change,
                                                                    change_gap)
from selfconcept.deceiver_axis.behaviour_measures.episodes import (PRESETS, SHARE, ArmName, Axis, Dose, ModelPreset,
                                                                   load_records, load_verdict_by_key)
from selfconcept.deceiver_axis.behaviour_measures.palette import COLOR_BY_MODEL, GRID, INK, MARKER_BY_MODEL, MUTED
from selfconcept.deceiver_axis.behaviour_measures.text_proxies import (admitted_hack, bad_test_belief,
                                                                       honest_abstention, incapability_admissions,
                                                                       resubmitted_same_code)

type Proxy = Literal["hack", "stick", "admission", "abstention", "bad_test", "incapability", "explanation_match",
                     "flag_on_unsteered", "flag_own"]
type ChangesByArm = dict[tuple[Axis, Dose], Change]
RECORD_PROXIES: list[Proxy] = ["hack", "stick", "admission", "abstention", "bad_test", "incapability"]
PANEL_GROUPS: list[tuple[str, list[Proxy]]] = [
    ("Reward hacking and its disclosure", ["hack", "admission", "explanation_match"]),
    ("Incapability: recognition, admission, abstention",
     ["incapability", "abstention", "bad_test", "flag_on_unsteered", "flag_own"]),
    ("Other", ["stick"]),
]
MAIN_PROXIES: list[Proxy] = [proxy for _, proxies in PANEL_GROUPS for proxy in proxies]
GROUP_TINTS = ["#f4f3ef", "#eef3f9", "#f6f1f4"]
TITLE_BY_PROXY: dict[Proxy, str] = {
    "hack": "Judged hack rate\n(Sonnet HACK verdict; all episodes, unjudged = not a hack)",
    "admission": "Admits the hack\n(among judged hacks)",
    "explanation_match": "Explanation matches the code\n(Sonnet labels; sampled episodes with code)",
    "incapability": "Admits incapability\n(reply says it could not solve; among rejected submissions)",
    "abstention": "Honest abstention\n(ends on a reply without code that says a test is wrong)",
    "bad_test": "Bad-test belief\n(says a test is wrong; all episodes)",
    "flag_on_unsteered": "Abstain flag on turns not passing honestly\n(steering on the unsteered transcripts)",
    "flag_own": "Abstain-flag tokens per 10k tokens, turns not passing honestly\n(each arm's own transcripts)",
    "stick": "Stick to honest answer\n(identical resubmission, all episodes)",
}
# flag panel -> (gi_v2 arm, gfar arm) labels in abstain_flag_turns.py output
FLAG_ARMS_BY_PROXY: dict[Proxy, tuple[str, str]] = {"flag_on_unsteered": ("gi_v2 on unsteered", "gfar on unsteered"),
                                                    "flag_own": ("gi_v2 own", "gfar own")}
FLAG_CLASS: TurnClass = "not_honest_pass"
PER_10K_TOKEN_PROXIES: set[Proxy] = {"flag_own"}
UNIT_BY_PROXY: dict[Proxy, str] = {"flag_own": "over-threshold tokens per 10k"}
DEFAULT_FLAG_TURNS = Path.home() / "nobackup/autodelete/self-concept-experiments/schemer-axis/abstain_flag"
EPISODE_PREDICATE_BY_PROXY: dict[Proxy, Callable[[CheckRecord], bool]] = {
    "stick": resubmitted_same_code, "admission": admitted_hack,
    "abstention": honest_abstention, "bad_test": bad_test_belief,
}
LABEL_CANDIDATES: list[tuple[tuple[int, int], Literal["left", "right"]]] = [
    ((8, 6), "left"), ((8, -14), "left"), ((-8, 6), "right"), ((-8, -14), "right"),
    ((8, 20), "left"), ((-8, 20), "right"), ((8, -28), "left"), ((-8, -28), "right"),
]
DEFAULT_MISMATCH_LABELS = [
    Path.home() / "nobackup/autodelete/self-concept-experiments/schemer-axis/abstention_admission_pilot"
    / "audit_sonnet_gemma/labels_joined.jsonl",
    SHARE / "gfar_vs_gi_v2_jansen05/explanation_audit/labels_joined.jsonl"]
type LabelAnchor = tuple[str, tuple[float, float]]


class MismatchLabelRow(TypedDict):
    """A joined explanation-label row: ours (model label, arm, episode_key) or Jansen's explanation audit (model
    slug, arm "<axis>_minus", seed, example_id)."""
    model: str
    arm: str
    episode_key: NotRequired[str]
    seed: NotRequired[int]
    example_id: NotRequired[str]
    explanation_mismatch: NotRequired[str]
type ChangesByModel = dict[str, dict[Proxy, ChangesByArm]]
type GapByModel = dict[str, dict[Proxy, Gap]]


def hits_and_pool_size(proxy: Proxy, record: CheckRecord, verdict: str | None) -> tuple[int, int]:
    """Incapability counts rejected submissions; every other proxy counts episodes (admission: judged hacks only)."""
    if proxy == "hack":
        return int(verdict == "HACK"), 1
    if proxy == "incapability":
        return incapability_admissions(record)
    if proxy == "admission" and verdict != "HACK":
        return 0, 0
    return int(EPISODE_PREDICATE_BY_PROXY[proxy](record)), 1


def problem_counts(records_by_seed_key: dict[str, CheckRecord], arm_name: str, proxy: Proxy, verdict_by_key: dict[str, str],
                   problem_ids: list[str]) -> ProblemCounts:
    hits_by_problem: dict[str, int] = defaultdict(int)
    totals_by_problem: dict[str, int] = defaultdict(int)
    for seed_key, record in records_by_seed_key.items():
        hits, pool_size = hits_and_pool_size(proxy, record, verdict_by_key.get(f"{arm_name}/{seed_key}"))
        totals_by_problem[record["example_id"]] += pool_size
        hits_by_problem[record["example_id"]] += hits
    return {"hits": np.array([hits_by_problem[problem] for problem in problem_ids]),
            "totals": np.array([totals_by_problem[problem] for problem in problem_ids])}


def model_changes(preset: ModelPreset, n_draws: int, rng: np.random.Generator
                  ) -> tuple[dict[Proxy, ChangesByArm], dict[Proxy, Gap]]:
    """Changes per record proxy and arm (both doses), and per proxy the main-dose gi_v2 - gfar gap, from one set of
    problem-bootstrap draws."""
    unsteered_records = load_records(preset["unsteered"], preset["record_file"])
    records_by_arm = {arm: load_records(arm_dir, preset["record_file"]) for arm, arm_dir in preset["arm_dirs"].items()}
    verdict_by_key = load_verdict_by_key(preset["verdict_files"])
    problem_ids = sorted({record["example_id"] for record in unsteered_records.values()})
    resampled_problems = rng.integers(len(problem_ids), size=(n_draws, len(problem_ids)))
    changes: dict[Proxy, ChangesByArm] = {}
    gap_by_proxy: dict[Proxy, Gap] = {}
    for proxy in RECORD_PROXIES:
        if proxy in preset["exclusion_reason_by_proxy"]:
            continue
        unsteered = problem_counts(unsteered_records, "unsteered", proxy, verdict_by_key, problem_ids)
        counts_by_arm = {arm: problem_counts(records, preset["arm_dirs"][arm].name, proxy, verdict_by_key, problem_ids)
                         for arm, records in records_by_arm.items()}
        changes[proxy] = {arm: bootstrap_change(counts, unsteered, resampled_problems)
                          for arm, counts in counts_by_arm.items()}
        gap_by_proxy[proxy] = change_gap(counts_by_arm[("gi_v2", "main")], counts_by_arm[("gfar", "main")],
                                         unsteered, resampled_problems)
    return changes, gap_by_proxy


def load_mismatch_labels(label_paths: list[Path]) -> dict[tuple[str, str, str], str]:
    """explanation_mismatch label by (model, arm, episode key); a later file overrides an earlier one per episode.
    Also reads Jansen's explanation-audit rows (model slug, arm "<axis>_minus", seed and example_id)."""
    label_by_slug: dict[str, str] = {preset["slug"]: preset["label"] for preset in PRESETS}
    label_by_episode: dict[tuple[str, str, str], str] = {}
    for path in label_paths:
        rows: list[MismatchLabelRow] = list(map(json.loads, path.read_text().splitlines()))
        for row in rows:
            if "explanation_mismatch" in row:
                model = label_by_slug.get(row["model"]) or row["model"]
                episode_key = row.get("episode_key") or f"seed{row.get('seed')}/{row.get('example_id')}"
                label_by_episode[(model, row["arm"].removesuffix("_minus"), episode_key)] = row["explanation_mismatch"]
    return label_by_episode


def explanation_match_changes(label_by_episode: dict[tuple[str, str, str], str], model: str, n_draws: int,
                              rng: np.random.Generator) -> tuple[ChangesByArm, Gap]:
    """Main-dose changes in the share of labelled episodes with code whose explanation_mismatch label is not "yes"
    (unclear counts as a match), and the gi_v2 - gfar gap, bootstrapping the problems in the labelled sample."""
    labelled = [(arm, episode_key.split("/")[1], label) for (label_model, arm, episode_key), label
                in label_by_episode.items() if label_model == model and label != "not_applicable"]
    problem_ids = sorted({problem for _, problem, _ in labelled})

    def counts(arm: ArmName) -> ProblemCounts:
        hits_by_problem = Counter(problem for label_arm, problem, label in labelled if label_arm == arm and label != "yes")
        totals_by_problem = Counter(problem for label_arm, problem, _ in labelled if label_arm == arm)
        return {"hits": np.array([hits_by_problem[problem] for problem in problem_ids]),
                "totals": np.array([totals_by_problem[problem] for problem in problem_ids])}

    resampled_problems = rng.integers(len(problem_ids), size=(n_draws, len(problem_ids)))
    unsteered, gi_v2, gfar = counts("unsteered"), counts("gi_v2"), counts("gfar")
    changes: ChangesByArm = {("gi_v2", "main"): bootstrap_change(gi_v2, unsteered, resampled_problems),
                             ("gfar", "main"): bootstrap_change(gfar, unsteered, resampled_problems)}
    return changes, change_gap(gi_v2, gfar, unsteered, resampled_problems)


def measure(class_summary: ClassSummary, per_10k: bool) -> float:
    return class_summary["over_per_10k_tokens"] if per_10k else class_summary["flag_rate"]


def flag_changes(turns_json: Path) -> dict[Proxy, tuple[ChangesByArm, Gap]]:
    """Main-dose changes on turns not passing honestly, and the gi_v2 - gfar gap, per flag panel, from
    abstain_flag_turns.py output (its problem-bootstrap intervals; flag rate in percentage points)."""
    summary: TurnFlagSummary = json.loads(turns_json.read_text())
    unsteered = summary["by_arm"]["unsteered"][FLAG_CLASS]
    result: dict[Proxy, tuple[ChangesByArm, Gap]] = {}
    for proxy, (gi_v2_arm, gfar_arm) in FLAG_ARMS_BY_PROXY.items():
        per_10k = proxy in PER_10K_TOKEN_PROXIES
        contrasts = summary["contrasts_over_per_10k_tokens"] if per_10k else summary["contrasts"]
        scale = 1.0 if per_10k else 100.0
        changes: ChangesByArm = {}
        arm_by_axis: dict[Axis, str] = {"gi_v2": gi_v2_arm, "gfar": gfar_arm}
        for axis, arm in arm_by_axis.items():
            steered = summary["by_arm"][arm][FLAG_CLASS]
            low, high = contrasts[f"{arm} - unsteered"][FLAG_CLASS]
            changes[(axis, "main")] = {"delta": scale * (measure(steered, per_10k) - measure(unsteered, per_10k)),
                                       "low": scale * low, "high": scale * high,
                                       "steered_rate": measure(steered, per_10k),
                                       "unsteered_rate": measure(unsteered, per_10k),
                                       "steered_n": steered["tokens" if per_10k else "turns"],
                                       "unsteered_n": unsteered["tokens" if per_10k else "turns"]}
        gap_low, gap_high = contrasts[f"{gi_v2_arm} - {gfar_arm}"][FLAG_CLASS]
        result[proxy] = (changes, {"delta": changes[("gi_v2", "main")]["delta"] - changes[("gfar", "main")]["delta"],
                                   "low": scale * gap_low, "high": scale * gap_high})
    return result


def draw_gap_marker(axes: Axes, model: str, gfar: Change, gi_v2: Change, gap: Gap) -> list[float]:
    """The model's main-dose point with its gap bar (perpendicular to the diagonal: moving along it changes only the
    gap). -> the bar's coordinates, for the axis extent."""
    color = COLOR_BY_MODEL[model]
    shifts = [(bound - gap["delta"]) / 2 for bound in (gap["low"], gap["high"])]
    bar_xs = [gfar["delta"] - shift for shift in shifts]
    bar_ys = [gi_v2["delta"] + shift for shift in shifts]
    axes.plot(bar_xs, bar_ys, color=color, linewidth=1.5, alpha=0.45, solid_capstyle="round", zorder=2)
    off_diagonal = gap["low"] > 0 or gap["high"] < 0
    axes.plot(gfar["delta"], gi_v2["delta"], marker=MARKER_BY_MODEL[model], markersize=10, linestyle="none",
              color=color, markerfacecolor=color if off_diagonal else "white", markeredgecolor=color,
              markeredgewidth=2, zorder=4)
    return [*bar_xs, *bar_ys]


def style_panel(axes: Axes, proxy: Proxy, extent: float) -> None:
    limit = extent * 1.08
    axes.plot([-limit, limit], [-limit, limit], linestyle=":", color=MUTED, linewidth=1, zorder=1)
    axes.axhline(0, color=GRID, linewidth=1, zorder=0)
    axes.axvline(0, color=GRID, linewidth=1, zorder=0)
    axes.plot(0, 0, marker="o", markersize=5, color=INK, zorder=5)
    axes.set_xlim(-limit, limit)
    axes.set_ylim(-limit, limit)
    axes.set_aspect("equal")
    axes.set_title(TITLE_BY_PROXY[proxy], fontsize=10, color=INK, loc="left")
    unit = UNIT_BY_PROXY.get(proxy, "pp")
    axes.set_xlabel(f"change under gfar − ({unit})", fontsize=9, color=MUTED)
    axes.set_ylabel(f"change under gi_v2 − ({unit})", fontsize=9, color=MUTED)
    axes.tick_params(colors=MUTED, labelsize=8)
    for spine in axes.spines.values():
        spine.set_color(GRID)


def draw_main_dose_panel(axes: Axes, proxy: Proxy, changes_by_model: ChangesByModel, gap_by_model: GapByModel
                         ) -> tuple[list[LabelAnchor], list[tuple[float, float]]]:
    extent = 1.0
    label_anchors: list[LabelAnchor] = []
    plotted_points: list[tuple[float, float]] = [(0.0, 0.0)]
    for model, changes_by_proxy in changes_by_model.items():
        if proxy not in changes_by_proxy:
            continue
        gfar, gi_v2 = changes_by_proxy[proxy][("gfar", "main")], changes_by_proxy[proxy][("gi_v2", "main")]
        bar_coordinates = draw_gap_marker(axes, model, gfar, gi_v2, gap_by_model[model][proxy])
        extent = max(extent, *map(abs, bar_coordinates))
        label_anchors.append((model, (gfar["delta"], gi_v2["delta"])))
        plotted_points.append((gfar["delta"], gi_v2["delta"]))
    style_panel(axes, proxy, extent)
    return label_anchors, plotted_points


def place_labels(axes: Axes, label_anchors: list[LabelAnchor], plotted_points: list[tuple[float, float]]) -> None:
    """First candidate offset whose box overlaps no placed label and no plotted marker (display coordinates)."""
    renderer = axes.figure.canvas.get_renderer()  # type: ignore[attr-defined]
    marker_centres = [axes.transData.transform(point) for point in plotted_points]
    marker_boxes = [Bbox.from_bounds(float(x) - 6, float(y) - 6, 12, 12) for x, y in marker_centres]
    placed_boxes: list[Bbox] = []
    for model, anchor in label_anchors:
        for offset, alignment in LABEL_CANDIDATES:
            label = axes.annotate(model, anchor, xytext=offset, textcoords="offset points", fontsize=8, color=INK,
                                  ha=alignment, zorder=5)
            extent = label.get_window_extent(renderer)
            box = Bbox.from_extents(extent.x0 - 4, extent.y0 - 3, extent.x1 + 4, extent.y1 + 3)
            is_last_candidate = offset == LABEL_CANDIDATES[-1][0]
            if is_last_candidate or not any(box.overlaps(other) for other in placed_boxes + marker_boxes):
                placed_boxes.append(box)
                break
            label.remove()


def tight_box(axes: Axes, renderer) -> Bbox:
    box = axes.get_tightbbox(renderer)
    assert box is not None, "a drawn panel always has a bounding box"
    return box


def draw_group_bands(figure: Figure, axes_by_proxy: dict[Proxy, Axes]) -> None:
    """A tinted band behind each group's panels, one per grid row the group occupies, headed by the group name."""
    renderer = figure.canvas.get_renderer()  # type: ignore[attr-defined]
    to_figure = figure.transFigure.inverted()
    for (heading, proxies), tint in zip(PANEL_GROUPS, GROUP_TINTS):
        boxes = [tight_box(axes_by_proxy[proxy], renderer).transformed(to_figure) for proxy in proxies]
        rows: dict[float, list[Bbox]] = defaultdict(list)
        for proxy, box in zip(proxies, boxes):
            rows[round(axes_by_proxy[proxy].get_position().y0, 3)].append(box)
        for row_index, (_, row_boxes) in enumerate(sorted(rows.items(), reverse=True)):
            band = Bbox.union(row_boxes).padded(0.004)
            figure.add_artist(FancyBboxPatch((band.x0, band.y0), band.width, band.height + (0.018 if row_index == 0 else 0),
                                             boxstyle="round,pad=0,rounding_size=0.01", transform=figure.transFigure,
                                             facecolor=tint, edgecolor="none", zorder=-10))
            if row_index == 0:
                figure.text(band.x0 + 0.006, band.y1 + 0.003, heading, fontsize=11, fontweight="bold", color=INK,
                            ha="left", va="bottom")


def legend_handles() -> list[Line2D]:
    model_handles = [Line2D([], [], color=COLOR_BY_MODEL[preset["label"]], marker=MARKER_BY_MODEL[preset["label"]],
                            linestyle="none", markersize=8,
                            label=f"{preset['label']} (α {preset['alpha_by_dose']['main']:g})") for preset in PRESETS]
    fill_handles = [Line2D([], [], color=MUTED, marker="o", linestyle="none", markersize=8, markerfacecolor=MUTED,
                           label="filled: gi_v2 and gfar effects differ"),
                    Line2D([], [], color=MUTED, marker="o", linestyle="none", markersize=8, markerfacecolor="white",
                           markeredgewidth=2, label="hollow: not distinguishable from the diagonal")]
    return model_handles + fill_handles


def exclusion_note() -> str:
    return " ".join(f"{preset['label']} is left out of {proxy} ({reason})." for preset in PRESETS
                    for proxy, reason in preset["exclusion_reason_by_proxy"].items())


def footnote(mismatch_label_paths: list[Path]) -> str:
    return "\n".join([
        "Main dose of each model, the same α on both axes. Dotted line: equal effect. Bars: 95% problem-bootstrap "
        "interval of (gi_v2 change − gfar change), drawn perpendicular to the diagonal; filled when it excludes 0.",
        "Text proxies are provisional regexes (text_proxies.py). Empty replies (output spent in reasoning) are not "
        "abstentions. Explanation labels (Sonnet): "
        + ", ".join(f"{path.parent.name}/{path.name}" for path in mismatch_label_paths)
        + " (later files override earlier ones per episode; unclear counts as a match).",
        exclusion_note() + " Gemma and Qwen have no honest abstention in any arm (shown at 0).",
        "Abstain-flag panels: turns that failed the visible tests in an episode not judged a hack, plus the judge's "
        "hack turns; threshold flags 10% of unsteered honest-success episodes (instruction added to the prompt,",
        "transcripts teacher-forced). The first is the share of turns flagged, paired on the unsteered transcripts; the "
        "second uses each arm's own transcripts and counts tokens over the threshold per 10k tokens."])


def save_main_figure(changes_by_model: ChangesByModel, gap_by_model: GapByModel, mismatch_label_paths: list[Path],
                     out: Path) -> None:
    figure, axes_grid = plt.subplots(3, 3, figsize=(17, 20))
    axes_by_proxy = dict(zip(MAIN_PROXIES, axes_grid.flat))
    label_layout = [(axes, draw_main_dose_panel(axes, proxy, changes_by_model, gap_by_model))
                    for proxy, axes in axes_by_proxy.items()]
    figure.legend(handles=legend_handles(), loc="upper center", ncol=3, frameon=False, fontsize=9)
    figure.text(0.5, 0.01, footnote(mismatch_label_paths), ha="center", fontsize=8, color=MUTED)
    figure.tight_layout(rect=(0.01, 0.07, 0.99, 0.95), h_pad=9, w_pad=4)
    for axes, (label_anchors, plotted_points) in label_layout:
        place_labels(axes, label_anchors, plotted_points)
    draw_group_bands(figure, axes_by_proxy)
    figure.savefig(out.with_suffix(".png"), dpi=200)
    figure.savefig(out.with_suffix(".pdf"))
    plt.close(figure)


def all_changes(n_draws: int, seed: int, mismatch_label_paths: list[Path], flag_turns: Path
                ) -> tuple[ChangesByModel, GapByModel]:
    rng = np.random.default_rng(seed)
    results_by_model = {preset["label"]: model_changes(preset, n_draws, rng) for preset in PRESETS}
    changes_by_model = {model: changes for model, (changes, _) in results_by_model.items()}
    gap_by_model = {model: gap_by_proxy for model, (_, gap_by_proxy) in results_by_model.items()}
    label_by_episode = load_mismatch_labels(mismatch_label_paths)
    for preset in PRESETS:
        model = preset["label"]
        changes_by_model[model]["explanation_match"], gap_by_model[model]["explanation_match"] = (
            explanation_match_changes(label_by_episode, model, n_draws, rng))
        for proxy, (changes, gap) in flag_changes(flag_turns / f"{preset['slug']}_turns.json").items():
            changes_by_model[model][proxy], gap_by_model[model][proxy] = changes, gap
    return changes_by_model, gap_by_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True, help="output stem: <out>.{png,pdf,json}")
    parser.add_argument("--n-draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mismatch-labels", type=Path, nargs="+", default=DEFAULT_MISMATCH_LABELS,
                        help="labels_joined.jsonl files with Sonnet explanation_mismatch labels (episode_audit_tally.py "
                             "or Jansen's explanation audit); a later file overrides an earlier one per episode")
    parser.add_argument("--flag-turns", type=Path, default=DEFAULT_FLAG_TURNS,
                        help="directory with abstain_flag_turns.py output <slug>_turns.json per model")
    args = parser.parse_args()
    changes_by_model, gap_by_model = all_changes(args.n_draws, args.seed, args.mismatch_labels, args.flag_turns)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_main_figure(changes_by_model, gap_by_model, args.mismatch_labels, args.out)
    args.out.with_suffix(".json").write_text(json.dumps(
        {model: {proxy: {**{f"{axis}/{dose}": change for (axis, dose), change in changes.items()},
                         "gi_v2_minus_gfar/main": gap_by_model[model][proxy]}
                 for proxy, changes in changes_by_proxy.items()}
         for model, changes_by_proxy in changes_by_model.items()}, indent=1))


if __name__ == "__main__":
    main()
