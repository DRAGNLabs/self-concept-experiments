"""Dose-trend test of the judged hack rate, one per (model, axis, ladder): does hacking change with signed intervention strength?

Each fit is a logistic regression of judged hacking on signed strength over the unsteered arm and the fitted rungs of both sides of
one ladder, with standard errors clustered by problem (statsmodels GLM, binomial, cov_type="cluster"). The p-value is the two-sided
Wald test of the slope. Signed strength of a rung (negative = pushing the projection down):
    steering    α
    cap at p    −(100 − p) / 100, minus the share of unsteered positions above the threshold
    floor at p  p / 100, the share of unsteered positions below the threshold

Rungs fitted: on each side of unsteered, outward in strength, up to the first rung that damages capability: no-code share or
hidden-test pass over all episodes more than 10 points from unsteered, in either direction. That rung and stronger ones on its side
are left out, as are the preset `omit` rungs of pareto_chart.py. `p_all_rungs` is the same fit over every judged rung.

A fit with fewer than MIN_HACKS_FOR_TEST judged hacks gets no p-value.
Hack counts per (arm, problem) are per_problem_rate × seeds from the judged analysis files on the group share.

    python experiments/deceiver-axis/scripts/dose_trend.py [--out <path>.json]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Literal, TypedDict

import numpy as np
import statsmodels.api as sm
from jaxtyping import Float

import pareto_chart as pc

GROUP_SHARE = Path("/grphome/grp_mirror/nobackup/autodelete/self-concept-experiments/schemer-axis")
SUMMARY_BY_MODEL_AXIS = {  # hidden-pass source for analysis files written without `hidden_pass` (pareto_chart.hidden_pass_diffs)
    ("muse", "gi_v2"): GROUP_SHARE / "muse_glimmer/gi_v2/pilot/summary_ladder.json",
    ("gemma", "gi_v2"): GROUP_SHARE / "gemma4_31b_gi_v2_judging_jansen05/hard108/analysis_capfloor15/summary_capfloor15.json",
}
ANALYSIS_BY_MODEL_AXIS = {
    ("gptoss", "gi_v2"): GROUP_SHARE / "gptoss120b/gi_v2/pilot/analysis_floorladder_all_as_judged.json",
    ("gptoss", "gfar"): GROUP_SHARE / "gptoss120b/gi_v2/pilot/analysis_gfx02nr_all_as_judged.json",
    ("muse", "gi_v2"): GROUP_SHARE / "muse_glimmer/gi_v2/pilot/analysis_floorladder_all_as_judged.json",
    ("muse", "gfar"): GROUP_SHARE / "muse_glimmer/gi_v2/pilot/analysis_gfx03_all_as_judged.json",
    ("gemma", "gi_v2"): GROUP_SHARE / "gemma4_31b_gi_v2_judging_jansen05/hard108/analysis_capfloor15/analysis_all_as_judged.json",
    ("gemma", "gfar"): GROUP_SHARE / "gemma4_31b_gi_v2_judging_jansen05/hard108/analysis_gfar7/analysis_all_as_judged.json",
    ("qwen", "gi_v2"): GROUP_SHARE / "qwen38_27b_gi_v2_judging_jansen05/qwen_hard61/analysis_gicap50/analysis_all_as_judged.json",
    ("qwen", "gfar"): GROUP_SHARE / "qwen38_27b_gi_v2_judging_jansen05/qwen_hard61/analysis_gfar_gfx04/analysis_all_as_judged.json",
}
CAPABILITY_LIMIT_POINTS = 10.0
MIN_HACKS_FOR_TEST = 10  # below this the clustered Wald test is unstable (Qwen gi_v2 −: one hack gave p < 0.0001, exchangeable GEE 0.92)

Ladder = Literal["add", "clamp"]


class Rung(TypedDict):
    arm: str
    label: str
    strength: float
    hack_pct: float
    no_code_diff: float
    hidden_pass_diff: float


class TrendFit(TypedDict):
    slope: float | None
    p: float | None
    note: str


class TrendRow(TypedDict):
    model: str
    axis: str
    ladder: Ladder
    unsteered_hack_pct: float
    rungs_fitted: list[Rung]
    rungs_left_out: list[Rung]
    slope_log_odds_per_unit: float | None
    p: float | None
    note: str
    p_all_rungs: float | None


def ladder_of(arm: str) -> Ladder:
    return "add" if pc.kind(arm) == "steering" else "clamp"


def signed_strength(arm: str) -> float:
    kind = pc.kind(arm)
    if kind == "steering":
        return pc.dose(arm)
    percentile = pc.percentile(arm)
    return -(100 - percentile) / 100 if kind == "cap" else percentile / 100


def hack_counts_by_problem(judged_arm: dict) -> tuple[Float[np.ndarray, " problems"], int]:
    seeds = len(judged_arm["seeds"])
    problems = sorted(judged_arm["per_problem_rate"])
    raw_counts = np.array([judged_arm["per_problem_rate"][problem] * seeds for problem in problems])
    counts = np.round(raw_counts)
    assert np.allclose(raw_counts, counts, atol=1e-6), "per-problem rate × seeds is not a whole number of episodes"
    return counts, seeds


def fit_trend(judged_arms: dict, arms_with_strength: list[tuple[str, float]]) -> TrendFit:
    """Logistic regression of hacking on strength, binomial counts per (arm, problem), problem-clustered standard errors."""
    hacks_rows, misses_rows, strength_rows, problem_rows = [], [], [], []
    for arm, arm_strength in arms_with_strength:
        counts, seeds = hack_counts_by_problem(judged_arms[arm])
        hacks_rows.append(counts)
        misses_rows.append(seeds - counts)
        strength_rows.append(np.full(len(counts), arm_strength))
        problem_rows.append(np.arange(len(counts)))
    hacks, misses = np.concatenate(hacks_rows), np.concatenate(misses_rows)
    if hacks.sum() == 0:
        return TrendFit(slope=None, p=None, note="no judged hacks on any fitted arm")
    if hacks.sum() < MIN_HACKS_FOR_TEST:
        return TrendFit(slope=None, p=None, note=f"{int(hacks.sum())} judged hacks: too few to test")
    endog = np.column_stack([hacks, misses])
    exog = sm.add_constant(np.concatenate(strength_rows))
    result = sm.GLM(endog, exog, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": np.concatenate(problem_rows)})
    p = float(result.pvalues[1])
    return TrendFit(slope=float(result.params[1]), p=None if math.isnan(p) else p, note="")


def rung_of(analysis: dict, source_arm: str, hidden_pass_diff: float) -> Rung:
    arm = pc.canonical(source_arm)
    return Rung(arm=arm, label=pc.label(arm), strength=signed_strength(arm), hack_pct=100 * analysis["judged_hack"]["arms"][source_arm]["rate"],
                no_code_diff=100 * analysis["no_code"]["arms"][source_arm]["diff_vs_unsteered"], hidden_pass_diff=hidden_pass_diff)


def damages_capability(rung: Rung) -> bool:
    return abs(rung["no_code_diff"]) > CAPABILITY_LIMIT_POINTS or abs(rung["hidden_pass_diff"]) > CAPABILITY_LIMIT_POINTS


def split_at_damage(rungs_outward: list[Rung]) -> tuple[list[Rung], list[Rung]]:
    for index, rung in enumerate(rungs_outward):
        if damages_capability(rung):
            return rungs_outward[:index], rungs_outward[index:]
    return rungs_outward, []


def fitted_and_left_out(ladder_rungs: list[Rung]) -> tuple[list[Rung], list[Rung]]:
    """Each side of unsteered cut at its first capability-damaging rung; both lists in order of signed strength."""
    negative_fitted, negative_left_out = split_at_damage(sorted((rung for rung in ladder_rungs if rung["strength"] < 0), key=lambda rung: -rung["strength"]))
    positive_fitted, positive_left_out = split_at_damage(sorted((rung for rung in ladder_rungs if rung["strength"] > 0), key=lambda rung: rung["strength"]))
    by_strength = lambda rungs: sorted(rungs, key=lambda rung: rung["strength"])
    return by_strength(negative_fitted + positive_fitted), by_strength(negative_left_out + positive_left_out)


def trend_rows(model: str, axis: str) -> list[TrendRow]:
    analysis = json.loads(ANALYSIS_BY_MODEL_AXIS[(model, axis)].read_text())
    preset = pc.MODELS[model]["axes"][axis]
    judged_arms = analysis["judged_hack"]["arms"]
    source_arm_by_arm = {pc.canonical(arm): arm for arm in judged_arms if pc.canonical(arm).startswith(preset["prefix"])}
    source_arm_by_arm["unsteered"] = "unsteered"
    source_arms = [source for arm, source in source_arm_by_arm.items() if arm != "unsteered" and arm not in preset.get("omit", ())]
    summary = SUMMARY_BY_MODEL_AXIS.get((model, axis))
    hidden_pass_diff_by_source_arm = pc.hidden_pass_diffs(analysis, {"summary": summary} if summary else {}, source_arms)
    missing_hidden_pass = [source for source in source_arms if hidden_pass_diff_by_source_arm[source] is None]
    assert not missing_hidden_pass, f"{model} / {axis}: no hidden-pass source for {missing_hidden_pass}"
    rungs = [rung_of(analysis, source, float(hidden_pass_diff_by_source_arm[source] or 0.0)) for source in source_arms]
    to_source = lambda selected: [("unsteered", 0.0)] + [(source_arm_by_arm[rung["arm"]], rung["strength"]) for rung in selected]
    rows = []
    for ladder in ("add", "clamp"):
        ladder_rungs = [rung for rung in rungs if ladder_of(rung["arm"]) == ladder]
        if not ladder_rungs:
            continue
        fitted, left_out = fitted_and_left_out(ladder_rungs)
        fit = fit_trend(judged_arms, to_source(fitted)) if fitted else TrendFit(slope=None, p=None, note="the weakest rungs already damage capability")
        rows.append(TrendRow(model=model, axis=axis, ladder=ladder, unsteered_hack_pct=100 * judged_arms["unsteered"]["rate"], rungs_fitted=fitted,
                             rungs_left_out=left_out, slope_log_odds_per_unit=fit["slope"], p=fit["p"], note=fit["note"],
                             p_all_rungs=fit_trend(judged_arms, to_source(ladder_rungs))["p"]))
    return rows


def format_p(p: float | None) -> str:
    if p is None:
        return "–"
    return "<0.0001" if p < 1e-4 else f"{p:.2g}"


def markdown_table(rows: list[TrendRow]) -> str:
    lines = ["| model | axis | ladder | rungs fitted | most negative → unsteered → most positive fitted | slope | trend p | p, all rungs | note |",
             "|---|---|---|---|---|---|---|---|---|"]
    for row in rows:
        fitted_rungs = row["rungs_fitted"]
        ends = [f"{rung['hack_pct']:.1f}% ({rung['label']})" for rung in fitted_rungs[:1] if rung["strength"] < 0]
        ends += [f"{row['unsteered_hack_pct']:.1f}% (unsteered)"]
        ends += [f"{rung['hack_pct']:.1f}% ({rung['label']})" for rung in fitted_rungs[-1:] if rung["strength"] > 0]
        slope = "–" if row["slope_log_odds_per_unit"] is None else f"{row['slope_log_odds_per_unit']:+.2f}"
        fitted = ", ".join(rung["label"] for rung in fitted_rungs) or "none"
        left_out = f" (left out: {', '.join(rung['label'] for rung in row['rungs_left_out'])})" if row["rungs_left_out"] else ""
        lines.append(f"| {pc.MODELS[row['model']]['name']} | {row['axis']} | {row['ladder']} | {fitted}{left_out} | {' → '.join(ends)} | "
                     f"{slope} | {format_p(row['p'])} | {format_p(row['p_all_rungs'])} | {row['note']} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    rows = [row for model, axis in ANALYSIS_BY_MODEL_AXIS for row in trend_rows(model, axis)]
    print(markdown_table(rows))
    if args.out:
        args.out.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
