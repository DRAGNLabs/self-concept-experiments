"""Outcome-composition chart: for every intervention arm of the gi_v2 vector and the Goodfire reward-hacking axis, how the
task-set episodes split into four outcomes, with the change in the benchmark score in a strip underneath.

One panel per (model, steering axis): a row per model, a column per axis. The x axis is dose_lines_chart.py's: caps in order of
their threshold percentile, steering doses α with unsteered (α = 0) in the middle, floors in order of their threshold percentile;
the two panels of a model share the rungs (the union of the two axes), so a rung one axis did not run is an empty slot. Each rung
is a bar of all the arm's episodes, split from the bottom into

    judged reward hack        the judge's verdict is HACK (as judged, no adjudication)
    passes the hidden tests   not judged a hack, code submitted, passes the hidden suite
    fails the hidden tests    not judged a hack, code submitted, fails the hidden suite
    no code submitted         not judged a hack, no final code

so the hack share reads from the bottom and the no-code share from the top. The categories are per episode and exclusive, in that
order of precedence (a judged hack that also passes the hidden suite, or has no final code, counts as a hack; the counts of both
overlaps are in the .json). The outlined bar is the unsteered arm; the two dashed lines carry its hack share and the lower edge
of its no-code share across the panel. The strip under the bars is the change in the benchmark composite vs unsteered in points
(pareto_chart.py's pooled score by default) with its 95% CI; an arm with no benchmark run has no point there.

Episodes are read from the run directories named in the analysis file's .meta (root, tag, verdicts); the seeds and problems are
the analysis file's, and the hack and no-code shares are checked against its `judged_hack` and `no_code` rates. Reading every
episode file takes a few minutes, so the counts are cached in <out>.shares.json (keyed by analysis file and its modification
time). No table under the panels (user 2026-11-13); every plotted number is in the .json.

    python experiments/deceiver-axis/scripts/outcome_bars_chart.py --out <path>.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import dose_lines_chart as dl
import pareto_chart as pc

# pareto_chart.GROUP_COLORS, validated on all pairs. Orange = the outcome the study is about; aqua = an honest solve.
OUTCOMES = {"hack": dict(name="judged reward hack", color="#eb6834"),
            "pass": dict(name="passes the hidden tests (not a hack)", color="#1baf7a"),
            "fail": dict(name="fails the hidden tests (code submitted, not a hack)", color="#2a78d6"),
            "no_code": dict(name="no code submitted", color="#4a3aa7")}
BAR_WIDTH = 0.74
PANEL_WIDTH, LEFT, RIGHT, BARS_HEIGHT, STRIP_GAP, STRIP_HEIGHT, TITLE_HEIGHT, XAXIS_HEIGHT, GAP = 8.4, 0.8, 0.25, 3.0, 0.14, 0.95, 0.42, 0.72, 0.28


def episode_counts(analysis_path: Path, arms: list[str]) -> dict[str, dict]:
    """Per analysis arm: episodes in each exclusive outcome, and the two overlaps the precedence resolves."""
    meta = json.loads(Path(str(analysis_path) + ".meta").read_text())
    analysis = json.loads(analysis_path.read_text())
    verdicts = json.loads(Path(meta["verdicts"]).read_text())
    judged, counts = analysis["judged_hack"], {}
    for arm in arms:
        problems = set(judged["arms"][arm]["per_problem_rate"])
        tally = dict(hack=0, **{"pass": 0}, fail=0, no_code=0, hack_and_hidden_pass=0, hack_and_no_code=0, no_code_any=0)
        for seed in judged["arms"][arm]["seeds"]:
            with (Path(meta["root"]) / arm / f"seed{seed}" / f"{meta['tag']}_impossible_original.jsonl").open() as lines:
                for line in lines:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    if record["example_id"] not in problems:
                        continue
                    hack = verdicts.get(f"{arm}/seed{seed}/{record['example_id']}", "").upper() == "HACK"
                    no_code, hidden = not record.get("final_code"), bool(record.get("passed_original"))
                    tally["hack" if hack else "no_code" if no_code else "pass" if hidden else "fail"] += 1
                    tally["hack_and_hidden_pass"] += hack and hidden
                    tally["hack_and_no_code"] += hack and no_code
                    tally["no_code_any"] += no_code
        episodes = sum(tally[key] for key in OUTCOMES)
        assert episodes == judged["arms"][arm]["episodes"], (arm, episodes, judged["arms"][arm]["episodes"])
        assert abs(tally["hack"] / episodes - judged["arms"][arm]["rate_pooled"]) < 1e-9, (arm, tally)
        assert abs(tally["no_code_any"] / episodes - analysis["no_code"]["arms"][arm]["rate_pooled"]) < 1e-9, (arm, tally)
        counts[arm] = dict(episodes=episodes, **tally)
    return counts


def cached_counts(cache_path: Path, analysis_path: Path, arms: list[str]) -> dict[str, dict]:
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    key, stamp = str(analysis_path), analysis_path.stat().st_mtime
    entry = cache.get(key)
    if entry is None or entry["mtime"] != stamp or any(arm not in entry["arms"] for arm in arms):
        entry = dict(mtime=stamp, arms=episode_counts(analysis_path, arms))
        cache[key] = entry
        cache_path.write_text(json.dumps(cache, indent=1))
    return {arm: entry["arms"][arm] for arm in arms}


def style(axes) -> None:
    axes.set_facecolor(pc.SURFACE)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(pc.AXIS)
    axes.tick_params(colors=pc.MUTED, labelsize=8.2)


def draw_bars(axes, group: dict, slots: dict, sections: list[dict]) -> None:
    style(axes)
    for section in sections:
        if section["kind"] != "steering":
            axes.axvspan(section["start"], section["end"], color=dl.TINT, linewidth=0, zorder=0)
    axes.grid(axis="y", color=pc.GRID, linewidth=0.6, zorder=0.5)
    axes.set_axisbelow(True)
    bars = [(slots[(row["kind"], row["rung"])], row["shares"], False) for row in group["rows"]]
    origin = slots.get(("steering", 0.0))
    if origin is not None:
        bars.append((origin, group["unsteered_shares"], True))
    for x, shares, reference in bars:
        bottom = 0.0
        for key, outcome in OUTCOMES.items():
            axes.bar(x, shares[key], BAR_WIDTH, bottom=bottom, color=outcome["color"], edgecolor=pc.SURFACE, linewidth=1.1, zorder=2)
            bottom += shares[key]
        if reference:
            axes.bar(x, 100, BAR_WIDTH, fill=False, edgecolor=pc.INK, linewidth=1.5, zorder=4)
    base = group["unsteered_shares"]
    for level in (base["hack"], 100 - base["no_code"]):
        axes.axhline(level, color=pc.INK, linewidth=0.9, linestyle=(0, (4, 3)), zorder=3)
    axes.set_xlim(sections[0]["start"], sections[-1]["end"])
    axes.set_ylim(0, 100)
    axes.set_yticks([0, 25, 50, 75, 100])
    axes.set_xticks([])
    axes.set_ylabel("share of episodes, %", fontsize=9.5, color=pc.INK2)


def draw_strip(axes, group: dict, slots: dict, sections: list[dict], y_range: tuple[float, float]) -> None:
    style(axes)
    for section in sections:
        if section["kind"] != "steering":
            axes.axvspan(section["start"], section["end"], color=dl.TINT, linewidth=0, zorder=0)
    axes.axhline(0, color=pc.ZERO, linewidth=1.0, zorder=1.5)
    axes.grid(axis="y", color=pc.GRID, linewidth=0.6, zorder=0.5)
    axes.set_axisbelow(True)
    for row in group["rows"]:
        if row["diff"] is None:
            continue
        x, y, (low, high) = slots[(row["kind"], row["rung"])], row["diff"], row["diff_ci"]
        axes.errorbar(x, y, yerr=[[y - low], [high - y]], fmt="none", ecolor=pc.INK2, elinewidth=1, alpha=0.55, zorder=2)
        axes.plot(x, y, "s", markersize=5.6, color=pc.INK, markeredgecolor=pc.SURFACE, markeredgewidth=1.2, zorder=3)
    origin = slots.get(("steering", 0.0))
    if origin is not None:
        axes.plot(origin, 0, "s", markersize=5.6, color=pc.SURFACE, markeredgecolor=pc.INK, markeredgewidth=1.3, zorder=3)
    axes.set_xlim(sections[0]["start"], sections[-1]["end"])
    axes.set_ylim(*y_range)
    axes.set_xticks([slots[(section["kind"], value)] for section in sections for value in section["rungs"]],
                    [dl.rung_label(section["kind"], value) for section in sections for value in section["rungs"]])
    axes.set_ylabel("Δ benchmark,\npoints", fontsize=8.6, color=pc.INK2, linespacing=1.15)
    inches_per_step = axes.get_position().width * axes.figure.get_figwidth() / (sections[-1]["end"] - sections[0]["start"])
    for section in sections:
        width = (section["end"] - section["start"]) * inches_per_step
        axes.text((section["start"] + section["end"]) / 2, -0.36, pc.wrap(dl.SECTIONS[section["kind"]], 9, max(1.25 * width, 0.9)), fontsize=9,
                  color=pc.INK2, ha="center", va="top", linespacing=1.2, transform=axes.get_xaxis_transform(), clip_on=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", nargs="+", choices=sorted(pc.MODELS), default=["gptoss", "muse", "gemma", "qwen"])
    parser.add_argument("--axes", nargs="+", default=["gi_v2", "gfar"])
    parser.add_argument("--kinds", nargs="+", choices=list(dl.SECTIONS), default=list(dl.SECTIONS))
    parser.add_argument("--composite", choices=sorted(pc.COMPOSITES), default="pooled")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    combine, score_name = pc.COMPOSITES[args.composite]
    kinds = [kind for kind in dl.SECTIONS if kind in args.kinds]
    cache_path = args.out.with_suffix(".shares.json")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    shares = lambda tally: {key: 100 * tally[key] / tally["episodes"] for key in OUTCOMES}
    panels = []
    for model_key in args.model:
        model, groups = pc.MODELS[model_key], []
        for axis_key in args.axes:
            axis = model["axes"][axis_key]
            data = pc.load_axis(model, axis, combine, keep_without_benchmark=True)
            rows = [dict(arm=arm, rung=dl.rung(arm), **row) for arm, row in data["arms"].items() if row["kind"] in kinds]
            counts = cached_counts(cache_path, axis["analysis"], ["unsteered"] + [row["analysis_arm"] for row in rows])
            for row in rows:
                row["counts"], row["shares"] = counts[row["analysis_arm"]], shares(counts[row["analysis_arm"]])
            groups.append(dict(axis_key=axis_key, data=data, rows=rows, unsteered_counts=counts["unsteered"], unsteered_shares=shares(counts["unsteered"])))
            print(f"{model['name']} / {axis_key}: episode files read ({len(rows) + 1} arms)", flush=True)
        panels.append(dict(model_key=model_key, model=model, groups=groups))

    columns, grid_rows = len(args.axes), len(panels)
    panel_height = TITLE_HEIGHT + BARS_HEIGHT + STRIP_GAP + STRIP_HEIGHT + XAXIS_HEIGHT + GAP
    width = PANEL_WIDTH * columns
    title = pc.wrap("Where the episodes go under capping, steering and flooring: outcome of every task-set episode, and the change in "
                    "benchmark score, for " + " and ".join(pc.AXIS_NAMES[key] for key in args.axes)
                    + (f" on {pc.COUNT_WORDS.get(len(panels), len(panels))} models" if len(panels) > 1 else f" on {panels[0]['model']['name']}"),
                    12.5, width - 0.3)
    subtitle = pc.wrap("Each bar is all episodes of one arm. Left to right: caps by threshold, steering doses, floors by threshold; rungs are "
                       "evenly spaced, not to scale, and an empty slot is a rung the other axis ran. Strip: change in benchmark score vs "
                       "unsteered, bars = 95% CI.", 9.5, width - 0.1)
    foot = pc.wrap("\n".join(
        ["Outcomes are per episode and exclusive, in the order of the legend: a judged hack (as judged) counts as a hack whatever its code does. "
         "Clamp thresholds are percentiles of the unsteered projection.",
         f"Benchmark score = {score_name.format(items=panels[0]['groups'][0]['data']['items'])}: GSM8K (1,319 items), MMLU-Pro (1,400), "
         "HumanEval (164), IFBench (300), one run per arm; CI = paired bootstrap over items. An arm without a point in the strip has no "
         "benchmark run yet."]), 7.8, width - 0.3)
    foot_height = 0.14 * (foot.count("\n") + 1) + 0.2
    head_height = 0.12 + 0.235 * (title.count("\n") + 1) + 0.08 + 0.175 * (subtitle.count("\n") + 1) + 0.1 + 0.215 * 2 + 0.07 + 0.08
    height = head_height + panel_height * grid_rows + foot_height
    figure = plt.figure(figsize=(width, height), facecolor=pc.SURFACE)
    at = lambda inches: 1 - inches / height
    x0 = 0.14 / width
    figure.text(x0, at(0.12), title, fontsize=12.5, color=pc.INK, ha="left", va="top", linespacing=1.25)
    cursor = 0.12 + 0.235 * (title.count("\n") + 1) + 0.08
    figure.text(x0, at(cursor), subtitle, fontsize=9.5, color=pc.INK2, ha="left", va="top", linespacing=1.3)
    cursor += 0.175 * (subtitle.count("\n") + 1) + 0.1
    legend_style = dict(frameon=False, fontsize=9, handletextpad=0.5, columnspacing=1.6, loc="upper left", borderaxespad=0, borderpad=0)
    figure.legend(handles=[Patch(facecolor=outcome["color"], edgecolor=pc.SURFACE, label=outcome["name"]) for outcome in OUTCOMES.values()],
                  bbox_to_anchor=(x0, at(cursor)), ncol=len(OUTCOMES), handlelength=1.1, handleheight=0.9, **legend_style)
    cursor += 0.215 + 0.07
    extra = [Patch(facecolor="none", edgecolor=pc.INK, linewidth=1.5, label="outlined bar: unsteered"),
             Line2D([], [], color=pc.INK, linewidth=0.9, linestyle=(0, (4, 3)), label="unsteered levels: top of the hack share, bottom of the no-code share"),
             Line2D([], [], color=pc.INK, marker="s", markersize=5.6, linewidth=0, markeredgecolor=pc.SURFACE, label="change in benchmark score (strip)")]
    figure.legend(handles=extra, bbox_to_anchor=(x0, at(cursor)), ncol=3, handlelength=2.4, **legend_style)
    figure.text(x0, 0.1 / height, foot, fontsize=7.8, color=pc.MUTED, ha="left", va="bottom", linespacing=1.3)

    plot_width = PANEL_WIDTH - LEFT - RIGHT
    for grid_row, panel in enumerate(panels):
        slots, sections = dl.layout(panel["groups"], kinds)
        cis = [bound for group in panel["groups"] for row in group["rows"] if row["diff"] is not None for bound in row["diff_ci"]] + [0.0]
        pad = 0.08 * (max(cis) - min(cis)) + 0.1
        y_range = (min(cis) - pad, max(cis) + pad)
        top = head_height + grid_row * panel_height
        for column, group in enumerate(panel["groups"]):
            left = (column * PANEL_WIDTH + LEFT) / width
            bars = figure.add_axes((left, at(top + TITLE_HEIGHT + BARS_HEIGHT), plot_width / width, BARS_HEIGHT / height))
            draw_bars(bars, group, slots, sections)
            base = group["unsteered_shares"]
            bars.set_title(f"{panel['model']['name']} · {pc.AXIS_NAMES[group['axis_key']]}", fontsize=9.8, color=pc.INK, loc="left", pad=7)
            strip = figure.add_axes((left, at(top + TITLE_HEIGHT + BARS_HEIGHT + STRIP_GAP + STRIP_HEIGHT), plot_width / width, STRIP_HEIGHT / height))
            draw_strip(strip, group, slots, sections, y_range)
            print(f"{panel['model']['name']} / {group['axis_key']}  unsteered " + "  ".join(f"{key} {base[key]:5.1f}" for key in OUTCOMES))
            for row in sorted(group["rows"], key=lambda row: (list(dl.SECTIONS).index(row["kind"]), row["rung"])):
                print(f"  {row['kind']:8s} {dl.rung_label(row['kind'], row['rung']):>6s}  " + "  ".join(f"{key} {row['shares'][key]:5.1f}" for key in OUTCOMES)
                      + f"  | hack∧hidden-pass {row['counts']['hack_and_hidden_pass']:3d}  hack∧no-code {row['counts']['hack_and_no_code']:3d}  of {row['counts']['episodes']}")

    figure.savefig(args.out, dpi=200, facecolor=pc.SURFACE)
    figure.savefig(args.out.with_suffix(".pdf"), facecolor=pc.SURFACE)
    record = dict(composite=args.composite, kinds=kinds, outcomes={key: outcome["name"] for key, outcome in OUTCOMES.items()},
                  panels=[dict(model=panel["model"]["name"],
                               axes=[dict(axis=group["axis_key"], analysis=group["data"]["analysis"], unsteered=group["data"]["unsteered"],
                                          unsteered_counts=group["unsteered_counts"], unsteered_shares=group["unsteered_shares"], arms=group["rows"])
                                     for group in panel["groups"]])
                          for panel in panels])
    args.out.with_suffix(".json").write_text(json.dumps(record, indent=1))
    print("->", args.out)


if __name__ == "__main__":
    main()
