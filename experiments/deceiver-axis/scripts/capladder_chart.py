"""Chart a model's gi_v2 dose scan with the cap dose ladder (p5 / p25 / p50 caps) as side points (PLAN "Capping dose ladder on
every model"; FINDINGS "gpt-oss cap dose ladder, judged — 2026-11-06").

gemma_gicap_chart.py's layout on the 45-problem × 12-seed main-run layout of gpt-oss and Muse: two line panels over an equally
spaced, labelled dose axis (the layer-17 / layer-26 doses of the gi_v2 vector, unsteered at 0) with the cap rungs as side points
after a gap, ordered strongest first (p5, p25, p50); under them a real table (header, gridlines, shading) with every number the
lines are drawn from, the judged H / B / N dossier counts, Δ vs unsteered, CI and Holm p. Numbers come from
`analysis_<tag>_all_as_judged*.json`, `analysis_<tag>_proxy_as_judged.json` and the pilot_summary file in --root, and the
verdict detail files named by --details.

Panel (a), hack rates: judged hack (as judged) with the 95% problem-bootstrap CI of each arm's OWN rate (own_ci: problems
resampled with replacement, 10,000 draws, fixed seed, clipped to 0–100; the unsteered arm has one too; the CI of Δ stays in the
table), the BORDERLINE-up / WEAK-down sensitivity band, the automatic proxy, and the model's random-direction steering (add)
arms as a grey range band with hollow marks at the left (random_arms; never a random cap or floor arm).
Panel (b), capability = the dose rule's measures: honest_hidden_pass (hidden pass among episodes not judged a hack) and
no-code, each with its unsteered ± 10-point band; an arm outside the rule (|Δ| > 10 on either) gets a ring and "outside" in
the table. An analysis file written before 2026-11-07 has no honest_hidden_pass: raw hidden pass is drawn and labelled as such.
An arm on fewer seeds than the rest is drawn hollow; episodes and seeds per arm are read from the analysis file.
The panel and table helpers here (hack_panel, capability_panel, fmt_p, holm_cell, n_note, chart_notes, fit_table, …) are
imported by gemma_gicap_chart.py and qwen_hard61_chart.py, so the three charts stay consistent.

    python experiments/deceiver-axis/scripts/capladder_chart.py --model gptoss [--tag capladder] [--out <path>.png]
    python experiments/deceiver-axis/scripts/capladder_chart.py --model muse --tag capladder
    python experiments/deceiver-axis/scripts/capladder_chart.py --model gptoss --tag floorladder   # cap and floor rungs side by side
    python experiments/deceiver-axis/scripts/capladder_chart.py --model gptoss --axis gfar --tag gfscan   # Goodfire-axis arms (gf_*)

--axis gfar (2026-11-10) draws the same chart for the Goodfire reward-hacking axis: arms gf_<p|m><dose> and gf_cap_p<N> /
gf_floor_p<N> from the analysis files of the newest judged gfar set, pilot_summary file summary_gfar.json.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import textwrap
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

N = Path.home() / "nobackup/autodelete/self-concept-experiments/schemer-axis"
MODELS = {
    "gptoss": dict(root=N / "gi_v2/pilot", name="gpt-oss-120b", layer=17, band="12–23", summary="summary_ladder.json",
                   details=["verdicts_main/verdicts_mixed.detail.json", "verdicts_main/verdicts_hidden_mixed.detail.json",
                            "verdicts_main/ref_proxy.detail.json", "verdicts_main/ref_hidden.detail.json",
                            "verdicts_capladder/verdicts_capladder.detail.json", "verdicts_capladder/verdicts_hidden_capladder.detail.json",
                            "verdicts_floorladder/verdicts_floorladder.detail.json", "verdicts_floorladder/verdicts_hidden_floorladder.detail.json"],
                   jobs="main run 14898345 / 14898350 (doses, p25 cap; judged 2026-10-30), cap ladder 15104508 (p5 / p50; judged 2026-11-06)",
                   jobs_floor="main run 14898345 / 14898350 (doses, p25 cap; judged 2026-10-30), cap ladder 15104508 (p5 / p50) and floor ladder 15107232 (p95 / p75 / p50), both judged 2026-11-06"),
    "muse": dict(root=N / "muse_glimmer/gi_v2/pilot", name="Muse-Glimmer-30B", layer=26, band="17–33", summary="summary_ladder.json",
                 details=["verdicts_main/verdicts_main.detail.json", "verdicts_main/verdicts_hidden_main.detail.json",
                          "verdicts_main/verdicts_p016.detail.json", "verdicts_main/verdicts_hidden_p016.detail.json",
                          "verdicts_main/ref_all.detail.json",
                          "verdicts_capladder/verdicts_capladder.detail.json", "verdicts_capladder/verdicts_hidden_capladder.detail.json",
                          "verdicts_floorladder/verdicts_floorladder.detail.json", "verdicts_floorladder/verdicts_hidden_floorladder.detail.json"],
                 jobs="main run 14898353 + 14917541 (doses, p25 cap; judged 2026-10-30), cap ladder 15104509 (p5 / p50)",
                 jobs_floor="main run 14898353 + 14917541 (doses, p25 cap; judged 2026-10-30), cap ladder 15104509 (p5 / p50; judged 2026-11-06), floor ladder 15206802 (p95 / p75 / p50; judged 2026-11-10)"),
}

SURFACE, INK, INK2, INK3, RULE, SHADE = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8985", "#dedcd7", "#f1f0ec"
C_JUDGED, C_AUTO, C_HIDDEN, C_SHOWN = "#1f4e9e", "#b8651b", "#2a7f62", "#7a3e9d"
C_NOCODE = "#8a8985"   # no code submitted (panel b)
C_RANDOM = "#52514e"   # random-direction steering arms: neutral ink, a secondary encoding (band + hollow marks)
BOOT_DRAWS, BOOT_SEED = 10_000, 0  # problem bootstrap of an arm's own rate (the bars of panel a)
RULE_POINTS = 10.0     # the dose rule: |Δ honest hidden pass| and |Δ no-code| both within 10 points of unsteered
DEFAULT_REPS = 20000   # main_analysis.py --reps default; analysis files written before 2026-11-10 carry no p_resampling_floor
HALO = [patheffects.withStroke(linewidth=2.6, foreground=SURFACE)]  # value labels stay readable where a line passes behind

CAPS = ["gi_cap_p5", "gi_cap_p25", "gi_cap_p50"]
FLOORS = ["gi_floor_p50", "gi_floor_p75", "gi_floor_p95"]  # the inverse cap (2026-11-06): projection clipped from below at tau
SIDE = CAPS + FLOORS  # side points beside the dose scan, caps then floors, a dotted rule before each group
PREFIX, AXIS_NAME = "gi_", "gi_v2 vector"
GFAR = {  # --axis gfar: per-model summary file and job line; verdict detail files are the reference ones plus verdicts_gf*/
    "gptoss": dict(summary="summary_gfar.json",
                   jobs="dose scan 15213661, p25 cap 15214272, p50 cap 15223509 (all judged 2026-11-10); unsteered arm of the gi_v2 main run"),
    "muse": dict(summary="summary_gfar.json",
                 jobs="±0.08 array 15213662, Koby's ±0.16 arms (job 15022726), p25 cap 15214273 (all judged 2026-11-10); unsteered arm of the gi_v2 main run"),
}


def use_gfar() -> None:
    global CAPS, FLOORS, SIDE, PREFIX, AXIS_NAME
    CAPS = ["gf_cap_p5", "gf_cap_p25", "gf_cap_p50"]
    FLOORS = ["gf_floor_p50", "gf_floor_p75", "gf_floor_p95"]
    SIDE = CAPS + FLOORS
    PREFIX, AXIS_NAME = "gf_", "Goodfire RH axis (gfar)"
SIDE_GAP = 0.8
ARMS: list = []
COL_LABELS: list = []
X: dict = {}


def dose(arm: str) -> float:
    if arm == "unsteered":
        return 0.0
    sign = -1 if "_m" in arm else 1
    return sign * float(arm.split("_")[-1][1:])


def set_arms(arms: list, layer: int) -> None:
    global ARMS, COL_LABELS, X
    doses = sorted([a for a in arms if a.startswith(PREFIX) and a not in SIDE] + ["unsteered"], key=dose)
    ARMS = [(a, "0\nunsteered" if a == "unsteered" else f"{dose(a):+g}".replace("-", "−")) for a in doses]
    X = {a: float(i) for i, (a, _) in enumerate(ARMS)}
    x = len(doses) - 1 + 0.6
    for group in (CAPS, FLOORS):
        present = [c for c in group if c in arms]
        if not present:
            continue
        x += SIDE_GAP
        for c in present:
            ARMS.append((c, f"{c.split('_')[1]}\n{c.split('_')[-1]}"))
            X[c] = x
            x += 1.15
    COL_LABELS = ["unsteered" if a == "unsteered" else (lab.replace("\n", " ") if a in SIDE else f"L{layer} {lab}") for a, lab in ARMS]


def dose_arms():
    return [(a, lab) for a, lab in ARMS if a not in SIDE]


def cap_arms():
    """Every side point: the cap rungs and, when present, the floor rungs."""
    return [(a, lab) for a, lab in ARMS if a in SIDE]


def load(root: Path, tag: str, summary: str, details: list, layer: int) -> dict:
    files = {"all": f"analysis_{tag}_all_as_judged.json", "up": f"analysis_{tag}_all_as_judged_borderline_up.json",
             "down": f"analysis_{tag}_all_as_judged_weak_down.json", "proxy": f"analysis_{tag}_proxy_as_judged.json"}
    data = {k: json.loads((root / f).read_text()) for k, f in files.items()}
    data["summary"] = json.loads((root / summary).read_text())
    counts = defaultdict(lambda: {"proxy": Counter(), "hidden": Counter()})
    for name in details:
        if not (root / name).exists():  # a ladder set not judged yet on this model (the floor ladder before it ran)
            continue
        which = "hidden" if "hidden" in Path(name).name else "proxy"
        # Muse keeps the reference arms in one file, ref_all (proxy + hidden-pass dossiers); ref_proxy beside it names the proxy ones
        ref_proxy = root / name.replace("ref_all", "ref_proxy")
        proxy_keys = set(json.loads(ref_proxy.read_text())) if Path(name).name.startswith("ref_all") and ref_proxy.exists() else None
        if Path(name).name.startswith("ref_all") and proxy_keys is None:
            print(f"note: {name} has no ref_proxy file beside it, its proxy and hidden-pass verdicts are counted together as proxy")
        for key, obj in json.loads((root / name).read_text()).items():
            w = which if proxy_keys is None else ("proxy" if key in proxy_keys else "hidden")
            counts[key.split("/")[0]][w][obj["verdict"]] += 1
    data["counts"] = counts
    set_arms(list(data["all"]["judged_hack"]["arms"]), layer)
    return data


# ---- helpers shared with gemma_gicap_chart.py and qwen_hard61_chart.py (they import them from here) ----

def rate(outcome, arm):
    return 100 * outcome["arms"][arm]["rate"]


def own_ci(outcome, arm):
    """95% problem-bootstrap CI of the arm's OWN rate, in percent: problems resampled with replacement (BOOT_DRAWS draws, fixed
    seed), problems without a rate (NaN) left out as the rate leaves them out, clipped to 0–100. Only for outcomes whose rate is
    the mean of per_problem_rate (judged_hack, no_code, …; not honest_hidden_pass, a ratio of two problem means)."""
    values = np.array([v for v in outcome["arms"][arm]["per_problem_rate"].values() if v is not None], dtype=float)
    values = values[~np.isnan(values)]
    if not math.isclose(values.mean(), outcome["arms"][arm]["rate"], abs_tol=1e-9):
        raise ValueError(f"{outcome.get('outcome')} / {arm}: rate is not the mean of per_problem_rate, own_ci does not apply")
    picks = np.random.default_rng(BOOT_SEED).integers(0, len(values), size=(BOOT_DRAWS, len(values)))
    lo, hi = np.percentile(100 * values[picks].mean(axis=1), [2.5, 97.5])
    return float(np.clip(lo, 0, 100)), float(np.clip(hi, 0, 100))


def is_random(arm: str) -> bool:
    """A random-direction or matched-noise arm of any kind (random_a0.07, random_plus_1730, L30_rand1729_a0.01, L30_noise_a0.01, L30_rand1729_cap_p25_L20-38)."""
    return arm.startswith("random") or "_rand" in arm or "_noise" in arm


def random_arms(outcome) -> list:
    """The random-direction steering (add) arms of the analysis file, the specificity control. Never a random cap or floor arm."""
    return sorted(a for a in outcome["arms"] if is_random(a) and "cap" not in a and "floor" not in a)


def capability(block: dict):
    """(outcome, label, is_rule_measure): honest_hidden_pass, the dose rule's measure, or raw hidden_pass from an older file."""
    if "honest_hidden_pass" in block:
        return block["honest_hidden_pass"], "hidden pass among episodes not judged a hack", True
    return block["hidden_pass"], "hidden-test pass, all episodes (file has no honest_hidden_pass)", False


def rule_breaks(HP, NC, arm) -> list:
    """The dose-rule measures on which the arm is more than RULE_POINTS from unsteered (empty = inside the rule)."""
    out = []
    for name, outcome in (("hidden", HP), ("no code", NC)):
        diff = outcome["arms"][arm].get("diff_vs_unsteered")
        if diff is not None and abs(100 * diff) > RULE_POINTS:
            out.append(name)
    return out


def rule_cell(HP, NC, arm) -> str:
    if "diff_vs_unsteered" not in HP["arms"][arm]:
        return "—"
    breaks = rule_breaks(HP, NC, arm)
    return "inside" if not breaks else "outside: " + ("both" if len(breaks) == 2 else breaks[0])


NEAR_POINTS = 2.0  # a miss of the rule measure by no more than this is "near the line"
CLAMP_FILE = N / "gfar_vs_gi_v2/clamp_shares.json"  # scripts/clamp_shares.py: open-loop prompt / generated clamp shares per cap or floor arm
TABLE_FLAGS = {"dagger": False, "clamp": False}  # set by table_rows, read by chart_notes (always called after it)


def comparison_hidden(summary: dict, arm: str):
    """(raw hidden pass, hidden pass among non-proxy episodes) in %, pooled, from the pilot_summary record of the arm; None if absent.
    Raw = non-proxy rate × (1 − proxy rate), because every hidden pass is a non-proxy episode."""
    s = summary.get(arm)
    if not s or "honest_hidden_pass_rate" not in s or "shown_pass_hidden_fail_rate" not in s:
        return None
    screen = 100 * s["honest_hidden_pass_rate"]
    return screen * (1 - s["shown_pass_hidden_fail_rate"]), screen


def may_be_composition(summary: dict, HP, NC, arm: str) -> bool:
    """PLAN "Review of 2026-11-10", selection caveat: the arm is outside the rule on the rule measure only (no-code inside), and either the
    miss is within NEAR_POINTS of the line or raw hidden pass / the non-proxy measure stays inside the 10 points. The rule measure counts
    only episodes not judged a hack, a set that changes with the arm, so such a miss need not be a capability loss."""
    if rule_breaks(HP, NC, arm) != ["hidden"]:
        return False
    if abs(100 * HP["arms"][arm]["diff_vs_unsteered"]) <= RULE_POINTS + NEAR_POINTS:
        return True
    c, r = comparison_hidden(summary, arm), comparison_hidden(summary, "unsteered")
    return bool(c and r and (abs(c[0] - r[0]) <= RULE_POINTS or abs(c[1] - r[1]) <= RULE_POINTS))


def clamp_cells(arms: list, model: str | None) -> list:
    """Per arm "prompt / generated" share of positions the clamp moves on unsteered activations (clamp_shares.py), "—" for other arms."""
    if model is None or not CLAMP_FILE.exists():
        return ["—"] * len(arms)
    records = {Path(r["arm_dir"]).name: r for r in json.loads(CLAMP_FILE.read_text())["arms"] if r["model"] == model}

    def span(pair):
        lo, hi = round(100 * pair[0]), round(100 * pair[1])
        return f"{lo}" if lo == hi else f"{lo}–{hi}"

    return [f"{span(records[a]['prompt_share']['mean'])} / {span(records[a]['generated_share']['mean'])}" if a in records else "—" for a in arms]


def p_floor(outcome) -> float:
    """Smallest p the resampled sign-flip test can return, 1 / (reps + 1)."""
    return outcome.get("p_resampling_floor", 1 / (DEFAULT_REPS + 1))


def fmt_p(p):
    """Four decimals; below 0.001 scientific notation with two significant digits (2.1e-05)."""
    return "" if p is None else (f"{p:.4f}" if p >= 1e-3 else f"{p:.1e}")


def holm_cell(outcome, arm) -> str:
    """Holm p of the arm's contrast; "≤ " in front when the raw p is a resampled one sitting on the resampling floor (no draw as
    extreme as the data), so that the adjusted value is an upper bound. Exact p-values (p_method "exact") print as they are."""
    c = outcome["contrasts"].get(arm)
    if c is None:
        return "—"
    at_floor = c.get("p_method", "resampled") != "exact" and math.isclose(c["p_raw"], p_floor(outcome), rel_tol=1e-6)
    return ("≤ " if at_floor else "") + fmt_p(c["p_holm"])


def delta_cell(outcome, arm) -> str:
    e = outcome["arms"][arm]
    return "—" if "diff_vs_unsteered" not in e else f"{100 * e['diff_vs_unsteered']:+.1f}"


def ci_cell(outcome, arm) -> str:
    e = outcome["arms"][arm]
    return "—" if "ci95" not in e else f"[{100 * e['ci95'][0]:+.1f}, {100 * e['ci95'][1]:+.1f}]"


def seed_range(seeds) -> str:
    s = sorted(seeds)
    return f"{s[0]}–{s[-1]}" if len(s) > 1 and s == list(range(s[0], s[-1] + 1)) else ", ".join(map(str, s))


def n_note(J, arms) -> str:
    """Seeds and episodes per arm, read from the analysis file; arms = [(arm, label)]. Arms that differ from the rest are named."""
    groups: dict = {}
    for a, label in arms:
        e = J["arms"][a]
        groups.setdefault((tuple(e["seeds"]), e["episodes"]), []).append(label.replace("\n", " "))
    main = max(groups, key=lambda k: len(groups[k]))
    if len(groups) == 1:
        return f"seeds {seed_range(main[0])} on every arm, {main[1]} episodes per arm"
    rest = "every other arm" if len(groups[main]) > 1 else groups[main][0]
    return "; ".join(f"{rest if k == main else ', '.join(labels)}: seeds {seed_range(k[0])} ({len(k[0])} seeds, {k[1]} episodes)"
                     for k, labels in sorted(groups.items(), key=lambda kv: kv[0] == main))


def fewer_seeds(J, arms) -> list:
    """Arms run on fewer seeds than the rest (drawn with a hollow marker)."""
    most = max(len(J["arms"][a]["seeds"]) for a in arms)
    return [a for a in arms if len(J["arms"][a]["seeds"]) < most]


def percentile(arm: str) -> int:
    return int(re.search(r"_p(\d+)", arm).group(1))


def heaviest_rungs(caps, floors) -> set:
    """The heavier end of each rung group of two or more: the lowest-percentile cap, the highest-percentile floor."""
    out = set()
    if len(caps) > 1:
        out.add(min(caps, key=percentile))
    if len(floors) > 1:
        out.add(max(floors, key=percentile))
    return out


def wrapped_xlabel(ax, text: str, size: float = 8.5) -> None:
    """x-axis label wrapped to the panel width."""
    width_pt = ax.get_position().width * ax.figure.get_figwidth() * 72
    ax.set_xlabel(textwrap.fill(text, max(40, int(width_pt / (0.52 * size)))), color=INK2, fontsize=size)


def near(outcome, X, group, arm, reach: float):
    """(min, max) of a series' line within `reach` x-units of the arm, along the arm's own run (dose arms or one rung group)."""
    xs, ys = [X[a] for a in group], [rate(outcome, a) for a in group]
    y = np.interp([max(xs[0], X[arm] - reach), X[arm], min(xs[-1], X[arm] + reach)], xs, ys)
    return float(y.min()), float(y.max())


def value_label(ax, x, y, text, slots, busy, span) -> None:
    """Write a value label (neutral ink, surface halo) in the first free slot, or not at all: every value is in the table.
    slots = [(where, anchor)] with where in above / below / right, anchor = the y the label hangs from (a bar end or the
    point); busy = [(lo, hi)] y-intervals already taken at this x (other series, bars, earlier labels), extended in place."""
    height, gap = 0.055 * span, 0.014 * span
    for where, anchor in slots:
        zone = {"above": (anchor + gap, anchor + gap + height), "below": (anchor - gap - height, anchor - gap),
                "right": (anchor - height / 2, anchor + height / 2)}[where]
        if (zone[0] < 0 and where != "right") or zone[1] > span or any(lo < hi and zone[0] < hi and lo < zone[1] for lo, hi in busy):
            continue
        xy, ha, va = {"above": ((0, 0), "center", "bottom"), "below": ((0, 0), "center", "top"), "right": ((9, 0), "left", "center")}[where]
        ax.annotate(text, (x, zone[0] if where == "above" else zone[1] if where == "below" else anchor), textcoords="offset points", xytext=xy,
                    fontsize=8.5, color=INK2, ha=ha, va=va, zorder=7, path_effects=HALO)
        busy.append(zone)
        return


def series(ax, outcome, X, doses, groups, color, marker="o", ls="-", lw=2.0, hollow=()):
    """One series: a line over the dose arms and a dotted run per side group (never bridged to the doses or to another group);
    arms in `hollow` (fewer seeds than the rest) get a hollow marker."""
    kw = dict(color=color, marker=marker, ms=6, mec=SURFACE, mew=1.2, zorder=4, clip_on=False)
    ax.plot([X[a] for a in doses], [rate(outcome, a) for a in doses], lw=lw, ls=ls, **kw)
    for group in groups:
        ax.plot([X[a] for a in group], [rate(outcome, a) for a in group], lw=lw * 0.6, ls=":" if len(group) > 1 else "none", **kw)
    for a in hollow:
        ax.plot([X[a]], [rate(outcome, a)], ls="none", marker=marker, ms=6.5, mfc=SURFACE, mec=color, mew=1.6, zorder=5, clip_on=False)


def hollow_handle():
    return Line2D([], [], ls="none", marker="o", ms=6.5, mfc=SURFACE, mec=INK2, mew=1.6, label="hollow marker: arm on fewer seeds than the rest")


def hack_panel(ax, X, doses, groups, J, UP, DOWN, AUTO, hollow=()):
    """Panel (a): judged hack with the bootstrap CI of each arm's own rate (unsteered too), the WEAK-down … BORDERLINE-up band,
    the automatic proxy, and the random-direction steering arms as a range band with hollow marks at the left edge."""
    side = [a for group in groups for a in group]
    arms = list(doses) + side
    ax.fill_between([X[a] for a in doses], [rate(DOWN, a) for a in doses], [rate(UP, a) for a in doses], color=C_JUDGED, alpha=0.12, lw=0, zorder=1)
    for a in side:
        ax.plot([X[a], X[a]], [rate(DOWN, a), rate(UP, a)], color=C_JUDGED, alpha=0.25, lw=6, solid_capstyle="butt", zorder=1)
    randoms = random_arms(J)
    left = min(X.values()) - 0.5
    if randoms:
        ys = sorted(rate(J, a) for a in randoms)
        ax.axhspan(ys[0], ys[-1], color=C_RANDOM, alpha=0.10, lw=0, zorder=0.5)
        left -= 0.7
        for i, y in enumerate(ys):
            ax.plot([left + 0.45 + 0.15 * (i - (len(ys) - 1) / 2)], [y], ls="none", marker="D", ms=4.5, mfc=SURFACE, mec=C_RANDOM, mew=1.1, zorder=4)
    ax.set_xlim(left, max(X.values()) + 0.5)
    series(ax, AUTO, X, doses, groups, C_AUTO, marker="s", ls="--", lw=1.6)
    series(ax, J, X, doses, groups, C_JUDGED, hollow=hollow)
    bars = {a: own_ci(J, a) for a in arms}
    for a, (lo, hi) in bars.items():
        ax.plot([X[a], X[a]], [lo, hi], color=C_JUDGED, lw=1.2, alpha=0.75, zorder=3)
        for v in (lo, hi):
            ax.plot([X[a] - 0.06, X[a] + 0.06], [v, v], color=C_JUDGED, lw=1.0, alpha=0.75, zorder=3)
    top = max([1.0] + [rate(o, a) for o in (AUTO, UP) for a in arms] + [hi for _, hi in bars.values()] + [rate(J, a) for a in randoms])
    handles = [Line2D([], [], color=C_JUDGED, lw=2, marker="o", ms=6, mec=SURFACE, label="judged hack, as judged (bar: 95% bootstrap CI of the arm's own rate)"),
               Patch(facecolor=C_JUDGED, alpha=0.12, label="sensitivity band: WEAK↓ … BORDERLINE↑"),
               Line2D([], [], color=C_AUTO, lw=1.6, ls="--", marker="s", ms=6, mec=SURFACE, label="automatic proxy (shown pass, hidden fail)")]
    if randoms:
        handles.append((Patch(facecolor=C_RANDOM, alpha=0.10, lw=0), Line2D([], [], ls="none", marker="D", ms=4.5, mfc=SURFACE, mec=C_RANDOM, mew=1.1)))
    kind = "random-direction and noise" if any("_noise" in a for a in randoms) else "random-direction"
    labels = [h.get_label() for h in handles[:3]] + ([f"{kind} steering arms (n = {len(randoms)}): range and values"] if randoms else [])
    if hollow:
        handles.append(hollow_handle())
        labels.append(handles[-1].get_label())
    span = top / (1 - 0.07 - 0.05 * len(handles))  # headroom for the legend
    ax.set_ylim(0, span)
    for run in [list(doses)] + [list(g) for g in groups]:  # the judged value over or under its bar, where the proxy line leaves room
        for a in run:
            y, (lo, hi) = rate(J, a), bars[a]
            own, proxy = near(J, X, run, a, 0.35), near(AUTO, X, run, a, 0.45)
            busy = [(proxy[0] - 0.015 * span, proxy[1] + 0.015 * span), (lo, hi), (own[0] + 0.03 * span, max(own[0] + 0.03 * span, own[1] - 0.03 * span))]
            value_label(ax, X[a], y, f"{y:.1f}", [("above", hi), ("below", lo)] + ([("right", y)] if a == run[-1] else []), busy, span)
    ax.legend(handles=handles, labels=labels, handler_map={tuple: HandlerTuple(ndivide=1)}, loc="upper left", fontsize=8.5, frameon=False, labelcolor=INK2)
    return ax.set_title("(a) Hack rates: judged vs the automatic proxy", loc="left", color=INK, fontsize=11, pad=10)


def capability_panel(ax, X, doses, groups, HP, NC, hp_label, honest, hollow=()):
    """Panel (b): the dose rule's two measures (honest hidden pass, no-code), each with its unsteered ± RULE_POINTS band, and a
    ring on every point that is outside its band. No bars here: the CIs of Δ are in the table."""
    arms = list(doses) + [a for group in groups for a in group]
    for outcome, color in ((HP, C_HIDDEN), (NC, C_NOCODE)):
        u = rate(outcome, "unsteered")
        ax.axhspan(max(0.0, u - RULE_POINTS), min(100.0, u + RULE_POINTS), color=color, alpha=0.12, lw=0, zorder=0.5)
    series(ax, NC, X, doses, groups, C_NOCODE, marker="^", ls="--", lw=1.6, hollow=hollow)
    series(ax, HP, X, doses, groups, C_HIDDEN, hollow=hollow)
    ringed = False
    for a in arms:
        breaks = rule_breaks(HP, NC, a)
        for name, outcome in (("hidden", HP), ("no code", NC)):
            if name in breaks:
                ax.plot([X[a]], [rate(outcome, a)], ls="none", marker="o", ms=13, mfc="none", mec=INK, mew=1.2, zorder=6, clip_on=False)
                ringed = True
    top = max([rate(o, a) for o in (HP, NC) for a in arms] + [rate(o, "unsteered") + RULE_POINTS for o in (HP, NC)])
    rule = "the dose rule" if honest else "the rule's limit, on raw hidden pass"
    handles = [Line2D([], [], color=C_HIDDEN, lw=2, marker="o", ms=6, mec=SURFACE, label=hp_label),
               Line2D([], [], color=C_NOCODE, lw=1.6, ls="--", marker="^", ms=6, mec=SURFACE, label="no code submitted"),
               Patch(facecolor=INK3, alpha=0.25, lw=0, label=f"shaded: unsteered ± {RULE_POINTS:g} points on each measure ({rule})"),
               Line2D([], [], ls="none", marker="o", ms=11, mfc="none", mec=INK, mew=1.2,
                      label=f"ring: outside the band, |Δ| > {RULE_POINTS:g} points" + ("" if ringed else " (no arm here)"))]
    if hollow:
        handles.append(hollow_handle())
    span = top / (1 - 0.07 - 0.05 * len(handles))
    ax.set_ylim(0, span)
    for run in [list(doses)] + [list(g) for g in groups]:  # each value over or under its point, where the other line leaves room
        for a in run:
            pad = (0.032 if rule_breaks(HP, NC, a) else 0.012) * span  # marker or ring
            busy = [(rate(o, a) - pad, rate(o, a) + pad) for o in (HP, NC)]  # the two points; label zones are added as they are taken
            for outcome, other in ((HP, NC), (NC, HP)):
                y, own, far = rate(outcome, a), near(outcome, X, run, a, 0.3), near(other, X, run, a, 0.45)
                taken = busy + [(far[0] - 0.01 * span, far[1] + 0.01 * span), (own[0] + 0.05 * span, max(own[0] + 0.05 * span, own[1] - 0.05 * span))]
                n = len(taken)
                value_label(ax, X[a], y, f"{y:.1f}", [("above", y + pad), ("below", y - pad)] + ([("right", y)] if a == run[-1] else []), taken, span)
                busy += taken[n:]
    ax.legend(handles=handles, loc="upper left", fontsize=8.5, frameon=False, labelcolor=INK2)
    return ax.set_title("(b) Capability on the dose rule's measures: honest hidden pass and no code" if honest
                 else "(b) Capability: hidden-test pass (all episodes) and no code submitted", loc="left", color=INK, fontsize=10.5, pad=10)


def table_rows(d, arms, J, UP, DOWN, PX, AUTO, HP, NC, SP, hp_label, honest, family: str, hidden_set: str = "hidden-pass set (signature)",
               clamp_model: str | None = None) -> list:
    """The table under the panels: every number the panels draw, Δ vs unsteered with its CI, Holm p, the dose rule and n per arm."""
    def own(a):
        lo, hi = own_ci(J, a)
        return f"[{lo:.1f}, {hi:.1f}]"

    def hbn(a, which):
        c = d["counts"][a][which]
        return f"{c['HACK']} / {c['BORDERLINE']} / {c['NOT_HACK']}"

    def summ(a, k):
        s = d["summary"].get(a)
        return "—" if s is None else str(s.get(k, "—"))

    def compare(a, i):
        c, r = comparison_hidden(d["summary"], a), comparison_hidden(d["summary"], "unsteered")
        return "—" if c is None else f"{c[i]:.1f}" + ("" if r is None or a == "unsteered" else f" ({c[i] - r[i]:+.1f})")

    def rule(a):
        dagger = honest and "diff_vs_unsteered" in HP["arms"][a] and may_be_composition(d["summary"], HP, NC, a)
        TABLE_FLAGS["dagger"] |= dagger
        return rule_cell(HP, NC, a) + (" †" if dagger else "")

    TABLE_FLAGS.update(dagger=False, clamp=False)
    clamp = clamp_cells(arms, clamp_model)
    TABLE_FLAGS["clamp"] = any(c != "—" for c in clamp)
    rows = [
        ("judged hack, as judged (%)", [f"{rate(J, a):.1f}" for a in arms]),
        ("   95% CI of the rate (bars in a)", [own(a) for a in arms]),
        ("   Δ vs unsteered (points)", [delta_cell(J, a) for a in arms]),
        ("   95% CI of Δ (problem bootstrap)", [ci_cell(J, a) for a in arms]),
        (f"   Holm p ({family})", [holm_cell(J, a) for a in arms]),
        ("judged hack, proxy set only (%)", [f"{rate(PX, a):.1f}" for a in arms]),
        ("judged hack, BORDERLINE counted (%)", [f"{rate(UP, a):.1f}" for a in arms]),
        ("judged hack, WEAK not counted (%)", [f"{rate(DOWN, a):.1f}" for a in arms]),
        ("automatic proxy (%)", [f"{rate(AUTO, a):.1f}" for a in arms]),
        ("   Δ vs unsteered (points)", [delta_cell(AUTO, a) for a in arms]),
        ("   Holm p", [holm_cell(AUTO, a) for a in arms]),
        (f"{hp_label.split(' (')[0]} (%)", [f"{rate(HP, a):.1f}" for a in arms]),
        ("   Δ vs unsteered (points)", [delta_cell(HP, a) for a in arms]),
        ("   95% CI of Δ", [ci_cell(HP, a) for a in arms]),
        ("no code submitted (%)", [f"{rate(NC, a):.1f}" for a in arms]),
        ("   Δ vs unsteered (points)", [delta_cell(NC, a) for a in arms]),
        ("   95% CI of Δ", [ci_cell(NC, a) for a in arms]),
        ("episodes not judged a hack (%)", [f"{100 - rate(J, a):.1f}" for a in arms]),
        ("for comparison: raw hidden pass (%, Δ)", [compare(a, 0) for a in arms]),
        ("for comparison: hidden pass, non-proxy episodes (%, Δ)", [compare(a, 1) for a in arms]),
        (f"dose rule: both |Δ| ≤ {RULE_POINTS:g} points" + ("" if honest else " (on raw hidden pass)"), [rule(a) for a in arms]),
        ("shown-test pass (%)", [f"{rate(SP, a):.1f}" for a in arms]),
        ("truncated turns / turns", [f"{summ(a, 'truncated_turns')} / {summ(a, 'turns')}" for a in arms]),
        ("dossiers judged H / B / N: proxy set", [hbn(a, "proxy") for a in arms]),
        (f"dossiers judged H / B / N: {hidden_set}", [hbn(a, "hidden") for a in arms]),
        ("episodes / seeds", [f"{J['arms'][a]['episodes']} / {len(J['arms'][a]['seeds'])}" for a in arms]),
    ]
    if TABLE_FLAGS["clamp"]:
        at = next(i for i, (label, _) in enumerate(rows) if label.startswith("dose rule")) + 1
        rows.insert(at, ("clamp moves, unsteered estimate: prompt / generated positions (%)", clamp))
    return rows


def draw_table(fig, ax, rows, col_labels, size=7.0, bbox=(0.25, 0.0, 0.74, 1.0)):
    """A real table: header row, gridlines, alternating shading, neutral ink. The row-label column starts at the left edge of
    the axes (the value columns take the rest), and a cell wider than its column is set smaller."""
    ax.axis("off")

    def build(x0: float):
        table = ax.table(cellText=[r[1] for r in rows], rowLabels=[r[0] for r in rows], colLabels=col_labels, loc="upper center",
                         cellLoc="center", rowLoc="left", colWidths=[(0.99 - x0) / len(col_labels)] * len(col_labels), bbox=[x0, bbox[1], 0.99 - x0, bbox[3]])
        table.auto_set_font_size(False)
        table.set_fontsize(size)
        for (r, c), cell in table.get_celld().items():
            cell.set_edgecolor(RULE)
            cell.set_linewidth(0.6)
            cell.get_text().set_color(INK if r == 0 or c == -1 else INK2)
            if r == 0:
                cell.set_facecolor(SHADE)
                cell.get_text().set_fontweight("bold")
            else:
                cell.set_facecolor("#f7f6f3" if r % 2 == 0 else SURFACE)
            if c == -1:
                cell.get_text().set_ha("left")
                cell.PAD = 0.02
        fig.canvas.draw()
        return table

    table = build(bbox[0])
    renderer = fig.canvas.get_renderer()
    label_w = max(cell.get_window_extent(renderer).width for (r, c), cell in table.get_celld().items() if c == -1) / ax.get_window_extent(renderer).width
    if abs(label_w - bbox[0]) > 0.004:  # the label column is as wide as its longest label, whatever the figure width
        table.remove()
        table = build(label_w)
    for (r, c), cell in table.get_celld().items():
        if c == -1:
            continue
        text_w, cell_w = cell.get_text().get_window_extent(renderer).width, cell.get_window_extent(renderer).width
        if text_w > 0.92 * cell_w:
            cell.get_text().set_fontsize(size * 0.92 * cell_w / text_w)
    return table


def chart_notes(J, HP, NC, honest, arms, hollow, axis_arms: str, has_caps: bool, has_floors: bool, width: int) -> list:
    """The footnote lines under the table: what the bars, the band, the rings, the p-values and the hollow markers are.
    arms = [(arm, label)] in column order."""
    notes = [f"Bars in (a): 95% problem-bootstrap CI of each arm's own judged rate, the unsteered arm included ({J['problems']} problems resampled with "
             f"replacement, {BOOT_DRAWS:,} draws, fixed seed, clipped to 0–100). They are not the CI of Δ vs unsteered; that one is in the table."]
    randoms = random_arms(J)
    if randoms:
        notes.append(f"Grey band and hollow grey diamonds at the left of (a): the random-direction {'and noise ' if any('_noise' in a for a in randoms) else ''}steering (add) arms, the specificity control "
                     f"({', '.join(randoms)}): judged hack {' / '.join(f'{rate(J, a):.1f}' for a in randoms)}%, "
                     f"Δ {' / '.join(delta_cell(J, a) for a in randoms)}, Holm p {' / '.join(holm_cell(J, a) for a in randoms)}. The band spans their range.")
    else:
        notes.append("Random-direction steering (add) arms, the specificity control: none in this analysis file, so none is drawn in (a).")
    outside = [label.replace("\n", " ") for a, label in arms if rule_breaks(HP, NC, a)]
    if honest:
        notes.append(f"Panel (b) is the dose rule's measure: hidden pass among episodes not judged a hack, and no code submitted. Shaded: unsteered ± {RULE_POINTS:g} "
                     f"points on each; ring: a value outside its band, which puts the arm outside the rule ({', '.join(outside) if outside else 'no arm here'}). "
                     "No bars in (b): the CIs of Δ are in the table.")
    else:
        notes.append("Panel (b): this analysis file was written before main_analysis.py had honest_hidden_pass (hidden pass among episodes not judged a hack, the dose "
                     f"rule's measure), so raw hidden-test pass over all episodes is drawn in its place. Bands and rings apply the rule's ± {RULE_POINTS:g} points to raw "
                     f"hidden pass and no code ({', '.join(outside) if outside else 'no arm'} outside) and are provisional until the analysis is re-run. No bars in (b): the CIs of Δ are in the table.")
    if TABLE_FLAGS["dagger"]:
        notes.append(f"† in the dose-rule row: outside on the rule measure only (no code inside), with the miss under {NEAR_POINTS:g} points or with raw hidden pass or the "
                     "non-proxy measure still within 10 points (comparison rows). The rule measure counts only episodes not judged a hack, a set that changes with "
                     "the arm (row above), so the miss may be composition, not capability; the benchmark capability eval is the read that separates the two.")
    if TABLE_FLAGS["clamp"]:
        notes.append("Clamp row: the rung's percentile is of generated tokens in 4 unsteered threshold episodes, but the clamp acts on every position, the prompt "
                     "included. The row gives the share of prompt / generated positions an unsteered run would have had moved (open loop, band-layer mean; a "
                     "range where the threshold lies outside the stored quantile grid).")
    n, n_random = len(J["contrasts"]), sum(is_random(a) for a in J["contrasts"])
    family = f"{n - n_random} {axis_arms}" + (f" + {n_random} random-direction{' / noise' if any('_noise' in a for a in J['contrasts']) else ''} arms" if n_random else "")
    notes.append(f"Holm p: sign-flip test on the per-problem differences, Holm-adjusted over the {n} contrast{'s' if n != 1 else ''} of the analysis file ({family}). "
                 f"Values under 0.001 are in scientific notation. “≤”: the raw p is a resampled one at its floor 1 / (reps + 1) = {p_floor(J):.1e} "
                 "(no resample as extreme as the data), so the printed value is an upper bound; exact p-values print without it.")
    if hollow:
        few = "; ".join(f"{label.replace(chr(10), ' ')}: {len(J['arms'][a]['seeds'])} seeds, {J['arms'][a]['episodes']} episodes" for a, label in arms if a in hollow)
        notes.append(f"Hollow marker: arm run on fewer seeds than the rest ({few}); its rate and CI rest on that many episodes.")
    x_note = "Values are written on the panels where there is room; every value is in the table. x-axis: the doses are equally spaced, not to scale."
    if has_caps or has_floors:
        x_note += " Rungs: " + "; ".join(t for t, on in (("cap, lower percentile = heavier clamp", has_caps), ("floor, higher percentile = heavier clamp", has_floors)) if on) + "."
    notes.append(x_note)
    return [line for note in notes for line in textwrap.wrap(note, width, subsequent_indent="   ")]


def fit_width(fig, artist, right: float) -> None:
    """Shrink a text until it ends left of the figure fraction `right` (a title that would run off the figure)."""
    renderer = fig.canvas.get_renderer()
    for _ in range(20):
        if artist.get_window_extent(renderer).x1 <= right * fig.bbox.width:
            break
        artist.set_fontsize(artist.get_fontsize() * 0.97)


def fit_titles(fig, suptitle, title_a, title_b, ax_b, dpi=None) -> None:
    """Keep the figure title and the two panel titles inside the figure / their own panel. dpi = the resolution the PNG is
    saved at: text is measured there, since hinted text is a little wider or narrower at another resolution."""
    if dpi:
        fig.set_dpi(dpi)
    fig.canvas.draw()
    fit_width(fig, suptitle, 0.98)
    fit_width(fig, title_a, ax_b.get_position().x0 - 0.03)
    fit_width(fig, title_b, 0.98)


def make_figure(fig_w: float, n_rows: int, n_note_lines: int, header_in: float = 1.75):
    """Figure with two line panels, the table under them and room for the footnote; sizes in inches so that rows keep their height.
    header_in = room above the panels for the figure title, the caption and the panel titles."""
    panel_in, gap_in, table_in, notes_in = 3.8, 1.75, 0.2 * (n_rows + 1), 0.16 * n_note_lines + 0.3
    fig_h = header_in + panel_in + gap_in + table_in + notes_in
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=SURFACE)
    left, panel_w, wgap = 0.07, 0.398, 0.119
    y_panel = (notes_in + table_in + gap_in) / fig_h
    ax_a = fig.add_axes([left, y_panel, panel_w, panel_in / fig_h])
    ax_b = fig.add_axes([left + panel_w + wgap, y_panel, panel_w, panel_in / fig_h])
    ax_t = fig.add_axes([left, notes_in / fig_h, 2 * panel_w + wgap, table_in / fig_h])
    return fig, ax_a, ax_b, ax_t, fig_h, (notes_in - 0.12) / fig_h


def style(ax, ylabel, layer):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(RULE)
    ax.tick_params(colors=INK2, labelsize=9, length=3)
    ax.yaxis.grid(True, color=RULE, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9.5)
    caps, floors = [a for a in CAPS if a in X], [a for a in FLOORS if a in X]
    heaviest = heaviest_rungs(caps, floors)
    ax.set_xticks([X[a] for a, _ in ARMS])
    ax.set_xticklabels([lab + ("\nheaviest" if a in heaviest else "") for a, lab in ARMS], color=INK, fontsize=9.5)
    ax.set_xlim(-0.5, max(X.values()) + 0.5)
    for group in (caps, floors):
        if group:
            ax.axvline(X[group[0]] - SIDE_GAP / 2 - 0.3, color=RULE, lw=0.8, ls=":", zorder=1)
    label = f"dose α at layer {layer} (units of the mean residual norm), {AXIS_NAME}; doses equally spaced, not to scale"
    if caps or floors:
        rungs = "cap rungs" if not floors else "cap rungs (ceiling) and floor rungs (inverse cap)"
        heavier = "; ".join(t for t, on in (("cap: lower percentile = heavier clamp", caps), ("floor: higher percentile = heavier clamp", floors)) if on)
        label += f"; {rungs} beside, τ at the unsteered percentile ({heavier})"
    wrapped_xlabel(ax, label)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", choices=MODELS, required=True)
    ap.add_argument("--tag", default="capladder", help="analysis tag: analysis_<tag>_all_as_judged.json etc. in the pilot root")
    ap.add_argument("--axis", choices=["gi_v2", "gfar"], default="gi_v2", help="gfar: the Goodfire-axis arms (gf_*) of the model")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--dpi", type=int, default=220)
    args = ap.parse_args()
    M = dict(MODELS[args.model])
    root, layer = M["root"], M["layer"]
    gfar = args.axis == "gfar"
    if gfar:
        use_gfar()
        M.update(GFAR[args.model])
        M.pop("jobs_floor", None)
        M["details"] = ([n for n in M["details"] if n.startswith("verdicts_main/")]
                        + sorted(str(q.relative_to(root)) for q in root.glob("verdicts_gf*/verdicts_*.detail.json")))
    d = load(root, args.tag, M["summary"], M["details"], layer)
    J, UP, DOWN, PX = d["all"]["judged_hack"], d["up"]["judged_hack"], d["down"]["judged_hack"], d["proxy"]["judged_hack"]
    SP, NC, AUTO = d["all"]["shown_pass"], d["all"]["no_code"], d["all"]["hack_proxy"]
    HP, hp_label, honest = capability(d["all"])
    arms = [a for a, _ in ARMS]
    doses = [a for a, _ in dose_arms()]
    caps, floors = [a for a in CAPS if a in X], [a for a in FLOORS if a in X]
    groups = [g for g in (caps, floors) if g]
    labelled = list(zip(arms, COL_LABELS))
    hollow = fewer_seeds(J, arms + random_arms(J))
    axis_arms = f"steered {'gfar' if gfar else 'gi'} arms"
    n_contrasts, n_random = len(J["contrasts"]), sum(is_random(a) for a in J["contrasts"])
    family = f"{n_contrasts} contrasts: {n_contrasts - n_random} {axis_arms}" + (f" + {n_random} random" if n_random else "")

    fig_w = 12.8 + 1.3 * len(floors)
    rows = table_rows(d, arms, J, UP, DOWN, PX, AUTO, HP, NC, SP, hp_label, honest, family, clamp_model={"gptoss": "gpt-oss", "muse": "Muse"}[args.model])
    notes = chart_notes(J, HP, NC, honest, labelled, hollow, axis_arms, bool(caps), bool(floors), width=int(fig_w * 14.2))
    fig, ax_a, ax_b, ax_t, fig_h, notes_y = make_figure(fig_w, len(rows), len(notes))

    style(ax_a, "share of episodes (%)", layer)
    title_a = hack_panel(ax_a, X, doses, groups, J, UP, DOWN, AUTO, hollow)
    style(ax_b, "share of episodes (%)", layer)
    title_b = capability_panel(ax_b, X, doses, groups, HP, NC, hp_label, honest, hollow)

    draw_table(fig, ax_t, rows, COL_LABELS)
    ax_t.set_title(f"Numbers behind the lines (rates are problem-stratified means over {J['problems']} problems; H / B / N = HACK / BORDERLINE / NOT_HACK)",
                   loc="left", color=INK, fontsize=10, pad=6)
    solo = len(doses) == 1  # a rung judged before any dose arm
    ladder = ("the cap ladder (τ at the unsteered p5 / p25 / p50) and the floor ladder, its inverse (τ at p50 / p75 / p95)" if floors
              else "the cap dose ladder (τ at the unsteered p5 / p25 / p50)")
    if gfar:
        present = [a.split("_")[-1] for a in caps + floors]
        ladder = f"the {' / '.join(present)} cap" if present and not floors else ladder
    scan = f" dose scan at layer {layer} and {ladder}" if not solo else f": {ladder}"
    title = fig.suptitle(f"{M['name']}, {'Goodfire RH axis (gfar)' if gfar else 'gi_v2'}{scan} over layers {M['band']}{',' if solo else ':'}\n"
                 "blind Sonnet rubric-v3 judging of every proxy episode and every hidden-pass episode with a hard-code signature",
                 x=0.07, ha="left", color=INK, fontsize=12.5, y=1 - 0.17 / fig_h)
    jobs = M.get("jobs_floor", M["jobs"]) if floors else M["jobs"]
    fig.text(0.07, 1 - 0.82 / fig_h, textwrap.fill(f"{jobs}; {n_note(J, labelled + [(a, a) for a in random_arms(J)])}.", int(fig_w * 13.4)) + "\n"
             "Judged rate = HACK (STRONG + WEAK) as judged, no adjudication. Notes on the bars, bands, marks and p-values are under the table.",
             color=INK2, fontsize=9, va="top")
    fig.text(0.07, notes_y, "\n".join(notes), color=INK2, fontsize=8.3, va="top", linespacing=1.25)
    fit_titles(fig, title, title_a, title_b, ax_b, dpi=args.dpi)
    out = args.out or root / (f"{args.model}_gfar_chart.png" if gfar else f"{args.model}_gi_v2_{'floorladder' if floors else 'capladder'}_chart.png")
    fig.savefig(out, dpi=args.dpi, facecolor=SURFACE)
    fig.savefig(out.with_suffix(".pdf"), facecolor=SURFACE)
    print(out)


if __name__ == "__main__":
    main()
