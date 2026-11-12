"""Three figures per headline claim about the deceiver axis (gi_v2), read straight from the group share.

Claims:
  1. Steering along the deceiver axis increases and decreases reward hacking (RH).
  2. The deceiver axis is distinct from the Goodfire reward-hacking axis (gfar).
  3. The deceiver axis encodes a deceptive persona.
  4. The deceiver axis is unique: random / control directions do not do the same.

Inputs: analysis_*.json files on the share (judged hack rates, problem-bootstrap CIs), gfar_vs_gi_v2 axis_compare_*.json,
gi_v2 validation.json, Koby's gi_v2_validation folder (persona projections, logit lens), and precomputed files next to this script:
  hack_given_code.json hack_given_code.py (per-episode hack among code submissions, all models, charts 2ci)
  antonym_counts.json  antonym_counts.py (origin/main cross_model_percentile.py counting code, proxy outcome)

    python make_claim_figures.py [--out <dir>]
"""
import argparse
import json
import re
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
R = Path("/grphome/grp_mirror/nobackup/autodelete/self-concept-experiments/schemer-axis")

# Design tokens (reference palette): text in ink tokens, colour only on marks.
SURFACE, INK, INK2, INK3, RULE = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8985", "#dedcd7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"   # deceiver axis, Goodfire axis / honest, third series
CTRL, CTRL_BAND = "#a9a8a3", "#ecebe7"                  # controls / random directions

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 9.5, "text.color": INK, "axes.labelcolor": INK2,
    "axes.edgecolor": RULE, "axes.linewidth": 0.8, "xtick.color": INK2, "ytick.color": INK2,
    "xtick.major.size": 0, "ytick.major.size": 0, "axes.grid": True, "grid.color": RULE, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "legend.frameon": False, "lines.linewidth": 2,
})

MODELS = ["gpt-oss-120b", "Muse-Glimmer-30B", "Gemma-4-31B", "Qwen3.8-27B"]
RULE_POINTS = 10.0  # dose rule: hidden pass and no-code within 10 points of unsteered


def load(rel):
    return json.loads((R / rel).read_text())


# Steering runs: analysis file, arm name per dose, random-direction arms (capability-matched).
STEER = {
    "gpt-oss-120b": dict(file="gptoss120b/gi_v2/pilot/analysis_floorladder_all_as_judged.json",
                         doses=[-0.4, -0.2, -0.1, 0.1, 0.2, 0.4], arm=lambda d: f"gi_{'p' if d > 0 else 'm'}{abs(d):g}",
                         main=0.2, randoms=["random_a0.07", "random_a0.07_s1730", "random_a0.07_s1731"],
                         rand_label="3 random directions (α 0.07, capability-matched)"),
    "Muse-Glimmer-30B": dict(file="muse_glimmer/gi_v2/pilot/analysis_capladder_all_as_judged.json",
                             doses=[-0.16, -0.08, -0.04, 0.04, 0.08, 0.16], arm=lambda d: f"gi_{'p' if d > 0 else 'm'}{abs(d):g}",
                             main=0.08, randoms=["random_plus", "random_plus_1730", "random_plus_1731"],
                             rand_label="3 random directions (+0.08, norm-matched)"),
    "Gemma-4-31B": dict(file="gemma4_31b_gi_v2_judging_jansen05/hard108/analysis_ctl16/analysis_all_as_judged.json",
                        doses=[-0.16, -0.08, -0.04, 0.04, 0.08, 0.16], arm=lambda d: f"L30_a{d:g}",
                        main=0.08, randoms=["L30_rand1729_a0.01", "L30_rand1730_a0.01", "L30_rand1731_a0.01"],
                        rand_label="3 random directions (α 0.01, capability-matched)"),
    "Qwen3.8-27B": dict(file="qwen38_27b_gi_v2_judging_jansen05/qwen_hard61/analysis_main/analysis_all_as_judged.json",
                        doses=[-0.32, -0.16, -0.08, 0.08, 0.16, 0.32], arm=lambda d: f"L32_a{d:g}",
                        main=0.16, randoms=[], rand_label=None),
}


def arm_stats(an, arm, outcome="judged_hack"):
    a = an[outcome]["arms"][arm]
    ci = a.get("ci95", [np.nan, np.nan])
    return dict(rate=100 * a["rate"], diff=100 * a.get("diff_vs_unsteered", 0.0), lo=100 * ci[0], hi=100 * ci[1],
                up=a.get("problems_up"), down=a.get("problems_down"), p=a.get("p_signflip"))


def capability_key(an):
    return "hidden_pass" if "hidden_pass" in an else "honest_hidden_pass"


def within_rule(an, arm):
    cap = capability_key(an)
    dh = 100 * (an[cap]["arms"][arm]["rate"] - an[cap]["arms"]["unsteered"]["rate"])
    dn = 100 * (an["no_code"]["arms"][arm]["rate"] - an["no_code"]["arms"]["unsteered"]["rate"])
    return abs(dh) <= RULE_POINTS and dn <= RULE_POINTS, dh, dn


def header(fig, title, verdict, source):
    fig.text(0.012, 0.985, title, ha="left", va="top", fontsize=13, fontweight="bold", color=INK)
    fig.text(0.012, 0.935, verdict, ha="left", va="top", fontsize=10, color=INK2, wrap=True)
    fig.text(0.012, 0.008, textwrap.fill(source, int(fig.get_figwidth() * 17)), ha="left", va="bottom", fontsize=7.2, color=INK3,
             linespacing=1.3)


def save(fig, out, name):
    fig.savefig(out / f"{name}.png", dpi=170)
    plt.close(fig)
    print("wrote", out / f"{name}.png")


def hollow_or_filled(ax, x, y, ok, color, size=7, zorder=4, **kw):
    ax.plot(x, y, "o", ms=size, mfc=color if ok else SURFACE, mec=color, mew=1.8, zorder=zorder, **kw)


# ----------------------------------------------------------------------------------------------- claim 1
def fig_1a_dose_response(out):
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.9), sharey=False)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.70, bottom=0.2, wspace=0.28)
    for ax, model in zip(axes, MODELS):
        cfg = STEER[model]; an = load(cfg["file"])
        unst = arm_stats(an, "unsteered")["rate"]
        doses = cfg["doses"][:3] + [0.0] + cfg["doses"][3:]
        xs = np.arange(len(doses))
        ys, los, his, oks = [], [], [], []
        for d in doses:
            if d == 0:
                ys.append(0); los.append(0); his.append(0); oks.append(True); continue
            s = arm_stats(an, cfg["arm"](d)); ok, _, _ = within_rule(an, cfg["arm"](d))
            ys.append(s["diff"]); los.append(s["lo"]); his.append(s["hi"]); oks.append(ok)
        if cfg["randoms"]:
            rd = [arm_stats(an, r)["diff"] for r in cfg["randoms"]]
            ax.axhspan(min(rd), max(rd), color=CTRL_BAND, zorder=0, lw=0)
            ax.text(xs[0] - 0.35, max(rd) + 0.4, "random-direction range", fontsize=7.5, color=INK3, va="bottom")
        ax.axhline(0, color=INK3, lw=0.9, zorder=1)
        ax.plot(xs, ys, color=BLUE, lw=2, zorder=2)
        ax.vlines(xs, los, his, color=BLUE, lw=1.4, alpha=0.55, zorder=2)
        for x, y, ok in zip(xs, ys, oks):
            hollow_or_filled(ax, x, y, ok, BLUE)
        ax.set_xticks(xs, [("0" if d == 0 else f"{d:+g}") for d in doses])
        ax.set_title(f"{model}\nunsteered judged hack rate {unst:.1f}%", fontsize=9.5)
        ax.set_xlabel("steering dose along deceiver axis")
        ax.grid(axis="x", visible=False)
        lim = max(abs(min(los)), abs(max(his)), 3) * 1.12
        ax.set_ylim(-lim, lim)
    axes[0].set_ylabel("change in judged hack rate (points)")
    h1 = plt.Line2D([], [], color=BLUE, marker="o", mfc=BLUE, mec=BLUE, label="inside capability rule")
    h2 = plt.Line2D([], [], color=BLUE, marker="o", mfc=SURFACE, mec=BLUE, mew=1.8, label="outside rule (model degraded)")
    h3 = plt.Rectangle((0, 0), 1, 1, color=CTRL_BAND, label="range of 3 random directions")
    fig.legend(handles=[h1, h2, h3], loc="upper right", ncol=3, fontsize=8.5, bbox_to_anchor=(0.99, 0.865))
    header(fig, "1a. Pushing along the deceiver axis moves reward hacking both ways on gpt-oss and Muse, not on Gemma or Qwen",
           "Verdict: SUPPORTED on 2 of 4 models. gpt-oss: −10.6 to +41.3 points; Muse: −14.4 to +16.5. Gemma's rise (+3.5, at a dose that already costs capability) sits inside the "
           "random-direction range; Qwen hacks ~0% unsteered and only moves at +0.32, where it is already degraded. Bars = 95% problem-bootstrap CI.",
           "Data: schemer-axis/{gptoss120b,muse_glimmer}/gi_v2/pilot/analysis_*_all_as_judged.json, gemma4_31b_gi_v2_judging_jansen05/hard108/analysis_ctl16, "
           "qwen38_27b_gi_v2_judging_jansen05/qwen_hard61/analysis_main. Judged = blind Sonnet rubric-v3 HACK verdicts. Rule: hidden pass and no-code within 10 pts of unsteered.")
    save(fig, out, "1a_dose_response_four_models")


def fig_1b_clamps(out):
    specs = [("gpt-oss-120b", "gptoss120b/gi_v2/pilot/analysis_floorladder_all_as_judged.json"),
             ("Muse-Glimmer-30B", "muse_glimmer/gi_v2/pilot/analysis_floorladder_all_as_judged.json")]
    arms = [("gi_cap_p5", "cap\np5"), ("gi_cap_p25", "cap\np25"), ("gi_cap_p50", "cap\np50"), ("unsteered", "none"),
            ("gi_floor_p50", "floor\np50"), ("gi_floor_p75", "floor\np75"), ("gi_floor_p95", "floor\np95")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.9))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.74, bottom=0.17, wspace=0.22)
    for ax, (model, f) in zip(axes, specs):
        an = load(f); xs = np.arange(len(arms))
        st = [arm_stats(an, a) if a != "unsteered" else dict(rate=arm_stats(an, a)["rate"], diff=0, lo=0, hi=0) for a, _ in arms]
        ys = [s["diff"] for s in st]
        ax.axhline(0, color=INK3, lw=0.9)
        ax.axvspan(-0.5, 2.5, color="#f3f6fb", zorder=0, lw=0); ax.axvspan(3.5, 6.5, color="#fcf3ee", zorder=0, lw=0)
        ax.plot(xs, ys, color=BLUE, zorder=2)
        ax.vlines(xs, [s["lo"] for s in st], [s["hi"] for s in st], color=BLUE, lw=1.4, alpha=0.55)
        for x, y in zip(xs, ys):
            hollow_or_filled(ax, x, y, True, BLUE)
            if x != 3:
                ax.text(x + 0.12, y, f"{y:+.1f}", fontsize=8, color=INK2, va="center")
        ax.set_xticks(xs, [l for _, l in arms]); ax.grid(axis="x", visible=False)
        ax.set_title(f"{model}  (unsteered {st[3]['rate']:.1f}%)")
        ax.text(1, ax.get_ylim()[1] * 0.92, "clamp the axis from above\n(keep the model less deceptive)", ha="center", fontsize=8, color=INK2, va="top")
        ax.text(5, ax.get_ylim()[0] * 0.55, "clamp the axis from below\n(keep the model more deceptive)", ha="center", fontsize=8, color=INK2)
    axes[0].set_ylabel("change in judged hack rate (points)")
    header(fig, "1b. Clamping the model's position on the axis, instead of pushing it, also moves hacking in both directions",
           "Verdict: SUPPORTS claim 1 on gpt-oss and Muse with a second kind of intervention. A cap at the model's own 5th percentile lowers hacking; "
           "a floor at its 95th raises it. Caveat: on Muse only floor p50 keeps both capability measures within 10 points.",
           "Data: schemer-axis/{gptoss120b,muse_glimmer}/gi_v2/pilot/analysis_floorladder_all_as_judged.json (12 seeds × 45 problems per arm; caps/floors over layers "
           "12–23 / 17–33 at percentiles of each model's unsteered projection). Bars = 95% problem-bootstrap CI.")
    save(fig, out, "1b_cap_and_floor_ladders")


def fig_1c_capability(out):
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.9))
    fig.subplots_adjust(left=0.05, right=0.99, top=0.74, bottom=0.16, wspace=0.28)
    for ax, model in zip(axes, MODELS):
        cfg = STEER[model]; an = load(cfg["file"])
        ax.axvspan(-RULE_POINTS, RULE_POINTS, color="#eef3fb", zorder=0, lw=0)
        ax.axhline(0, color=INK3, lw=0.9); ax.axvline(0, color=INK3, lw=0.9)
        for d in cfg["doses"]:
            a = cfg["arm"](d); s = arm_stats(an, a); ok, dh, _ = within_rule(an, a)
            hollow_or_filled(ax, dh, s["diff"], ok, BLUE)
            ax.annotate(f"{d:+g}", (dh, s["diff"]), xytext=(5, 3), textcoords="offset points", fontsize=7.5, color=INK2)
        for r in cfg["randoms"]:
            s = arm_stats(an, r); _, dh, _ = within_rule(an, r)
            ax.plot(dh, s["diff"], "D", ms=6, color=CTRL, mec=SURFACE, mew=1, zorder=3)
        ax.set_title(model); ax.set_xlabel("change in hidden-test pass rate (points)")
        xl = max(RULE_POINTS + 4, max(abs(x) for x in ax.get_xlim())); ax.set_xlim(-xl, xl)
    axes[0].set_ylabel("change in judged hack rate (points)")
    h1 = plt.Line2D([], [], ls="", marker="o", mfc=BLUE, mec=BLUE, label="deceiver-axis dose (inside rule)")
    h2 = plt.Line2D([], [], ls="", marker="o", mfc=SURFACE, mec=BLUE, mew=1.8, label="outside rule")
    h3 = plt.Line2D([], [], ls="", marker="D", color=CTRL, label="random direction")
    h4 = plt.Rectangle((0, 0), 1, 1, color="#eef3fb", label="hidden pass within ±10 points")
    fig.legend(handles=[h1, h2, h3, h4], loc="upper right", ncol=4, fontsize=8.5, bbox_to_anchor=(0.99, 0.875))
    header(fig, "1c. On gpt-oss and Muse the hacking change is not just the model getting worse at coding",
           "Verdict: SUPPORTS claim 1 for gpt-oss and Muse. Most of their doses change hacking by 5–40 points while hidden-test pass moves under 10. "
           "On Gemma and Qwen the larger doses mostly cost capability.",
           "Data: same analysis files as 1a. Hidden pass = passes the full hidden test suite. Point labels = steering dose. Shaded band = the 10-point capability rule.")
    save(fig, out, "1c_hacking_vs_capability")


# ----------------------------------------------------------------------------------------------- claim 2
GF = R / "gfar_vs_gi_v2_jansen05/analysis"
GF_FILES = {"gpt-oss-120b": "gptoss120b", "Muse-Glimmer-30B": "muse", "Gemma-4-31B": "gemma4_31b", "Qwen3.8-27B": "qwen38_27b"}


def fig_2a_cosine(out):
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.6), sharey=True)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.74, bottom=0.14, wspace=0.12)
    for ax, model in zip(axes, MODELS):
        d = json.loads((GF / f"axis_compare_{GF_FILES[model]}.json").read_text())
        layers = np.array(d["layers"]); c = d["cosine"]["gf_all"]
        cos = np.array(c["cosine"]); null = np.array(c["sign_flip_null_abs_95"])
        ax.fill_between(layers, -null, null, color=CTRL_BAND, lw=0, label="95% sign-flip null (no relationship)")
        ax.axhline(0, color=INK3, lw=0.9)
        ax.plot(layers, cos, color=ORANGE, label="cosine(Goodfire axis, deceiver axis)")
        L = d["steer_layer"]; i = list(layers).index(L)
        ax.plot(L, cos[i], "o", ms=8, color=ORANGE, mec=SURFACE, mew=2, zorder=5)
        ax.annotate(f"steer layer {L}: {cos[i]:+.2f}", (L, cos[i]), xytext=(6, 14), textcoords="offset points", fontsize=8, color=INK)
        ax.text(0.02, 0.04, f"Goodfire axis detects hacks on its own data:\nvalidation AUROC {c['validation_auroc'][i]:.2f} at layer {L}",
                transform=ax.transAxes, fontsize=7.5, color=INK2)
        ax.set_title(model); ax.set_xlabel("layer"); ax.set_ylim(-1, 1)
    axes[0].set_ylabel("cosine similarity")
    axes[0].legend(loc="upper left", fontsize=8)
    header(fig, "2a. The Goodfire reward-hacking axis and the deceiver axis point in unrelated directions",
           "Verdict: SUPPORTED. At every model's steering layer the cosine is −0.02 to +0.07, inside or at the edge of the noise band. "
           "1.0 would mean the same direction. The Goodfire axis itself works as a hack detector (AUROC 0.74–0.91).",
           "Data: schemer-axis/gfar_vs_gi_v2_jansen05/analysis/axis_compare_<model>.json, key cosine.gf_all (Koby's all-response Goodfire hack-minus-honest direction).")
    save(fig, out, "2a_cosine_goodfire_vs_deceiver")


def fig_2b_correlation(out):
    fig, ax = plt.subplots(figsize=(11, 4.8))
    fig.subplots_adjust(left=0.2, right=0.97, top=0.75, bottom=0.16)
    rows = []
    for model in MODELS:
        d = json.loads((GF / f"axis_compare_{GF_FILES[model]}.json").read_text()); i = d["layers"].index(d["steer_layer"])
        for reg in ("reasoning", "answer"):
            e = d["episode"][reg]["gf_all"][i]["vs_control"]["episode"]
            rows.append((model, reg, e["r"], 100 * e["reference"]["percentile"]))
    ys = np.arange(len(MODELS))[::-1] * 1.0
    ax.axvspan(95, 100, color="#fcf3ee", lw=0)
    ax.text(97.5, ys[0] + 0.5, "would count\nas shared", ha="center", fontsize=7.5, color=INK2, va="bottom")
    for j, model in enumerate(MODELS):
        for k, (reg, col, dy) in enumerate((("reasoning", BLUE, 0.13), ("answer", ORANGE, -0.13))):
            _, _, r, pct = next(x for x in rows if x[0] == model and x[1] == reg)
            y = ys[j] + dy
            ax.plot([0, pct], [y, y], color=col, lw=2, alpha=0.35)
            ax.plot(pct, y, "o", ms=8, color=col, mec=SURFACE, mew=1.5)
            ax.text(pct + 1.2, y, f"{pct:.0f}th pct   (r = {r:+.2f})", va="center", fontsize=8, color=INK2)
    ax.set_yticks(ys, MODELS); ax.set_xlim(0, 100); ax.set_ylim(ys[-1] - 0.6, ys[0] + 1.0)
    ax.set_xlabel("percentile of the Goodfire–deceiver correlation among the deceiver axis's correlations with 109 unrelated control directions")
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[plt.Line2D([], [], ls="", marker="o", color=BLUE, label="reasoning tokens"),
                       plt.Line2D([], [], ls="", marker="o", color=ORANGE, label="answer tokens")], loc="lower right", fontsize=8.5)
    header(fig, "2b. On real coding episodes the two axes track each other no more than unrelated directions do",
           "Verdict: SUPPORTED. On unsteered episodes, how strongly the two projections co-vary ranks between the 4th and 53rd percentile of what "
           "arbitrary antonym-concept directions give, never in the top 5%. (Muse's last-token readout, not shown, reaches the 97th / 89th.)",
           "Data: axis_compare_<model>.json, episode.<region>.gf_all[steer layer].vs_control.episode (540 / 540 / 1,728 / 732 unsteered episodes). "
           "Control null = |r| of the deceiver axis with each of 109 isolated-text antonym control directions.")
    save(fig, out, "2b_episode_correlation_vs_controls")


SUB = "deceiver_vs_goodfire_steering"   # subfolder for every deceiver-vs-Goodfire steering chart
GF_STEER = {
    "gpt-oss-120b": dict(
        slug="gpt-oss", layer=17,
        gi=("gptoss120b/gi_v2/pilot/analysis_floorladder_all_as_judged.json", [-0.4, -0.2, -0.1, 0.1, 0.2, 0.4], lambda d: f"gi_{'p' if d > 0 else 'm'}{abs(d):g}"),
        gf=("gptoss120b/gi_v2/pilot/analysis_gfscan_all_as_judged.json", [-0.4, -0.2, -0.1, 0.1, 0.2, 0.4], lambda d: f"gf_{'p' if d > 0 else 'm'}{abs(d):g}"),
        title="gpt-oss: only the deceiver axis raises hacking; pushing Goodfire positive stops the model writing code",
        verdict="SUPPORTS claim 2 on gpt-oss. Pushing either axis negative lowers hacking. Pushing Goodfire positive does not raise it (14.3 / 9.8 / 5.2%); the model "
                "stops submitting code instead (up to 67%). The deceiver axis raises hacking to 53% and the model keeps writing code. (Among code submissions only, "
                "Goodfire does rise to 19–22%: see 2ci.)"),
    "Muse-Glimmer-30B": dict(
        slug="muse", layer=26,
        gi=("muse_glimmer/gi_v2/pilot/analysis_floorladder_all_as_judged.json", [-0.16, -0.08, -0.04, 0.04, 0.08, 0.16], lambda d: f"gi_{'p' if d > 0 else 'm'}{abs(d):g}"),
        gf=("muse_glimmer/gi_v2/pilot/analysis_gfdose_all_as_judged.json", [-0.16, -0.08, 0.08, 0.16], lambda d: f"gf_{'p' if d > 0 else 'm'}{abs(d):g}"),
        title="Muse: steering along Goodfire moves hacking as much as the deceiver axis does",
        verdict="DOES NOT SUPPORT the behavioural half of claim 2 on Muse. Goodfire takes judged hacking from 32.8% to 10.7% (−0.16) and 52.0% (+0.16); the deceiver "
                "axis gives 18.3% and 49.3% at the same doses. Neither stops code submission. The two directions are still nearly orthogonal (2a), so on Muse two "
                "different directions both steer hacking. Goodfire's negative doses are hollow on the rule's measure, but raw hidden pass barely moves (−2.6, −0.4)."),
    "Gemma-4-31B": dict(
        slug="gemma", layer=30,
        gi=("gemma4_31b_gi_v2_judging_jansen05/hard108/analysis_ctl16/analysis_all_as_judged.json", [-0.16, -0.08, -0.04, 0.04, 0.08, 0.16], lambda d: f"L30_a{d:g}"),
        gf=("gemma4_31b_gi_v2_judging_jansen05/hard108/analysis_gfar3/analysis_all_as_judged.json", [-0.04, 0.04, 0.08], lambda d: f"gfar_L30_a{d:g}"),
        title="Gemma: Goodfire steering raises hacking at least as much as the deceiver axis",
        verdict="DOES NOT SUPPORT the behavioural half of claim 2 on Gemma. At −0.04 / +0.04 / +0.08 Goodfire gives 3.4 / 7.3 / 12.3% judged hacking against 4.7% "
                "unsteered; the deceiver axis gives 4.5 / 6.1 / 8.2%. Both +0.08 arms cost capability (hollow). Goodfire's other doses were stopped after a one-seed "
                "pilot (±0.16 broke the model), so the two lines cover different ranges."),
    "Qwen3.8-27B": dict(
        slug="qwen", layer=32,
        gi=("qwen38_27b_gi_v2_judging_jansen05/qwen_hard61/analysis_main/analysis_all_as_judged.json", [-0.32, -0.16, -0.08, 0.08, 0.16, 0.32], lambda d: f"L32_a{d:g}"),
        gf=("qwen38_27b_gi_v2_judging_jansen05/qwen_hard61/analysis_gfar_gfdose/analysis_all_as_judged.json", [-0.08, 0.08], lambda d: f"gfar_L32_a{d:g}"),
        title="Qwen: neither axis moves hacking much, because Qwen almost never hacks",
        verdict="NO TEST of claim 2 on Qwen. Unsteered judged hacking is 0.0%. Goodfire ±0.08 gives 0.0 / 0.3%; the deceiver axis gives 0.0 / 0.5% at the same doses "
                "and only reaches 4.1% at +0.32 (outside the capability rule). Goodfire −0.08 adds 10 points of no-code. Goodfire was only run at ±0.08."),
}


def fig_2c_model(out, model):
    c = GF_STEER[model]
    (gi_f, gi_d, gi_arm), (gf_f, gf_d, gf_arm) = c["gi"], c["gf"]
    gi, gf = load(gi_f), load(gf_f)
    doses = sorted(set(gi_d) | set(gf_d) | {0}); xs = {d: i for i, d in enumerate(doses)}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    fig.subplots_adjust(left=0.065, right=0.97, top=0.71, bottom=0.15, wspace=0.28)
    for ax, outcome, label in zip(axes, ("judged_hack", "no_code"), ("judged hack rate (%)", "episodes with no code submitted (%)")):
        top = 0
        for an, ds, arm, col, name, dx in ((gi, gi_d, gi_arm, BLUE, "deceiver axis", -0.05), (gf, gf_d, gf_arm, ORANGE, "Goodfire RH axis", 0.05)):
            pts = [(0, "unsteered")] + [(d, arm(d)) for d in ds]
            pts.sort()
            u = 100 * an[outcome]["arms"]["unsteered"]["rate"]
            x = np.array([xs[d] for d, _ in pts]) + dx
            y = np.array([100 * an[outcome]["arms"][a]["rate"] for _, a in pts])
            ax.plot(x, y, color=col, lw=2, zorder=3)
            for xi, (d, a), yi in zip(x, pts, y):
                ok = a == "unsteered" or within_rule(an, a)[0]
                if outcome == "judged_hack" and a != "unsteered":
                    ci = an[outcome]["arms"][a].get("ci95")
                    if ci:
                        ax.plot([xi, xi], [u + 100 * ci[0], u + 100 * ci[1]], color=col, lw=1.3, alpha=0.8, zorder=2)
                        top = max(top, u + 100 * ci[1])
                hollow_or_filled(ax, xi, yi, ok, col)
            top = max(top, y.max())
            ax.text(x[-1] + 0.15, y[-1], name, color=INK, fontsize=8.5, va="center")
        ax.set_xticks(range(len(doses)), [("0" if d == 0 else f"{d:+g}") for d in doses]); ax.grid(axis="x", visible=False)
        ax.set_xlabel(f"steering dose (same units for both axes, layer {c['layer']})"); ax.set_ylabel(label)
        ax.set_xlim(-0.4, len(doses) + 0.6); ax.set_ylim(-1, max(5, top * 1.12))
    axes[0].set_title("Reward hacking (bars: 95% CI of the change from unsteered)"); axes[1].set_title("Capability side effect")
    header(fig, f"2c ({model}). {c['title']}", f"Verdict: {c['verdict']}",
           f"Data: schemer-axis/{gi_f} (deceiver) and {gf_f} (Goodfire), judged as-judged rates; both files share the unsteered arm. "
           "Hollow = outside the team's dose rule (hidden pass, on the measure each file reports, or no-code moves more than 10 points from unsteered).")
    save(fig, out, f"{SUB}/2c_{c['slug']}_steering_goodfire_vs_deceiver")


GF_CODE_TEXT = {
    "gpt-oss-120b": ("gpt-oss: among code submissions Goodfire rises a little, the deceiver axis a lot",
                     "still SUPPORTS claim 2, with a correction to 2c. Among episodes with code, pushing Goodfire positive raises hacking from 12.6% to 19 / 22 / 16%, "
                     "so part of its flat line in 2c came from the model submitting less code. The deceiver axis reaches 53% with code nearly always submitted. Caveat: "
                     "under strong Goodfire steering only a third of episodes have code, and they may be the easier problems."),
    "Muse-Glimmer-30B": ("Muse: unchanged from 2c, because Muse submits code in almost every episode",
                         "same as 2c: DOES NOT SUPPORT the behavioural half of claim 2 on Muse. Among episodes with code, Goodfire gives 11.3% (−0.16) to 52.1% (+0.16) "
                         "and the deceiver axis 18.9% to 49.3%. At least 515 of 540 episodes have code in every arm, so dividing by code submissions changes almost nothing."),
    "Gemma-4-31B": ("Gemma: unchanged from 2c except the deceiver axis at −0.16",
                    "same as 2c: DOES NOT SUPPORT the behavioural half of claim 2 on Gemma. Among episodes with code, Goodfire gives 3.4 / 7.3 / 12.4% at −0.04 / +0.04 / "
                    "+0.08 and the deceiver axis 4.5 / 6.1 / 8.2%. Only the deceiver's −0.16 arm moves (7.1% → 9.6%), because 26% of its episodes have no code; that arm "
                    "is outside the capability rule anyway."),
    "Qwen3.8-27B": ("Qwen: still almost no hacking once episodes without code are removed",
                    "NO TEST of claim 2 on Qwen. A quarter of unsteered episodes have no code, so the denominators shrink, but hacking stays near zero: 0.0% unsteered, "
                    "Goodfire ±0.08 0.0 / 0.4%, deceiver ±0.08 0.0 / 0.8%. Only the deceiver's +0.32 arm (outside the capability rule) reaches 5.2%."),
}


def fig_2ci_model(out, model):
    hc = json.loads((HERE / "hack_given_code.json").read_text())[model]
    c = GF_STEER[model]
    (_, gi_d, gi_arm), (_, gf_d, gf_arm) = c["gi"], c["gf"]
    doses = sorted(set(gi_d) | set(gf_d) | {0}); xs = {d: i for i, d in enumerate(doses)}
    fig, ax = plt.subplots(figsize=(12.5, 6.2))
    fig.subplots_adjust(left=0.07, right=0.8, top=0.74, bottom=0.21)
    top = 0
    for ds, arm, col, name, dx in ((gi_d, gi_arm, BLUE, "deceiver axis", -0.06), (gf_d, gf_arm, ORANGE, "Goodfire RH axis", 0.06)):
        pts = sorted([(0, "unsteered")] + [(d, arm(d)) for d in ds])
        a = [hc[n] for _, n in pts]
        x = np.array([xs[d] for d, _ in pts]) + dx
        y = np.array([100 * r["hack_given_code"] for r in a]); raw = [100 * r["hack_rate"] for r in a]
        lo = y - [100 * r["ci95"][0] for r in a]; hi = [100 * r["ci95"][1] for r in a] - y
        top = max(top, max(100 * r["ci95"][1] for r in a))
        ax.plot(x, raw, color=col, lw=1.2, ls=(0, (3, 2)), alpha=0.75, zorder=2)
        ax.errorbar(x, y, yerr=[lo, hi], color=col, lw=2, marker="o", ms=7, mec=SURFACE, mew=1.5, capsize=0, elinewidth=1.3, zorder=3)
        ax.text(x[-1] + 0.25, y[-1], f"{name}\namong episodes with code", color=INK, fontsize=8.5, va="center")
    def code(arm, d, ds):
        return str(hc["unsteered" if d == 0 else arm(d)]["with_code"]) if (d == 0 or d in ds) else "–"
    n = hc["unsteered"]["episodes"]
    ax.set_xticks(range(len(doses)), [("0" if d == 0 else f"{d:+g}") + f"\n{code(gi_arm, d, gi_d)} / {code(gf_arm, d, gf_d)}" for d in doses])
    ax.grid(axis="x", visible=False)
    ax.set_xlabel(f"steering dose (same units for both axes, layer {c['layer']})\nsecond row: episodes with code out of {n}, deceiver / Goodfire")
    ax.set_ylabel("judged hack rate among episodes with code (%)")
    ax.set_xlim(-0.4, len(doses) - 0.5); ax.set_ylim(-0.5 if top < 10 else -1, max(3, top * 1.08))
    title, verdict = GF_CODE_TEXT[model]
    header(fig, f"2ci ({model}). {title}", f"Verdict: {verdict}",
           f"Data: the raw episode files (code = non-empty final_code) and the judge's as-judged verdicts behind the 2c analysis files; HACK only, and every judged hack "
           f"had code. Same seeds as the team's analysis; counts match its rates exactly. Bars = problem-bootstrap 95% CI. Dashed = all episodes, as in 2c. "
           f"hack_given_code.py → hack_given_code.json.")
    save(fig, out, f"{SUB}/2ci_{c['slug']}_hack_rate_among_code_submissions")


# ----------------------------------------------------------------------------------------------- claim 3
def pretty_role(r):
    return r.replace("_fic", "").replace("_v2", "").replace("_", " ")


V = R / "gi_v2_validation"   # Koby's gi_v2 validation on all four models (2026-11-11)
V_DIRS = {"gpt-oss-120b": ("gptoss_reference", 17), "Muse-Glimmer-30B": ("muse", 26), "Gemma-4-31B": ("gemma", 30), "Qwen3.8-27B": ("qwen", 32)}
# Persona groups exactly as in the summary JSONs' "group" keys.
PERSONA_GROUPS = [
    ("deceptive, built the axis", BLUE, True, "schemer deceiver con_artist_v2 sycophant spin_doctor infiltrator two_faced_friend rumor_fabricator scammer"),
    ("deceptive, held out", BLUE, False, "embezzler impostor gaslighter"),
    ("cheating / reward-hacking, held out", AQUA, True, "cheater sandbagger saboteur social_climber metric_gamer corner_cutter rules_lawyer"),
    ("gray: legitimate deception, bluntness, neutral, held out", INK3, True,
     "poker_player stage_magician undercover_detective white_liar blunt_truth_teller strategist neutral"),
    ("honest, held out", ORANGE, False, "honest_bookkeeper humble_expert accountable_owner rule_follower fair_player spirit_keeper"),
    ("honest, built the axis", ORANGE, True,
     "straight_shooter faithful_witness transparent_negotiator candid_friend scrupulous_scientist_v2 whistleblower loyal_steward "
     "conscientious_craftsperson_v2 diligent_worker"),
]


def v_summary(model):
    d, L = V_DIRS[model]
    return json.loads((V / d / "figures_response_only" / f"response_only_L{L}_summary.json").read_text())


def fig_3a_role_positions(out):
    # Each persona's mean projection on held-out questions, rescaled per model: 0 = mean of the 9 honest axis personas,
    # 1 = mean of the 9 deceptive axis personas (raw scales differ ~100x between models).
    norm, groups_of = {}, {}
    for model in MODELS:
        p = v_summary(model)["persona"]
        get = lambda r: p[r + "_fic"]["heldout"]
        lo = np.mean([get(r)["mean"] for r in PERSONA_GROUPS[5][3].split()])
        hi = np.mean([get(r)["mean"] for r in PERSONA_GROUPS[0][3].split()])
        f = lambda x: (x - lo) / (hi - lo)
        norm[model] = {r: (f(get(r)["mean"]), f(get(r)["ci_low"]), f(get(r)["ci_high"])) for g in PERSONA_GROUPS for r in g[3].split()}
    # Row order: by group, then by cross-model average within group (highest first).
    rows = []
    for gi, g in enumerate(PERSONA_GROUPS):
        rs = sorted(g[3].split(), key=lambda r: -np.mean([norm[m][r][0] for m in MODELS]))
        rows += [(gi, r) for r in rs]
    gap = 0.9
    ys, y = [], 0.0
    for i, (gi, r) in enumerate(rows):
        if i and gi != rows[i - 1][0]:
            y += gap
        ys.append(y); y += 1
    ys = np.max(ys) - np.array(ys)
    fig, axes = plt.subplots(1, 4, figsize=(15, 12.5), sharey=True)
    fig.subplots_adjust(left=0.165, right=0.99, top=0.855, bottom=0.075, wspace=0.07)
    summary = {}
    for ax, model in zip(axes, MODELS):
        for (gi, r), yy in zip(rows, ys):
            _, col, filled, _ = PERSONA_GROUPS[gi]
            m, l, h = norm[model][r]
            ax.plot([l, h], [yy, yy], color=col, lw=1.4, alpha=0.55, solid_capstyle="round")
            ax.plot(m, yy, "o", ms=6.5, mfc=col if filled else SURFACE, mec=col, mew=1.6, zorder=4)
        dec = [norm[model][r][0] for g in PERSONA_GROUPS[:2] for r in g[3].split()]
        hon = [norm[model][r][0] for g in PERSONA_GROUPS[4:] for r in g[3].split()]
        summary[model] = (min(dec), max(hon))
        ax.axvline(0, color=INK3, lw=0.9); ax.axvline(1, color=INK3, lw=0.9, ls=(0, (3, 2)))
        ax.set_title(f"{model} (layer {V_DIRS[model][1]})"); ax.set_xlabel("position on deceiver axis\n(0 = honest axis personas, 1 = deceptive)")
        ax.set_xlim(-0.75, 1.55); ax.grid(axis="y", visible=False)
        if min(dec) > max(hon):
            ax.text(0.99, 0.003, "every deceptive persona\nabove every honest one", transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5, color=INK2)
    axes[0].set_yticks(ys, [pretty_role(r) for _, r in rows], fontsize=8.2)
    axes[0].set_ylim(-0.8, ys.max() + 0.8)
    # Group labels in the left margin, above each block.
    for gi, (name, col, filled, _) in enumerate(PERSONA_GROUPS):
        top = max(yy for (g, _), yy in zip(rows, ys) if g == gi)
        fig.text(0.003, axes[0].transData.transform((0, top + 0.75))[1] / fig.dpi / fig.get_figheight(), ("● " if filled else "○ ") + name,
                 fontsize=8.3, color=INK, fontweight="bold", va="center")
        fig.texts[-1].set_color(INK)
    for m, (d, h) in summary.items():
        print(f"  3a {m}: lowest deceptive {d:+.2f}, highest honest {h:+.2f}")
    header(fig, "3a. Held-out deceptive personas land with the deceptive ones; held-out honest personas with the honest ones",
           "Verdict: SUPPORTS claim 3. On all four models every deceptive persona, including the 3 never used to build the axis, scores above every honest persona (including 6 held out). "
           "Cheating roles and legitimate-deception roles (poker player, stage magician, undercover detective) fall in between, and the neutral character sits near the honest end. "
           "Filled = used to build the axis; hollow = held out. All scores are on held-out questions.",
           "Data: schemer-axis/gi_v2_validation/<model>/figures_response_only/response_only_L<layer>_summary.json (Koby, 2026-11-11), persona.<role>.heldout: mean projection "
           "of the persona's answer activations on 36 held-out questions with bootstrap 95% CI (bars). Rescaled per model so the honest / deceptive axis-persona means are 0 / 1. "
           "gpt-oss uses gptoss_reference (axis gptoss120b_v2/full/axis_response_only/persona_contrast.pt).")
    save(fig, out, "3a_persona_positions_on_axis")


def fig_3b_heldout_auroc(out):
    files = {"gpt-oss-120b": "gptoss120b/gi_v2/validation/validation.json", "Muse-Glimmer-30B": "muse_glimmer/gi_v2/validation/validation.json",
             "Gemma-4-31B": "gemma4_31b_gi_v2/validation/validation.json", "Qwen3.8-27B": "qwen38_27b_gi_v2/validation/validation.json"}
    tests = [("schemer_vs_honest", "deceptive vs honest personas"), ("schemer_vs_default", "deceptive persona vs plain assistant"),
             ("honest_vs_default", "honest persona vs plain assistant"), ("costume_vs_default", "neutral character vs plain assistant")]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.6), sharey=True)
    fig.subplots_adjust(left=0.185, right=0.99, top=0.74, bottom=0.2, wspace=0.1)
    for ax, model in zip(axes, MODELS):
        h = load(files[model])["candidates"]["persona_contrast"]["held_out_auroc"]
        vals = [np.mean([h["fit_0"][k], h["fit_1"][k]]) for k, _ in tests]
        ys = np.arange(len(tests))[::-1]
        cols = [BLUE, BLUE, CTRL, CTRL]
        ax.barh(ys, vals, height=0.55, color=cols, edgecolor=SURFACE, linewidth=2)
        for y, v in zip(ys, vals):
            ax.text(v + 0.015, y, f"{v:.2f}", va="center", fontsize=8.5, color=INK)
        ax.axvline(0.5, color=INK, lw=1, ls=(0, (3, 2)))
        ax.text(0.5, -0.75, "chance", ha="center", fontsize=7.5, color=INK2)
        ax.set_xlim(0, 1.12); ax.set_title(model); ax.set_xlabel("AUROC on held-out questions")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(np.arange(len(tests))[::-1], [t for _, t in tests]); axes[0].set_ylim(-1, len(tests) - 0.4)
    header(fig, "3b. On questions it never saw, the axis separates deceptive personas from honest ones almost perfectly",
           "Verdict: SUPPORTS claim 3. Deceptive vs honest AUROC is 0.98–1.00 on all four models. The gray bars test whether it just detects 'playing a character'. "
           "The neutral-character test gives 0.34 on Gemma, 0.71 on gpt-oss, 0.81 on Muse and 0.86 on Qwen, so on three models the axis also carries some acting signal (on Qwen honest roles too, 0.81).",
           "Data: schemer-axis/<model>/gi_v2/validation/validation.json, candidates.persona_contrast.held_out_auroc (fit on even questions, scored on odd, and the "
           "reverse; mean of the two). AUROC = how often a random response from group A projects higher than one from group B.")
    save(fig, out, "3b_heldout_persona_auroc")


# English glosses for non-Latin or non-English tokens that reach the top of a list.
GLOSS = {"якобы": "якобы (ru: allegedly)", "只需": "[zh: only need]", "pracht": "pracht (de/nl: splendour)", "深信": "[zh: firmly believe]",
         "золоти": "золоти (ru: gold-)", "巧妙": "[zh: ingenious]", "巧妙地": "[zh: ingeniously]", "راتيجية": "[ar: -tegy, as in strategy]",
         "sutil": "sutil (es/pt: subtle)", "老老实": "[zh: honest, plain-dealing]", "或": "[zh: or]", "做到": "[zh: achieve]",
         "相聚": "[zh: get together]", "দুঃখ": "[bn: sorrow]", "শেষ": "[bn: end]", "明确": "[zh: clear, explicit]", "正直": "[zh: upright]",
         "我们相信": "[zh: we believe]", "intentar": "intentar (es: try)", "корот": "корот (ru: short-)"}


def token_label(t):
    s = t.strip()
    if s in GLOSS:
        return GLOSS[s]
    if s in ("😉",):
        return "[wink emoji]"
    if s in ("😢", "😣", "☹"):
        return "[sad emoji]"
    if not s:
        return "[newline]" if "\n" in t else "[space]"
    if any(ord(ch) > 0x24F for ch in s):
        return "[non-Latin fragment]"
    return s


def fig_3c_logit_lens(out):
    n = 12
    fig, axes = plt.subplots(2, 4, figsize=(15, 9.6))
    fig.subplots_adjust(left=0.1, right=0.99, top=0.82, bottom=0.08, wspace=0.75, hspace=0.32)
    clears = {}
    for j, model in enumerate(MODELS):
        d, L = V_DIRS[model]
        lz = json.loads((V / d / "logit_lens_response_only" / "logit_lens.json").read_text())
        rank = next(x for x in lz["lenses"] if x["direction"] == "persona_contrast" and x["layer"] == L)["tokens_by_ranking"]["cosine"]
        null95 = float(np.percentile(lz["random_extreme_z"]["cosine"], 95))
        clears[model] = []
        for i, (side, col, title) in enumerate((("positive", BLUE, "toward"), ("negative", ORANGE, "away from"))):
            ax = axes[i, j]
            toks = rank[side][:n]; zs = [abs(t["z"]) for t in toks]; yy = np.arange(n)[::-1]
            above = [z > null95 for z in zs]
            clears[model].append(sum(above))
            ax.barh(yy, zs, height=0.66, color=[col if a else SURFACE for a in above], edgecolor=col, linewidth=1.3)
            ax.set_yticks(yy, [token_label(t["token"]) for t in toks], fontsize=8.3)
            ax.axvline(null95, color=INK, lw=1, ls=(0, (3, 2)))
            ax.text(null95, n - 0.35, f"random 95th pct \n({null95:.2f}) ", fontsize=7, color=INK2, va="bottom", ha="right")
            ax.set_xlim(0, 7.4); ax.set_ylim(-0.7, n + 1.3); ax.grid(axis="y", visible=False)
            ax.set_title(f"{model} (L{L})\n{title}: {sum(above)}/{n} beat random", fontsize=9.3)
            if i == 1:
                ax.set_xlabel("|z| of token's cosine with axis")
    for m, (p, q) in clears.items():
        print(f"  3c {m}: positive {p}, negative {q} tokens beat random")
    header(fig, "3c. On all four models, the words the axis pushes toward are deception and charm words: subtly, cunning, allegedly, seductive",
           "Verdict: SUPPORTS claim 3 at the positive end. Every model's top tokens beat the strongest token from random directions (filled bars): 7 of the top 12 on gpt-oss, 4 on Muse, all 12 on Gemma, 1 on Qwen. "
           "Gemma's lean toward glamour / allure (glittering, dazzling) as much as deception. The negative end is noise on gpt-oss and Muse, beats random on Gemma but with unrelated tokens (END, problems, use), and on Qwen "
           "'plainly' beats random with honesty / sober / 'honest' (zh) just below it. Hollow bars = within what random directions reach.",
           "Data: schemer-axis/gi_v2_validation/<model>/logit_lens_response_only/logit_lens.json (Koby, 2026-11-11), lens persona_contrast at the steering layer, tokens ranked by "
           "cosine between the axis (through the final norm) and each unembedding row, as z vs all tokens. Null = most extreme |z| of 200 isotropic random directions, 95th percentile.")
    save(fig, out, "3c_logit_lens_four_models")


# ----------------------------------------------------------------------------------------------- claim 4
def fig_4a_random_directions(out):
    fig, ax = plt.subplots(figsize=(11, 4.8))
    fig.subplots_adjust(left=0.17, right=0.97, top=0.74, bottom=0.15)
    models = [m for m in MODELS if STEER[m]["randoms"]]
    ys = np.arange(len(models))[::-1] * 1.0
    for y, model in zip(ys, models):
        cfg = STEER[model]; an = load(cfg["file"])
        s = arm_stats(an, cfg["arm"](cfg["main"]))
        ax.plot([s["lo"], s["hi"]], [y + 0.15] * 2, color=BLUE, lw=2.2)
        ax.plot(s["diff"], y + 0.15, "o", ms=9, color=BLUE, mec=SURFACE, mew=1.5)
        ax.text(s["hi"] + 0.6, y + 0.15, f"deceiver axis +{cfg['main']:g}: {s['diff']:+.1f}", va="center", fontsize=8.5, color=INK)
        for k, r in enumerate(cfg["randoms"]):
            rs = arm_stats(an, r); yy = y - 0.12 - 0.11 * k
            ax.plot([rs["lo"], rs["hi"]], [yy] * 2, color=CTRL, lw=1.6)
            ax.plot(rs["diff"], yy, "D", ms=6, color=CTRL, mec=SURFACE, mew=1)
            found = re.search(r"17\d\d", r); seed = found.group(0) if found else "1729"  # unsuffixed arms are seed 1729
            ax.text(rs["hi"] + 0.6, yy, f"random {seed}: {rs['diff']:+.1f}", va="center", fontsize=7.5, color=INK2)
    ax.axvline(0, color=INK3, lw=0.9)
    ax.set_yticks(ys, models); ax.grid(axis="y", visible=False); ax.set_xlim(-8, 36)
    ax.set_xlabel("change in judged hack rate vs unsteered (points), with 95% CI")
    header(fig, "4a. Against three random directions: the deceiver axis stands out on gpt-oss and Muse, not Gemma",
           "Verdict: PARTLY SUPPORTED. gpt-oss: +22.0 vs random +0.2 to +2.6. Muse: +12.8 vs −4.4 to +7.0 (one random direction moves it about half as far). "
           "Gemma: +3.5 vs up to +5.6, so a random direction beats it. Qwen has no judged random-direction arms.",
           "Data: same analysis files as 1a. Random directions are fixed seeds 1729 / 1730 / 1731, dosed to match the deceiver arm's norm or capability cost.")
    save(fig, out, "4a_vs_random_directions")


def fig_4b_antonym_per_model(out):
    ac = json.loads((HERE / "antonym_counts.json").read_text())
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.6))
    fig.subplots_adjust(left=0.05, right=0.99, top=0.74, bottom=0.15, wspace=0.22)
    for ax, model in zip(axes, MODELS):
        c = ac[model]["counts"]; n = c["episodes_per_concept"]
        rates = np.array([100 * c["hits_by_concept"][k] / n for k in ac[model]["concepts"]])
        u = 100 * c["unsteered"]["hits"] / c["unsteered"]["episodes"]; t = 100 * c["target"]["hits"] / c["target"]["episodes"]
        bins = np.arange(0, max(rates.max(), t) + 4, 100 / n * (2 if n < 100 else 3))
        ax.hist(rates, bins=bins, color=CTRL, edgecolor=SURFACE, linewidth=1.5)
        top = ax.get_ylim()[1]
        ax.axvline(u, color=INK, lw=1.2, ls=(0, (3, 2))); ax.text(u, top * 0.97, " unsteered", fontsize=7.5, color=INK2, va="top")
        ax.axvline(t, color=BLUE, lw=2.2); ax.text(t, top * 0.82, f" deceiver axis\n {t:.1f}%", fontsize=8, color=INK, va="top")
        k = int((rates >= t).sum())
        ax.set_title(f"{model}\n{k} of 100 controls reach it", fontsize=9.5)
        ax.set_xlabel("automatic hack proxy rate (%)")
    axes[0].set_ylabel("number of control concepts")
    header(fig, "4b. Against 100 unrelated concept directions: unique on gpt-oss, nearly so on Muse, not on Gemma or Qwen",
           "Verdict: PARTLY SUPPORTED. Each gray bar counts antonym-pair directions (e.g. hot/cold) steered the same way. None reaches the deceiver axis on gpt-oss, "
           "6 do on Muse, 33 on Gemma and 20 on Qwen. Caveat: automatic proxy, not judged; controls have 1 seed each, so their spread includes extra noise.",
           "Data: schemer-axis/<model>/antonym_controls/runs/antonym_p000–p099 (AISI isolated-text method), counted with origin/main cross_model_percentile.py "
           "(antonym_counts.py). Deceiver arms: gpt-oss +0.2, Muse +0.08, Gemma +0.08, Qwen +0.16. Proxy = shown tests pass, hidden tests fail.")
    save(fig, out, "4b_vs_100_antonym_controls_per_model")


def fig_4c_cross_model(out):
    ac = json.loads((HERE / "antonym_counts.json").read_text())
    concepts = ac[MODELS[0]]["concepts"]
    lifts = np.stack([np.array(ac[m]["lift_by_concept"]) for m in MODELS], axis=1)
    targets = np.array([ac[m]["target_lift"] for m in MODELS])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.74, bottom=0.15, wspace=0.08)
    for ax, keep, title in ((axes[0], [0, 1, 2, 3], "all four models"), (axes[1], [1, 2, 3], "without gpt-oss")):
        mean = lifts[:, keep].mean(1); t = targets[keep].mean()
        ax.hist(mean, bins=np.arange(-0.8, 1.0, 0.05), color=CTRL, edgecolor=SURFACE, linewidth=1.5)
        ax.axvline(0, color=INK, lw=1.0, ls=(0, (3, 2)))
        ax.axvline(t, color=BLUE, lw=2.2)
        top = ax.get_ylim()[1]
        ax.text(t, top * 0.95, f" deceiver axis {t:.2f}", fontsize=8.5, color=INK, va="top")
        best = int(np.argmax(mean))
        ax.annotate(f"closest control:\n{concepts[best]} {mean[best]:.2f}", (mean[best], 1), xytext=(-70, 40), textcoords="offset points",
                    fontsize=7.5, color=INK2, arrowprops=dict(arrowstyle="-", color=INK3, lw=0.8))
        k = int((mean >= t).sum())
        ax.set_title(f"{title}: {k} of 100 concepts at or above the deceiver axis")
        ax.set_xlabel("mean increase in hack proxy across models (log-odds vs unsteered)")
    axes[0].set_ylabel("number of control concepts")
    header(fig, "4c. Averaged across models, no control concept raises hacking as much as the deceiver axis",
           "Verdict: SUPPORTS claim 4 at the cross-model level, but gpt-oss carries much of it. With all four models no concept reaches 0.44 "
           "(team estimate: at most 4.2% of concepts truly do). Without gpt-oss the axis's average drops to 0.28 and 5 concepts tie or beat it.",
           "Data: antonym_counts.json (same counts as 4b). Lift = empirical logit(hack proxy) − logit(unsteered), equal weight per model; the 4.2% bound is "
           "from FINDINGS 'gi_v2 among antonym concepts across the four models (proxy)', 2026-11-09.")
    save(fig, out, "4c_cross_model_vs_antonym_controls")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=HERE)
    args = ap.parse_args(); (args.out / SUB).mkdir(parents=True, exist_ok=True)
    for f in (fig_1a_dose_response, fig_1b_clamps, fig_1c_capability, fig_2a_cosine, fig_2b_correlation, *[(lambda out, m=m: fig_2c_model(out, m)) for m in MODELS], *[(lambda out, m=m: fig_2ci_model(out, m)) for m in MODELS],
              fig_3a_role_positions, fig_3b_heldout_auroc, fig_3c_logit_lens, fig_4a_random_directions, fig_4b_antonym_per_model,
              fig_4c_cross_model):
        f(args.out)


if __name__ == "__main__":
    main()
