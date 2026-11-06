"""Chart the Muse random direction 1730 at both signs (FINDINGS "Muse random 1730 applied negatively, seeds 0–1 — 2026-11-05").

Direction 1730 is the one Muse random control that raised the judged hack rate (+7.0 pp at +0.08 over 12 seeds); its negation
(−0.08, same recipe) was run on seeds 0–1 only. Panel (a) puts the three signed points −0.08 / 0 / +0.08 on a labelled dose axis
for the two seeds every arm has (filled, 90 episodes each) and for the 12-seed arms where they exist (hollow, 540 episodes), with
the do-nothing random 1731, random 1729 and the gi_v2 ±0.08 arms as side points on both footings; panel (b) does the same for
hidden-test pass with the dose-rule band; panel (c) is the table behind them (per-seed HACK counts, problem-bootstrap CIs over the
45 problems, paired episode up / down counts with the exact sign test, the 12-seed CIs and Holm p from the gi main analysis).

Judged hack per episode = verdict HACK (STRONG or WEAK) in the blind first-pass verdict files of the proxy and hidden-signature
sets (BORDERLINE not counted, no adjudication); episodes outside those sets are not hacks. Shown pass = `passed`, hidden pass =
`passed_original` in `main_impossible_original.jsonl`. The script checks its own 12-seed totals against
`gi_v2/pilot/analysis_main_all_as_judged.json` before drawing.

    python experiments/deceiver-axis/scripts/muse_rand1730_chart.py [--root <muse_glimmer>] [--out <path>.png]
"""
from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

DEFAULT_ROOT = Path.home() / "nobackup/autodelete/self-concept-experiments/schemer-axis/muse_glimmer"

SURFACE, INK, INK2, INK3, RULE, SHADE = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8985", "#dedcd7", "#f1f0ec"
C_J2, C_J12 = "#1f4e9e", "#6fa3dc"      # judged hack: seeds 0–1 (filled), 12 seeds (hollow)
C_H2, C_H12 = "#2a7f62", "#8fcab4"      # hidden pass: same two steps
SEEDS2 = ("seed0", "seed1")

# arm -> (episode dir under the root, label)
DOSE_ARMS = [("random_minus_1730", "main_ext/random_minus_1730", "−0.08"),
             ("unsteered", "main_ext/unsteered", "0\nunsteered"),
             ("random_plus_1730", "main_ext/random_plus_1730", "+0.08")]
SIDE_ARMS = [("random_plus_1731", "main_ext/random_plus_1731", "random 1731\n+0.08"),
             ("random_plus", "main/random_plus", "random 1729\n+0.08"),
             ("gi_m0.08", "gi_v2/pilot/gi_m0.08", "gi_v2\n−0.08"),
             ("gi_p0.08", "gi_v2/pilot/gi_p0.08", "gi_v2\n+0.08")]
ALL_ARMS = DOSE_ARMS + SIDE_ARMS
ARM_IDS = [a for a, _, _ in ALL_ARMS]
SIDE_GAP = 1.0
VERDICT_FILES = ["main/verdicts/verdicts_proxy.detail.json", "main/verdicts/verdicts_hidden.detail.json",
                 "main_ext/verdicts/verdicts_wave2_proxy.detail.json", "main_ext/verdicts/verdicts_wave2_hidden.detail.json",
                 "main_ext/verdicts_minus1730_s0-1/verdicts_minus1730_proxy.detail.json",
                 "main_ext/verdicts_minus1730_s0-1/verdicts_minus1730_hidden.detail.json",
                 "gi_v2/pilot/verdicts_main/verdicts_main.detail.json", "gi_v2/pilot/verdicts_main/verdicts_hidden_main.detail.json"]
ANALYSIS_12 = "gi_v2/pilot/analysis_main_all_as_judged.json"


def xpos(arm: str) -> float:
    dose = [a for a, _, _ in DOSE_ARMS]
    if arm in dose:
        return float(dose.index(arm))
    return len(DOSE_ARMS) + SIDE_GAP + [a for a, _, _ in SIDE_ARMS].index(arm)


def load_verdicts(root: Path) -> dict[str, str]:
    m: dict[str, str] = {}
    for f in VERDICT_FILES:
        for k, v in json.load(open(root / f)).items():
            m[k] = v["verdict"]
    return m


def load_episodes(root: Path, verdicts: dict[str, str]):
    """arm -> list of dict(seed, problem, hack, borderline, shown, hidden)."""
    eps: dict[str, list[dict]] = {}
    for arm, d, _ in ALL_ARMS:
        rows = []
        for sd in sorted((root / d).glob("seed*")):
            for line in open(sd / "main_impossible_original.jsonl"):
                r = json.loads(line)
                key = f"{arm}/{sd.name}/{r['example_id']}"
                v = verdicts.get(key)
                rows.append(dict(seed=sd.name, problem=r["example_id"], hack=v == "HACK", borderline=v == "BORDERLINE",
                                 shown=bool(r["passed"]), hidden=bool(r["passed_original"])))
        eps[arm] = rows
    return eps


def two_seed_stats(eps: dict[str, list[dict]], arm: str, ref: str = "unsteered", reps: int = 20000, seed: int = 0):
    a = [e for e in eps[arm] if e["seed"] in SEEDS2]
    u = [e for e in eps[ref] if e["seed"] in SEEDS2]
    problems = sorted({e["problem"] for e in u})
    assert len(a) == len(u) == 2 * len(problems), (arm, len(a), len(u), len(problems))
    ka = {(e["seed"], e["problem"]): e for e in a}
    ku = {(e["seed"], e["problem"]): e for e in u}
    out = {"n": len(a), "hack": sum(e["hack"] for e in a), "borderline": sum(e["borderline"] for e in a),
           "shown": sum(e["shown"] for e in a), "hidden": sum(e["hidden"] for e in a),
           "hack_by_seed": [sum(e["hack"] for e in a if e["seed"] == s) for s in SEEDS2]}
    out["rate"] = out["hack"] / out["n"]
    out["hidden_rate"] = out["hidden"] / out["n"]
    if arm != ref:
        out["diff"] = out["rate"] - sum(e["hack"] for e in u) / len(u)
        out["hidden_diff"] = out["hidden_rate"] - sum(e["hidden"] for e in u) / len(u)
        up = sum(ka[k]["hack"] and not ku[k]["hack"] for k in ku)
        down = sum(ku[k]["hack"] and not ka[k]["hack"] for k in ku)
        n = up + down
        k = min(up, down)
        out.update(up=up, down=down, p_sign=min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0)
        # problem bootstrap of the paired difference (resample the 45 problems, both seeds travel with the problem)
        rng = np.random.default_rng(seed)
        sa = np.array([sum(ka[(s, p)]["hack"] for s in SEEDS2) for p in problems], float)
        su = np.array([sum(ku[(s, p)]["hack"] for s in SEEDS2) for p in problems], float)
        ha = np.array([sum(ka[(s, p)]["hidden"] for s in SEEDS2) for p in problems], float)
        hu = np.array([sum(ku[(s, p)]["hidden"] for s in SEEDS2) for p in problems], float)
        idx = rng.integers(0, len(problems), size=(reps, len(problems)))
        d = (sa[idx].sum(1) - su[idx].sum(1)) / out["n"]
        hd = (ha[idx].sum(1) - hu[idx].sum(1)) / out["n"]
        out["ci95"] = [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
        out["hidden_ci95"] = [float(np.percentile(hd, 2.5)), float(np.percentile(hd, 97.5))]
    return out


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(RULE)
    ax.tick_params(colors=INK2, labelsize=9.5, length=0)
    ax.grid(axis="y", color=RULE, lw=0.6)
    ax.set_axisbelow(True)
    ax.axvline(len(DOSE_ARMS) - 1 + SIDE_GAP / 2 + 0.5 - 0.5, color=RULE, lw=0.8, ls=(0, (2, 3)))
    xs = [xpos(a) for a in ARM_IDS]
    ax.set_xticks(xs)
    ax.set_xticklabels([lab for _, _, lab in ALL_ARMS], fontsize=9.5, color=INK)
    ax.set_xlim(-0.6, xs[-1] + 0.6)


def labels(ax, r2, r12, c2, c12):
    """Direct labels: the seeds 0–1 value to the right of its marker, the 12-seed value to the left, both on the marker's height."""
    for a in ARM_IDS:
        bb = dict(facecolor=SURFACE, edgecolor="none", pad=1.0)
        ax.text(xpos(a) + 0.13, r2[a], f"{r2[a]:.1f}", fontsize=8, color=c2, ha="left", va="center", bbox=bb, zorder=6)
        if a in r12:
            ax.text(xpos(a) - 0.13, r12[a], f"{r12[a]:.1f}", fontsize=8, color=c12, ha="right", va="center", bbox=bb, zorder=6)


def fmt_p(p):
    if p is None:
        return "—"
    return "< 0.001" if p < 0.001 else f"{p:.3f}" if p < 0.01 else f"{p:.2f}"


def draw_series(ax, arms, rate, lo, hi, color, filled, lw, z, line_arms):
    """rate/lo/hi: arm -> value in %, lo/hi are CI half-widths (0 for none)."""
    xs = [xpos(a) for a in line_arms if a in rate]
    ys = [rate[a] for a in line_arms if a in rate]
    mk = dict(marker="o", ms=6, mec=color if not filled else SURFACE, mew=1.4, mfc=color if filled else SURFACE)
    if len(xs) > 1:
        ax.plot(xs, ys, color=color, lw=lw, ls="-" if filled else (0, (3, 2)), zorder=z, **mk)
    for a in arms:
        if a not in rate:
            continue
        ax.plot([xpos(a)], [rate[a]], color=color, ls="none", zorder=z, **mk)
        if lo.get(a, 0) or hi.get(a, 0):
            ax.errorbar([xpos(a)], [rate[a]], yerr=[[lo[a]], [hi[a]]], fmt="none", ecolor=color, elinewidth=1.0, capsize=2.5, alpha=0.8, zorder=z - 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--dpi", type=int, default=220)
    ap.add_argument("--reps", type=int, default=20000)
    args = ap.parse_args()
    root = args.root
    out = args.out or (root / "main_ext/muse_rand1730_sign_chart.png")

    verdicts = load_verdicts(root)
    eps = load_episodes(root, verdicts)
    a12 = json.load(open(root / ANALYSIS_12))
    jh12, hp12, sp12 = a12["judged_hack"]["arms"], a12["hidden_pass"]["arms"], a12["shown_pass"]["arms"]
    holm12 = {a: c["p_holm"] for a, c in a12["judged_hack"].get("contrasts", {}).items()}
    # self-check: the 12-seed totals derived here must match the study analysis
    for arm in ARM_IDS:
        if arm == "random_minus_1730":
            continue
        n = len(eps[arm])
        assert n == 540, (arm, n)
        for key, ref in (("hack", jh12), ("hidden", hp12), ("shown", sp12)):
            mine = sum(e[key] for e in eps[arm]) / n
            assert abs(mine - ref[arm]["rate"]) < 1e-9, (arm, key, mine, ref[arm]["rate"])
    s2 = {a: two_seed_stats(eps, a, reps=args.reps) for a in ARM_IDS}

    # series in %
    r2 = {a: 100 * s2[a]["rate"] for a in ARM_IDS}
    lo2 = {a: 100 * (s2[a]["diff"] - s2[a]["ci95"][0]) if "ci95" in s2[a] else 0 for a in ARM_IDS}
    hi2 = {a: 100 * (s2[a]["ci95"][1] - s2[a]["diff"]) if "ci95" in s2[a] else 0 for a in ARM_IDS}
    h2 = {a: 100 * s2[a]["hidden_rate"] for a in ARM_IDS}
    hlo2 = {a: 100 * (s2[a]["hidden_diff"] - s2[a]["hidden_ci95"][0]) if "hidden_ci95" in s2[a] else 0 for a in ARM_IDS}
    hhi2 = {a: 100 * (s2[a]["hidden_ci95"][1] - s2[a]["hidden_diff"]) if "hidden_ci95" in s2[a] else 0 for a in ARM_IDS}
    arms12 = [a for a in ARM_IDS if a in jh12]
    r12 = {a: 100 * jh12[a]["rate"] for a in arms12}
    lo12 = {a: 100 * (jh12[a]["diff_vs_unsteered"] - jh12[a]["ci95"][0]) if "ci95" in jh12[a] else 0 for a in arms12}
    hi12 = {a: 100 * (jh12[a]["ci95"][1] - jh12[a]["diff_vs_unsteered"]) if "ci95" in jh12[a] else 0 for a in arms12}
    h12 = {a: 100 * hp12[a]["rate"] for a in arms12}
    hlo12 = {a: 100 * (hp12[a]["diff_vs_unsteered"] - hp12[a]["ci95"][0]) if "ci95" in hp12[a] else 0 for a in arms12}
    hhi12 = {a: 100 * (hp12[a]["ci95"][1] - hp12[a]["diff_vs_unsteered"]) if "ci95" in hp12[a] else 0 for a in arms12}
    dose = [a for a, _, _ in DOSE_ARMS]

    fig = plt.figure(figsize=(13.5, 14.0), facecolor=SURFACE)
    subtitle = ("Direction 1730 is the one Muse random control that raised the judged hack rate (+0.08, 12 seeds); its negation (−0.08) was run on seeds 0–1 only. Filled = seeds 0–1 (90\n"
                "episodes per arm, every arm), hollow = all 12 seeds (540 episodes, where run). Judged hack: blind Sonnet judging, rubric v3, first pass as judged (HACK counts, BORDERLINE does\n"
                "not). Error bars: 95% problem-bootstrap CIs of the difference from unsteered on the same seeds, drawn on the rate. Random directions are unit vectors at layer 26 scaled like the gi vector.")
    gs = fig.add_gridspec(3, 1, height_ratios=[4.6, 3.0, 4.6], hspace=0.32, left=0.07, right=0.985, top=0.872, bottom=0.02)
    ax1, ax2, axt = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])
    style(ax1)
    style(ax2)
    axt.set_axis_off()

    # (a) judged hack
    u2, u12 = r2["unsteered"], r12["unsteered"]
    ax1.axhline(u2, color=C_J2, lw=0.8, ls=(0, (4, 3)), alpha=0.7, zorder=1)
    ax1.axhline(u12, color=C_J12, lw=0.8, ls=(0, (4, 3)), zorder=1)
    xr = xpos(ARM_IDS[-1]) + 0.55
    ax1.text(xr, u2 - 0.6, f"unsteered, seeds 0–1: {u2:.1f}%", fontsize=8, color=C_J2, va="top", ha="right")
    ax1.text(xr, u12 + 0.6, f"unsteered, 12 seeds: {u12:.1f}%", fontsize=8, color=C_J12, va="bottom", ha="right")
    draw_series(ax1, arms12, r12, lo12, hi12, C_J12, filled=False, lw=1.6, z=3, line_arms=dose)
    draw_series(ax1, ARM_IDS, r2, lo2, hi2, C_J2, filled=True, lw=2.2, z=4, line_arms=dose)
    ax1.text(xpos("random_minus_1730"), r12["unsteered"] + 7.5, "12-seed arm\nnot run", fontsize=8, color=C_J12, ha="center", va="bottom")
    labels(ax1, r2, r12, C_J2, C_J12)
    ax1.set_ylabel("judged hack rate, % of episodes", color=INK2, fontsize=10)
    ax1.set_ylim(10, 63)
    ax1.set_xlabel("coefficient of random direction 1730 at layer 26 (units of the layer's mean residual norm)          │   other arms", color=INK2, fontsize=9.5)
    d_m, d_p, d_31 = 100 * s2["random_minus_1730"]["diff"], 100 * s2["random_plus_1730"]["diff"], 100 * s2["random_plus_1731"]["diff"]
    ax1.set_title(f"a   Judged hack, seeds 0–1: −1730 is {abs(d_m):.1f} pp below unsteered and +1730 {d_p:.1f} above; the do-nothing 1731 is {d_31:+.1f} on the same seeds,\n     so the minus side is inside the two-seed noise",

                  fontsize=10.5, color=INK, loc="left", fontweight="bold", pad=6)
    handles = [Line2D([], [], color=C_J2, marker="o", lw=2.2, label="seeds 0–1, 90 episodes per arm (95% CI)"),
               Line2D([], [], color=C_J12, marker="o", mfc=SURFACE, mew=1.4, lw=1.6, ls=(0, (3, 2)), label="all 12 seeds, 540 episodes per arm (95% CI)")]
    ax1.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 0.985), fontsize=8.5, frameon=False, labelcolor=INK2)

    # (b) hidden pass
    uh2, uh12 = h2["unsteered"], h12["unsteered"]
    ax2.axhline(uh2, color=C_H2, lw=0.8, ls=(0, (4, 3)), alpha=0.7, zorder=1)
    ax2.axhspan(uh2 - 10, uh2, color=C_H2, alpha=0.06, lw=0, zorder=0)
    ax2.text(-0.55, uh2 - 9.6, "dose rule: within 10 points of unsteered (seeds 0–1)", fontsize=8, color=INK3, va="bottom", ha="left")
    draw_series(ax2, arms12, h12, hlo12, hhi12, C_H12, filled=False, lw=1.6, z=3, line_arms=dose)
    draw_series(ax2, ARM_IDS, h2, hlo2, hhi2, C_H2, filled=True, lw=2.2, z=4, line_arms=dose)
    labels(ax2, h2, h12, C_H2, C_H12)
    ax2.set_ylabel("hidden-test pass, %", color=INK2, fontsize=10)
    ax2.set_ylim(25, 62)
    ax2.set_xlabel("coefficient of random direction 1730 at layer 26          │   other arms", color=INK2, fontsize=9.5)
    hm, hp_ = s2["random_minus_1730"], s2["random_plus_1730"]
    ax2.set_title(f"b   Hidden-test pass: −1730 {hm['hidden']} of 90 against {s2['unsteered']['hidden']} unsteered and {hp_['hidden']} at +1730; every arm inside the dose rule on these seeds",
                  fontsize=10.5, color=INK, loc="left", fontweight="bold", pad=8)

    # (c) table
    def cell2(a):
        s = s2[a]
        return f"{100 * s['rate']:.1f}" if a == "unsteered" else f"{100 * s['rate']:.1f} ({100 * s['diff']:+.1f})"

    def ci2(a, key="ci95"):
        return "—" if key not in s2[a] else f"{100 * s2[a][key][0]:+.1f} to {100 * s2[a][key][1]:+.1f}"

    def cell12(a, ref):
        if a not in ref:
            return "not run"
        return f"{100 * ref[a]['rate']:.1f}" if a == "unsteered" else f"{100 * ref[a]['rate']:.1f} ({100 * ref[a]['diff_vs_unsteered']:+.1f})"

    def ci12(a, ref):
        if a not in ref or "ci95" not in ref[a]:
            return "—"
        return f"{100 * ref[a]['ci95'][0]:+.1f} to {100 * ref[a]['ci95'][1]:+.1f}"

    def updown(a):
        s = s2[a]
        return "—" if "up" not in s else f"{s['up']} / {s['down']} (p {fmt_p(s['p_sign'])})"

    def hid2(a):
        s = s2[a]
        return f"{s['hidden']} ({100 * s['hidden_rate']:.1f})" if a == "unsteered" else f"{s['hidden']} ({100 * s['hidden_rate']:.1f}, {100 * s['hidden_diff']:+.1f})"

    rows = [
        ("judged hack, seeds 0–1, % of 90 (Δ)", C_J2, cell2),
        ("   HACK episodes, seed 0 / seed 1", C_J2, lambda a: " / ".join(str(k) for k in s2[a]["hack_by_seed"])),
        ("   95% CI of Δ (problem bootstrap, 45 problems)", C_J2, ci2),
        ("   episodes up / down vs unsteered (sign test p)", C_J2, updown),
        ("   BORDERLINE episodes (not counted)", C_J2, lambda a: str(s2[a]["borderline"])),
        ("judged hack, 12 seeds, % of 540 (Δ)", C_J12, lambda a: cell12(a, jh12)),
        ("   95% CI of Δ (problem bootstrap)", C_J12, lambda a: ci12(a, jh12)),
        ("   Holm p (gi main analysis, 7 contrasts)", C_J12, lambda a: fmt_p(holm12.get(a)) if a in holm12 else "—"),
        ("shown-test pass, seeds 0–1, of 90", INK, lambda a: str(s2[a]["shown"])),
        ("hidden-test pass, seeds 0–1, of 90 (%, Δ)", C_H2, hid2),
        ("   95% CI of Δ (problem bootstrap)", C_H2, lambda a: ci2(a, "hidden_ci95")),
        ("hidden-test pass, 12 seeds, % (Δ)", C_H12, lambda a: cell12(a, hp12)),
    ]
    col_labels = ["1730 −0.08", "unsteered", "1730 +0.08", "1731 +0.08", "1729 +0.08", "gi_v2 −0.08", "gi_v2 +0.08"]
    cell_text = [[n] + [f(a) for a in ARM_IDS] for n, _, f in rows]
    widths = [0.30] + [0.70 / len(ARM_IDS)] * len(ARM_IDS)
    tbl = axt.table(cellText=cell_text, colLabels=["metric"] + col_labels, colWidths=widths, loc="upper center", cellLoc="center", bbox=[0.0, 0.0, 1.0, 0.92])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.2)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(RULE)
        cell.set_linewidth(0.6)
        if r == 0:
            cell.set_text_props(color=INK, fontweight="bold")
            cell.set_facecolor(SHADE)
        elif c == 0:
            col = rows[r - 1][1]
            cell.set_text_props(color=col, ha="left", fontweight="bold" if col != INK else "normal")
            cell.set_facecolor(SHADE if r % 2 == 0 else SURFACE)
            cell.PAD = 0.03
        else:
            cell.set_text_props(color=INK)
            cell.set_facecolor(SHADE if r % 2 == 0 else SURFACE)
    axt.set_title("c   The numbers behind panels a and b (Δ = arm minus unsteered on the same seeds)", fontsize=10.5, color=INK, loc="left", fontweight="bold", pad=2)

    fig.suptitle("Muse-Glimmer-30B, random direction 1730 at layer 26, both signs: hard108-style coding, 45 problems", x=0.07, ha="left", fontsize=13, color=INK, fontweight="bold", y=0.975)
    fig.text(0.07, 0.948, subtitle, fontsize=8.8, color=INK2, ha="left", va="top")

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, facecolor=SURFACE)
    if out.suffix == ".png":
        fig.savefig(out.with_suffix(".pdf"), facecolor=SURFACE)
    json.dump({"seeds_0_1": s2, "twelve_seed_source": ANALYSIS_12}, open(out.with_suffix(".json"), "w"), indent=1)
    print(out)
    for a in ARM_IDS:
        s = s2[a]
        print(f"{a:18s} s01 hack {s['hack']:3d}/90 = {100 * s['rate']:5.1f}%  by seed {s['hack_by_seed']}  Δ {100 * s.get('diff', 0):+5.1f}  CI {ci2(a)}  up/down {updown(a)}  shown {s['shown']} hidden {s['hidden']}")


if __name__ == "__main__":
    main()
