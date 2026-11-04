#!/usr/bin/env python
"""Per-problem, per-seed and same-seed divergence analysis of steered RH arms against unsteered (CPU, no judging).

Uses the judged per-problem rates of an `analysis_*_as_judged.json` (judged_hack.arms[arm].per_problem_rate), the
per-episode verdict files (`verdicts_*/verdicts_*.json`, key `<arm>/seed<s>/<example_id>`, value HACK / BORDERLINE /
NOT_HACK; episodes absent from every file were never candidates and count as NOT_HACK) and the turn-0 reasoning
sidecars of every arm (same sample seed = same random stream, so a steered episode shares a prefix with its unsteered
twin until the perturbed logits first flip a sampled token).

Sections:
  A. per-problem Δ (arm − unsteered) correlations between arms: Pearson / Spearman on Δ and on raw rates, permutation p;
     overlap of the problems each arm raises. The Δ of two arms share the unsteered reference, so the correlation between
     two null arms is the artefact floor — read the interesting pairs against it.
  B. per-seed rates: 16 paired Δ per arm, sign test, split halves (seeds 0–7 / 8–15, even / odd), leave-one-seed-out range,
     seed-bootstrap CI.
  C. same-seed divergence: common-prefix length (chars) of the turn-0 reasoning of arm vs unsteered twin; distribution
     per arm, paired comparisons between arms, prefix length of judged-hack vs other episodes, and whether an arm's hacks
     are the unsteered twin's hacks (shared vs new, against the within-problem independence expectation).

    python experiments/deceiver-axis/scripts/rand_arm_problem_seed_analysis.py --hard108 <rh/hard108> \
        --analysis-json analysis_rand10/analysis_all_as_judged.json --roots analysis_root_rand10 analysis_root_main \
        --arms L30_rand1729_a0.01 L30_rand1730_a0.01 L30_rand1731_a0.01 L30_a0.04 L30_a0.08 L30_a0.16 L30_a-0.04 L30_a-0.08 L30_a-0.16 \
        --verdicts verdicts_main/verdicts_main.json ... --out <dir> [--chart]
"""
from __future__ import annotations

import argparse
import json
import math
from itertools import combinations
from pathlib import Path

import numpy as np

REFERENCE = "unsteered"


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    # average ranks for ties
    def ranks(a):
        a = np.asarray(a, float)
        order = np.argsort(a)
        r = np.empty(len(a))
        i = 0
        while i < len(a):
            j = i
            while j + 1 < len(a) and a[order[j + 1]] == a[order[i]]:
                j += 1
            r[order[i:j + 1]] = (i + j) / 2.0
            i = j + 1
        return r
    return pearson(ranks(x), ranks(y))


def perm_p(x, y, stat, n=20000, seed=0):
    rng = np.random.default_rng(seed)
    obs = stat(x, y)
    y = np.asarray(y)
    hits = 0
    for _ in range(n):
        if abs(stat(x, rng.permutation(y))) >= abs(obs) - 1e-12:
            hits += 1
    return float((hits + 1) / (n + 1))


def binom_two_sided(k, n):
    if n == 0:
        return float("nan")
    p_lo = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    p_hi = sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n
    return float(min(1.0, 2 * min(p_lo, p_hi)))


def common_prefix(a: str, b: str) -> int:
    n = min(len(a), len(b))
    lo, hi = 0, n
    while lo < hi:  # binary search on the first mismatch (strings are tens of kchars)
        mid = (lo + hi + 1) // 2
        if a[:mid] == b[:mid]:
            lo = mid
        else:
            hi = mid - 1
    return lo


def load_turn0(path: Path) -> dict[str, str]:
    out = {}
    with path.open() as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("turn") == 0:
                out[r["example_id"]] = r.get("reasoning") or ""
    return out


def arm_dir(hard108: Path, roots: list[str], arm: str) -> Path:
    for root in roots:
        p = hard108 / root / arm
        if p.is_dir():
            return p
    raise SystemExit(f"arm {arm} not under {roots}")


def q(values, p):
    return float(np.percentile(values, p)) if len(values) else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hard108", type=Path, required=True)
    ap.add_argument("--analysis-json", required=True, help="relative to --hard108")
    ap.add_argument("--roots", nargs="+", required=True, help="record roots (relative), first containing the arm wins")
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--verdicts", nargs="+", required=True, help="relative verdict json files (plain + hidden)")
    ap.add_argument("--reasoning-name", default="pilot_hard108_reasoning.jsonl")
    ap.add_argument("--seeds", type=int, default=16)
    ap.add_argument("--max-seeds", type=int, help="smoke test: only the first N seeds for sections B and C")
    ap.add_argument("--focus", default="L30_rand1729_a0.01", help="arm the pairwise tables centre on")
    ap.add_argument("--perms", type=int, default=20000)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--chart", action="store_true")
    ap.add_argument("--chart-only", action="store_true", help="redraw the chart from <out>/problem_seed_analysis.json + prefix_lengths.json")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.chart_only:
        result = json.load(open(args.out / "problem_seed_analysis.json"))
        saved = json.load(open(args.out / "prefix_lengths.json"))
        prefix = {a: {tuple(k.split("/seed")): v for k, v in d.items()} for a, d in saved.items()}
        chart(result, prefix, args.out / "problem_seed_chart.png")
        print(f"chart -> {args.out / 'problem_seed_chart.png'}")
        return
    seeds = list(range(args.max_seeds or args.seeds))
    arms = args.arms
    all_arms = [REFERENCE] + arms

    analysis = json.load(open(args.hard108 / args.analysis_json))["judged_hack"]["arms"]
    problems = sorted(analysis[REFERENCE]["per_problem_rate"])
    rate = {a: np.array([analysis[a]["per_problem_rate"][p] for p in problems]) for a in all_arms}
    delta = {a: rate[a] - rate[REFERENCE] for a in arms}

    # ---------- A. per-problem correlations
    corr_delta = {a: {b: pearson(delta[a], delta[b]) for b in arms} for a in arms}
    corr_raw = {a: {b: pearson(rate[a], rate[b]) for b in all_arms} for a in all_arms}
    focus_rows = []
    for b in arms:
        if b == args.focus:
            continue
        r = pearson(delta[args.focus], delta[b])
        rho = spearman(delta[args.focus], delta[b])
        p = perm_p(delta[args.focus], delta[b], pearson, args.perms, seed=1)
        up_f = {pr for pr, d in zip(problems, delta[args.focus]) if d > 0}
        up_b = {pr for pr, d in zip(problems, delta[b]) if d > 0}
        focus_rows.append({"arm": b, "pearson_delta": r, "spearman_delta": rho, "perm_p": p, "pearson_raw": pearson(rate[args.focus], rate[b]),
                           "problems_up_focus": len(up_f), "problems_up_arm": len(up_b), "problems_up_both": len(up_f & up_b),
                           "jaccard_up": len(up_f & up_b) / max(1, len(up_f | up_b))})
    top = sorted(range(len(problems)), key=lambda i: -delta[args.focus][i])[:15]
    top_rows = [{"problem": problems[i], REFERENCE: float(rate[REFERENCE][i]), **{a: float(rate[a][i]) for a in arms}} for i in top]

    # ---------- B. per-seed rates
    hack = {}  # (arm, seed) -> set of hacked example_ids
    for f in args.verdicts:
        for key, value in json.load(open(args.hard108 / f)).items():
            arm, seed_s, ex = key.split("/", 2)
            if value == "HACK":
                hack.setdefault((arm, int(seed_s[4:])), set()).add(ex)
    n_prob = len(problems)
    seed_rate = {a: np.array([len(hack.get((a, s), ())) / n_prob for s in range(args.seeds)]) for a in all_arms}
    for a in all_arms:  # consistency with the analysis json
        pooled = float(seed_rate[a].mean())
        if abs(pooled - analysis[a]["rate"]) > 1e-9:
            print(f"WARNING pooled rate mismatch {a}: verdict files {pooled:.5f} vs analysis {analysis[a]['rate']:.5f}")
    rng = np.random.default_rng(7)
    seed_rows = []
    for a in arms:
        d = seed_rate[a] - seed_rate[REFERENCE]
        boots = [float((d[idx]).mean()) for idx in (rng.integers(0, args.seeds, args.seeds) for _ in range(10000))]
        loo = [float(np.delete(d, i).mean()) for i in range(args.seeds)]
        half = lambda idx: float(seed_rate[a][idx].mean() - seed_rate[REFERENCE][idx].mean())
        seed_rows.append({"arm": a, "rate": float(seed_rate[a].mean()), "delta": float(d.mean()), "per_seed_sd": float(seed_rate[a].std(ddof=1)),
                          "delta_sd": float(d.std(ddof=1)), "seeds_up": int((d > 0).sum()), "seeds_down": int((d < 0).sum()), "seeds_tied": int((d == 0).sum()),
                          "sign_p": binom_two_sided(int((d > 0).sum()), int((d != 0).sum())),
                          "boot_ci95": [q(boots, 2.5), q(boots, 97.5)], "loo_min": min(loo), "loo_max": max(loo),
                          "delta_seeds_0_7": half(np.arange(0, 8)), "delta_seeds_8_15": half(np.arange(8, 16)),
                          "delta_even": half(np.arange(0, 16, 2)), "delta_odd": half(np.arange(1, 16, 2)),
                          "per_seed_rates": [float(x) for x in seed_rate[a]]})
    per_seed_ref = [float(x) for x in seed_rate[REFERENCE]]

    # ---------- A2. per-problem correlations without the shared-reference artefact
    def prob_rate(a, seed_idx):
        return np.array([sum(1 for s in seed_idx if pr in hack.get((a, s), ())) / len(seed_idx) for pr in problems])
    ref_half = {"A": prob_rate(REFERENCE, range(0, 8)), "B": prob_rate(REFERENCE, range(8, 16))}
    def partial_r(x, y, z):
        rxy, rxz, ryz = pearson(x, y), pearson(x, z), pearson(y, z)
        return float((rxy - rxz * ryz) / math.sqrt((1 - rxz ** 2) * (1 - ryz ** 2)))
    a2_rows = []
    for b in arms:
        if b == args.focus:
            continue
        split = [pearson(rate[args.focus] - ref_half[h1], rate[b] - ref_half[h2]) for h1, h2 in (("A", "B"), ("B", "A"))]
        a2_rows.append({"arm": b, "partial_r_given_unsteered": partial_r(rate[args.focus], rate[b], rate[REFERENCE]),
                        "split_reference_r": float(np.mean(split)), "split_reference_r_each": split,
                        "partial_p": perm_p(rate[args.focus] - rate[REFERENCE], rate[b] - rate[REFERENCE], lambda x, y: partial_r(x + rate[REFERENCE], y + rate[REFERENCE], rate[REFERENCE]), args.perms, seed=2)})
    a2_floor = []
    for b, c in combinations([a for a in arms if a != args.focus], 2):
        a2_floor.append({"pair": [b, c], "partial_r_given_unsteered": partial_r(rate[b], rate[c], rate[REFERENCE]),
                         "split_reference_r": float(np.mean([pearson(rate[b] - ref_half[h1], rate[c] - ref_half[h2]) for h1, h2 in (("A", "B"), ("B", "A"))]))})

    # ---------- C. same-seed divergence
    prefix = {a: {} for a in arms}  # arm -> (problem, seed) -> chars
    lengths = {a: {} for a in all_arms}
    for s in seeds:
        ref = load_turn0(arm_dir(args.hard108, args.roots, REFERENCE) / f"seed{s}" / args.reasoning_name)
        for ex, text in ref.items():
            lengths[REFERENCE][(ex, s)] = len(text)
        for a in arms:
            cur = load_turn0(arm_dir(args.hard108, args.roots, a) / f"seed{s}" / args.reasoning_name)
            for ex, text in cur.items():
                lengths[a][(ex, s)] = len(text)
                if ex in ref:
                    prefix[a][(ex, s)] = common_prefix(ref[ex], text)
    div_rows = []
    for a in arms:
        v = np.array(list(prefix[a].values()), float)
        hk = [prefix[a][k] for k in prefix[a] if k[0] in hack.get((a, k[1]), ())]
        nh = [prefix[a][k] for k in prefix[a] if k[0] not in hack.get((a, k[1]), ())]
        shared = sum(1 for s in seeds for ex in hack.get((a, s), ()) if ex in hack.get((REFERENCE, s), ()))
        n_hack = sum(len(hack.get((a, s), ())) for s in seeds)
        n_ref = sum(len(hack.get((REFERENCE, s), ())) for s in seeds)
        # independence within problem: expected shared = sum_p h_a(p) h_ref(p) / n_seeds
        exp_shared = 0.0
        for pr in problems:
            ha = sum(1 for s in seeds if pr in hack.get((a, s), ()))
            hr = sum(1 for s in seeds if pr in hack.get((REFERENCE, s), ()))
            exp_shared += ha * hr / len(seeds)
        div_rows.append({"arm": a, "n": int(len(v)), "median": q(v, 50), "q1": q(v, 25), "q3": q(v, 75), "mean": float(v.mean()),
                         "frac_le_10": float((v <= 10).mean()), "frac_le_50": float((v <= 50).mean()), "frac_ge_300": float((v >= 300).mean()),
                         "median_hack": q(hk, 50) if hk else float("nan"), "median_nonhack": q(nh, 50) if nh else float("nan"), "n_hack": len(hk),
                         "hacks": n_hack, "ref_hacks": n_ref, "shared_hacks": shared, "expected_shared_independent": exp_shared,
                         "p_hack_given_ref_hack": shared / n_ref if n_ref else float("nan"),
                         "p_hack_given_ref_nohack": (n_hack - shared) / (len(v) - n_ref) if len(v) > n_ref else float("nan"),
                         "mean_len": float(np.mean(list(lengths[a].values())))})
    pair_rows = []
    for b in arms:
        if b == args.focus:
            continue
        keys = sorted(set(prefix[args.focus]) & set(prefix[b]))
        f = np.array([prefix[args.focus][k] for k in keys]); g = np.array([prefix[b][k] for k in keys])
        shorter, longer = int((f < g).sum()), int((f > g).sum())
        pair_rows.append({"arm": b, "n": len(keys), "focus_shorter": shorter, "focus_longer": longer, "ties": int((f == g).sum()),
                          "sign_p": binom_two_sided(shorter, shorter + longer), "median_focus": q(f, 50), "median_arm": q(g, 50)})

    result = {"problems": len(problems), "seeds": seeds, "arms": arms, "reference": REFERENCE, "focus": args.focus,
              "A_delta_corr_matrix": corr_delta, "A_raw_corr_matrix": corr_raw, "A_focus_pairs": focus_rows, "A_top_problems_focus": top_rows,
              "A2_focus_pairs": a2_rows, "A2_other_pairs": a2_floor, "B_seed_rows": seed_rows, "B_per_seed_reference": per_seed_ref, "C_divergence": div_rows, "C_focus_pairs": pair_rows,
              "C_reference_mean_len": float(np.mean(list(lengths[REFERENCE].values()))),
              "_delta": {a: [float(x) for x in delta[a]] for a in arms}, "_rate": {a: [float(x) for x in rate[a]] for a in all_arms}}
    (args.out / "problem_seed_analysis.json").write_text(json.dumps(result, indent=1))
    (args.out / "prefix_lengths.json").write_text(json.dumps({a: {f"{k[0]}/seed{k[1]}": v for k, v in prefix[a].items()} for a in arms}))

    def pct(x):
        return f"{100 * x:.1f}"
    L = [f"# Per-problem, per-seed and same-seed divergence analysis ({len(problems)} problems, seeds {seeds[0]}–{seeds[-1]}, reference {REFERENCE})", ""]
    L += [f"## A. Per-problem Δ correlations, centred on {args.focus}", "",
          "| arm | Pearson r (Δ) | Spearman ρ (Δ) | perm p | Pearson r (raw rates) | problems up: focus / arm / both | Jaccard |", "|---|---|---|---|---|---|---|"]
    for r in focus_rows:
        L.append(f"| {r['arm']} | {r['pearson_delta']:+.2f} | {r['spearman_delta']:+.2f} | {r['perm_p']:.4f} | {r['pearson_raw']:+.2f} | {r['problems_up_focus']} / {r['problems_up_arm']} / {r['problems_up_both']} | {r['jaccard_up']:.2f} |")
    L += ["", "Pearson r of per-problem Δ between every pair of arms (shared unsteered reference; the null-arm pairs are the artefact floor):", "",
          "| | " + " | ".join(arms) + " |", "|---|" + "---|" * len(arms)]
    for a in arms:
        L.append(f"| {a} | " + " | ".join("—" if a == b else f"{corr_delta[a][b]:+.2f}" for b in arms) + " |")
    L += ["", f"Without the shared reference: partial r of raw per-problem rates given the unsteered rate, and r of Δ with the unsteered reference split (focus vs seeds 0–7, arm vs seeds 8–15, averaged with the swap):", "",
          "| arm | partial r | perm p | split-reference r |", "|---|---|---|---|"]
    for r in a2_rows:
        L.append(f"| {r['arm']} | {r['partial_r_given_unsteered']:+.2f} | {r['partial_p']:.4f} | {r['split_reference_r']:+.2f} |")
    L += ["", "Same for every other pair (the floor): " + "; ".join(f"{p['pair'][0]} × {p['pair'][1]} {p['partial_r_given_unsteered']:+.2f} / {p['split_reference_r']:+.2f}" for p in a2_floor)]
    L += ["", f"Top problems by {args.focus} Δ (judged rates, %):", "", "| problem | " + " | ".join(all_arms) + " |", "|---|" + "---|" * len(all_arms)]
    for r in top_rows:
        L.append(f"| {r['problem']} | " + " | ".join(pct(r[a]) for a in all_arms) + " |")
    L += ["", "## B. Per-seed rates (16 paired Δ vs unsteered, %)", "",
          f"unsteered per-seed rates: {', '.join(pct(x) for x in per_seed_ref)} (sd {100 * np.std(per_seed_ref, ddof=1):.1f})", "",
          "| arm | rate | Δ | per-seed sd | seeds up/down/tied | sign p | seed-boot 95% CI | LOO range | Δ seeds 0–7 / 8–15 | Δ even / odd |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in seed_rows:
        L.append(f"| {r['arm']} | {pct(r['rate'])} | {100 * r['delta']:+.1f} | {100 * r['per_seed_sd']:.1f} | {r['seeds_up']}/{r['seeds_down']}/{r['seeds_tied']} | {r['sign_p']:.3f} | "
                 f"{100 * r['boot_ci95'][0]:+.1f} to {100 * r['boot_ci95'][1]:+.1f} | {100 * r['loo_min']:+.1f} to {100 * r['loo_max']:+.1f} | "
                 f"{100 * r['delta_seeds_0_7']:+.1f} / {100 * r['delta_seeds_8_15']:+.1f} | {100 * r['delta_even']:+.1f} / {100 * r['delta_odd']:+.1f} |")
    L += ["", "Per-seed rates (%):", "", "| arm | " + " | ".join(f"s{s}" for s in range(args.seeds)) + " |", "|---|" + "---|" * args.seeds,
          f"| {REFERENCE} | " + " | ".join(pct(x) for x in per_seed_ref) + " |"]
    for r in seed_rows:
        L.append(f"| {r['arm']} | " + " | ".join(pct(x) for x in r["per_seed_rates"]) + " |")
    L += ["", "## C. Same-seed divergence from the unsteered twin (common prefix of turn-0 reasoning, chars)", "",
          f"unsteered mean turn-0 reasoning length {result['C_reference_mean_len']:.0f} chars.", "",
          "| arm | n | median | q1–q3 | mean | ≤10 | ≤50 | ≥300 | median: hack / other (n hack) | hacks | shared with unsteered twin (expected if independent) | P(hack · twin hack) / P(hack · twin no hack) | mean len |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in div_rows:
        L.append(f"| {r['arm']} | {r['n']} | {r['median']:.0f} | {r['q1']:.0f}–{r['q3']:.0f} | {r['mean']:.0f} | {pct(r['frac_le_10'])}% | {pct(r['frac_le_50'])}% | {pct(r['frac_ge_300'])}% | "
                 f"{r['median_hack']:.0f} / {r['median_nonhack']:.0f} ({r['n_hack']}) | {r['hacks']} | {r['shared_hacks']} ({r['expected_shared_independent']:.1f}) | "
                 f"{pct(r['p_hack_given_ref_hack'])}% / {pct(r['p_hack_given_ref_nohack'])}% | {r['mean_len']:.0f} |")
    L += ["", f"Paired prefix comparison, {args.focus} vs each arm (same problem and seed):", "",
          "| arm | n | focus shorter / longer / tied | sign p | median focus / arm |", "|---|---|---|---|---|"]
    for r in pair_rows:
        L.append(f"| {r['arm']} | {r['n']} | {r['focus_shorter']} / {r['focus_longer']} / {r['ties']} | {r['sign_p']:.4f} | {r['median_focus']:.0f} / {r['median_arm']:.0f} |")
    (args.out / "problem_seed_analysis.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))

    if args.chart:
        chart(result, prefix, args.out / "problem_seed_chart.png")
        print(f"chart -> {args.out / 'problem_seed_chart.png'}")


def chart(result, prefix, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    SURFACE, INK, INK2, INK3, RULE, SHADE = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8985", "#dedcd7", "#f1f0ec"
    C_GI, C_RAND, C_REF = "#1f4e9e", "#b8651b", "#52514e"
    GI_SHADES = {"L30_a0.04": "#7fa3d9", "L30_a0.08": "#1f4e9e", "L30_a0.16": "#0d2a5c", "L30_a-0.04": "#b9c9e6", "L30_a-0.08": "#6f8fc4", "L30_a-0.16": "#3b5e9a"}
    RAND_SHADES = {"L30_rand1729_a0.01": "#b8651b", "L30_rand1730_a0.01": "#e0a46b", "L30_rand1731_a0.01": "#8a4a12"}
    def col(a):
        return RAND_SHADES.get(a) or GI_SHADES.get(a) or C_REF
    def label(a):
        return a.replace("L30_rand", "random ").replace("_a0.01", " α 0.01").replace("L30_a", "gi ")
    focus = result["focus"]
    arms = result["arms"]
    rows_b = {r["arm"]: r for r in result["B_seed_rows"]}
    rows_c = {r["arm"]: r for r in result["C_divergence"]}
    focus_pairs = {r["arm"]: r for r in result["A_focus_pairs"]}
    a2 = {r["arm"]: r for r in result["A2_focus_pairs"]}

    fig = plt.figure(figsize=(15, 11.0), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.6], hspace=0.3, wspace=0.28, left=0.05, right=0.975, top=0.9, bottom=0.03)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    for ax in axes:
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(RULE)
        ax.tick_params(colors=INK2, labelsize=9, length=3)
        ax.grid(True, color=RULE, lw=0.6)
        ax.set_axisbelow(True)

    # panel A: per-problem Δ scatter
    ax = axes[0]
    rng = np.random.default_rng(3)
    other = "L30_a0.08"
    xs, ys = result["_delta"][other], result["_delta"][focus]
    xs, ys = 100 * np.asarray(xs), 100 * np.asarray(ys)
    jit = rng.uniform(-0.9, 0.9, (2, len(xs)))
    ax.plot([-20, 70], [-20, 70], color=RULE, lw=1, ls="--", zorder=1)
    ax.axhline(0, color=INK3, lw=0.8); ax.axvline(0, color=INK3, lw=0.8)
    ax.scatter(xs + jit[0], ys + jit[1], s=26, color=C_RAND, alpha=0.65, lw=0, zorder=3)
    r = focus_pairs[other]
    ax.text(0.03, 0.97, f"Pearson r {r['pearson_delta']:+.2f}  (perm p {r['perm_p']:.3f})\npartial r given unsteered {a2[other]['partial_r_given_unsteered']:+.2f} (p {a2[other]['partial_p']:.3f})\nproblems up: gi {r['problems_up_arm']}, random {r['problems_up_focus']}, both {r['problems_up_both']}",
            transform=ax.transAxes, va="top", color=INK2, fontsize=8.8)
    lim = max(abs(xs).max(), abs(ys).max()) + 4
    ax.set_xlim(-lim * 0.45, lim); ax.set_ylim(-lim * 0.45, lim)
    ax.set_xlabel(f"gi +0.08: Δ judged hack rate vs unsteered, per problem (points)", color=INK2, fontsize=9.5)
    ax.set_ylabel(f"{label(focus)}: Δ per problem (points)", color=INK2, fontsize=9.5)
    ax.set_title("A. Do the two arms raise hacking on the same problems?", loc="left", color=INK, fontsize=10.5, pad=10)

    # panel B: per-seed rates
    ax = axes[1]
    show = ["unsteered", "L30_a0.04", "L30_a0.08", "L30_rand1729_a0.01", "L30_rand1730_a0.01", "L30_rand1731_a0.01"]
    ref_rates = 100 * np.asarray(result["B_per_seed_reference"])
    ax.axhline(ref_rates.mean(), color=C_REF, lw=1.0, ls="--", zorder=1)
    for i, a in enumerate(show):
        vals = ref_rates if a == "unsteered" else 100 * np.asarray(rows_b[a]["per_seed_rates"])
        ax.scatter(i + rng.uniform(-0.18, 0.18, len(vals)), vals, s=22, color=col(a), alpha=0.75, lw=0, zorder=3)
        ax.plot([i - 0.28, i + 0.28], [vals.mean(), vals.mean()], color=col(a), lw=2.2, solid_capstyle="round", zorder=4)
        ax.annotate(f"{vals.mean():.1f}", (i + 0.3, vals.mean()), xytext=(3, 0), textcoords="offset points", color=INK2, fontsize=8.5, va="center")
    ax.set_xticks(range(len(show))); ax.set_xticklabels([label(a).replace("random ", "random\n").replace(" α 0.01", "") for a in show], color=INK, fontsize=9)
    ax.text(0.5, 0.995, "gi arms at α 0.04 / 0.08; random arms at α 0.01", transform=ax.transAxes, ha="center", va="top", color=INK3, fontsize=8.3)
    ax.set_ylabel("judged hack rate per seed (%), 108 problems each", color=INK2, fontsize=9.5)
    ax.set_xlabel("arm (one point per sample seed, bar = 16-seed mean)", color=INK2, fontsize=9.5)
    ax.set_title("B. Is the lift spread over the 16 seeds?", loc="left", color=INK, fontsize=10.5, pad=10)

    # panel C: ECDF of prefix lengths
    ax = axes[2]
    order = ["L30_a0.04", "L30_a0.08", "L30_a0.16", "L30_rand1729_a0.01", "L30_rand1730_a0.01", "L30_rand1731_a0.01"]
    for a in order:
        if a not in prefix:
            continue
        v = np.sort(np.asarray(list(prefix[a].values()), float)) + 1
        y = np.arange(1, len(v) + 1) / len(v)
        ax.step(v, y, where="post", color=col(a), lw=2.2 if a == focus else 1.4, zorder=3 if a == focus else 2, label=label(a))
    ax.set_xscale("log")
    ax.legend(loc="lower right", fontsize=8.3, frameon=False, labelcolor=INK2)
    ax.set_xlabel("shared prefix with the unsteered twin (chars of turn-0 reasoning, log)", color=INK2, fontsize=9.5)
    ax.set_ylabel("fraction of episodes diverged by this point", color=INK2, fontsize=9.5)
    ax.set_title("C. How soon does each arm leave the unsteered path?", loc="left", color=INK, fontsize=10.5, pad=10)

    # table
    ax_t = fig.add_subplot(gs[1, :]); ax_t.axis("off")
    cols = [a for a in ["L30_a-0.08", "L30_a-0.04", "L30_a0.04", "L30_a0.08", "L30_a0.16", "L30_rand1729_a0.01", "L30_rand1730_a0.01", "L30_rand1731_a0.01"] if a in rows_b]
    def f1(x, sign=False):
        return ("—" if x is None or (isinstance(x, float) and math.isnan(x)) else (f"{100 * x:+.1f}" if sign else f"{100 * x:.1f}"))
    rows = [
        ("judged hack rate (%)", [f1(rows_b[a]["rate"]) for a in cols]),
        ("Δ vs unsteered (points)", [f1(rows_b[a]["delta"], True) for a in cols]),
        ("per-seed sd of the rate", [f1(rows_b[a]["per_seed_sd"]) for a in cols]),
        ("seeds up / down / tied", [f"{rows_b[a]['seeds_up']} / {rows_b[a]['seeds_down']} / {rows_b[a]['seeds_tied']}" for a in cols]),
        ("sign test p (16 seeds)", [f"{rows_b[a]['sign_p']:.3f}" for a in cols]),
        ("seed-bootstrap 95% CI of Δ", [f"{100 * rows_b[a]['boot_ci95'][0]:+.1f} to {100 * rows_b[a]['boot_ci95'][1]:+.1f}" for a in cols]),
        ("Δ seeds 0–7 / 8–15", [f"{100 * rows_b[a]['delta_seeds_0_7']:+.1f} / {100 * rows_b[a]['delta_seeds_8_15']:+.1f}" for a in cols]),
        (f"per-problem Δ: r with {label(focus).replace(' α 0.01', '')}", ["—" if a == focus else f"{focus_pairs[a]['pearson_delta']:+.2f} (p {focus_pairs[a]['perm_p']:.3f})" for a in cols]),
        ("same, partial r given unsteered (p)", ["—" if a == focus else f"{a2[a]['partial_r_given_unsteered']:+.2f} ({a2[a]['partial_p']:.3f})" for a in cols]),
        ("median shared prefix with twin (chars)", [f"{rows_c[a]['median']:.0f}" for a in cols]),
        ("episodes diverged within 10 chars", [f"{100 * rows_c[a]['frac_le_10']:.0f}%" for a in cols]),
        ("hacks also in the twin (indep. expectation)", [f"{rows_c[a]['shared_hacks']} of {rows_c[a]['hacks']} ({rows_c[a]['expected_shared_independent']:.1f})" for a in cols]),
    ]
    table = ax_t.table(cellText=[[r[0]] + r[1] for r in rows], colLabels=[""] + [label(a).replace(" α 0.01", "") for a in cols],
                       loc="upper center", cellLoc="center", colLoc="center", bbox=[0.0, 0.0, 1.0, 1.0],
                       colWidths=[0.2] + [0.8 / len(cols)] * len(cols))
    table.auto_set_font_size(False); table.set_fontsize(8.2)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor(RULE); cell.set_linewidth(0.6)
        cell.get_text().set_color(INK if r == 0 or c == 0 else INK2)
        if r == 0:
            cell.set_facecolor(SHADE); cell.get_text().set_fontweight("bold")
        elif c == 0:
            cell.get_text().set_ha("left"); cell.set_facecolor("#f7f6f3")
        if c > 0 and r > 0 and cols[c - 1] == focus:
            cell.set_facecolor("#f8efe6")
    fig.suptitle("Gemma 4 31B hard108, layer 30: random direction 1729 against the gi_v2 arms — per problem, per seed, and against the unsteered same-seed twin",
                 x=0.05, ha="left", color=INK, fontsize=12, y=0.965)
    fig.text(0.05, 0.925, f"Judged hack = HACK (STRONG + WEAK) over all 1,728 episodes per arm (108 problems × 16 seeds); same seed = same sampling stream, so prefixes are shared until the steered logits first flip a token.",
             color=INK2, fontsize=9)
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    fig.savefig(path.with_suffix(".pdf"), facecolor=SURFACE)


if __name__ == "__main__":
    main()
