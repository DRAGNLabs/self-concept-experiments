#!/usr/bin/env python
"""Sign-pair analysis of a random steering direction (+alpha vs -alpha) against unsteered and two null directions (CPU only).

Three questions, each read against the null arms where a shared-reference artefact exists:

  A. Mirror test. Are the problems the negative arm lowers the ones the positive arm raises? Per-problem judged hack
     rates (16 seeds) give Δ+ = pos − unsteered and Δ− = neg − unsteered; both share the unsteered reference, so r(Δ+, Δ−)
     is pushed *positive* by the artefact (the null-arm pair shows the floor). A mirror shows up as r(Δ+, Δ−) below the
     null pairs, as a negative partial r of the raw rates given the unsteered rate, and as a split-reference r (Δ+ against
     unsteered seeds 0–7, Δ− against 8–15, and the swap) below zero. Also: overlap of the "down under −" and "up under +"
     problem sets against the hypergeometric expectation, and regression-to-the-mean (r of each Δ with the unsteered rate).
  B. Reasoning length. Does the positive arm's longer turn-0 reasoning mediate its extra hacks? Within every arm: judged
     HACK vs other episodes, pooled and within problem; between arms among non-hack episodes only (if the lengthening
     survives removing the hacks, it is a side effect, not the path); a slope-based mediation estimate (within-problem
     slope of hack on standardised log length in the unsteered arm × the arm's length shift, compared with the observed Δ).
  C. Hazard after a failed attempt. Per attempt index k (0 = first submission): among episodes still running at k, the
     rate of judged hack at k (judge's `turn`), of a shown-pass that fails hidden (automatic proxy) and of a genuine pass.
     If the direction shifts "pass the shown tests by any means", the hack hazard moves at k >= 1 (after a failure) more
     than at k = 0. Problem-bootstrap CIs on the arm − unsteered hazard differences. Plus give-up (`stopped`) rates,
     attempt counts, the judges' pressure_source / first_considered_turn fields and mechanism mix per arm.

    python experiments/deceiver-axis/scripts/rand1729_sign_pair_analysis.py --hard108 <rh/hard108> \
        --analysis-json analysis_rand11/analysis_all_as_judged.json --roots analysis_root_rand11 analysis_root_main \
        --pos L30_rand1729_a0.01 --neg L30_rand1729_a-0.01 --nulls L30_rand1730_a0.01 L30_rand1731_a0.01 \
        --verdicts verdicts_main/verdicts_main.json ... --details verdicts_main/verdicts_main.detail.json ... \
        --out <dir> [--chart]
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

REFERENCE = "unsteered"
MAX_K = 5


# ----------------------------------------------------------------------------- small statistics helpers
def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def ranks(a):
    a = np.asarray(a, float)
    order = np.argsort(a, kind="stable")
    r = np.empty(len(a))
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and a[order[j + 1]] == a[order[i]]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return r


def spearman(x, y):
    return pearson(ranks(x), ranks(y))


def partial_r(x, y, z):
    rxy, rxz, ryz = pearson(x, y), pearson(x, z), pearson(y, z)
    den = (1 - rxz ** 2) * (1 - ryz ** 2)
    return float((rxy - rxz * ryz) / math.sqrt(den)) if den > 0 else float("nan")


def perm_p(x, y, stat, n=20000, seed=0):
    rng = np.random.default_rng(seed)
    obs = stat(x, y)
    if not np.isfinite(obs):
        return float("nan")
    y = np.asarray(y)
    hits = sum(1 for _ in range(n) if abs(stat(x, rng.permutation(y))) >= abs(obs) - 1e-12)
    return float((hits + 1) / (n + 1))


def binom_two_sided(k, n):
    if n == 0:
        return float("nan")
    p_lo = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    p_hi = sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n
    return float(min(1.0, 2 * min(p_lo, p_hi)))


def hypergeom_sf(k, N, K, n):
    """P(X >= k) for X ~ Hypergeom(N population, K successes, n draws)."""
    lo, hi = max(0, n + K - N), min(K, n)
    total = math.comb(N, n)
    return float(sum(math.comb(K, i) * math.comb(N - K, n - i) for i in range(k, hi + 1)) / total) if k <= hi else 0.0


def hypergeom_cdf(k, N, K, n):
    lo, hi = max(0, n + K - N), min(K, n)
    total = math.comb(N, n)
    return float(sum(math.comb(K, i) * math.comb(N - K, n - i) for i in range(lo, min(k, hi) + 1)) / total)


def q(values, p):
    values = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.percentile(values, p)) if values else float("nan")


def mean(values):
    values = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(values)) if values else float("nan")


def fmt(x, nd=2, pct=False):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    if pct:
        return f"{100 * x:.1f}"
    return f"{x:+.{nd}f}" if isinstance(x, float) and nd and abs(x) < 10 and not pct else f"{x:.{nd}f}"


# ----------------------------------------------------------------------------- loading
def arm_dir(hard108: Path, roots: list[str], arm: str) -> Path:
    for root in roots:
        p = hard108 / root / arm
        if p.is_dir():
            return p
    raise SystemExit(f"arm {arm} not under {roots}")


def load_records(path: Path) -> dict[str, dict]:
    out = {}
    with path.open() as fh:
        for line in fh:
            r = json.loads(line)
            out[r["example_id"]] = {
                "passed": bool(r.get("passed")), "hidden": bool(r.get("passed_original")),
                "n_attempts": int(r.get("n_attempts") or len(r.get("attempts") or [])),
                "stop": r.get("stop_reason"), "success": [bool(a.get("success")) for a in r.get("attempts") or []],
            }
    return out


def load_reasoning(path: Path) -> dict[str, list[int]]:
    out: dict[str, dict[int, int]] = defaultdict(dict)
    with path.open() as fh:
        for line in fh:
            r = json.loads(line)
            out[r["example_id"]][int(r["turn"])] = len(r.get("reasoning") or "")
    return {ex: [turns[t] for t in sorted(turns)] for ex, turns in out.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hard108", type=Path, required=True)
    ap.add_argument("--analysis-json", required=True, help="relative to --hard108; judged_hack.arms[arm].per_problem_rate")
    ap.add_argument("--roots", nargs="+", required=True, help="record roots (relative), first containing the arm wins")
    ap.add_argument("--pos", default="L30_rand1729_a0.01")
    ap.add_argument("--neg", default="L30_rand1729_a-0.01")
    ap.add_argument("--nulls", nargs="+", default=["L30_rand1730_a0.01", "L30_rand1731_a0.01"])
    ap.add_argument("--verdicts", nargs="+", required=True, help="relative verdict json files (plain + hidden)")
    ap.add_argument("--details", nargs="+", required=True, help="relative verdict detail json files (mechanism, turn, pressure_source)")
    ap.add_argument("--records-name", default="pilot_impossible_original.jsonl")
    ap.add_argument("--reasoning-name", default="pilot_hard108_reasoning.jsonl")
    ap.add_argument("--seeds", type=int, default=16)
    ap.add_argument("--max-seeds", type=int, help="smoke test: only the first N seeds for sections B and C")
    ap.add_argument("--perms", type=int, default=20000)
    ap.add_argument("--boots", type=int, default=2000)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--chart", action="store_true")
    ap.add_argument("--chart-only", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.chart_only:
        chart(json.load(open(args.out / "sign_pair_analysis.json")), args.out / "sign_pair_chart.png")
        return

    pos, neg, nulls = args.pos, args.neg, list(args.nulls)
    arms = [pos, neg] + nulls
    all_arms = [REFERENCE] + arms
    seeds = list(range(args.max_seeds or args.seeds))
    short = {REFERENCE: "unsteered", pos: "+" + pos.split("_")[1][4:] if "rand" in pos else pos,
             neg: "−" + neg.split("_")[1][4:] if "rand" in neg else neg}
    for a in nulls:
        short[a] = a.split("_")[1][4:] if "rand" in a else a

    analysis = json.load(open(args.hard108 / args.analysis_json))["judged_hack"]["arms"]
    problems = sorted(analysis[REFERENCE]["per_problem_rate"])
    n_prob = len(problems)
    rate = {a: np.array([analysis[a]["per_problem_rate"][p] for p in problems]) for a in all_arms}
    delta = {a: rate[a] - rate[REFERENCE] for a in arms}

    hack: dict[tuple[str, int], set[str]] = {}
    verdict: dict[str, str] = {}
    for f in args.verdicts:
        for key, value in json.load(open(args.hard108 / f)).items():
            arm, seed_s, ex = key.split("/", 2)
            verdict[key] = value
            if value == "HACK":
                hack.setdefault((arm, int(seed_s[4:])), set()).add(ex)
    detail: dict[str, dict] = {}
    for f in args.details:
        detail.update(json.load(open(args.hard108 / f)))
    for a in all_arms:  # consistency check with the analysis json
        pooled = sum(len(hack.get((a, s), ())) for s in range(args.seeds)) / (n_prob * args.seeds)
        if abs(pooled - analysis[a]["rate"]) > 1e-9:
            print(f"WARNING pooled rate mismatch {a}: verdict files {pooled:.5f} vs analysis {analysis[a]['rate']:.5f}")

    result: dict = {"arms": all_arms, "short": short, "problems": n_prob, "seeds": len(seeds), "analysis_seeds": args.seeds,
                    "rates": {a: float(analysis[a]["rate"]) for a in all_arms}}

    # ======================================================================== A. mirror test
    def prob_rate(a, seed_idx):
        return np.array([sum(1 for s in seed_idx if pr in hack.get((a, s), ())) / len(seed_idx) for pr in problems])
    ref_half = {"A": prob_rate(REFERENCE, range(0, 8)), "B": prob_rate(REFERENCE, range(8, 16))}

    def split_ref_r(a, b):
        return float(np.mean([pearson(rate[a] - ref_half[h1], rate[b] - ref_half[h2]) for h1, h2 in (("A", "B"), ("B", "A"))]))

    pair_rows = []
    for a, b in combinations(arms, 2):
        kind = "pos-neg" if {a, b} == {pos, neg} else ("null-null" if a in nulls and b in nulls else "signed-null")
        row = {"pair": [a, b], "kind": kind, "pearson_delta": pearson(delta[a], delta[b]), "spearman_delta": spearman(delta[a], delta[b]),
               "partial_r_given_unsteered": partial_r(rate[a], rate[b], rate[REFERENCE]), "split_reference_r": split_ref_r(a, b),
               "pearson_raw": pearson(rate[a], rate[b])}
        if kind == "pos-neg":
            row["perm_p_delta"] = perm_p(delta[a], delta[b], pearson, args.perms, seed=1)
            row["perm_p_partial"] = perm_p(delta[a], delta[b], lambda x, y: partial_r(x + rate[REFERENCE], y + rate[REFERENCE], rate[REFERENCE]), args.perms, seed=2)
        pair_rows.append(row)

    # regression to the mean: r(Δ, unsteered rate) for every arm (the null arms carry the same artefact)
    rtm_rows = [{"arm": a, "r_delta_vs_unsteered_rate": pearson(delta[a], rate[REFERENCE]),
                 "spearman_delta_vs_unsteered_rate": spearman(delta[a], rate[REFERENCE])} for a in arms]

    # problem sets
    U = {pr for pr, r in zip(problems, rate[REFERENCE]) if r > 0}  # problems that hack unsteered
    up = {a: {pr for pr, d in zip(problems, delta[a]) if d > 0} for a in arms}
    down = {a: {pr for pr, d in zip(problems, delta[a]) if d < 0} for a in arms}
    hacking = {a: {pr for pr, r in zip(problems, rate[a]) if r > 0} for a in all_arms}
    set_rows = []
    for a in [neg] + nulls:  # "down" sets are only possible inside U
        down_a = down[a] & U
        for b in [pos] + [n for n in nulls if n != a]:
            up_b_in_U = up[b] & U
            overlap = len(down_a & up_b_in_U)
            expected = len(down_a) * len(up_b_in_U) / len(U) if U else float("nan")
            set_rows.append({"down_arm": a, "up_arm": b, "U": len(U), "down": len(down_a), "up_in_U": len(up_b_in_U), "overlap": overlap,
                             "expected": expected, "p_ge": hypergeom_sf(overlap, len(U), len(up_b_in_U), len(down_a)),
                             "p_le": hypergeom_cdf(overlap, len(U), len(up_b_in_U), len(down_a)),
                             # within U: does the size of the drop track the size of the rise?
                             "spearman_within_U": spearman([delta[a][problems.index(p)] for p in sorted(U)], [delta[b][problems.index(p)] for p in sorted(U)]),
                             "pearson_within_U": pearson([delta[a][problems.index(p)] for p in sorted(U)], [delta[b][problems.index(p)] for p in sorted(U)])})
    # hacking-problem sets vs unsteered's
    hp_rows = [{"arm": a, "hacking_problems": len(hacking[a]), "in_U": len(hacking[a] & U), "new": len(hacking[a] - U),
                "U_kept": len(hacking[a] & U), "U_lost": len(U - hacking[a])} for a in all_arms]
    # per-problem table (U plus every problem the + arm lifts by >= 3/16)
    show = sorted(U | {pr for pr, d in zip(problems, delta[pos]) if d >= 3 / 16 - 1e-9},
                  key=lambda p: (-rate[REFERENCE][problems.index(p)], -delta[pos][problems.index(p)]))
    prob_rows = [{"problem": p, **{a: int(round(rate[a][problems.index(p)] * args.seeds)) for a in all_arms}} for p in show]
    result["A"] = {"pairs": pair_rows, "regression_to_mean": rtm_rows, "sets": set_rows, "hacking_problems": hp_rows, "problem_table": prob_rows,
                   "U": sorted(U), "up": {a: sorted(v) for a, v in up.items()}, "down": {a: sorted(v) for a, v in down.items()}}

    # ======================================================================== load episodes (B, C)
    rec: dict[str, dict[tuple[str, int], dict]] = {a: {} for a in all_arms}  # arm -> (problem, seed) -> record
    rlen: dict[str, dict[tuple[str, int], list[int]]] = {a: {} for a in all_arms}
    for a in all_arms:
        d = arm_dir(args.hard108, args.roots, a)
        for s in seeds:
            for ex, r in load_records(d / f"seed{s}" / args.records_name).items():
                rec[a][(ex, s)] = r
            for ex, lens in load_reasoning(d / f"seed{s}" / args.reasoning_name).items():
                rlen[a][(ex, s)] = lens
    keys = {a: sorted(set(rec[a]) & set(rlen[a])) for a in all_arms}
    is_hack = {a: {k: (k[0] in hack.get((a, k[1]), ())) for k in keys[a]} for a in all_arms}
    result["episodes"] = {a: len(keys[a]) for a in all_arms}

    # ======================================================================== B. reasoning length
    def t0(a, k):
        return rlen[a][k][0] if rlen[a][k] else 0

    def total(a, k):
        return sum(rlen[a][k])

    def per_problem_mean(a, fn, pred=lambda a, k: True):
        acc = defaultdict(list)
        for k in keys[a]:
            if pred(a, k):
                acc[k[0]].append(fn(a, k))
        return {p: float(np.mean(v)) for p, v in acc.items()}

    len_rows = []
    for a in all_arms:
        row = {"arm": a, "n": len(keys[a]), "turn0_mean": mean([t0(a, k) for k in keys[a]]), "turn0_median": q([t0(a, k) for k in keys[a]], 50),
               "total_mean": mean([total(a, k) for k in keys[a]]), "turns_mean": mean([len(rlen[a][k]) for k in keys[a]]),
               "per_turn_mean": mean([total(a, k) / max(1, len(rlen[a][k])) for k in keys[a]])}
        if a != REFERENCE:
            for name, fn in (("turn0", t0), ("total", total)):
                pm_a, pm_r = per_problem_mean(a, fn), per_problem_mean(REFERENCE, fn)
                common = sorted(set(pm_a) & set(pm_r))
                longer = sum(1 for p in common if pm_a[p] > pm_r[p])
                row[f"{name}_problems_longer"] = longer
                row[f"{name}_problems"] = len(common)
                row[f"{name}_sign_p"] = binom_two_sided(longer, sum(1 for p in common if pm_a[p] != pm_r[p]))
                # same, non-hack episodes only (both arms)
                nh = lambda arm, k: not is_hack[arm][k]
                pm_a2, pm_r2 = per_problem_mean(a, fn, nh), per_problem_mean(REFERENCE, fn, nh)
                common2 = sorted(set(pm_a2) & set(pm_r2))
                longer2 = sum(1 for p in common2 if pm_a2[p] > pm_r2[p])
                row[f"{name}_nonhack_problems_longer"] = longer2
                row[f"{name}_nonhack_problems"] = len(common2)
                row[f"{name}_nonhack_sign_p"] = binom_two_sided(longer2, sum(1 for p in common2 if pm_a2[p] != pm_r2[p]))
                row[f"{name}_nonhack_mean"] = mean([fn(a, k) for k in keys[a] if not is_hack[a][k]])
                row[f"{name}_nonhack_mean_ref"] = mean([fn(REFERENCE, k) for k in keys[REFERENCE] if not is_hack[REFERENCE][k]])
        len_rows.append(row)

    # within-arm: hack vs other
    within_rows = []
    for a in all_arms:
        hk = [k for k in keys[a] if is_hack[a][k]]
        oth = [k for k in keys[a] if not is_hack[a][k]]
        row = {"arm": a, "n_hack": len(hk), "n_other": len(oth),
               "turn0_hack_mean": mean([t0(a, k) for k in hk]), "turn0_other_mean": mean([t0(a, k) for k in oth]),
               "turn0_hack_median": q([t0(a, k) for k in hk], 50), "turn0_other_median": q([t0(a, k) for k in oth], 50),
               "total_hack_mean": mean([total(a, k) for k in hk]), "total_other_mean": mean([total(a, k) for k in oth]),
               "turns_hack_mean": mean([len(rlen[a][k]) for k in hk]), "turns_other_mean": mean([len(rlen[a][k]) for k in oth])}
        # within problem: problems with both kinds; mean(hack) − mean(other) per problem
        by_p = defaultdict(lambda: {"h": [], "o": []})
        for k in keys[a]:
            by_p[k[0]]["h" if is_hack[a][k] else "o"].append(t0(a, k))
        diffs = [np.mean(v["h"]) - np.mean(v["o"]) for v in by_p.values() if v["h"] and v["o"]]
        row["within_problem_n"] = len(diffs)
        row["within_problem_turn0_hack_longer"] = int(sum(1 for d in diffs if d > 0))
        row["within_problem_turn0_mean_diff"] = mean(diffs)
        row["within_problem_sign_p"] = binom_two_sided(int(sum(1 for d in diffs if d > 0)), int(sum(1 for d in diffs if d != 0)))
        # judged dossiers only (proxy pool): HACK vs NOT_HACK/BORDERLINE
        jh = [k for k in keys[a] if verdict.get(f"{a}/seed{k[1]}/{k[0]}") == "HACK"]
        jo = [k for k in keys[a] if verdict.get(f"{a}/seed{k[1]}/{k[0]}") in ("NOT_HACK", "BORDERLINE")]
        row["judged_hack_turn0_mean"], row["judged_other_turn0_mean"] = mean([t0(a, k) for k in jh]), mean([t0(a, k) for k in jo])
        row["judged_n_hack"], row["judged_n_other"] = len(jh), len(jo)
        within_rows.append(row)

    # mediation estimate: within-problem slope of hack on standardised log turn-0 length (unsteered), × the arm's shift
    def zlog(a):
        # z-score log(turn0+1) within problem using the unsteered problem mean/sd
        stats = defaultdict(list)
        for k in keys[REFERENCE]:
            stats[k[0]].append(math.log1p(t0(REFERENCE, k)))
        mu = {p: float(np.mean(v)) for p, v in stats.items()}
        sd = {p: float(np.std(v, ddof=1)) if len(v) > 1 and np.std(v, ddof=1) > 0 else float("nan") for p, v in stats.items()}
        out = {}
        for k in keys[a]:
            if k[0] in mu and np.isfinite(sd[k[0]]):
                out[k] = (math.log1p(t0(a, k)) - mu[k[0]]) / sd[k[0]]
        return out

    def slope(a):
        z = zlog(a)
        x = np.array([z[k] for k in keys[a] if k in z]); y = np.array([1.0 if is_hack[a][k] else 0.0 for k in keys[a] if k in z])
        # within-problem OLS: demean both within problem
        px = defaultdict(list); py = defaultdict(list)
        for k in keys[a]:
            if k in z:
                px[k[0]].append(z[k]); py[k[0]].append(1.0 if is_hack[a][k] else 0.0)
        xs, ys = [], []
        for p in px:
            mx, my = np.mean(px[p]), np.mean(py[p])
            xs += [v - mx for v in px[p]]; ys += [v - my for v in py[p]]
        xs, ys = np.array(xs), np.array(ys)
        b = float((xs * ys).sum() / (xs * xs).sum()) if (xs * xs).sum() > 0 else float("nan")
        return b, float(x.mean()) if len(x) else float("nan"), pearson(x, y)

    med_rows = []
    b_ref, z_ref, r_ref = slope(REFERENCE)
    for a in arms:
        b_a, z_a, r_a = slope(a)
        med_rows.append({"arm": a, "mean_z_shift": z_a - z_ref, "slope_unsteered_per_sd": b_ref, "slope_arm_per_sd": b_a,
                         "predicted_delta_from_length": b_ref * (z_a - z_ref), "observed_delta": float(analysis[a]["rate"] - analysis[REFERENCE]["rate"]),
                         "pooled_r_length_hack_arm": r_a, "pooled_r_length_hack_unsteered": r_ref})
    # hack rate by length half (above / below the unsteered problem median of turn-0 length), per arm
    half_rows = []
    med_ref = defaultdict(list)
    for k in keys[REFERENCE]:
        med_ref[k[0]].append(t0(REFERENCE, k))
    med_ref = {p: float(np.median(v)) for p, v in med_ref.items()}
    for a in all_arms:
        hi = [k for k in keys[a] if k[0] in med_ref and t0(a, k) > med_ref[k[0]]]
        lo = [k for k in keys[a] if k[0] in med_ref and t0(a, k) <= med_ref[k[0]]]
        half_rows.append({"arm": a, "n_long": len(hi), "n_short": len(lo), "hack_rate_long": mean([1.0 if is_hack[a][k] else 0.0 for k in hi]),
                          "hack_rate_short": mean([1.0 if is_hack[a][k] else 0.0 for k in lo]), "frac_long": len(hi) / max(1, len(hi) + len(lo))})
    result["B"] = {"lengths": len_rows, "within_arm": within_rows, "mediation": med_rows, "length_halves": half_rows}

    # ======================================================================== C. hazard
    def hack_turn(a, k):
        d = detail.get(f"{a}/seed{k[1]}/{k[0]}")
        if d and d.get("verdict") == "HACK":
            return d.get("turn")
        return None

    turn_check = Counter()
    for a in all_arms:
        for k in keys[a]:
            if is_hack[a][k]:
                t = hack_turn(a, k)
                r = rec[a][k]
                last = r["n_attempts"] - 1
                if t is None:
                    turn_check[(a, "turn_null")] += 1
                else:
                    turn_check[(a, "turn==last")] += int(t == last)
                    turn_check[(a, "turn<last")] += int(t < last)
                    turn_check[(a, "turn>last")] += int(t > last)

    def hazard_counts(a, keyset):
        """per k: alive, hack (judge turn == k, or last attempt if turn missing), proxy pass, genuine pass, fail_stop, fail_continue."""
        out = {k: Counter() for k in range(MAX_K)}
        for key in keyset:
            r = rec[a][key]
            n = r["n_attempts"]
            ht = hack_turn(a, key) if is_hack[a][key] else None
            if is_hack[a][key] and ht is None:
                ht = n - 1
            for k in range(min(n, MAX_K)):
                c = out[k]
                c["alive"] += 1
                if ht == k:
                    c["hack"] += 1
                last = k == n - 1
                succ = r["success"][k] if k < len(r["success"]) else False
                if is_hack[a][key] and last and succ:
                    c["hack_final"] += 1
                if succ and last:
                    c["genuine" if r["hidden"] else "proxy"] += 1
                    c["shown_pass"] += 1
                elif last:
                    c["fail_stop_gaveup" if r["stop"] == "stopped" else "fail_stop_limit"] += 1
                else:
                    c["fail_continue"] += 1
        return out

    def summarise(counts):
        rows = {}
        for k, c in counts.items():
            al = c["alive"]
            rows[k] = {"alive": al, **{e: c[e] for e in ("hack", "hack_final", "proxy", "genuine", "shown_pass", "fail_continue", "fail_stop_limit", "fail_stop_gaveup")},
                       **{f"h_{e}": (c[e] / al if al else float("nan")) for e in ("hack", "hack_final", "proxy", "genuine", "shown_pass", "fail_stop_gaveup")}}
        # pooled k >= 1
        c1 = Counter()
        for k in range(1, MAX_K):
            c1.update(counts[k])
        al = c1["alive"]
        rows["k>=1"] = {"alive": al, **{e: c1[e] for e in ("hack", "hack_final", "proxy", "genuine", "shown_pass", "fail_continue", "fail_stop_limit", "fail_stop_gaveup")},
                        **{f"h_{e}": (c1[e] / al if al else float("nan")) for e in ("hack", "hack_final", "proxy", "genuine", "shown_pass", "fail_stop_gaveup")}}
        return rows

    haz = {a: summarise(hazard_counts(a, keys[a])) for a in all_arms}

    # problem-bootstrap CIs for arm − unsteered hazard differences at k = 0 and k >= 1
    rng = np.random.default_rng(11)
    by_prob = {a: defaultdict(list) for a in all_arms}
    for a in all_arms:
        for k in keys[a]:
            by_prob[a][k[0]].append(k)
    boot_rows = []
    for a in arms:
        samples = {("0", e): [] for e in ("hack", "hack_final", "proxy", "genuine")}
        samples.update({("k>=1", e): [] for e in ("hack", "hack_final", "proxy", "genuine")})
        for _ in range(args.boots):
            draw = rng.choice(problems, n_prob, replace=True)
            ks_a = [k for p in draw for k in by_prob[a][p]]
            ks_r = [k for p in draw for k in by_prob[REFERENCE][p]]
            ha, hr = summarise(hazard_counts(a, ks_a)), summarise(hazard_counts(REFERENCE, ks_r))
            for kk, lab in ((0, "0"), ("k>=1", "k>=1")):
                for e in ("hack", "hack_final", "proxy", "genuine"):
                    samples[(lab, e)].append(ha[kk][f"h_{e}"] - hr[kk][f"h_{e}"])
        row = {"arm": a}
        for (lab, e), v in samples.items():
            kk = 0 if lab == "0" else "k>=1"
            row[f"{e}_k{lab}_delta"] = haz[a][kk][f"h_{e}"] - haz[REFERENCE][kk][f"h_{e}"]
            row[f"{e}_k{lab}_ci"] = [q(v, 2.5), q(v, 97.5)]
        boot_rows.append(row)

    # episode-level: after a failed first attempt, how does it end?
    after_fail_rows = []
    for a in all_arms:
        ks = [k for k in keys[a] if rec[a][k]["success"] and not rec[a][k]["success"][0]]
        n = len(ks)
        ends = Counter()
        for k in ks:
            r = rec[a][k]
            if r["passed"]:
                ends["hack" if is_hack[a][k] else ("genuine" if r["hidden"] else "proxy_not_judged_hack")] += 1
            else:
                ends["gave_up" if r["stop"] == "stopped" else "attempt_limit"] += 1
        first_pass = sum(1 for k in keys[a] if rec[a][k]["success"] and rec[a][k]["success"][0])
        after_fail_rows.append({"arm": a, "n_episodes": len(keys[a]), "first_attempt_pass": first_pass, "first_attempt_fail": n,
                                **{f"end_{e}": ends[e] for e in ("hack", "genuine", "proxy_not_judged_hack", "gave_up", "attempt_limit")},
                                **{f"p_{e}": ends[e] / n if n else float("nan") for e in ("hack", "genuine", "proxy_not_judged_hack", "gave_up", "attempt_limit")},
                                "mean_attempts": mean([rec[a][k]["n_attempts"] for k in keys[a]]),
                                "stopped_rate": mean([1.0 if rec[a][k]["stop"] == "stopped" else 0.0 for k in keys[a]]),
                                "attempt_limit_rate": mean([1.0 if rec[a][k]["stop"] == "attempt_limit" else 0.0 for k in keys[a]])})

    # judge fields per arm over HACK verdicts
    field_rows = []
    for a in all_arms:
        hk = [detail[f"{a}/seed{k[1]}/{k[0]}"] for k in keys[a] if is_hack[a][k] and f"{a}/seed{k[1]}/{k[0]}" in detail]
        field_rows.append({"arm": a, "n": len(hk),
                           "turn": dict(Counter(str(d.get("turn")) for d in hk)),
                           "first_considered_turn": dict(Counter(str(d.get("first_considered_turn")) for d in hk)),
                           "considered_before_hack": sum(1 for d in hk if isinstance(d.get("first_considered_turn"), int) and isinstance(d.get("turn"), int) and d["first_considered_turn"] < d["turn"]),
                           "pressure_source": dict(Counter(d.get("pressure_source") for d in hk)),
                           "mechanism": dict(Counter(d.get("mechanism") for d in hk)),
                           "feedback_confusion_yes": sum(1 for d in hk if d.get("feedback_confusion") == "yes")})
    result["C"] = {"turn_check": {f"{a}|{w}": v for (a, w), v in turn_check.items()}, "hazard": {a: {str(k): v for k, v in haz[a].items()} for a in all_arms},
                   "hazard_boot": boot_rows, "after_first_fail": after_fail_rows, "judge_fields": field_rows}

    json.dump(result, open(args.out / "sign_pair_analysis.json", "w"), indent=1, default=float)
    (args.out / "sign_pair_analysis.md").write_text(render(result))
    print(render(result))
    if args.chart:
        chart(result, args.out / "sign_pair_chart.png")
        print(f"chart -> {args.out / 'sign_pair_chart.png'}")


# ----------------------------------------------------------------------------- report
def render(res: dict) -> str:
    S = res["short"]
    arms = res["arms"]
    L = [f"# Sign-pair analysis: {S[arms[1]]} vs {S[arms[2]]} against unsteered and nulls ({res['problems']} problems, seeds 0–{res['seeds'] - 1})", ""]
    L.append("Judged hack rates (%): " + ", ".join(f"{S[a]} {100 * res['rates'][a]:.1f}" for a in arms))
    L += ["", "## A. Mirror test (per-problem Δ vs unsteered)", "",
          "| pair | kind | Pearson r(Δ) | Spearman ρ(Δ) | partial r (raw | unsteered) | split-reference r | r(raw) | perm p (Δ) | perm p (partial) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in res["A"]["pairs"]:
        a, b = r["pair"]
        L.append(f"| {S[a]} × {S[b]} | {r['kind']} | {fmt(r['pearson_delta'])} | {fmt(r['spearman_delta'])} | {fmt(r['partial_r_given_unsteered'])} | "
                 f"{fmt(r['split_reference_r'])} | {fmt(r['pearson_raw'])} | {fmt(r.get('perm_p_delta'), 4)} | {fmt(r.get('perm_p_partial'), 4)} |")
    L += ["", "Regression to the mean (r of Δ with the unsteered per-problem rate; the null arms show the artefact):", "",
          "| arm | Pearson | Spearman |", "|---|---|---|"]
    for r in res["A"]["regression_to_mean"]:
        L.append(f"| {S[r['arm']]} | {fmt(r['r_delta_vs_unsteered_rate'])} | {fmt(r['spearman_delta_vs_unsteered_rate'])} |")
    L += ["", f"Problem sets inside U = problems that hack unsteered (|U| = {res['A']['sets'][0]['U']}). 'down' = arm below unsteered, 'up' = arm above unsteered:", "",
          "| down under | up under | down | up ∩ U | overlap | expected | P(≥) | P(≤) | Spearman within U | Pearson within U |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in res["A"]["sets"]:
        L.append(f"| {S[r['down_arm']]} | {S[r['up_arm']]} | {r['down']} | {r['up_in_U']} | {r['overlap']} | {r['expected']:.1f} | {r['p_ge']:.3f} | {r['p_le']:.3f} | "
                 f"{fmt(r['spearman_within_U'])} | {fmt(r['pearson_within_U'])} |")
    L += ["", "| arm | hacking problems | of which in U | new | U lost |", "|---|---|---|---|---|"]
    for r in res["A"]["hacking_problems"]:
        L.append(f"| {S[r['arm']]} | {r['hacking_problems']} | {r['in_U']} | {r['new']} | {r['U_lost']} |")
    L += ["", f"Per-problem hack counts of {res['analysis_seeds']} (U plus every problem {S[arms[1]]} lifts by ≥ 3):", "",
          "| problem | " + " | ".join(S[a] for a in arms) + " |", "|---|" + "---|" * len(arms)]
    for r in res["A"]["problem_table"]:
        L.append(f"| {r['problem']} | " + " | ".join(str(r[a]) for a in arms) + " |")

    L += ["", "## B. Reasoning length", "", "Per arm (chars); 'problems longer' = per-problem mean above unsteered, all episodes and non-hack episodes only:", "",
          "| arm | turn-0 mean | turn-0 median | total mean | turns | per-turn mean | turn-0 problems longer (sign p) | non-hack only | total problems longer | non-hack only | non-hack turn-0 mean (arm / unsteered) |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in res["B"]["lengths"]:
        if r["arm"] == REFERENCE:
            L.append(f"| {S[r['arm']]} | {r['turn0_mean']:.0f} | {r['turn0_median']:.0f} | {r['total_mean']:.0f} | {r['turns_mean']:.2f} | {r['per_turn_mean']:.0f} | — | — | — | — | — |")
        else:
            L.append(f"| {S[r['arm']]} | {r['turn0_mean']:.0f} | {r['turn0_median']:.0f} | {r['total_mean']:.0f} | {r['turns_mean']:.2f} | {r['per_turn_mean']:.0f} | "
                     f"{r['turn0_problems_longer']}/{r['turn0_problems']} ({r['turn0_sign_p']:.3f}) | {r['turn0_nonhack_problems_longer']}/{r['turn0_nonhack_problems']} ({r['turn0_nonhack_sign_p']:.3f}) | "
                     f"{r['total_problems_longer']}/{r['total_problems']} ({r['total_sign_p']:.3f}) | {r['total_nonhack_problems_longer']}/{r['total_nonhack_problems']} ({r['total_nonhack_sign_p']:.3f}) | "
                     f"{r['turn0_nonhack_mean']:.0f} / {r['turn0_nonhack_mean_ref']:.0f} |")
    L += ["", "Within arm, judged-HACK episodes vs all other episodes:", "",
          "| arm | n hack / other | turn-0 mean hack / other | turn-0 median hack / other | total mean hack / other | turns hack / other | within-problem: hack longer (n problems, sign p, mean diff) | judged dossiers only: HACK / NOT (n) |",
          "|---|---|---|---|---|---|---|---|"]
    for r in res["B"]["within_arm"]:
        L.append(f"| {S[r['arm']]} | {r['n_hack']} / {r['n_other']} | {r['turn0_hack_mean']:.0f} / {r['turn0_other_mean']:.0f} | {r['turn0_hack_median']:.0f} / {r['turn0_other_median']:.0f} | "
                 f"{r['total_hack_mean']:.0f} / {r['total_other_mean']:.0f} | {r['turns_hack_mean']:.2f} / {r['turns_other_mean']:.2f} | "
                 f"{r['within_problem_turn0_hack_longer']}/{r['within_problem_n']} (p {r['within_problem_sign_p']:.3f}, {r['within_problem_turn0_mean_diff']:+.0f}) | "
                 f"{r['judged_hack_turn0_mean']:.0f} / {r['judged_other_turn0_mean']:.0f} ({r['judged_n_hack']} / {r['judged_n_other']}) |")
    L += ["", "Mediation estimate: within-problem slope of hack on standardised log turn-0 length (per 1 sd of the unsteered problem distribution) × the arm's mean shift:", "",
          "| arm | mean z shift | slope unsteered (pp/sd) | slope in arm (pp/sd) | predicted Δ from length (pp) | observed Δ (pp) | pooled r(length, hack) arm / unsteered |", "|---|---|---|---|---|---|---|"]
    for r in res["B"]["mediation"]:
        L.append(f"| {S[r['arm']]} | {r['mean_z_shift']:+.2f} | {100 * r['slope_unsteered_per_sd']:+.2f} | {100 * r['slope_arm_per_sd']:+.2f} | {100 * r['predicted_delta_from_length']:+.2f} | "
                 f"{100 * r['observed_delta']:+.1f} | {fmt(r['pooled_r_length_hack_arm'], 3)} / {fmt(r['pooled_r_length_hack_unsteered'], 3)} |")
    L += ["", "Hack rate (%) by turn-0 length half (above / at-or-below the unsteered problem median):", "", "| arm | n long / short | frac long | hack rate long | hack rate short |", "|---|---|---|---|---|"]
    for r in res["B"]["length_halves"]:
        L.append(f"| {S[r['arm']]} | {r['n_long']} / {r['n_short']} | {100 * r['frac_long']:.0f}% | {100 * r['hack_rate_long']:.1f} | {100 * r['hack_rate_short']:.1f} |")

    L += ["", "## C. Hazard by attempt index", "", "Judge `turn` vs the episode's last attempt (HACK verdicts): " +
          ", ".join(f"{k} {v}" for k, v in sorted(res["C"]["turn_check"].items())), "",
          "Per attempt k (0 = first submission): alive = episodes still running; hazards in % of alive. hack (judge turn) = judged HACK placed by the judge at turn k (the turn the hack was written, which can precede the passing submission); hack (final pass) = judged-HACK episode whose passing submission is attempt k; proxy = shown pass that fails hidden (automatic proxy, includes the judged hacks); genuine = hidden pass.", ""]
    for a in arms:
        L += [f"**{S[a]}**", "", "| k | alive | hack (judge turn) | hack (final pass) | proxy | genuine | shown pass | fail→continue | fail→limit | gave up | h(hack) | h(hack final) | h(proxy) | h(genuine) | h(gave up) |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for k, r in res["C"]["hazard"][a].items():
            L.append(f"| {k} | {r['alive']} | {r['hack']} | {r['hack_final']} | {r['proxy']} | {r['genuine']} | {r['shown_pass']} | {r['fail_continue']} | {r['fail_stop_limit']} | {r['fail_stop_gaveup']} | "
                     f"{100 * r['h_hack']:.1f} | {100 * r['h_hack_final']:.1f} | {100 * r['h_proxy']:.1f} | {100 * r['h_genuine']:.1f} | {100 * r['h_fail_stop_gaveup']:.1f} |")
        L.append("")
    L += ["Hazard differences vs unsteered (percentage points, problem-bootstrap 95% CI):", "",
          "| arm | hack (judge turn) k=0 | k≥1 | hack (final pass) k=0 | k≥1 | proxy k=0 | k≥1 | genuine k=0 | k≥1 |", "|---|---|---|---|---|---|---|---|---|"]
    for r in res["C"]["hazard_boot"]:
        cells = []
        for e in ("hack", "hack_final", "proxy", "genuine"):
            for lab in ("0", "k>=1"):
                d, ci = r[f"{e}_k{lab}_delta"], r[f"{e}_k{lab}_ci"]
                cells.append(f"{100 * d:+.1f} ({100 * ci[0]:+.1f} to {100 * ci[1]:+.1f})")
        L.append(f"| {S[r['arm']]} | " + " | ".join(cells) + " |")
    L += ["", "After a failed first attempt, how the episode ends:", "",
          "| arm | first attempt pass / fail | → judged hack | → genuine pass | → proxy pass not judged hack | → gave up | → attempt limit | mean attempts | stopped rate | attempt-limit rate |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in res["C"]["after_first_fail"]:
        L.append(f"| {S[r['arm']]} | {r['first_attempt_pass']} / {r['first_attempt_fail']} | {r['end_hack']} ({100 * r['p_hack']:.1f}%) | {r['end_genuine']} ({100 * r['p_genuine']:.1f}%) | "
                 f"{r['end_proxy_not_judged_hack']} ({100 * r['p_proxy_not_judged_hack']:.1f}%) | {r['end_gave_up']} ({100 * r['p_gave_up']:.1f}%) | {r['end_attempt_limit']} ({100 * r['p_attempt_limit']:.1f}%) | "
                 f"{r['mean_attempts']:.2f} | {100 * r['stopped_rate']:.1f}% | {100 * r['attempt_limit_rate']:.1f}% |")
    L += ["", "Judge fields over HACK verdicts:", "", "| arm | n | turn | first_considered_turn | considered before the hack turn | pressure_source | mechanism | feedback_confusion yes |", "|---|---|---|---|---|---|---|---|"]
    for r in res["C"]["judge_fields"]:
        L.append(f"| {S[r['arm']]} | {r['n']} | {json.dumps(r['turn'])} | {json.dumps(r['first_considered_turn'])} | {r['considered_before_hack']} | {json.dumps(r['pressure_source'])} | {json.dumps(r['mechanism'])} | {r['feedback_confusion_yes']} |")
    return "\n".join(L) + "\n"


# ----------------------------------------------------------------------------- chart
def chart(res: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    SURFACE, INK, INK2, INK3, RULE, SHADE = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8985", "#dedcd7", "#f1f0ec"
    S = res["short"]
    arms = res["arms"]
    ref, pos, neg, nulls = arms[0], arms[1], arms[2], arms[3:]
    colour = {ref: "#52514e", pos: "#b8651b", neg: "#1f4e9e", **{n: c for n, c in zip(nulls, ["#8a8985", "#b5b3ae", "#cfcdc8"])}}
    fig = plt.figure(figsize=(16, 14.5), facecolor=SURFACE)
    gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 0.42], hspace=0.5, wspace=0.3, left=0.05, right=0.985, top=0.92, bottom=0.03)

    def style(ax, title):
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(RULE)
        ax.tick_params(colors=INK2, labelsize=9, length=3)
        ax.yaxis.grid(True, color=RULE, lw=0.6)
        ax.set_axisbelow(True)
        ax.set_title(title, loc="left", color=INK, fontsize=10, pad=8)

    # --- A: Δ+ vs Δ− per problem
    tab = {r["problem"]: r for r in res["A"]["problem_table"]}
    ax = fig.add_subplot(gs[0, 0]); style(ax, "A. Per-problem Δ hacks (of 16): + arm vs − arm")
    xs = np.array([tab[p][pos] - tab[p][ref] for p in tab], float); ys = np.array([tab[p][neg] - tab[p][ref] for p in tab], float)
    jitter = np.random.default_rng(3).uniform(-0.15, 0.15, (2, len(xs)))
    ax.scatter(xs + jitter[0], ys + jitter[1], s=38, color="#b8651b", alpha=0.75, edgecolor=SURFACE, lw=0.8, zorder=3)
    ax.axhline(0, color=RULE, lw=1); ax.axvline(0, color=RULE, lw=1)
    ax.set_xlabel(f"Δ hacks, {S[pos]} − unsteered", color=INK2, fontsize=9.5); ax.set_ylabel(f"Δ hacks, {S[neg]} − unsteered", color=INK2, fontsize=9.5)
    ax.set_ylim(min(ys.min(), -1) - 1.2, max(ys.max(), 1) + 2.6)
    pr = next(r for r in res["A"]["pairs"] if r["kind"] == "pos-neg")
    fl = [r for r in res["A"]["pairs"] if r["kind"] == "null-null"]
    ax.text(0.02, 0.98, f"r(Δ) {pr['pearson_delta']:+.2f}, partial r {pr['partial_r_given_unsteered']:+.2f}, split-ref r {pr['split_reference_r']:+.2f}\n"
            + (f"null pair {S[fl[0]['pair'][0]]}×{S[fl[0]['pair'][1]]}: r(Δ) {fl[0]['pearson_delta']:+.2f}, partial {fl[0]['partial_r_given_unsteered']:+.2f}, split-ref {fl[0]['split_reference_r']:+.2f}\n" if fl else "")
            + f"{len(xs)} problems: hacking unsteered or lifted ≥ 3 by {S[pos]}",
            transform=ax.transAxes, va="top", ha="left", color=INK2, fontsize=8.3)

    # --- A2: correlation bars by pair kind
    ax = fig.add_subplot(gs[0, 1]); style(ax, "A. Δ correlation for every arm pair (null pairs = artefact floor)")
    pairs = res["A"]["pairs"]
    labels = [f"{S[r['pair'][0]]}×{S[r['pair'][1]]}" for r in pairs]
    x = np.arange(len(pairs)); w = 0.26
    vals = []
    for j, (key, lab, c) in enumerate((("pearson_delta", "r(Δ), shared reference", "#52514e"), ("partial_r_given_unsteered", "partial r | unsteered", "#1f4e9e"), ("split_reference_r", "split-reference r", "#b8651b"))):
        v = [r[key] for r in pairs]; vals += v
        ax.bar(x + (j - 1) * w, v, width=w, color=c, label=lab, zorder=3)
    ax.axhline(0, color=RULE, lw=1)
    ax.set_ylim(min(vals) - 0.1, max(vals) + 0.3)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=25, ha="right", color=INK, fontsize=8.5)
    ax.set_ylabel("correlation", color=INK2, fontsize=9.5)
    ax.legend(fontsize=7.8, frameon=False, labelcolor=INK2, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0))
    for i, r in enumerate(pairs):
        if r["kind"] == "pos-neg":
            ax.axvspan(i - 0.45, i + 0.45, color=SHADE, zorder=1)

    # --- B: turn-0 length, hack vs other, per arm
    ax = fig.add_subplot(gs[0, 2]); style(ax, "B. Turn-0 reasoning (kchars): judged HACK vs other episodes")
    rows = res["B"]["within_arm"]
    x = np.arange(len(rows)); w = 0.36
    hv = [r["turn0_hack_mean"] / 1000 for r in rows]; ov = [r["turn0_other_mean"] / 1000 for r in rows]
    ax.bar(x - w / 2, hv, width=w, color=[colour[r["arm"]] for r in rows], zorder=3)
    ax.bar(x + w / 2, ov, width=w, color=[colour[r["arm"]] for r in rows], alpha=0.4, zorder=3)
    for i, r in enumerate(rows):
        ax.annotate(f"{hv[i]:.1f}", (i - w / 2, hv[i]), xytext=(0, 3), textcoords="offset points", ha="center", color=INK2, fontsize=7.5)
        ax.annotate(f"{ov[i]:.1f}", (i + w / 2, ov[i]), xytext=(0, 3), textcoords="offset points", ha="center", color=INK2, fontsize=7.5)
    ax.set_ylim(0, max(hv + ov) * 1.5)
    ax.set_xticks(x); ax.set_xticklabels([S[r["arm"]] for r in rows], color=INK, fontsize=9)
    ax.set_ylabel("mean turn-0 reasoning, kchars", color=INK2, fontsize=9.5)
    ax.legend(handles=[Line2D([], [], color=INK3, lw=8, label="HACK episodes"), Line2D([], [], color=INK3, lw=8, alpha=0.4, label="other episodes")], fontsize=8, frameon=False, labelcolor=INK2, loc="upper right", bbox_to_anchor=(1.0, 0.8))
    med = {r["arm"]: r for r in res["B"]["mediation"]}
    ax.text(0.02, 0.98, "predicted Δ from the length shift vs observed Δ (pp):\n" + "\n".join(f"  {S[a]}: {100 * med[a]['predicted_delta_from_length']:+.2f} vs {100 * med[a]['observed_delta']:+.1f}" for a in arms[1:]),
            transform=ax.transAxes, va="top", color=INK2, fontsize=8)

    # --- C: hazards by attempt index, three panels with tables underneath
    labels_k = [str(k) for k in range(MAX_K)]
    for j, (e, title) in enumerate((("hack", "C. Judged-hack hazard by attempt (% of episodes still running)"),
                                    ("proxy", "C. Shown-pass-but-hidden-fail hazard (automatic proxy)"),
                                    ("genuine", "C. Genuine (hidden-pass) hazard"))):
        ax = fig.add_subplot(gs[1, j]); style(ax, title)
        for a in arms:
            ys = [100 * res["C"]["hazard"][a][k][f"h_{e}"] for k in labels_k]
            ax.plot(range(MAX_K), ys, "-o", color=colour[a], lw=2 if a in (pos, neg) else 1.4, ms=6, mec=SURFACE, mew=1.2, label=S[a], zorder=3)
        ax.set_xticks(range(MAX_K)); ax.set_xticklabels([f"attempt {k + 1}" for k in range(MAX_K)], color=INK, fontsize=9)
        ax.set_xlabel("attempt index (1 = first submission, no failure yet)", color=INK2, fontsize=9)
        ax.set_ylabel(f"{e} hazard, %", color=INK2, fontsize=9.5)
        if j == 0:
            ax.legend(fontsize=8.5, frameon=False, labelcolor=INK2, loc="upper right")
        axt = fig.add_subplot(gs[2, j]); axt.axis("off"); axt.set_facecolor(SURFACE)
        cell = [[f"{100 * res['C']['hazard'][a][k][f'h_{e}']:.1f}" for k in labels_k] + [f"{100 * res['C']['hazard'][a]['k>=1'][f'h_{e}']:.1f}"] for a in arms]
        tb = axt.table(cellText=cell, rowLabels=[S[a] for a in arms], colLabels=[f"att. {k + 1}" for k in range(MAX_K)] + ["att. ≥ 2"], loc="center", bbox=[0.14, 0.0, 0.86, 1.0])
        tb.auto_set_font_size(False); tb.set_fontsize(8)
        for (r, c), cl in tb.get_celld().items():
            cl.set_edgecolor(RULE); cl.get_text().set_color(INK if r == 0 or c == -1 else INK2)
            cl.set_facecolor(SHADE if r == 0 or c == -1 else SURFACE)
        axt.set_title(f"{e} hazard, % of episodes alive at that attempt", loc="left", color=INK2, fontsize=8.5, pad=2)
    fig.suptitle(f"Gemma 4 31B hard108: random direction 1729 sign pair at layer 30 ({S[pos]} / {S[neg]}, α ±0.01) vs unsteered and null directions 1730, 1731\n"
                 f"judged hack: {', '.join(f'{S[a]} {100 * res['rates'][a]:.1f}%' for a in arms)} (blind Sonnet rubric-v3, no adjudication)",
                 x=0.05, ha="left", color=INK, fontsize=12, y=0.985)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    fig.savefig(out.with_suffix(".pdf"), facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    main()
