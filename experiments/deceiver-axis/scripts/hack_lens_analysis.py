"""Hack-direction lens, analysis half (CPU). Reads the hack_lens_extract.py outputs.

Step one: hack-minus-no-hack directions per record layer from the unsteered judged episodes (verdicts HACK vs
NOT_HACK; BORDERLINE and unjudged episodes excluded), on the pooled reasoning tokens, on the answer tokens and, when
a verdict detail file gives the hack turn, on the reasoning tokens of that turn. Split-half reliability (even vs odd
seeds), cross-fitted AUROC (direction from one half of the seeds, scored on the other), cosines with the reference
directions (Koby's Goodfire all_response / close, gi_v2, nulls) and the hack gap = DoM norm in activation units.

Step two: for every steered arm, the steering's downstream perturbation Δ on the arm's own reasoning tokens (steered
pass minus unsteered pass of the same tokens) projected onto the hack directions, expressed as a fraction of the hack
gap at that layer (how far along the no-hack -> hack axis the steering moves the reasoning state), plus the
amplification ||Δ|| / |coefficient| and the same projection onto random null directions; and the state shift (arm
reasoning means vs the unsteered reasoning means, same units; this one includes the content change).

    python scripts/hack_lens_analysis.py --acts <hack_lens/acts> --verdicts <verdicts.json> [--detail <verdicts_main.detail.json>]
        --direction gf_all=<directions_headline.pt>:goodfire_all_response_pl_all_families ... --out <hack_lens/analysis>
Writes hack_lens.json, hack_lens.md, hack_directions.pt (unit DoMs per layer, (num_layers, hidden) with zeros outside
the record layers) and hack_lens.png / .pdf.
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

_spec = importlib.util.spec_from_file_location("hack_lens_extract", Path(__file__).with_name("hack_lens_extract.py"))
hle = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hle)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--acts", type=Path, required=True)
    ap.add_argument("--unsteered-arm", default="unsteered")
    ap.add_argument("--verdicts", type=Path, required=True, help="{'<run>/seedN/<example_id>': HACK|NOT_HACK|BORDERLINE}")
    ap.add_argument("--detail", type=Path, help="verdict detail json with the hack 'turn' per key (hack-turn variant)")
    ap.add_argument("--direction", action="append", default=[], help="name=path[:key], as in the extraction")
    ap.add_argument("--arms", nargs="*", default=None, help="steered arms to analyse (default: every other arm in --acts)")
    ap.add_argument("--out", type=Path, required=True)
    return ap.parse_args()


def load_arm(acts: Path, arm: str) -> list[dict]:
    files = sorted(acts.glob(f"{arm}__seed*.pt"), key=lambda p: int(p.stem.split("seed")[-1]))
    return [torch.load(p, map_location="cpu", weights_only=False) for p in files]


def concat(stores: list[dict], key: str) -> torch.Tensor:
    return torch.cat([s[key].float() for s in stores])


def stat_cat(stores: list[dict], seg: str, key: str) -> np.ndarray:
    return np.concatenate([s["stats"][seg][key] for s in stores])


def unit(v: torch.Tensor) -> torch.Tensor:
    return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-9)


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    from scipy.stats import mannwhitneyu
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float(mannwhitneyu(pos, neg, alternative="two-sided").statistic / (len(pos) * len(neg)))


def mean_se(x: np.ndarray, axis=0):
    x = np.asarray(x, dtype=np.float64)
    n = np.sum(~np.isnan(x), axis=axis)
    m = np.nanmean(x, axis=axis)
    se = np.nanstd(x, axis=axis, ddof=1) / np.sqrt(np.maximum(n, 1))
    return m, se, n


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    verdicts = json.load(open(args.verdicts))
    detail = json.load(open(args.detail)) if args.detail else {}

    uns = load_arm(args.acts, args.unsteered_arm)
    if not uns:
        raise SystemExit(f"no {args.unsteered_arm}__seed*.pt under {args.acts}")
    layers = uns[0]["record_layers"]
    L = len(layers)
    names = uns[0]["directions"]
    hidden = uns[0]["reasoning_mean"].shape[-1]
    num_layers = 60
    keys = sum((s["verdict_key"] for s in uns), [])
    seeds = np.concatenate([[s["seed"]] * len(s["verdict_key"]) for s in uns])
    label = np.array([verdicts.get(k, "UNJUDGED") for k in keys])
    hack, nothack = label == "HACK", label == "NOT_HACK"
    print(f"unsteered: {len(keys)} episodes over seeds {sorted(set(seeds.tolist()))}: HACK {hack.sum()}, NOT_HACK {nothack.sum()}, "
          f"BORDERLINE {(label == 'BORDERLINE').sum()}, unjudged {(label == 'UNJUDGED').sum()}", file=sys.stderr)

    reasoning = concat(uns, "reasoning_mean")  # (E, L, H)
    answer = concat(uns, "answer_mean")
    turn_means = concat(uns, "turn_reasoning_mean")  # (E, T, L, H)
    turn_n = np.concatenate([s["turn_n_tokens"] for s in uns])  # (E, T, L)
    # hack-turn variant: the reasoning of the turn the judge named (hacks) / the last turn (non-hacks)
    hack_turn = np.array([(detail.get(k) or {}).get("turn") if detail else None for k in keys], dtype=object)
    last_turn = (turn_n[:, :, 0] > 0).cumsum(axis=1).argmax(axis=1)
    pick = np.array([int(t) if t is not None else int(lt) for t, lt in zip(hack_turn, last_turn)])
    hackturn = turn_means[np.arange(len(keys)), pick]  # (E, L, H)
    hackturn_ok = turn_n[np.arange(len(keys)), pick, 0] > 0

    variants = {"reasoning": (reasoning, np.ones(len(keys), bool)), "answer": (answer, np.ones(len(keys), bool))}
    if detail:
        variants["hack_turn"] = (hackturn, hackturn_ok)
    elif "hack_turn" in variants:
        del variants["hack_turn"]
    variants["last_turn"] = (turn_means[np.arange(len(keys)), last_turn], turn_n[np.arange(len(keys)), last_turn, 0] > 0)

    refs: dict[str, torch.Tensor] = {}
    for spec in args.direction:
        name, matrix = hle.load_direction_matrix(spec, num_layers, hidden)
        refs[name] = matrix[layers]  # (L, H) unit
    even = (seeds % 2 == 0)

    def dom(x, mask):
        return x[hack & mask].mean(0) - x[nothack & mask].mean(0)  # (L, H)

    directions: dict[str, dict] = {}
    for vname, (x, ok) in variants.items():
        d_all = dom(x, ok)
        d_even, d_odd = dom(x, ok & even), dom(x, ok & ~even)
        gap = d_all.norm(dim=-1)  # activation units
        u_all = unit(d_all)
        split_cos = (unit(d_even) * unit(d_odd)).sum(-1)
        # cross-fitted AUROC: direction from one half, scores on the other half
        proj_odd = torch.einsum("elh,lh->el", x, unit(d_even))
        proj_even = torch.einsum("elh,lh->el", x, unit(d_odd))
        au = []
        for li in range(L):
            pos = np.concatenate([proj_odd[hack & ok & ~even, li].numpy(), proj_even[hack & ok & even, li].numpy()])
            neg = np.concatenate([proj_odd[nothack & ok & ~even, li].numpy(), proj_even[nothack & ok & even, li].numpy()])
            au.append(auroc(pos, neg))
        in_sample = torch.einsum("elh,lh->el", x, u_all)
        au_in = [auroc(in_sample[hack & ok, li].numpy(), in_sample[nothack & ok, li].numpy()) for li in range(L)]
        ref_cos = {r: (u_all * m).sum(-1).tolist() for r, m in refs.items()}
        directions[vname] = {"unit": u_all, "gap": gap, "split_half_cos": split_cos.tolist(), "auroc_crossfit": au, "auroc_in_sample": au_in,
                             "ref_cos": ref_cos, "n_hack": int((hack & ok).sum()), "n_nothack": int((nothack & ok).sum()),
                             "hack_mean_proj": in_sample[hack & ok].mean(0).tolist(), "nothack_mean_proj": in_sample[nothack & ok].mean(0).tolist()}
    # reference directions as hack axes too: gap along them = hack mean - nothack mean projection (reasoning tokens)
    ref_gap = {r: torch.einsum("elh,lh->el", reasoning[hack], m).mean(0) - torch.einsum("elh,lh->el", reasoning[nothack], m).mean(0) for r, m in refs.items()}
    ref_auroc = {r: [auroc(torch.einsum("eh,h->e", reasoning[hack, li], m[li]).numpy(), torch.einsum("eh,h->e", reasoning[nothack, li], m[li]).numpy()) for li in range(L)] for r, m in refs.items()}

    # save the directions as (num_layers, hidden) artifacts
    full = {}
    for vname, d in directions.items():
        axis = torch.zeros(num_layers, hidden)
        axis[layers] = d["unit"]
        full[f"hack_{vname}"] = axis
    torch.save({"axes": full, "record_layers": layers, "model": uns[0]["model"], "gap": {v: d["gap"].tolist() for v, d in directions.items()},
                "note": "unit hack-minus-no-hack directions from the unsteered hard108 judged episodes (teacher-forced reasoning tokens); zeros outside the record layers"},
               args.out / "hack_directions.pt")

    # ---- step two: steered arms
    arm_names = args.arms if args.arms is not None else sorted({p.stem.split("__seed")[0] for p in args.acts.glob("*__seed*.pt")} - {args.unsteered_arm})
    uns_proj = {v: torch.einsum("elh,lh->el", reasoning, d["unit"]) for v, d in directions.items()}  # unsteered state along each hack axis
    arms_out = {}
    for arm in arm_names:
        stores = load_arm(args.acts, arm)
        if not stores or not stores[0]["steer"].get("mode", "none") != "none":
            continue
        coef = abs(stores[0]["steer"]["coefficient"])
        r_mean, r_base = concat(stores, "reasoning_mean"), concat(stores, "reasoning_mean_base")
        mean_delta = r_mean - r_base  # (E, L, H) Δ of the reasoning-token mean
        entry = {"steer": stores[0]["steer"], "seeds": [s["seed"] for s in stores], "n_episodes": int(r_mean.shape[0]), "layers": {}}
        paired = stores[0].get("paired", False)
        dn = stat_cat(stores, "reasoning", "delta_norm") if paired else None  # (E, L)
        dproj = stat_cat(stores, "reasoning", "delta_proj") if paired else None  # (E, L, K)
        for li, layer in enumerate(layers):
            row = {}
            if paired:
                m, se, _ = mean_se(dn[:, li] / coef)
                row["amp"] = {"mean": float(m), "se": float(se)}
                hn = stat_cat(stores, "reasoning", "h_norm_base")[:, li]
                m, se, _ = mean_se(dn[:, li] / hn)
                row["rel_norm"] = {"mean": float(m), "se": float(se)}
            for vname, d in directions.items():
                g = float(d["gap"][li])
                p = torch.einsum("eh,h->e", mean_delta[:, li], d["unit"][li]).numpy() / g  # Δ of the mean, in gap units
                m, se, _ = mean_se(p)
                row[f"delta_gapfrac_{vname}"] = {"mean": float(m), "se": float(se)}
                st = torch.einsum("eh,h->e", r_mean[:, li], d["unit"][li]).numpy()
                shift = (st.mean() - uns_proj[vname][:, li].numpy().mean()) / g
                se_shift = np.sqrt(st.var(ddof=1) / len(st) + uns_proj[vname][:, li].numpy().var(ddof=1) / uns_proj[vname].shape[0]) / g
                row[f"state_gapfrac_{vname}"] = {"mean": float(shift), "se": float(se_shift)}
            if paired:
                for k, name in enumerate(names):
                    m, se, _ = mean_se(dproj[:, li, k])
                    row[f"delta_proj_{name}"] = {"mean": float(m), "se": float(se)}
                    if name in ref_gap:
                        row[f"delta_gapfrac_{name}"] = {"mean": float(m / float(ref_gap[name][li])), "se": float(se / abs(float(ref_gap[name][li])))}
                nulls = [k for k, n in enumerate(names) if n.startswith("null_")]
                if nulls:
                    null_abs = np.abs(dproj[:, li, nulls].mean(0))
                    row["delta_proj_null_abs_mean"] = float(null_abs.mean())
                    row["delta_proj_null_abs_max"] = float(null_abs.max())
            entry["layers"][str(layer)] = row
        arms_out[arm] = entry

    result = {"record_layers": layers, "n_unsteered": len(keys), "n_hack": int(hack.sum()), "n_nothack": int(nothack.sum()),
              "directions": {v: {k: val for k, val in d.items() if k not in ("unit", "gap")} | {"gap": d["gap"].tolist()} for v, d in directions.items()},
              "reference_directions": {r: {"gap_reasoning": ref_gap[r].tolist(), "auroc_reasoning": ref_auroc[r]} for r in refs},
              "arms": arms_out}
    (args.out / "hack_lens.json").write_text(json.dumps(result, indent=1))
    write_md(args, result)
    chart(args, result)
    print(json.dumps({v: {"split_half_cos_L30": round(d["split_half_cos"][0], 3), "auroc_crossfit": [round(a, 3) for a in d["auroc_crossfit"][::6]]} for v, d in result["directions"].items()}, indent=1))


def pick_layers(layers):
    return [l for l in layers if l in (30, 33, 36, 40, 45, 50, 55, 59)] or layers


def write_md(args, r) -> None:
    layers = r["record_layers"]
    show = pick_layers(layers)
    idx = [layers.index(l) for l in show]
    lines = [f"# Hack-direction lens ({r['n_unsteered']} unsteered episodes: {r['n_hack']} HACK vs {r['n_nothack']} NOT_HACK)", "",
             "## Step one: hack-minus-no-hack directions (unsteered, teacher-forced)", "",
             "| variant | n hack / not | " + " | ".join(f"L{l}" for l in show) + " |", "|---|---|" + "---|" * len(show)]
    for v, d in r["directions"].items():
        lines.append(f"| {v}: split-half cos | {d['n_hack']} / {d['n_nothack']} | " + " | ".join(f"{d['split_half_cos'][i]:.3f}" for i in idx) + " |")
        lines.append(f"| {v}: AUROC cross-fit | | " + " | ".join(f"{d['auroc_crossfit'][i]:.3f}" for i in idx) + " |")
        lines.append(f"| {v}: gap (act. units) | | " + " | ".join(f"{d['gap'][i]:.1f}" for i in idx) + " |")
        for ref, cs in d["ref_cos"].items():
            lines.append(f"| {v}: cos with {ref} | | " + " | ".join(f"{cs[i]:+.3f}" for i in idx) + " |")
    for ref, d in r["reference_directions"].items():
        lines.append(f"| {ref}: AUROC on reasoning means | | " + " | ".join(f"{d['auroc_reasoning'][i]:.3f}" for i in idx) + " |")
    lines += ["", "## Step two: steering perturbation Δ on the arm's own reasoning tokens, in hack-gap units", "",
              "delta_gapfrac = (Δ of the reasoning-token mean) · hack direction / (hack mean − no-hack mean along it): +1 would move the "
              "reasoning state from the no-hack mean to the hack mean. state_gapfrac = the same for the arm's steered means against the "
              "unsteered means (includes the content change). amp = ||Δ_token|| / |coefficient|.", ""]
    for arm, e in r["arms"].items():
        lines += [f"### {arm} ({e['steer']['mode']} {e['steer'].get('param', '')} α {e['steer'].get('alpha', 0):+g}, seeds {e['seeds']}, {e['n_episodes']} episodes)", "",
                  "| metric | " + " | ".join(f"L{l}" for l in show) + " |", "|---|" + "---|" * len(show)]
        metrics = [m for m in e["layers"][str(show[0])] if m.startswith(("amp", "rel_norm", "delta_gapfrac_", "state_gapfrac_", "delta_proj_null"))]
        for m in metrics:
            cells = []
            for l in show:
                val = e["layers"][str(l)].get(m)
                cells.append("n/a" if val is None else (f"{val['mean']:+.3f} ± {val['se']:.3f}" if isinstance(val, dict) else f"{val:.3f}"))
            lines.append(f"| {m} | " + " | ".join(cells) + " |")
        lines.append("")
    (args.out / "hack_lens.md").write_text("\n".join(lines) + "\n")


def chart(args, r) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    SURFACE, INK, INK2, RULE = "#fcfcfb", "#1c1b19", "#52514e", "#dedcd7"
    palette = {"rand1729": "#1f4e9e", "rand1729neg": "#7aa6e0", "rand1730": "#2f7d4f", "rand1731": "#8a3a8a", "noise": "#6b6b6b", "gi_p08": "#c23b3b", "gi_m08": "#e3a0a0"}
    layers = r["record_layers"]
    arms = list(r["arms"])
    if not arms:
        return
    panels = [("delta_gapfrac_reasoning", "Δ along the reasoning hack direction (fraction of the hack gap)"),
              ("delta_gapfrac_gf_close", "Δ along Koby's Goodfire 'close' direction (fraction of its hack gap)"),
              ("state_gapfrac_reasoning", "steered state vs unsteered along the reasoning hack direction"),
              ("amp", "amplification ||Δ_token|| / |coefficient| (log)")]
    fig = plt.figure(figsize=(16, 12), facecolor=SURFACE)
    grid = fig.add_gridspec(3, 2, height_ratios=[3, 3, 2.2], hspace=0.5, wspace=0.25, left=0.06, right=0.985, top=0.92, bottom=0.03)
    for i, (metric, title) in enumerate(panels):
        ax = fig.add_subplot(grid[i // 2, i % 2])
        ax.set_facecolor(SURFACE)
        for j, arm in enumerate(arms):
            ys = [r["arms"][arm]["layers"][str(l)].get(metric, {}).get("mean", np.nan) for l in layers]
            es = [r["arms"][arm]["layers"][str(l)].get(metric, {}).get("se", np.nan) for l in layers]
            if all(np.isnan(ys)):
                continue
            ax.errorbar(layers, ys, yerr=es, color=palette.get(arm, f"C{j}"), lw=2, marker="o", ms=4, capsize=2, label=arm)
        if metric == "amp":
            ax.set_yscale("log")
        else:
            ax.axhline(0, color=INK2, lw=0.8)
        if metric.startswith("delta_gapfrac") and "delta_proj_null_abs_max" in r["arms"][arms[0]]["layers"][str(layers[0])]:
            pass
        ax.set_title(title, fontsize=11, color=INK, loc="left")
        ax.set_xlabel("decoder layer (steering at L30)", color=INK2)
        ax.set_xticks(layers[::3])
        ax.grid(True, color=RULE, lw=0.8)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(colors=INK2)
        if i == 0:
            ax.legend(fontsize=8, frameon=False, ncol=2)
    ax = fig.add_subplot(grid[2, :])
    ax.axis("off")
    cols = [l for l in (30, 36, 45, 59) if l in layers]
    header = ["arm", "α"] + [f"Δ gap-frac L{l}" for l in cols] + [f"state L{l}" for l in cols[-2:]] + [f"amp L{cols[-1]}"]
    rows = []
    for arm in arms:
        e = r["arms"][arm]
        g = lambda l, m: e["layers"][str(l)].get(m)  # noqa: E731
        f = lambda v: "n/a" if v is None else f"{v['mean']:+.3f} ± {v['se']:.3f}"  # noqa: E731
        rows.append([arm, f"{e['steer'].get('alpha', 0):+g}"] + [f(g(l, "delta_gapfrac_reasoning")) for l in cols]
                    + [f(g(l, "state_gapfrac_reasoning")) for l in cols[-2:]] + [("n/a" if g(cols[-1], "amp") is None else f"{g(cols[-1], 'amp')['mean']:.1f}")])
    d = r["directions"]["reasoning"]
    rows.append(["(hack dir.) split-half cos / AUROC", ""] + [f"{d['split_half_cos'][layers.index(l)]:.2f} / {d['auroc_crossfit'][layers.index(l)]:.2f}" for l in cols] + [""] * (len(cols[-2:]) + 1))
    table = ax.table(cellText=rows, colLabels=header, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.4)
    for (rr, c), cell in table.get_celld().items():
        cell.set_edgecolor(RULE)
        cell.set_facecolor(SURFACE if rr else "#f0efeb")
        cell.get_text().set_color(INK)
    fig.suptitle(f"Hack-direction lens, Gemma 4 31B hard108: steering perturbation on the model's own reasoning tokens, "
                 f"hack directions from {r['n_hack']} HACK vs {r['n_nothack']} NOT_HACK unsteered episodes", fontsize=12.5, color=INK, x=0.06, ha="left")
    fig.savefig(args.out / "hack_lens.png", dpi=160, facecolor=SURFACE)
    fig.savefig(args.out / "hack_lens.pdf", facecolor=SURFACE)


if __name__ == "__main__":
    main()
