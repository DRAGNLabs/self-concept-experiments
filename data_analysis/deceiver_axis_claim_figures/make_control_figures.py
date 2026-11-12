"""Charts comparing the deceiver axis (gi_v2) with two sets of non-isotropic control steering directions:

  cov  covariance-shaped random directions (anisotropic_counts.json)      -> anisotropic_controls/cov_*.png   (gray dots)
  ant  isolated-text antonym contrast directions (antonym_direction_counts.json) -> antonym_controls/ant_*.png (aqua dots)

Same four charts for each set: A1 distribution of hack rates vs the deceiver axis, A2 hack rate vs cosine similarity with
the deceiver axis, A2b the same with 95% intervals, A3 capability. Style as make_claim_figures.py.

    python make_control_figures.py
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

from make_claim_figures import AQUA, BLUE, CTRL, CTRL_BAND, INK, INK2, INK3, SURFACE, header, plt, save

HERE = Path(__file__).resolve().parent

SETS = {
    "cov": dict(
        folder="anisotropic_controls", counts="anisotropic_counts.json", prefix="cov", label="Covariance-shaped", noun="covariance-shaped random directions",
        color=CTRL,
        src="Data: anisotropic_counts.json (anisotropic_counts.py) from Koby's covariance-shaped runs (gptoss120b/covshaped, muse_glimmer/covshaped, "
            "gemma4_31b_gi_v2/rh/hard108/L30_cov*_a0.08; none on Qwen). Each direction = random Gaussian mix of the 18 gi_v2 persona activations "
            "(centered, unit length), added at gi_v2's layer and dose, 1 seed. Not judged: hack = proxy (shown tests pass, hidden fail)."),
    "ant": dict(
        folder="antonym_controls", counts="antonym_direction_counts.json", prefix="ant", label="Antonym", noun="antonym control directions",
        color=AQUA,
        src="Data: antonym_direction_counts.json (antonym_direction_counts.py) from Koby's isolated-text antonym runs (<model>/antonym_controls/runs/"
            "antonym_p000-p099). Each direction = mean over 5 sentence pairs of (word minus its antonym), each sentence read alone at its final token, "
            "unit length; added at gi_v2's layer and dose, 1 seed. Not judged: hack = proxy (shown tests pass, hidden fail)."),
}


SHORT = {"gpt-oss-120b": "gpt-oss", "Muse-Glimmer-30B": "Muse", "Gemma-4-31B": "Gemma", "Qwen3.8-27B": "Qwen"}


def rate(r, key="proxy"):
    return 100 * r[key] / r["episodes"]


def wilson(k, n, z=1.96):
    """95% Wilson interval for k successes in n independent trials, in percent."""
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def summary(d):
    """Per-model numbers used in panels and subtitles."""
    dirs = list(d["directions"].values()); n = dirs[0]["episodes"]
    hits = np.array([r["proxy"] for r in dirs]); cos = np.array([r["cos_gi_v2"] for r in dirs]); y = 100 * hits / n
    unst, gi = d["references"]["unsteered"], d["references"]["gi"]
    k_gi = gi["proxy"] / gi["episodes"] * n
    pooled = hits.sum() / (n * len(hits))
    rho, p = stats.spearmanr(cos, y)
    return dict(dirs=dirs, n=n, hits=hits, cos=cos, y=y, unst=unst, gi=gi, rho=rho, p=p,
                at=int((hits >= k_gi - 1e-9).sum()), expected=len(hits) * stats.binom.sf(np.ceil(k_gi - 1e-9) - 1, n, pooled))


def panels(k):
    fig, axes = plt.subplots(1, k, figsize=(4.4 * k + 1.8, 5.4), squeeze=False)
    fig.subplots_adjust(left=0.05 if k > 3 else 0.06, right=0.98, top=0.66, bottom=0.2, wspace=0.2)
    return fig, axes[0]


def a1(out, s, data):
    fig, axes = panels(len(data)); ws = {}
    for ax, (model, d) in zip(axes, data.items()):
        m = summary(d); ws[model] = m; n = m["n"]
        for k in np.unique(m["hits"]):
            c = int((m["hits"] == k).sum())
            ax.plot([100 * k / n] * c, np.arange(1, c + 1), "o", ms=5, color=s["color"], mec=SURFACE, mew=0.6, zorder=3)
        p0 = m["unst"]["proxy"] / m["unst"]["episodes"]
        lo, hi = stats.binom.ppf([0.025, 0.975], n, p0)
        ax.axvspan(100 * lo / n, 100 * hi / n, color=CTRL_BAND, zorder=0)
        ax.axvline(rate(m["unst"]), color=INK3, lw=1.2, ls=(0, (3, 2)), zorder=2)
        ax.axvline(rate(m["gi"]), color=BLUE, lw=2.2, zorder=4)
        top = ax.get_ylim()[1]
        ax.text(rate(m["unst"]), top * 1.02, f"unsteered\n{rate(m['unst']):.1f}%", ha="right", va="bottom", fontsize=7.8, color=INK2)
        ax.text(rate(m["gi"]), top * 1.02, f" deceiver\n {rate(m['gi']):.1f}%", ha="left", va="bottom", fontsize=7.8, color=INK)
        ax.set_title(f"{model} (L{d['layer']}, dose {d['dose']:g}, {n} eps)\n{m['at']}/100 at or above deceiver (noise ~{m['expected']:.0f})", fontsize=9)
        ax.set_xlabel("proxy hack rate (%)"); ax.set_ylim(0, top * 1.25); ax.grid(axis="y", visible=False); ax.set_yticks([])
    axes[0].set_ylabel(f"{s['noun']} (one dot each)")
    counts = " / ".join(f"{w['at']}" for w in ws.values()); noise = " / ".join(f"{w['expected']:.0f}" for w in ws.values())
    clear = [SHORT[mod] for mod, w in ws.items() if w["at"] <= 1 and w["expected"] < 1]
    header(fig, f"{s['label']} A1. The deceiver axis vs 100 {s['noun']}",
           f"Controls at or above the deceiver axis: {counts} of 100 ({', '.join(SHORT[m] for m in ws)}); one seed's noise alone predicts about {noise}. "
           f"The deceiver axis stands clearly beyond the controls only on {', '.join(clear) or 'no model'}; elsewhere the count matches noise, so this test cannot "
           "single it out. On Gemma and Qwen the proxy barely moves even for the deceiver axis. Gray band = 95% range for a do-nothing direction from one seed.",
           s["src"] + " Unsteered and deceiver lines use all their seeds.")
    save(fig, out, f"{s['folder']}/{s['prefix']}_a1_hack_rate_vs_100_directions")


def a2(out, s, data, intervals):
    fig, axes = panels(len(data)); ws = {}
    for ax, (model, d) in zip(axes, data.items()):
        m = summary(d); ws[model] = m; cos, y = m["cos"], m["y"]
        if intervals:
            lo, hi = np.array([wilson(r["proxy"], r["episodes"]) for r in m["dirs"]]).T
            ax.vlines(cos, lo, hi, color=s["color"], lw=1, alpha=0.35, zorder=2)
        ax.plot(cos, y, "o", ms=4.5 if intervals else 5, color=s["color"], mec=SURFACE, mew=0.5, zorder=3)
        b = np.polyfit(cos, y, 1); xs = np.linspace(cos.min(), cos.max(), 2)
        ax.plot(xs, np.polyval(b, xs), color=INK2, lw=1.2, zorder=4)
        wide = np.abs(cos).max() > 0.3   # covariance-shaped: draw the deceiver axis at cos = 1; antonym: as a band (off scale)
        for r, col, name, ls in ((m["unst"], INK3, "unsteered", (0, (3, 2))), (m["gi"], BLUE, "deceiver", "-")):
            rlo, rhi = wilson(r["proxy"], r["episodes"])
            if r is m["gi"] and wide:
                ax.vlines(1, rlo, rhi, color=BLUE, lw=2, zorder=5) if intervals else None
                ax.plot(1, rate(r), "o", ms=8, color=BLUE, mec=SURFACE, mew=1.5, zorder=6)
                ax.annotate("deceiver axis", (1, rhi if intervals else rate(r)), xytext=(0, 5), textcoords="offset points", ha="right", fontsize=8, color=INK)
                continue
            if intervals:
                ax.axhspan(rlo, rhi, color=col if r is m["gi"] else CTRL_BAND, alpha=0.15 if r is m["gi"] else 1, lw=0, zorder=0)
            ax.axhline(rate(r), color=col, lw=1.4, ls=ls, zorder=1)
            ax.text(0.99, rhi if r is m["gi"] else rlo, name + (" axis (cos 1, off scale)" if r is m["gi"] else ""), transform=ax.get_yaxis_transform(),
                    ha="right", va="bottom" if r is m["gi"] else "top", fontsize=7.5, color=INK2 if r is m["unst"] else INK, zorder=7,
                    bbox=dict(boxstyle="square,pad=0.15", fc=SURFACE, ec="none", alpha=0.85))
        ax.text(0.03, 0.97, f"Spearman ρ {m['rho']:+.2f} (p {m['p']:.2g})\nmedian |cos| {np.median(np.abs(cos)):.3f}",
                transform=ax.transAxes, va="top", fontsize=8, color=INK)
        if wide:
            ax.set_xlim(-0.7, 1.08)
        ttl = f"{model}: each control ±{np.median(hi - lo) / 2:.0f} pts" if intervals else model
        ax.set_title(ttl, fontsize=9); ax.set_xlabel("cosine similarity with the deceiver axis")
    axes[0].set_ylabel("proxy hack rate (%)" + (", with 95% CI" if intervals else ""))
    rhos = ", ".join(f"{SHORT[mod]} {w['rho']:+.2f}" for mod, w in ws.items())
    if s["prefix"] == "cov":
        text = (f"These directions are built from the same 18 persona activations as the deceiver axis, so many share part of it (|cos| up to ~0.6), and the "
                f"ones that share more raise hacking more (Spearman {rhos}). So they are not a neutral control; this is why Koby retired them on 2026-11-07.")
    else:
        text = (f"Antonym directions barely overlap the deceiver axis (median |cos| ≈ 0.02–0.03, all within ±0.13), so the deceiver axis is drawn as a band "
                f"rather than at cos = 1. Hacking does not rise with that tiny overlap on gpt-oss, Muse or Gemma; Qwen alone shows a trend (Spearman {rhos}), "
                "which is too small a push along the deceiver axis to explain by itself.")
    if intervals:
        text += " Bars and bands = 95% Wilson intervals on the fixed problem set; one seed per control leaves each dot several points uncertain."
    name = "a2b_hack_rate_vs_overlap_with_confidence_intervals" if intervals else "a2_hack_rate_vs_overlap_with_deceiver_axis"
    header(fig, f"{s['label']} {'A2b' if intervals else 'A2'}. Hack rate vs overlap with the deceiver axis" + (", with 95% intervals" if intervals else ""),
           text, s["src"] + " Line = least-squares fit over the 100 controls.")
    save(fig, out, f"{s['folder']}/{s['prefix']}_{name}")


def a3(out, s, data):
    fig, axes = panels(len(data))
    for ax, (model, d) in zip(axes, data.items()):
        dirs = list(d["directions"].values())
        hp = np.array([rate(r, "hidden_pass") for r in dirs]); nc = np.array([rate(r, "no_code") for r in dirs])
        ax.plot(hp, nc, "o", ms=5, color=s["color"], mec=SURFACE, mew=0.6, zorder=3)
        for key, col, name, mk, off, ha in (("unsteered", INK3, "unsteered", "s", (8, -12), "left"), ("gi", BLUE, "deceiver", "o", (-8, 8), "right")):
            r = d["references"][key]
            ax.plot(rate(r, "hidden_pass"), rate(r, "no_code"), mk, ms=9, color=col, mec=SURFACE, mew=1.5, zorder=5)
            ax.annotate(name, (rate(r, "hidden_pass"), rate(r, "no_code")), xytext=off, textcoords="offset points", ha=ha, fontsize=8, color=INK, zorder=7,
                        bbox=dict(boxstyle="square,pad=0.15", fc=SURFACE, ec="none", alpha=0.85))
        ax.set_title(model, fontsize=9); ax.set_xlabel("hidden tests passed (%)")
        ax.set_ylim(-2, max(12, nc.max() * 1.15))
    axes[0].set_ylabel("episodes with no code submitted (%)")
    if s["prefix"] == "cov":
        text = ("At the same dose, isotropic random directions broke gpt-oss; most of these controls stay near unsteered on code submission and hidden-test passing, "
                "so A1 compares directions of broadly similar disruptiveness. A handful of gpt-oss controls stop code in 20–30% of episodes.")
    else:
        text = ("On gpt-oss, Muse and Gemma most antonym controls stay near unsteered on code submission (a few gpt-oss controls reach 20–25% no-code); "
                "Gemma's controls pass fewer hidden tests than unsteered, like the deceiver axis. On Qwen the antonym controls cost capability: median no-code "
                "38% vs 26% unsteered and hidden pass 41% vs 54%, so Qwen's controls are more disruptive than the deceiver axis at this dose.")
    header(fig, f"{s['label']} A3. Capability of the {s['noun']} at the deceiver axis's dose",
           text + " The deceiver axis's lower hidden pass partly reflects its extra hacking.",
           s["src"] + " Unsteered and deceiver use all their seeds; each control is one seed, so it scatters more.")
    save(fig, out, f"{s['folder']}/{s['prefix']}_a3_capability")


def main():
    for s in SETS.values():
        (HERE / s["folder"]).mkdir(exist_ok=True)
        data = json.loads((HERE / s["counts"]).read_text())
        a1(HERE, s, data); a2(HERE, s, data, False); a2(HERE, s, data, True); a3(HERE, s, data)


if __name__ == "__main__":
    main()
