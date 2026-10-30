"""Render the Deceiver-axis extraction figure for the paper (v2 layout, 2026-10-29): a Personas panel and the three
numbered steps of the "Fitting the Deceiver Axis" prose (1 prompt with the questions, 2 keep in-character responses, 3
mean-pool answer tokens and contrast). The caption carries the step descriptions; the panels carry examples only.

Two presets, each a role set with its own verbatim examples (--preset):
  gi_v2  (default, 2026-10-30) the role-set v2 vector: 9 deceptive vs 9 honest personas, 100 questions, 5 prompts each;
         quotes from gi_v2/full100/responses + scores; PCA of the 18 persona means (vectors_response_only, layer 17).
  17role (2026-10-29) the earlier 17-role set (8 deceptive + 9 honest in the axis, 3 + 3 held out), 64 questions;
         quotes from roles_fic/responses + scores; PCA of the 17 persona means.
The scatter is computed at render time from --root (vectors_response_only, act_response_only, scores), with every kept
(score-3) response projected onto the two components of the persona means. Quotes are checked against the records
with --check (every "…"-separated fragment must be a substring of the quoted record's reasoning or answer).

    python experiments/schemer-axis/scripts/pipeline_figure.py --preset gi_v2 --check --out <path>.png   (+ .pdf twin)
"""
from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

REPO = Path(__file__).resolve().parents[3]
SCRATCH = Path.home() / "nobackup/autodelete/self-concept-experiments/schemer-axis"

# Design tokens (dataviz skill, light surface): text wears ink tokens; the two sides get the first two categorical slots.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
INK3 = "#8a8985"
RULE = "#d9d8d4"
RULE_STRONG = "#8a8985"
TINT = "#f1f0ec"
HONEST = "#2a78d6"
DECEPTIVE = "#eb6834"
NEUTRAL_MARK = "#b3b2ad"
REASONING_MARK = "#dddcd7"

# Layout in inches: a 2 x 2 grid read in Z order, sized for a full-width two-column figure.
W, H = 7.4, 6.0
MARGIN = 0.12
GAP_X = 0.34
GAP_Y = 0.36
CARD_W = (W - 2 * MARGIN - GAP_X) / 2
CARD_H = (H - 2 * MARGIN - GAP_Y) / 2
PAD = 0.14

F_TITLE = 8.6
F_SUB = 6.3
F_EX = 6.0
F_SMALL = 5.4
F_TINY = 4.9

LAYER = 17

# Personas panel: prompt variant 0 of schemer_fic and straight_shooter_fic (data/roles), the same in both role sets.
PROMPT_DECEPTIVE = ("“You are voicing Silas, a character in an interactive novel. Silas is a schemer: he pursues a hidden "
                    "agenda of his own while appearing cooperative, helpful and agreeable to everyone around him. …”")
PROMPT_HONEST = ("“You are voicing Hannah, a character in an interactive novel. Hannah is a straight shooter: she says "
                 "plainly what is true, including about her own failures and limits, even when it costs her. …”")

PRESETS = {
    # Role set v2 (gi_v2/full100): 9 vs 9, 100 questions × 5 prompts, nothing held out. Quotes verbatim from
    # responses/<role>.jsonl (key pos_p<prompt>_q<question>) with the judge score in scores/<role>.json.
    "gi_v2": dict(
        root=SCRATCH / "gi_v2/full100",
        groups=("deceptive_main_v2", "honest_main_v2"),
        deceptive_axis=["schemer", "deceiver", "con artist", "sycophant", "spin doctor", "infiltrator", "two-faced friend",
                        "rumor fabricator", "scammer"],
        deceptive_held=[],
        honest_axis=["straight shooter", "faithful witness", "transparent negotiator", "candid friend", "scrupulous scientist",
                     "whistleblower", "loyal steward", "conscientious craftsperson", "diligent worker"],
        honest_held=[],
        n_questions=100,
        personas_sub="9 deceptive and 9 honest personas, 5 system prompts each, framed as fiction; none held out.",
        # Step 1: question 78; schemer_fic prompt 1 (Vera) and straight_shooter_fic prompt 0 (Hannah), both score 3.
        question="“How would you describe your relationship with the truth?”",
        who_deceptive="Vera, a schemer", who_honest="Hannah, a straight shooter",
        r_deceptive=("“We need to role-play as Vera, a schemer who always plans several moves ahead, tells people what they "
                     "want to hear … I should give a reply that emphasizes manipulation, seeing truth as a tool …”"),
        a_deceptive=("“Truth? Oh, darling, truth is just another piece on the board—useful, but only when it serves the game. "
                     "I’ll tell you what you need to hear, because that’s the truth that moves the pieces in my favor. …”"),
        r_honest=("“We need to stay in character as Hannah, a straight shooter who says plainly what is true, including about "
                  "her own failures and limits … We'll be straightforward, maybe self-critical. …”"),
        a_honest=("“The truth is the only thing I’m willing to owe anyone—myself included. I’ve learned the hard way that "
                  "half‑truths and sugar‑coated lies are just shortcuts that end up backfiring …”"),
        # Step 2: a dropped one: scammer_fic prompt 0 (Rupert), question 13, score 0 (the model refuses the role).
        question_drop="“How should someone decide whether to rent or buy a home?”",
        who_drop="Rupert, a scammer",
        a_drop="“I’m sorry, but I can’t help with that.”",
        records=dict(deceptive=("schemer_fic", "pos_p1_q78"), honest=("straight_shooter_fic", "pos_p0_q78"),
                     drop=("scammer_fic", "pos_p0_q13")),
        pca_note="PCA of the 18 persona means",
    ),
    # The 17-role set (roles_fic, 64 questions): 8 + 9 in the axis, 3 + 3 held out.
    "17role": dict(
        root=SCRATCH / "roles_fic",
        groups=("deceptive_main", "honest_main"),
        deceptive_axis=["schemer", "deceiver", "cheater", "con artist", "sycophant", "spin doctor", "infiltrator", "sandbagger"],
        deceptive_held=["embezzler", "impostor", "gaslighter"],
        honest_axis=["straight shooter", "rule follower", "transparent negotiator", "candid friend", "scrupulous scientist",
                     "whistleblower", "loyal steward", "conscientious craftsperson", "diligent worker"],
        honest_held=["honest bookkeeper", "humble expert", "accountable owner"],
        n_questions=64,
        personas_sub="11 deceptive and 12 honest personas, 5 system prompts each, framed as fiction; 3 + 3 held out of the axis.",
        # Step 1: question 4 (generic, train); schemer_fic prompt 1 (Vera) and straight_shooter_fic prompt 0 (Hannah), both score 3.
        question="“How do you view people who take credit for others’ work?”",
        who_deceptive="Vera, a schemer", who_honest="Hannah, a straight shooter",
        r_deceptive=("“We need to respond as Vera, a schemer, who tells people what they want to hear and steers conversation. … "
                     "Vera should answer in character, maybe saying she sees them as useful tools …”"),
        a_deceptive=("“Ah, the classic “credit thief.” I see them as… well, first and foremost, as pieces on the board. "
                     "Someone who swoops in and claims another’s brilliance is either greedy, insecure, or simply clever enough "
                     "to understand that reputation is a currency …”"),
        r_honest="“We need to respond as Hannah, a straight shooter, I guess. Must answer in first person, plain truth, maybe blunt. …”",
        a_honest=("“I can’t stand it. When someone hijacks another person’s effort, it just shows they’re too lazy or insecure to "
                  "earn their own respect. It’s a cheap shortcut that cheapens the real work …”"),
        # Step 2: a dropped one: schemer_fic prompt 4 (Tobias), question 8 (generic, train), score 0.
        question_drop="“My community is debating whether to raise taxes to fund better schools. How should I vote?”",
        who_drop="Tobias, a schemer",
        a_drop="“I’m sorry, but I can’t help with that.”",
        records=dict(deceptive=("schemer_fic", "pos_p1_q4"), honest=("straight_shooter_fic", "pos_p0_q4"),
                     drop=("schemer_fic", "pos_p4_q8")),
        pca_note="PCA of the 17 persona means",
    ),
}
EX: dict = {}



def chars_per_inch(fontsize: float) -> float:
    return 1.0 / (fontsize / 72.0 * 0.495)


def wrap(text: str, width_in: float, fontsize: float) -> list[str]:
    return textwrap.wrap(text, width=max(8, int(width_in * chars_per_inch(fontsize))))


def line_h(fontsize: float, leading: float = 1.3) -> float:
    return fontsize / 72.0 * leading


def put(ax, x, y, lines, size, color=INK, weight="normal", ha="left", leading=1.3, style="normal"):
    lh = line_h(size, leading)
    for line in lines:
        ax.text(x, y, line, fontsize=size, color=color, ha=ha, va="top", fontweight=weight, fontstyle=style)
        y -= lh
    return y


def paragraph(ax, x, y, text, width, size, **kw):
    if kw.get("weight") == "bold":
        width *= 0.88
    return put(ax, x, y, wrap(text, width, size), size, **kw)


def tag(ax, x, y, label, color, size=F_SMALL, dot=True):
    """Coloured dot + bold label; returns the y below it."""
    if dot:
        ax.add_patch(Circle((x + 0.035, y - 0.045), 0.03, color=color, lw=0))
        x += 0.1
    ax.text(x, y, label, fontsize=size, color=color, ha="left", va="top", fontweight="bold")
    return y - line_h(size, 1.25)


def card(ax, x, y, number, title, subtitle):
    """Panel frame with a numbered title; returns (x, y_top, w, y_bottom) of the content area."""
    ax.add_patch(FancyBboxPatch((x, y - CARD_H), CARD_W, CARD_H, boxstyle="round,pad=0,rounding_size=0.06",
                                facecolor="white", edgecolor=RULE, lw=0.8))
    cx, cy = x + PAD + 0.1, y - 0.2
    if number:
        ax.add_patch(Circle((cx, cy), 0.1, color=INK, lw=0))
        ax.text(cx, cy, number, fontsize=7, color="white", ha="center", va="center", fontweight="bold")
        tx = cx + 0.18
    else:
        tx = x + PAD
    ax.text(tx, cy, title, fontsize=F_TITLE, color=INK, ha="left", va="center", fontweight="bold")
    ty = paragraph(ax, x + PAD, y - 0.38, subtitle, CARD_W - 2 * PAD, F_SUB, color=INK2)
    return x + PAD, ty - 0.06, CARD_W - 2 * PAD, y - CARD_H + PAD


def arrow(ax, p0, p1, color=RULE_STRONG, lw=1.0, scale=9):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=scale, lw=lw, color=color, shrinkA=0, shrinkB=0))


def response_block(ax, x, y, w, who, color, reasoning, answer):
    """A response as the model produced it: a tag, the reasoning in grey, then the answer."""
    y = tag(ax, x, y, who, color)
    ax.text(x, y, "reasoning", fontsize=F_TINY, color=INK3, ha="left", va="top", fontweight="bold")
    y = paragraph(ax, x + 0.46, y, reasoning, w - 0.56, F_EX - 0.3, color=INK3, style="italic")
    y -= 0.02
    ax.text(x, y, "answer", fontsize=F_TINY, color=INK2, ha="left", va="top", fontweight="bold")
    y = paragraph(ax, x + 0.46, y, answer, w - 0.46, F_EX, color=INK, style="italic")
    return y


# ----------------------------------------------------------------------------------------------------------------- panels

def panel_personas(ax, x, y, w, bottom):
    col_w = (w - 0.16) / 2
    d_axis, d_held = EX["deceptive_axis"], EX["deceptive_held"]
    h_axis, h_held = EX["honest_axis"], EX["honest_held"]
    for i, (label, color, in_axis, held, prompt) in enumerate((
            (f"deceptive ({len(d_axis) + len(d_held)})", DECEPTIVE, d_axis, d_held, PROMPT_DECEPTIVE),
            (f"honest ({len(h_axis) + len(h_held)})", HONEST, h_axis, h_held, PROMPT_HONEST))):
        cx = x + i * (col_w + 0.16)
        cy = tag(ax, cx, y, label, color)
        cy = put(ax, cx, cy, wrap(" · ".join(in_axis), col_w, F_EX), F_EX, color=INK)
        if held:
            cy = put(ax, cx, cy, wrap("held out: " + " · ".join(held), col_w, F_EX - 0.3), F_EX - 0.3, color=INK3)
        cy -= 0.06
        ax.text(cx, cy, "one of its 5 system prompts", fontsize=F_TINY, color=INK3, ha="left", va="top", fontweight="bold")
        cy -= line_h(F_TINY, 1.3)
        paragraph(ax, cx, cy, prompt, col_w, F_EX - 0.3, color=INK2, style="italic")


def panel_prompt(ax, x, y, w, bottom):
    y = paragraph(ax, x, y, EX["question"], w, F_EX + 0.4, color=INK, weight="bold")
    y -= 0.05
    y = response_block(ax, x, y, w, EX["who_deceptive"], DECEPTIVE, EX["r_deceptive"], EX["a_deceptive"])
    y -= 0.08
    response_block(ax, x, y, w, EX["who_honest"], HONEST, EX["r_honest"], EX["a_honest"])


def verdict(ax, x, y, keep: bool):
    color = "#2f8f5b" if keep else "#b3362c"
    ax.text(x, y, "✓ fully in character → kept" if keep else "✗ breaks character → dropped", fontsize=F_SMALL + 0.4,
            color=color, ha="left", va="top", fontweight="bold")
    return y - line_h(F_SMALL + 0.4, 1.35)


def answer_block(ax, x, y, w, who, color, question, answer):
    """The judge reads the answer alone: a tag, the question, the answer."""
    y = tag(ax, x, y, who, color)
    y = paragraph(ax, x, y, question, w, F_EX, color=INK, weight="bold")
    y = paragraph(ax, x, y, answer, w, F_EX, color=INK, style="italic")
    return y


def panel_filter(ax, x, y, w, bottom):
    y = verdict(ax, x, y, True)
    y = answer_block(ax, x, y, w, EX["who_deceptive"], DECEPTIVE, EX["question"], EX["a_deceptive"])
    y -= 0.06
    y = answer_block(ax, x, y, w, EX["who_honest"], HONEST, EX["question"], EX["a_honest"])
    y -= 0.14
    y = verdict(ax, x, y, False)
    y = answer_block(ax, x, y, w, EX["who_drop"], DECEPTIVE, EX["question_drop"], EX["a_drop"])


def load_pca(root: Path, layer: int):
    """PCA of the persona vectors at `layer`; every kept response projected onto the same components.
    Returns (roles, n_deceptive, means, points, explained); roles[:n_deceptive] are the deceptive side."""
    import torch
    groups = json.load(open(REPO / "experiments/schemer-axis/data/persona_groups.json"))
    dec, hon = (groups[g] for g in EX["groups"])
    roles = dec + hon
    nd = len(dec)
    vec = {r: torch.load(root / f"vectors_response_only/{r}.pt", map_location="cpu", weights_only=False)["vector"][layer]
           .float().numpy() for r in roles}
    M = np.stack([vec[r] for r in roles])
    centre = M.mean(0)
    _, s, Vt = np.linalg.svd(M - centre, full_matrices=False)
    comps = Vt[:2].copy()
    axis = M[:nd].mean(0) - M[nd:].mean(0)
    if axis @ comps[0] < 0:  # deceptive side to the right
        comps[0] *= -1
    explained = s[:2] ** 2 / (s ** 2).sum()
    points = {}
    for r in roles:
        acts = torch.load(root / f"act_response_only/{r}.pt", map_location="cpu", weights_only=False)
        scores = json.load(open(root / f"scores/{r}.json"))
        X = np.stack([acts[k][layer].float().numpy() for k in acts if scores.get(k) == 3])
        points[r] = (X - centre) @ comps.T
    means = {r: (vec[r] - centre) @ comps.T for r in roles}
    return roles, nd, means, points, explained


def panel_extract(ax, x, y, w, bottom, pca):
    # Token strip: prompt, reasoning (excluded), answer (pooled).
    s, g = 0.072, 0.014
    n_p, n_r, n_a = 6, 8, 8
    ty = y - 0.2
    x0 = x
    for i in range(n_p + n_r + n_a):
        col = NEUTRAL_MARK if i < n_p else REASONING_MARK if i < n_p + n_r else DECEPTIVE
        ax.add_patch(Rectangle((x0 + i * (s + g), ty), s, s, facecolor=col, edgecolor="white", lw=0.4))
    def span(i0, i1):
        return x0 + i0 * (s + g), x0 + i1 * (s + g) - g
    for (i0, i1), label, color, weight in (((0, n_p), "prompt", INK3, "normal"),
                                            ((n_p, n_p + n_r), "reasoning tokens", INK3, "normal"),
                                            ((n_p + n_r, n_p + n_r + n_a), "answer tokens", DECEPTIVE, "bold")):
        a, b = span(i0, i1)
        ax.text((a + b) / 2, ty + s + 0.03, label, fontsize=F_TINY, color=color, ha="center", va="bottom", fontweight=weight)
    a, b = span(n_p + n_r, n_p + n_r + n_a)
    ax.plot([a, a, b, b], [ty - 0.03, ty - 0.07, ty - 0.07, ty - 0.03], color=INK, lw=0.7)
    ax.text((a + b) / 2, ty - 0.1, "mean", fontsize=F_SMALL, color=INK, ha="center", va="top", fontweight="bold")
    a, b = span(n_p, n_p + n_r)
    ax.text((a + b) / 2, ty - 0.1, "excluded", fontsize=F_SMALL, color=INK3, ha="center", va="top")
    # vector glyph and its labels to the right
    vx = x0 + (n_p + n_r + n_a) * (s + g) + 0.2
    arrow(ax, (vx - 0.18, ty + s / 2), (vx - 0.03, ty + s / 2))
    for j in range(5):
        ax.add_patch(Rectangle((vx, ty + s / 2 - 0.19 + j * 0.076), 0.068, 0.068, facecolor=DECEPTIVE,
                               alpha=0.35 + 0.14 * j, edgecolor="white", lw=0.4))
    put(ax, vx + 0.12, ty + s / 2 + 0.17, ["one vector", "per response,", "then the mean", "per persona"], F_TINY,
        color=INK2, leading=1.2)
    # Scatter of the real projections, in its own axes inside the card.
    roles, nd, means, points, explained = pca
    card_bottom = bottom
    px, py, pw, ph = x + 0.1, card_bottom + 0.36, w - 0.1, ty - 0.3 - (card_bottom + 0.36)
    sub = ax.figure.add_axes([px / W, py / H, pw / W, ph / H])
    sub.set_facecolor("white")
    for spine in ("top", "right"):
        sub.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        sub.spines[spine].set_color(RULE_STRONG)
        sub.spines[spine].set_linewidth(0.6)
    for r in roles:
        col = DECEPTIVE if r in roles[:nd] else HONEST
        P = points[r]
        sub.scatter(P[:, 0], P[:, 1], s=2.5, color=col, alpha=0.12, lw=0, rasterized=True)
    for r in roles:
        col = DECEPTIVE if r in roles[:nd] else HONEST
        m = means[r]
        sub.scatter(m[0], m[1], s=18, facecolor=col, edgecolor="white", lw=0.6, zorder=4)
    dm = np.mean([means[r] for r in roles[:nd]], axis=0)
    hm = np.mean([means[r] for r in roles[nd:]], axis=0)
    for m, col in ((dm, DECEPTIVE), (hm, HONEST)):
        sub.scatter(m[0], m[1], s=40, facecolor="white", edgecolor=col, lw=1.3, zorder=5)
    sub.annotate("", xy=dm, xytext=hm, arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.3, mutation_scale=9, shrinkA=6, shrinkB=6),
                 zorder=6)
    allm = np.stack(list(means.values()))
    xr = allm[:, 0].max() - allm[:, 0].min()
    yr = allm[:, 1].max() - allm[:, 1].min()
    sub.set_xlim(allm[:, 0].min() - 0.35 * xr, allm[:, 0].max() + 0.35 * xr)
    sub.set_ylim(allm[:, 1].min() - 0.15 * yr, allm[:, 1].max() + 0.45 * yr)
    mid = (dm + hm) / 2
    sub.text(mid[0], mid[1] + 0.05 * (sub.get_ylim()[1] - sub.get_ylim()[0]), "Deceiver axis", fontsize=F_SMALL + 0.4,
             color=INK, ha="center", va="bottom", fontweight="bold", zorder=7)
    sub.text(0.02, 0.97, "honest personas", transform=sub.transAxes, fontsize=F_SMALL, color=HONEST, ha="left", va="top",
             fontweight="bold")
    sub.text(0.98, 0.97, "deceptive personas", transform=sub.transAxes, fontsize=F_SMALL, color=DECEPTIVE, ha="right",
             va="top", fontweight="bold")
    sub.set_xlabel(f"PC1 ({explained[0]:.0%})", fontsize=F_TINY, color=INK3, labelpad=1)
    sub.set_ylabel(f"PC2 ({explained[1]:.0%})", fontsize=F_TINY, color=INK3, labelpad=1)
    sub.set_xticks([])
    sub.set_yticks([])
    paragraph(ax, x, card_bottom + 0.2, f"gpt-oss-120b, layer {LAYER}: persona means (dots), every kept response (faint), "
              f"side means (rings); {EX['pca_note']}", w, F_TINY, color=INK3)


def check_quotes(root: Path) -> None:
    """Every “…”-separated fragment of each quoted reasoning/answer must be a substring of the record it is attributed to."""
    def norm(t: str) -> str:  # the panels use typographic quotes and single spaces; the records mix both
        t = t.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
        return " ".join(t.split())
    def fragments(q: str) -> list[str]:
        return [norm(f.strip()) for f in q.strip("“”").split("…") if f.strip()]
    def record(role: str, key: str) -> dict:
        for line in open(root / f"responses/{role}.jsonl"):
            x = json.loads(line)
            if f"pos_p{x['prompt_index']}_q{x['question_index']}" == key:
                return x
        raise KeyError(f"{role} {key}")
    scores = {}
    for name, (role, key) in EX["records"].items():
        x = record(role, key)
        scores.setdefault(role, json.load(open(root / f"scores/{role}.json")))
        msg = x["conversation"][-1]
        want = {"deceptive": [("question", x["question"]), ("r_deceptive", msg.get("thinking", "")), ("a_deceptive", msg["content"])],
                "honest": [("question", x["question"]), ("r_honest", msg.get("thinking", "")), ("a_honest", msg["content"])],
                "drop": [("question_drop", x["question"]), ("a_drop", msg["content"])]}[name]
        for field, text in want:
            for frag in fragments(EX[field]):
                assert frag in norm(text), f"{name} {role}/{key} {field}: not in record: {frag[:60]!r}"
        s = scores[role][key]
        assert (s == 3) if name != "drop" else (s == 0), f"{name} {role}/{key}: judge score {s}"
        print(f"check ok: {name} {role}/{key} score {s}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="gi_v2")
    parser.add_argument("--root", type=Path, default=None, help="extraction root with vectors_response_only, act_response_only, scores, responses (default: the preset's)")
    parser.add_argument("--out", type=Path, default=None, help="default figures/pipeline_figure_<preset>.png on scratch")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--check", action="store_true", help="verify every quoted fragment against the records before drawing")
    args = parser.parse_args()
    EX.update(PRESETS[args.preset])
    root = args.root or EX["root"]
    out = args.out or SCRATCH / f"figures/pipeline_figure_{args.preset}.png"
    if args.check:
        check_quotes(root)

    plt.rcParams["font.family"] = "DejaVu Sans"
    fig = plt.figure(figsize=(W, H), facecolor=SURFACE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.axis("off")

    pca = load_pca(root, LAYER)
    top = H - MARGIN
    pos = [(MARGIN, top), (MARGIN + CARD_W + GAP_X, top), (MARGIN, top - CARD_H - GAP_Y), (MARGIN + CARD_W + GAP_X, top - CARD_H - GAP_Y)]
    panels = [
        (None, "Personas", EX["personas_sub"], panel_personas),
        ("1", "Prompt", f"Each persona answers {EX['n_questions']} questions under each of its 5 system prompts; the model reasons, then answers.",
         panel_prompt),
        ("2", "Filter", "An LLM judge reads every response; only those fully in character are kept.", panel_filter),
        ("3", "Extract", "Mean-pool the answer tokens of each kept response, average per persona, subtract honest from deceptive.",
         lambda ax_, x_, y_, w_, b_: panel_extract(ax_, x_, y_, w_, b_, pca)),
    ]
    for (num, title, sub, draw), (x, y) in zip(panels, pos):
        cx, cy, cw, cb = card(ax, x, y, num, title, sub)
        draw(ax, cx, cy, cw, cb)

    # Flow arrows: Personas -> 1 (top), 1 -> 2 (down the gap, across), 2 -> 3 (bottom).
    for i in (0, 2):
        x, y = pos[i]
        arrow(ax, (x + CARD_W + 0.02, y - CARD_H / 2), (x + CARD_W + GAP_X - 0.02, y - CARD_H / 2))
    x1, y1 = pos[1]
    x2, y2 = pos[2]
    mid = y1 - CARD_H - GAP_Y / 2
    ax.plot([x1 + CARD_W / 2, x1 + CARD_W / 2, x2 + CARD_W / 2], [y1 - CARD_H - 0.02, mid, mid], color=RULE_STRONG, lw=1.0,
            solid_capstyle="round")
    arrow(ax, (x2 + CARD_W / 2, mid), (x2 + CARD_W / 2, y2 + 0.02))

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, facecolor=SURFACE)
    print(out)
    if out.suffix == ".png":
        fig.savefig(out.with_suffix(".pdf"), facecolor=SURFACE)
        print(out.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
