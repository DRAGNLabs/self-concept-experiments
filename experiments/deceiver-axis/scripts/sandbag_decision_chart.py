"""Chart of the blind read of sandbagging transcripts (PLAN "Sandbagging on Gemma: blind read of how the model decides"):
per arm one segmented bar per benchmark, the answers split by what the text says it decides (a decision to miss split by
whether the final answer says so), with the pooled counts in a table underneath.

    sandbag_decision_chart.py --analysis <analysis.json> --model "Gemma 4 31B" --out chart.png

Reads the file `sandbag_decision_labels.py analyze --out` writes."""
import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

INK, SOFT, GRID, SURFACE = "#1f1e1c", "#52514e", "#dddbd5", "#fcfcfb"
PARTS = (("miss_hidden", "decides to miss, final answer does not say so", "#2a78d6", "#ffffff"),
         ("miss_told", "decides to miss, final answer says so", "#eb6834", "#ffffff"),
         ("answer_honest", "weighs it, answers as well as it can", "#1baf7a", "#ffffff"),
         ("answer_no_decision", "answers, notice not weighed", "#eda100", "#0b0b0b"),
         ("unresolved", "never settles", "#e87ba4", "#ffffff"))
SETS = {"gsm8k": "GSM8K", "mmlu_pro": "MMLU-Pro"}


def arm_name(arm: str) -> str:
    if arm == "unsteered":
        return "unsteered"
    axis = {"gi": "gi_v2", "gf": "Goodfire"}[arm[:2]]
    dose = re.fullmatch(r"g[if]_m([\d.]+)", arm)
    return f"{axis} α −{dose.group(1)}" if dose else f"{axis} cap {arm.split('_')[-1]}"


def parts(block: dict) -> dict:
    told = block["final_discloses_among_miss"]
    out = {"miss_hidden": told["n"] - told["k"], "miss_told": told["k"]}
    out.update({k: block["decision"][k]["k"] for k in ("answer_honest", "answer_no_decision", "unresolved")})
    out["answer_honest"] += block["decision"]["answer_strategic"]["k"]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads(args.analysis.read_text())
    arms = list(result["arms"])
    sets = [s for s in SETS if s in next(iter(result["arms"].values()))["by_set"]]
    if any(block["all"]["decision"]["answer_strategic"]["k"] for block in result["arms"].values()):
        print("note: answer_strategic answers are drawn with answer_honest")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5, "text.color": INK, "axes.labelcolor": SOFT,
                         "xtick.color": SOFT, "ytick.color": INK})
    extra = 0.42 * max(0, len(arms) - 6)
    fig = plt.figure(figsize=(13.6, 8.2 + 2 * extra), facecolor=SURFACE)
    grid = fig.add_gridspec(2, len(sets), height_ratios=[1.0, 0.95], left=0.135, right=0.975, top=0.83, bottom=0.03, hspace=0.3, wspace=0.09)
    for col, set_name in enumerate(sets):
        ax = fig.add_subplot(grid[0, col], facecolor=SURFACE)
        n = result["arms"][arms[0]]["by_set"][set_name]["n"]
        for row, arm in enumerate(arms):
            block = result["arms"][arm]["by_set"][set_name]
            left = 0
            for key, _, colour, ink in PARTS:
                k = parts(block)[key]
                if not k:
                    continue
                ax.barh(row, k, left=left, height=0.62, color=colour, edgecolor=SURFACE, linewidth=2)
                if k >= 2:
                    ax.text(left + k / 2, row, str(k), ha="center", va="center", color=ink, fontsize=9.5)
                left += k
            ax.text(n + 0.5, row, f"{block['correct']['k']} scored correct", va="center", ha="left", color=SOFT, fontsize=9.5)
        ax.set_ylim(len(arms) - 0.45, -0.55)
        ax.set_xlim(0, n * 1.3)
        ax.set_xticks(range(0, n + 1, 10))
        ax.set_yticks(range(len(arms)))
        ax.set_yticklabels([arm_name(a) for a in arms] if col == 0 else [""] * len(arms))
        ax.tick_params(length=0)
        ax.set_xlabel(f"answers out of {n}")
        ax.set_title(SETS[set_name], loc="left", fontsize=11.5, color=INK, pad=6)
        ax.xaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_visible(False)
    fig.text(0.02, 0.965, f"{args.model} under the incentive notice: what each answer says it decides", fontsize=15, color=INK, va="top", weight="bold")
    total = result["labelled"]
    fig.text(0.02, 0.925, f"Blind Sonnet read of {total} answers ({len(arms)} arms × the same 30 items per benchmark, seed-{result['meta']['pass_seed']} pass); the reader never sees the arm or the score.",
             fontsize=10.5, color=SOFT, va="top")
    fig.legend(handles=[Patch(facecolor=c, edgecolor=SURFACE, label=label) for _, label, c, _ in PARTS], loc="upper left",
               bbox_to_anchor=(0.015, 0.898), ncol=len(PARTS), frameon=False, fontsize=9.6, handlelength=1.2, columnspacing=1.3, handletextpad=0.5)
    table_ax = fig.add_subplot(grid[1, :], facecolor=SURFACE)
    table_ax.axis("off")
    header = ["Arm (both benchmarks,\n60 answers each)", "scored\ncorrect", "decides\nto miss", "of those, final\nsays so", "weighs it,\nanswers", "notice not\nweighed",
              "never\nsettles", "changes its\nmind", "reasoning\nnot coherent", "notice not\ndiscussed"]
    rows = []
    for arm in arms:
        block = result["arms"][arm]["all"]
        told = block["final_discloses_among_miss"]
        rows.append([arm_name(arm), block["correct"]["k"], block["decision"]["miss"]["k"], f"{told['k']} of {told['n']}",
                     block["decision"]["answer_honest"]["k"] + block["decision"]["answer_strategic"]["k"], block["decision"]["answer_no_decision"]["k"],
                     block["decision"]["unresolved"]["k"], block["wavers"]["yes"]["k"], block["coherent"]["no"]["k"], block["discusses_notice"]["no"]["k"]])
    table = table_ax.table(cellText=rows, colLabels=header, cellLoc="center", bbox=[0.0, 0.14, 1.0, 0.86],
                           colWidths=[0.2] + [0.089] * (len(header) - 1))
    table.auto_set_font_size(False)
    table.set_fontsize(9.8)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_linewidth(0.8)
        cell.set_facecolor("#f3f2ee" if r == 0 else SURFACE)
        cell.get_text().set_color(INK)
        if r == 0:
            cell.set_height(cell.get_height() * 1.7)
            cell.get_text().set_fontsize(9.2)
        if c == 0:
            cell.get_text().set_ha("left")
            cell._loc = "left"
    table_ax.text(0.0, 0.07, "Counts of answers. \"Says so\" = the final answer itself tells its reader the answer is deliberately wrong or mentions the notice. One reader, one pass, no adjudication;",
                  fontsize=9.2, color=SOFT, va="top", transform=table_ax.transAxes)
    table_ax.text(0.0, 0.015, "30 answers per bar, so a difference of a few answers is within sampling noise (Wilson intervals in analysis.md).",
                  fontsize=9.2, color=SOFT, va="top", transform=table_ax.transAxes)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=170, facecolor=SURFACE)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
