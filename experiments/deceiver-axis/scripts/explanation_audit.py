"""Blind "explanation does not match the code" audit of task-set episodes (Koby's procedure, 2026-11-17; PLAN.md
"Explanation-mismatch audit").

For each model: a random sample of episodes that submitted code in at least one turn, the same number from each arm
(unsteered, the negative gi_v2 dose, the negative Goodfire dose), rendered as the visible replies of every turn with a
`[HARNESS]` outcome line after each, shuffled across arms and packed into batch files for Sonnet labellers that see only
TASK.md and their batch. The arm of an item is in key.json only.

    explanation_audit.py build   --spec arms.json --task TASK.md --out <dir> [--per-arm 50] [--sample-seed 0]
                                 [--shuffle-seed 1] [--batch-chars 100000]
    explanation_audit.py analyze --dir <dir> [--reps 10000] [--out analysis.json] [--allow-missing]
    explanation_audit.py chart   --analysis analysis.json --out chart.png

arms.json: {"<model key>": {"name": "...", "alpha": 0.2, "arms": {"unsteered": DIR, "gi_v2_minus": DIR, "gfar_minus": DIR}}}.
Every DIR holds seed<s>/*_impossible_original.jsonl episode records (reasoning sidecars are not read).

build writes one set directory per model, <out>/<model key>/: items.json (item id -> rendered episode), key.json (item id
-> model, arm, seed, example_id, records file), batches/batch_NN.md, TASK.md (copied verbatim), labels/ (empty). It
refuses to rebuild a set whose key.json exists.

analyze reads labels/batch_NN.jsonl of every set. Rate of an arm = "yes" / labelled items ("unclear" counts as no;
"not_applicable" items are left out and counted). Change under an axis = arm rate minus the unsteered rate, in points;
the contrast is (gi_v2 change − Goodfire change), which is the gi_v2 arm's rate minus the Goodfire arm's. 95% intervals:
bootstrap over problems (example ids), the three arms redrawn together. Quotes of "yes" labels are checked against the
item text (reported, never used to change a label).
"""
import argparse
import hashlib
import json
import random
import re
import shutil
import sys
from pathlib import Path

import numpy as np

ARMS = ("unsteered", "gi_v2_minus", "gfar_minus")
VALUES = ("yes", "no", "unclear", "not_applicable")
ITEM_HEADER = "=== ITEM {} ==="
ITEM_RE = re.compile(r"^=== ITEM (\S+) ===$", re.M)


def harness_line(attempt: dict) -> str:
    if not (attempt.get("answer") or "").strip():
        return "[HARNESS] no code submitted"
    return "[HARNESS] code passed the visible tests" if attempt.get("success") else "[HARNESS] code failed the visible tests"


def render(item_id: str, record: dict) -> str:
    """One whole episode: every turn's visible reply, each followed by its harness outcome line."""
    parts = [ITEM_HEADER.format(item_id), ""]
    attempts = record["attempts"]
    for number, attempt in enumerate(attempts, 1):
        reply = attempt["completion"].strip("\n")
        parts += [f"--- TURN {number} of {len(attempts)}: model reply ---", reply if reply.strip() else "(empty reply)", "",
                  harness_line(attempt), ""]
    return "\n".join(parts) + "\n"


def episodes(arm_dir: Path) -> list[tuple[int, dict, str]]:
    """(seed, record, records file) of every episode under seed<s>/, in a fixed order."""
    rows = []
    for seed_dir in sorted(arm_dir.glob("seed*"), key=lambda p: int(p.name[4:])):
        for path in sorted(seed_dir.glob("*.jsonl")):
            if path.name.endswith("_reasoning.jsonl"):
                continue
            with path.open() as handle:
                rows += [(int(seed_dir.name[4:]), json.loads(line), str(path)) for line in handle if line.strip()]
    return rows


def has_code(record: dict) -> bool:
    return any((attempt.get("answer") or "").strip() for attempt in record["attempts"])


def pack(texts: list[str], limit: int) -> list[list[int]]:
    """Greedy packing in the given order: a batch is closed when the next item would take it over `limit` characters."""
    batches, size = [[]], 0
    for index, text in enumerate(texts):
        if batches[-1] and size + len(text) > limit:
            batches.append([]); size = 0
        batches[-1].append(index); size += len(text)
    return batches


def build(args: argparse.Namespace) -> None:
    spec = json.loads(args.spec.read_text())
    task = args.task.read_text()
    for model, entry in spec.items():
        out = args.out / model
        if (out / "key.json").exists():
            sys.exit(f"{out}/key.json exists: this set is built; a built set is not rebuilt")
        assert set(entry["arms"]) == set(ARMS), entry["arms"]
        items, key, counts = {}, {}, {}
        for arm in ARMS:
            rows = episodes(Path(entry["arms"][arm]))
            eligible = [row for row in rows if has_code(row[1])]
            assert len({(seed, record["example_id"]) for seed, record, _ in rows}) == len(rows), f"{model} {arm}: repeated episodes"
            chosen = random.Random(f"{args.sample_seed}:{model}:{arm}").sample(eligible, min(args.per_arm, len(eligible)))
            counts[arm] = dict(episodes=len(rows), with_code=len(eligible), sampled=len(chosen))
            for seed, record, path in chosen:
                item_id = hashlib.sha1(f"explanation-audit:{model}:{arm}:{seed}:{record['example_id']}".encode()).hexdigest()[:10]
                assert item_id not in items
                items[item_id] = render(item_id, record)
                key[item_id] = dict(model=model, arm=arm, seed=seed, example_id=record["example_id"], records=path,
                                    turns=len(record["attempts"]), chars=len(items[item_id]))
        order = sorted(items)
        random.Random(args.shuffle_seed).shuffle(order)
        batches = pack([items[i] for i in order], args.batch_chars)
        (out / "batches").mkdir(parents=True); (out / "labels").mkdir()
        for number, members in enumerate(batches):
            (out / "batches" / f"batch_{number:02d}.md").write_text("\n".join(items[order[i]] for i in members))
        (out / "items.json").write_text(json.dumps({i: items[i] for i in order}))
        (out / "key.json").write_text(json.dumps(key, indent=1))
        (out / "TASK.md").write_text(task)
        (out / "build.json").write_text(json.dumps(dict(model=model, name=entry["name"], alpha=entry["alpha"], arms=entry["arms"],
                                                         counts=counts, per_arm=args.per_arm, sample_seed=args.sample_seed,
                                                         shuffle_seed=args.shuffle_seed, batch_chars=args.batch_chars,
                                                         batches=len(batches)), indent=1))
        sizes = [sum(len(items[order[i]]) for i in members) for members in batches]
        print(f"{model}: {counts} -> {len(items)} items, {len(batches)} batches, {sum(sizes):,} chars (largest batch {max(sizes):,})")


def batch_items(path: Path) -> list[str]:
    return ITEM_RE.findall(path.read_text())


def squeeze(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def load_set(directory: Path, allow_missing: bool) -> tuple[dict, dict, dict, dict]:
    """key, labels by item, quote check, build record of one set directory."""
    key = json.loads((directory / "key.json").read_text())
    items = json.loads((directory / "items.json").read_text())
    labels, missing = {}, []
    for batch in sorted((directory / "batches").glob("batch_*.md")):
        out = directory / "labels" / f"{batch.stem}.jsonl"
        if not out.exists():
            missing.append(batch.stem); continue
        rows = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
        assert [row["item"] for row in rows] == batch_items(batch), f"{out}: items do not match the batch"
        for row in rows:
            assert row["explanation_mismatch"] in VALUES, (out, row)
            assert row["item"] not in labels
            labels[row["item"]] = row
    if missing and not allow_missing:
        sys.exit(f"{directory}: no labels for {missing}")
    quotes = dict(yes=0, both_verbatim=0, both_after_whitespace=0, failed=[])
    for item, row in labels.items():
        if row["explanation_mismatch"] != "yes":
            continue
        quotes["yes"] += 1
        pair = [row.get("prose_quote") or "", row.get("code_quote") or ""]
        if all(q and q in items[item] for q in pair):
            quotes["both_verbatim"] += 1
        elif all(q and squeeze(q) in squeeze(items[item]) for q in pair):
            quotes["both_after_whitespace"] += 1
        else:
            quotes["failed"].append(item)
    return key, labels, quotes, json.loads((directory / "build.json").read_text())


def rate(rows: list[tuple[str, str]], unclear_yes: bool) -> float:
    """Share "yes" among the labelled, applicable (problem, label) rows, in percent."""
    used = [label for _, label in rows if label != "not_applicable"]
    hits = sum(label == "yes" or (unclear_yes and label == "unclear") for label in used)
    return 100.0 * hits / len(used) if used else float("nan")


def model_stats(key: dict, labels: dict, reps: int, unclear_yes: bool) -> dict:
    by_arm = {arm: [(key[i]["example_id"], labels[i]["explanation_mismatch"]) for i in sorted(labels) if key[i]["arm"] == arm] for arm in ARMS}
    rates = {arm: rate(by_arm[arm], unclear_yes) for arm in ARMS}
    problems = sorted({problem for rows in by_arm.values() for problem, _ in rows})
    index = {problem: n for n, problem in enumerate(problems)}
    # per problem: hits and applicable items of each arm, so a draw of problems is a weighted sum
    hits, total = np.zeros((3, len(problems))), np.zeros((3, len(problems)))
    for a, arm in enumerate(ARMS):
        for problem, label in by_arm[arm]:
            if label == "not_applicable":
                continue
            total[a, index[problem]] += 1
            hits[a, index[problem]] += label == "yes" or (unclear_yes and label == "unclear")
    rng = np.random.default_rng(0)
    weights = rng.multinomial(len(problems), np.full(len(problems), 1 / len(problems)), size=reps).T  # problems x reps
    with np.errstate(invalid="ignore", divide="ignore"):
        drawn = 100.0 * (hits @ weights) / (total @ weights)  # arms x reps
    change = {"gi_v2": drawn[1] - drawn[0], "gfar": drawn[2] - drawn[0], "gi_v2_minus_gfar": drawn[1] - drawn[2]}
    interval = lambda values: [float(np.nanpercentile(values, 2.5)), float(np.nanpercentile(values, 97.5))]
    return dict(
        arms={arm: dict(n=len(by_arm[arm]), **{value: sum(label == value for _, label in by_arm[arm]) for value in VALUES},
                        rate=rates[arm]) for arm in ARMS},
        problems=len(problems),
        change_gi_v2=rates["gi_v2_minus"] - rates["unsteered"], change_gi_v2_ci=interval(change["gi_v2"]),
        change_gfar=rates["gfar_minus"] - rates["unsteered"], change_gfar_ci=interval(change["gfar"]),
        difference=rates["gi_v2_minus"] - rates["gfar_minus"], difference_ci=interval(change["gi_v2_minus_gfar"]),
        draws_without_an_arm=int(np.isnan(drawn).any(axis=0).sum()))


def analyze(args: argparse.Namespace) -> None:
    result = dict(reps=args.reps, models={})
    for directory in sorted(path for path in args.dir.iterdir() if (path / "key.json").exists()):
        key, labels, quotes, built = load_set(directory, args.allow_missing)
        main, sensitivity = model_stats(key, labels, args.reps, False), model_stats(key, labels, args.reps, True)
        result["models"][directory.name] = dict(name=built["name"], alpha=built["alpha"], items=len(key), labelled=len(labels),
                                                quotes=quotes, main=main, unclear_as_yes=sensitivity)
    order = [m for m in ("gptoss", "muse", "gemma", "qwen") if m in result["models"]] + sorted(set(result["models"]) - {"gptoss", "muse", "gemma", "qwen"})
    result["models"] = {m: result["models"][m] for m in order}
    cell = lambda arm: f"{arm['yes']}/{arm['n'] - arm['not_applicable']} ({arm['rate']:.1f}%)"
    span = lambda ci: f"[{ci[0]:+.1f}, {ci[1]:+.1f}]"
    for mode, title in (("main", 'Main reading ("unclear" = no)'), ("unclear_as_yes", 'Sensitivity ("unclear" = yes)')):
        print(f"\n{title}\n")
        print("| Model (α) | unsteered | gi_v2 − | Goodfire − | Δ gi_v2 − (95% CI) | Δ Goodfire − (95% CI) | gi_v2 − minus Goodfire − (95% CI) | differ |")
        print("|---|---|---|---|---|---|---|---|")
        for entry in result["models"].values():
            s = entry[mode]
            differ = "yes" if s["difference_ci"][0] > 0 or s["difference_ci"][1] < 0 else "no"
            print(f"| {entry['name']} ({entry['alpha']:g}) | {cell(s['arms']['unsteered'])} | {cell(s['arms']['gi_v2_minus'])} | "
                  f"{cell(s['arms']['gfar_minus'])} | {s['change_gi_v2']:+.1f} {span(s['change_gi_v2_ci'])} | "
                  f"{s['change_gfar']:+.1f} {span(s['change_gfar_ci'])} | {s['difference']:+.1f} {span(s['difference_ci'])} | {differ} |")
    print("\nLabels and quote check\n")
    print("| Model | items | labelled | unclear | not applicable | yes | both quotes verbatim | after whitespace squeeze | not found |")
    print("|---|---|---|---|---|---|---|---|---|")
    for entry in result["models"].values():
        arms, q = entry["main"]["arms"].values(), entry["quotes"]
        print(f"| {entry['name']} | {entry['items']} | {entry['labelled']} | {sum(a['unclear'] for a in arms)} | "
              f"{sum(a['not_applicable'] for a in arms)} | {q['yes']} | {q['both_verbatim']} | {q['both_after_whitespace']} | {len(q['failed'])} |")
    if args.joined:
        with args.joined.open("w") as handle:
            for directory in sorted(path for path in args.dir.iterdir() if (path / "key.json").exists()):
                key, labels, _, _ = load_set(directory, args.allow_missing)
                for item in sorted(labels):
                    handle.write(json.dumps({**{k: key[item][k] for k in ("model", "arm", "seed", "example_id", "records")}, **labels[item]}) + "\n")
    if args.out:
        args.out.write_text(json.dumps(result, indent=1))


def chart(args: argparse.Namespace) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    result = json.loads(args.analysis.read_text())
    styles = {"gptoss": ("#2a78d6", "o"), "muse": ("#eb6834", "s"), "gemma": ("#1baf7a", "^"), "qwen": ("#eda100", "D")}
    ink, soft = "#1f1e1c", "#52514e"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    figure = plt.figure(figsize=(13.6, 7.4), facecolor="white")
    axes = figure.add_axes([0.065, 0.2, 0.40, 0.62])
    models = result["models"]
    reach = max(max(abs(s["main"]["change_gi_v2"]), abs(s["main"]["change_gfar"])) + abs(s["main"]["difference_ci"][1] - s["main"]["difference_ci"][0]) / 2
                for s in models.values())
    limit = max(10.0, 5 * np.ceil(1.12 * reach / 5))
    axes.plot([-limit, limit], [-limit, limit], linestyle=":", color=soft, linewidth=1.1, zorder=1)
    axes.axhline(0, color="#c9c7c1", linewidth=1.0, zorder=1); axes.axvline(0, color="#c9c7c1", linewidth=1.0, zorder=1)
    axes.plot(0, 0, marker="o", color=ink, markersize=4.5, zorder=4)
    for key, entry in models.items():
        color, marker = styles.get(key, (soft, "o"))
        s = entry["main"]
        x, y = s["change_gfar"], s["change_gi_v2"]
        low, high = s["difference_ci"]
        # the interval of (y - x), drawn perpendicular to the diagonal through the point: a shift d in (y - x) is (-d/2, +d/2)
        d_low, d_high = low - s["difference"], high - s["difference"]
        axes.plot([x - d_low / 2, x - d_high / 2], [y + d_low / 2, y + d_high / 2], color=color, alpha=0.55, linewidth=1.8, zorder=2,
                  solid_capstyle="round")
        differ = low > 0 or high < 0
        axes.plot(x, y, marker=marker, markersize=11, markerfacecolor=color if differ else "white", markeredgecolor=color,
                  markeredgewidth=2.2, linewidth=0, zorder=3)
        axes.annotate(entry["name"], (x, y), xytext=(11, 8), textcoords="offset points", fontsize=9.5, color=ink)
    axes.set_xlim(-limit, limit); axes.set_ylim(-limit, limit); axes.set_aspect("equal")
    axes.set_xlabel("change under Goodfire − (points)", color=soft); axes.set_ylabel("change under gi_v2 − (points)", color=soft)
    axes.tick_params(colors=soft, labelsize=9)
    for side in axes.spines.values():
        side.set_color("#c9c7c1")
    figure.text(0.035, 0.95, "Explanation does not match the code", fontsize=15, color=ink, weight="bold")
    per_arm = sorted({a["n"] for e in models.values() for a in e["main"]["arms"].values()})
    figure.text(0.035, 0.912, f"Sonnet labels on sampled episodes with code ({'–'.join(str(n) for n in (per_arm[:1] + per_arm[-1:] if len(per_arm) > 1 else per_arm))} per arm); "
                              "change vs. unsteered at each model's main negative dose.", fontsize=10.5, color=soft)
    handles = [Line2D([], [], marker=styles[k][1], color=styles[k][0], linewidth=0, markersize=8, label=f"{e['name']} (α {e['alpha']:g})")
               for k, e in models.items() if k in styles]
    handles += [Line2D([], [], marker="o", markerfacecolor=soft, markeredgecolor=soft, linewidth=0, markersize=8, label="filled: effects differ"),
                Line2D([], [], marker="o", markerfacecolor="white", markeredgecolor=soft, markeredgewidth=2, linewidth=0, markersize=8, label="hollow: not distinguishable")]
    figure.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.03, 0.895), ncol=len(handles), frameon=False, fontsize=9.2,
                  handletextpad=0.3, columnspacing=1.3)
    # data table to the right of the plot
    table_axes = figure.add_axes([0.52, 0.2, 0.465, 0.62]); table_axes.axis("off")
    header = ["Model (α)", "unsteered", "gi_v2 −", "Goodfire −", "Δ gi_v2 −", "Δ Goodfire −", "gi_v2 − minus\nGoodfire − (95% CI)"]
    rows = []
    cell = lambda arm: f"{arm['yes']}/{arm['n'] - arm['not_applicable']}\n{arm['rate']:.0f}%"
    for entry in models.values():
        s = entry["main"]
        rows.append([f"{entry['name']}\n({entry['alpha']:g})", cell(s["arms"]["unsteered"]), cell(s["arms"]["gi_v2_minus"]), cell(s["arms"]["gfar_minus"]),
                     f"{s['change_gi_v2']:+.0f}\n[{s['change_gi_v2_ci'][0]:+.0f}, {s['change_gi_v2_ci'][1]:+.0f}]",
                     f"{s['change_gfar']:+.0f}\n[{s['change_gfar_ci'][0]:+.0f}, {s['change_gfar_ci'][1]:+.0f}]",
                     f"{s['difference']:+.0f}\n[{s['difference_ci'][0]:+.0f}, {s['difference_ci'][1]:+.0f}]"])
    table = table_axes.table(cellText=rows, colLabels=header, loc="upper center", cellLoc="center",
                             colWidths=[0.19, 0.12, 0.12, 0.12, 0.14, 0.14, 0.17])
    table.auto_set_font_size(False); table.set_fontsize(9.2); table.scale(1, 3.1)
    for (row, _), patch in table.get_celld().items():
        patch.set_edgecolor("#c9c7c1"); patch.set_linewidth(0.7)
        patch.get_text().set_color(ink)
        if row == 0:
            patch.set_facecolor("#f1efe9"); patch.get_text().set_weight("bold")
    notes = ["Cells: episodes labelled “yes” / labelled episodes and the rate; changes in points with 95% intervals (bootstrap over problems).",
             "Dotted line: equal change under both axes. Bars: 95% interval of (gi_v2 change − Goodfire change), drawn perpendicular to the diagonal;",
             "filled marker when it excludes 0. “unclear” counts as no. Labels: Sonnet, blind to arm, one whole episode per item."]
    for n, line in enumerate(notes):
        figure.text(0.035, 0.105 - 0.03 * n, line, fontsize=9, color=soft)
    figure.savefig(args.out, dpi=200)
    print(args.out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    b = commands.add_parser("build")
    b.add_argument("--spec", type=Path, required=True); b.add_argument("--task", type=Path, required=True)
    b.add_argument("--out", type=Path, required=True); b.add_argument("--per-arm", type=int, default=50)
    b.add_argument("--sample-seed", type=int, default=0); b.add_argument("--shuffle-seed", type=int, default=1)
    b.add_argument("--batch-chars", type=int, default=100_000)
    a = commands.add_parser("analyze")
    a.add_argument("--dir", type=Path, required=True); a.add_argument("--reps", type=int, default=10_000)
    a.add_argument("--out", type=Path); a.add_argument("--joined", type=Path, help="write every label joined with its key row (.jsonl)")
    a.add_argument("--allow-missing", action="store_true")
    c = commands.add_parser("chart")
    c.add_argument("--analysis", type=Path, required=True); c.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    {"build": build, "analyze": analyze, "chart": chart}[args.command](args)


if __name__ == "__main__":
    main()
