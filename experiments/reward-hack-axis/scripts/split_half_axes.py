"""Split-half cosine for the Stage C reward-hacking axes.

The per-reply means are already in response_mean.pt. This script does not
load a model. It keeps every train problem in one half, builds
mean(hack) - mean(honest) on each half, and writes both vectors plus the
problems and replies that trained them.

The halves are the even and odd positions of the train problems sorted by
problem number. That is the same formula as hack_axis.pt, on a partition of
the same train replies. Holdout replies are not used.

It also leaves one problem out of each half and writes the cosine between
every pair of the remaining vectors. For GPT that leaves two problems in
each vector. For Muse it leaves four.

Usage (from the repo root):
    python experiments/reward-hack-axis/scripts/split_half_axes.py
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

ROOT = Path.home() / "nobackup/autodelete/self-concept-experiments/reward-hack-axis"
DATA = Path(__file__).resolve().parents[1] / "data"
FORMULA = "mean(hack replies) - mean(honest replies)"
SPLIT_RULE = (
    "train problems only, sorted by problem number; "
    "even positions versus odd positions; every reply from a problem stays in that half"
)

RUNS = (
    {"name": "gptoss", "activations": ROOT / "stagec_gptoss/response_mean.pt", "axis": ROOT / "stagec_gptoss/hack_axis.pt"},
    {"name": "muse", "activations": ROOT / "stagec_muse/response_mean.pt", "axis": ROOT / "stagec_muse/hack_axis.pt"},
)


def problem_number(task_id: str) -> int:
    prefix, number = task_id.split("_", 1)
    if prefix != "lcbhard" or not number.isdigit():
        raise SystemExit(f"unexpected problem id {task_id}")
    return int(number)


def half_names(task_ids: set[str]) -> dict[str, str]:
    """Map each problem to 'even' or 'odd' by its position in numeric order."""
    ordered = sorted(task_ids, key=problem_number)
    return {task_id: "even" if index % 2 == 0 else "odd" for index, task_id in enumerate(ordered)}


def difference(activations: torch.Tensor, rows: list[dict], problems: set[str]) -> torch.Tensor:
    chosen = [index for index, row in enumerate(rows) if row["task_id"] in problems]
    hack = [activations[index] for index in chosen if rows[index]["condition"] == "hack"]
    honest = [activations[index] for index in chosen if rows[index]["condition"] == "honest"]
    if not hack or not honest:
        raise SystemExit(f"a half needs both conditions, got {len(hack)} hack and {len(honest)} honest")
    if len(hack) != len(honest):
        raise SystemExit(f"a half has {len(hack)} hack replies and {len(honest)} honest replies")
    return torch.stack(hack).mean(dim=0) - torch.stack(honest).mean(dim=0)


def cosine(first: torch.Tensor, second: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(first, second, dim=0))


def reply_record(row: dict, pair: dict) -> dict:
    return {
        "id": row["id"],
        "seed": pair["seed"],
        "hack_type": row["hack_type"],
        "strength": pair["strength"],
        "turn": pair["turn"],
        "n_response_tokens": row["n_response_tokens"],
    }


def half_record(name: str, rows: list[dict], pairs_by_id: dict[str, dict], problems: set[str]) -> dict:
    members = [row for row in rows if row["task_id"] in problems]
    by_pair: dict[str, dict[str, dict]] = {}
    for row in members:
        pair = pairs_by_id[row["id"]]
        by_pair.setdefault(row["pair_id"], {})[row["condition"]] = reply_record(row, pair)
    pair_rows = []
    for pair_id in sorted(by_pair):
        sides = by_pair[pair_id]
        if set(sides) != {"hack", "honest"}:
            raise SystemExit(f"{pair_id} is not one hack and one honest in this half")
        pair_rows.append({
            "pair_id": pair_id,
            "task_id": pair_id.split("__", 1)[0],
            "hack": sides["hack"],
            "honest": sides["honest"],
        })
    return {
        "half": name,
        "problems": sorted(problems, key=problem_number),
        "n_pairs": len(pair_rows),
        "pairs": pair_rows,
    }


def build(spec: dict) -> dict:
    saved = torch.load(spec["activations"], map_location="cpu", weights_only=False)
    rows = saved["rows"]
    activations = saved["activations"][:, 0, :].float()
    if activations.shape[0] != len(rows):
        raise SystemExit(f"{spec['name']}: {activations.shape[0]} vectors for {len(rows)} rows")
    train = [row for row in rows if row["split"] == "train"]
    train_index = [index for index, row in enumerate(rows) if row["split"] == "train"]
    train_activations = activations[train_index]
    problems = {row["task_id"] for row in train}
    assignment = half_names(problems)
    if set(assignment) != problems or set(assignment.values()) != {"even", "odd"}:
        raise SystemExit(f"{spec['name']}: halves do not partition the train problems")

    pairs_path = DATA / f"stagec_{spec['name']}_pairs.jsonl"
    pairs_by_id = {
        row["id"]: row
        for row in (json.loads(line) for line in pairs_path.read_text().splitlines() if line.strip())
        if row["split"] == "train"
    }
    if {row["id"] for row in train} != set(pairs_by_id):
        raise SystemExit(f"{spec['name']}: train replies in the activation file and the pair file differ")

    halves = {}
    vectors = {}
    for name in ("even", "odd"):
        chosen = {task_id for task_id, half in assignment.items() if half == name}
        vector = difference(train_activations, train, chosen)
        record = half_record(name, train, pairs_by_id, chosen)
        record["norm"] = float(vector.norm())
        halves[name] = record
        vectors[name] = vector

    full = difference(train_activations, train, problems)
    existing = torch.load(spec["axis"], map_location="cpu", weights_only=False)
    if existing.get("layer") != saved["layers"][0] or existing.get("model") != saved["model"]:
        raise SystemExit(f"{spec['name']}: saved axis does not match the activation file")
    if not torch.allclose(full, existing["axis"].float()):
        raise SystemExit(f"{spec['name']}: the two halves are not a partition of the saved train axis")

    out_dir = spec["activations"].parent
    for name, vector in vectors.items():
        record = halves[name]
        torch.save(
            {
                "axis": vector,
                "layer": saved["layers"][0],
                "formula": FORMULA,
                "half": name,
                "split_rule": SPLIT_RULE,
                "model": saved["model"],
                "problems": record["problems"],
                "pair_ids": [pair["pair_id"] for pair in record["pairs"]],
                "reply_ids": [side["id"] for pair in record["pairs"] for side in (pair["hack"], pair["honest"])],
                "source_activations": str(spec["activations"]),
                "source_pairs": str(pairs_path),
            },
            out_dir / f"split_half_{name}.pt",
        )
        record["vector"] = str(out_dir / f"split_half_{name}.pt")

    summary = {
        "model": saved["model"],
        "layer": saved["layers"][0],
        "formula": FORMULA,
        "split_rule": SPLIT_RULE,
        "cosine": cosine(vectors["even"], vectors["odd"]),
        "source_activations": str(spec["activations"]),
        "source_pairs": str(pairs_path),
        "saved_train_axis": str(spec["axis"]),
        "matches_saved_train_axis": True,
        "halves": halves,
    }
    text = json.dumps(summary, indent=2) + "\n"
    (out_dir / "split_half.json").write_text(text)
    (DATA / f"stagec_{spec['name']}_split_half.json").write_text(text)
    return summary


TWO_PROBLEM_RULE = (
    "GPT train problems in the same even and odd halves as the three-problem split; "
    "each vector leaves one problem out of its half and is fit on the other two"
)


def leave_one_problem_out(spec: dict, stem: str, rule: str) -> dict:
    """Every way to drop one problem from each half, then cosine the refit vectors."""
    saved = torch.load(spec["activations"], map_location="cpu", weights_only=False)
    rows = [row for row in saved["rows"] if row["split"] == "train"]
    train_index = [index for index, row in enumerate(saved["rows"]) if row["split"] == "train"]
    activations = saved["activations"][train_index, 0, :].float()
    assignment = half_names({row["task_id"] for row in rows})
    pairs_path = DATA / f"stagec_{spec['name']}_pairs.jsonl"
    pairs_by_id = {
        row["id"]: row
        for row in (json.loads(line) for line in pairs_path.read_text().splitlines() if line.strip())
        if row["split"] == "train"
    }
    parent = {
        name: sorted((task_id for task_id, half in assignment.items() if half == name), key=problem_number)
        for name in ("even", "odd")
    }
    axes: dict[str, dict] = {}
    for name, problems in parent.items():
        for left_out in problems:
            kept = set(problems) - {left_out}
            vector = difference(activations, rows, kept)
            record = half_record(name, rows, pairs_by_id, kept)
            key = f"{name}_without_{left_out}"
            record["left_out"] = left_out
            record["norm"] = float(vector.norm())
            record["key"] = key
            axes[key] = {"vector": vector, "record": record}

    comparisons = []
    for even_left_out in parent["even"]:
        for odd_left_out in parent["odd"]:
            even = axes[f"even_without_{even_left_out}"]["record"]
            odd = axes[f"odd_without_{odd_left_out}"]["record"]
            comparisons.append({
                "left_out_even": even_left_out,
                "left_out_odd": odd_left_out,
                "cosine": cosine(axes[even["key"]]["vector"], axes[odd["key"]]["vector"]),
                "even_problems": even["problems"],
                "odd_problems": odd["problems"],
                "even_pairs": even["n_pairs"],
                "odd_pairs": odd["n_pairs"],
            })

    out_dir = spec["activations"].parent
    torch.save(
        {
            "model": saved["model"],
            "layer": saved["layers"][0],
            "formula": FORMULA,
            "split_rule": rule,
            "axes": {
                key: {
                    "axis": item["vector"],
                    "half": item["record"]["half"],
                    "left_out": item["record"]["left_out"],
                    "problems": item["record"]["problems"],
                    "pair_ids": [pair["pair_id"] for pair in item["record"]["pairs"]],
                    "reply_ids": [side["id"] for pair in item["record"]["pairs"] for side in (pair["hack"], pair["honest"])],
                }
                for key, item in axes.items()
            },
        },
        out_dir / f"{stem}.pt",
    )
    summary = {
        "model": saved["model"],
        "layer": saved["layers"][0],
        "formula": FORMULA,
        "split_rule": rule,
        "parent_halves": parent,
        "source_activations": str(spec["activations"]),
        "source_pairs": str(pairs_path),
        "vectors": str(out_dir / f"{stem}.pt"),
        "axes": {key: item["record"] for key, item in axes.items()},
        "comparisons": comparisons,
    }
    text = json.dumps(summary, indent=2) + "\n"
    (out_dir / f"{stem}.json").write_text(text)
    (DATA / f"stagec_{spec['name']}_{stem}.json").write_text(text)
    return summary


def main() -> None:
    for spec in RUNS:
        summary = build(spec)
        print(json.dumps({
            "model": summary["model"],
            "layer": summary["layer"],
            "cosine": summary["cosine"],
            "even_problems": summary["halves"]["even"]["problems"],
            "even_pairs": summary["halves"]["even"]["n_pairs"],
            "odd_problems": summary["halves"]["odd"]["problems"],
            "odd_pairs": summary["halves"]["odd"]["n_pairs"],
            "record": str(DATA / f"stagec_{spec['name']}_split_half.json"),
        }, indent=2))
    gpt = next(spec for spec in RUNS if spec["name"] == "gptoss")
    muse = next(spec for spec in RUNS if spec["name"] == "muse")
    leave_one_problem_out(gpt, "split_half_two", TWO_PROBLEM_RULE)
    muse_left_out = leave_one_problem_out(
        muse,
        "split_half_leave_one",
        "Muse train problems in the same even and odd halves as the five-problem split; "
        "each vector leaves one problem out of its half and is fit on the other four",
    )
    cosines = [row["cosine"] for row in muse_left_out["comparisons"]]
    print(json.dumps({
        "record": str(DATA / "stagec_muse_split_half_leave_one.json"),
        "n_comparisons": len(cosines),
        "cosine_min": min(cosines),
        "cosine_max": max(cosines),
        "comparisons": [
            {
                "left_out_even": row["left_out_even"],
                "left_out_odd": row["left_out_odd"],
                "even_pairs": row["even_pairs"],
                "odd_pairs": row["odd_pairs"],
                "cosine": row["cosine"],
            }
            for row in muse_left_out["comparisons"]
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
