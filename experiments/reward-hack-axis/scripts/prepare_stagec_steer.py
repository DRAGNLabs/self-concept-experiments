"""Wrap each Stage C axis for the steering harness and write the eval problems.

The axis file from the extraction is one vector. Steering wants a row per layer,
nonzero only at the layer it was measured. The eval set is the 45 schemer hard
problems minus the problems used to fit that axis.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from selfconcept.assistant_axis.projection import load_unit_axes

MIRROR = Path("/grphome/grp_mirror/nobackup/autodelete/self-concept-experiments/rh_hillclimb/data")
ROOT = Path.home() / "nobackup/autodelete/self-concept-experiments/reward-hack-axis"
SHARDS = ("ib_solvhard_s0", "ib_solvhard_s1", "ib_solvhard_s2")

RUNS = (
    {
        "name": "gptoss",
        "model": "openai/gpt-oss-120b",
        "layers": 36,
        "hidden": 2880,
        "layer": 18,
        "arms": (
            ("rh_plus", "--steer-mode add --steer-layer 18 --steer-alpha 0.2"),
            ("rh_minus", "--steer-mode add --steer-layer 18 --steer-alpha -0.2"),
        ),
    },
    {
        "name": "muse",
        "model": "meta-models/Muse-Glimmer-30B",
        "layers": 52,
        "hidden": 6656,
        "layer": 26,
        "arms": (
            ("rh_plus", "--steer-mode add --steer-layer 26 --steer-alpha 0.08"),
            ("rh_minus", "--steer-mode add --steer-layer 26 --steer-alpha -0.16"),
        ),
    },
)


def main() -> None:
    by_shard: dict[str, list[dict]] = {}
    for shard in SHARDS:
        by_shard[shard] = [json.loads(line) for line in (MIRROR / f"{shard}.jsonl").read_text().splitlines() if line.strip()]
    for spec in RUNS:
        summary_path = Path(__file__).resolve().parents[1] / "data" / f"stagec_{spec['name']}_summary.json"
        summary = json.loads(summary_path.read_text())
        train = set(summary["problems_train"])
        run_dir = ROOT / f"stagec_{spec['name']}"
        saved = torch.load(run_dir / "hack_axis.pt", map_location="cpu", weights_only=False)
        vector = saved["axis"].float()
        if vector.shape != (spec["hidden"],) or saved.get("layer") != spec["layer"]:
            raise SystemExit(f"{spec['name']}: expected layer {spec['layer']} vector {(spec['hidden'],)}, got {tuple(vector.shape)} layer {saved.get('layer')}")
        wrapped = torch.zeros(spec["layers"], spec["hidden"])
        wrapped[spec["layer"]] = vector
        axis_path = run_dir / "hack_axis_layers.pt"
        torch.save({"model": spec["model"], "axis": wrapped}, axis_path)
        load_unit_axes(axis_path, [spec["layer"]], spec["layers"], spec["hidden"], spec["model"])

        steer = run_dir / "steer"
        data = steer / "data"
        data.mkdir(parents=True, exist_ok=True)
        kept: list[str] = []
        scenarios: list[str] = []
        for shard, rows in by_shard.items():
            chosen = [row for row in rows if row["example_id"] not in train]
            if not chosen:
                continue
            scenarios.append(shard)
            kept.extend(row["example_id"] for row in chosen)
            with (data / f"{shard}.jsonl").open("w") as handle:
                for row in chosen:
                    handle.write(json.dumps(row) + "\n")
        (data / "scenarios.txt").write_text("\n".join(scenarios) + "\n")
        (steer / "arms.tsv").write_text("".join(f"{name}\t{extra}\n" for name, extra in spec["arms"]))
        dropped = sorted(train & {row["example_id"] for rows in by_shard.values() for row in rows})
        print(json.dumps({
            "model": spec["model"],
            "axis": str(axis_path),
            "eval_problems": len(kept),
            "excluded_train_problems": dropped,
            "scenarios": scenarios,
        }, indent=2))


if __name__ == "__main__":
    main()
