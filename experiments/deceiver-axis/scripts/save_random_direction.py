"""Save a seeded random unit direction as a steering artifact that applies the SAME vector at every layer.

``--random-direction-seed`` in the steering harnesses draws a different isotropic vector per layer
(``random_directions(hidden, 1, seed, layer)`` seeds the RNG with (seed, layer)), which is what an add arm at one layer
wants. A cap over a band of layers must clamp one direction, so this writes the layer-``--layer`` vector tiled over
``--num-layers`` rows in the ``{model, axis, note}`` format ``load_unit_axes`` accepts (as ``make_split_direction.py``
does for the split parts), and checks it against the harness's own draw and any ``--check name=path`` artifacts.

Example (Gemma 4 31B, direction 1729 at layer 30, the one whose add arms moved the hack rate):
  python scripts/save_random_direction.py --seed 1729 --layer 30 --model google/gemma-4-31B-it --num-layers 60 \
      --hidden 5376 --out <dir>/rand1729_L30_alllayers.pt --check pc50=<split>/rand1729_pc50.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from selfconcept.correlation.random_vectors import random_directions


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--layer", type=int, required=True, help="layer whose draw is saved at every row")
    ap.add_argument("--model", required=True)
    ap.add_argument("--num-layers", type=int, required=True)
    ap.add_argument("--hidden", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--check", action="append", default=[], help="name=path of a (layers, hidden) artifact to report cosines with")
    args = ap.parse_args()

    unit = random_directions(args.hidden, 1, args.seed, args.layer)[0].astype(np.float32)
    assert abs(float(np.linalg.norm(unit)) - 1.0) < 1e-5
    axis = torch.as_tensor(np.tile(unit, (args.num_layers, 1)))
    note = (f"random direction seed {args.seed} as drawn at layer {args.layer} (random_directions(hidden, 1, seed, layer)), "
            f"unit, tiled over all {args.num_layers} layers for cap bands; identical to --random-direction-seed {args.seed} "
            f"at --steer-layer {args.layer} only")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": args.model, "axis": axis, "note": note}, args.out)

    report = {"seed": args.seed, "layer": args.layer, "model": args.model, "num_layers": args.num_layers,
              "hidden": args.hidden, "out": str(args.out), "note": note,
              "differs_from_other_layers_draw": {str(l): float(1 - unit @ random_directions(args.hidden, 1, args.seed, l)[0])
                                                for l in (args.layer - 1, args.layer + 1)},
              "checks": {}}
    for pair in args.check:
        name, path = pair.split("=", 1)
        other = torch.load(path, map_location="cpu", weights_only=True)
        other = other["axis"] if isinstance(other, dict) else other
        row = other[args.layer].float().numpy()
        report["checks"][name] = float(unit @ (row / np.linalg.norm(row)))
    args.out.with_suffix(".json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
